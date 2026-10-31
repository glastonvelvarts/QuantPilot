"""Model facts and fit. Sizes come from the model's own tensor shapes, not from rules of thumb."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

from ocq import measurements
from ocq.engine import default_group_size
from ocq.gguf_file import (
    SPLIT_PART,
    GgufHeader,
    config_from_gguf,
    hf_style_tensors,
    quant_from_filename,
    read_local,
    read_remote,
)
from ocq.llmfit_bridge import LlmfitBridge, LlmfitError
from ocq.local_model import fetch_local_model_files, gguf_files, parse_model_ref
from ocq.model_size import (
    EMBEDDING,
    QUANT_METADATA,
    Tensor,
    bnb_scale_bits,
    bnb_weights_bytes,
    experts_in_names,
    gguf_tensor_bytes,
    gguf_weights_bytes,
    group_size_that_fits,
    kv_value_bytes,
    layer_index,
    nvfp4_weights_bytes,
    output_buffer_bytes,
    packed_weights_bytes,
    parameters_by_type,
    routed_expert,
    stored_bytes,
    weights_per_token,
)
from ocq.types import (
    CONVERSATIONS,
    DTYPE_BITS,
    INTEGER_TYPES,
    KV_BYTES_PER_VALUE,
    LAYER_SPLIT,
    LLAMA_CPP,
    ONE_GPU,
    SLIDING_WINDOW_UBATCH,
    TENSOR_PARALLEL,
    TRANSFORMERS,
    VLLM,
    Fact,
    FactKind,
    FeasibilityEstimate,
    FitLevel,
    HardwareProfile,
    KvLayer,
    ModelFacts,
    ModelSnapshot,
    OcqError,
    QuantAlgorithm,
    QuantConfig,
    QuantOption,
    RunMode,
)

HF_API = "https://huggingface.co/api/models"
VLLM_ROWS = (QuantAlgorithm.AWQ.value, QuantAlgorithm.GPTQ.value, QuantAlgorithm.NVFP4.value)
MODEL_CACHE = Path.home() / ".cache" / "ocq" / "models"
GB = 1024**3
DEFAULT_CONTEXT = 4096

UNQUANTIZED = ("F32", "BF16", "F16", "FP16")
QUANTIZED_LEVELS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")

# Rows shown after GGUF levels — packed size from tensor shapes + default group sizes.
# (backend, bits, double quantization)
HF_QUANT_ROWS = (
    (QuantAlgorithm.AWQ, 4, False),
    (QuantAlgorithm.GPTQ, 4, False),
    (QuantAlgorithm.GPTQ, 8, False),
    (QuantAlgorithm.BNB, 4, False),
    (QuantAlgorithm.BNB, 4, True),
    (QuantAlgorithm.BNB, 8, False),
)

# Only used when a repo has no readable tensor shapes. Computed with model_size.py on
# Qwen2.5-7B, so they are real averages for a typical model rather than nominal bit widths.
AVERAGE_BITS = {
    "F32": 32.0,
    "BF16": 16.0,
    "F16": 16.0,
    "FP16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.71,
    "Q5_K_S": 5.58,
    "Q5_0": 5.58,
    "Q4_K_M": 4.91,
    "Q4_K_S": 4.68,
    "Q4_0": 4.65,
    "Q3_K_L": 4.29,
    "Q3_K_M": 3.99,
    "Q3_K_S": 3.66,
    "Q2_K": 3.16,
}

ALREADY_QUANTIZED = re.compile(r"awq|gptq|gguf|int4|int8|4bit|8bit|bnb|mlx|exl2|fp8", re.IGNORECASE)

# Open, ungated models used for recommendations when llmfit is not available.
STARTER_MODELS = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": 0.135,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 0.36,
    "Qwen/Qwen2.5-0.5B-Instruct": 0.49,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.54,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1.71,
    "Qwen/Qwen2.5-3B-Instruct": 3.09,
    "microsoft/Phi-3.5-mini-instruct": 3.82,
    "Qwen/Qwen2.5-7B-Instruct": 7.62,
    "Qwen/Qwen2.5-14B-Instruct": 14.8,
    "Qwen/Qwen2.5-32B-Instruct": 32.8,
    "Qwen/Qwen2.5-72B-Instruct": 72.7,
}


def resolve_model(
    model_ref: str,
    hardware: HardwareProfile | None,
    llmfit_bin: str | None = None,
    context: int = DEFAULT_CONTEXT,
    kv_cache: str = "f16",
) -> ModelSnapshot:
    """Read a model's facts from its files. Without hardware, the fit is not computed."""
    display_id, local_dir = parse_model_ref(model_ref)
    if local_dir is not None and (local_dir.is_file() or gguf_files(local_dir)):
        if local_dir.is_file() or not (local_dir / "config.json").is_file():
            return local_gguf_snapshot(display_id, local_dir, hardware, context, kv_cache)
    if local_dir is not None:
        config, tensors, dtypes = fetch_local_model_files(local_dir)
        meta = {"pipeline_tag": "local"}
        snapshot = build_snapshot(
            display_id, meta, config, tensors, hardware, context, dtypes, kv_cache
        )
        return replace(
            snapshot,
            provider="local",
            use_case="Fine-tuned / local checkpoint",
            catalog_source="local",
            weights_dir=local_dir,
        )

    model_id = display_id
    catalog = catalog_entry(model_id, llmfit_bin)
    try:
        meta = fetch_hf_model(model_id)
    except OcqError:
        if catalog:
            return catalog
        raise

    if not meta.get("safetensors") and any(
        str(sibling.get("rfilename", "")).endswith(".gguf") for sibling in meta.get("siblings", [])
    ):
        return hub_gguf_snapshot(model_id, meta, hardware, context, kv_cache)
    config, tensors, dtypes = fetch_model_files(model_id, meta.get("sha", ""))
    if quantization_config(config):
        meta = meta | {"base_model_parameters": base_model_parameters(meta)}
    snapshot = build_snapshot(model_id, meta, config, tensors, hardware, context, dtypes, kv_cache)
    if catalog:
        snapshot = replace(
            snapshot,
            provider=catalog.provider or snapshot.provider,
            use_case=catalog.use_case,
            catalog_source="llmfit",
        )
    return snapshot


def base_model_parameters(meta: dict) -> tuple[str, int] | None:
    """(repo, count) of the model a quantized repo was made from, as the HF API reports it.

    A second source for the count of a quantized repo: the HF API cannot be trusted there,
    it counts bitsandbytes scales as weights and does not unpack NVFP4 weights.
    """
    base = (meta.get("cardData") or {}).get("base_model")
    if isinstance(base, list):
        base = base[0] if len(base) == 1 else None
    if not isinstance(base, str):
        return None
    try:
        total = parameters_from_hf(fetch_hf_model(base))
    except OcqError:
        return None
    return (base, total) if total else None


def catalog_entry(model_id: str, llmfit_bin: str | None) -> ModelSnapshot | None:
    try:
        _, snapshot = LlmfitBridge(llmfit_bin).model_fit(model_id)
    except LlmfitError:
        return None
    return snapshot if snapshot.model_id.lower() == model_id.lower() else None


def recommend_models(
    hardware: HardwareProfile,
    limit: int = 10,
    use_case: str | None = None,
    llmfit_bin: str | None = None,
) -> list[ModelSnapshot]:
    try:
        catalog = LlmfitBridge(llmfit_bin).recommend(limit * 8, use_case)
        originals = [m for m in catalog if not ALREADY_QUANTIZED.search(m.model_id)]
        if originals:
            return originals[:limit]
    except LlmfitError:
        pass

    starters = [
        build_snapshot(model_id, {"safetensors": {"total": params_b * 1e9}}, {}, [], hardware)
        for model_id, params_b in STARTER_MODELS.items()
    ]
    return starters[:limit]


def fetch_hf_model(model_id: str) -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": "ocq/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(f"{HF_API}/{model_id}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            raise OcqError(
                f"Model {model_id!r} was not found on HuggingFace. Check the repo id (org/name). "
                "If it is gated or private, accept its license and set HF_TOKEN."
            ) from exc
        raise OcqError(f"HuggingFace API error {exc.code} for {model_id}") from exc
    except urllib.error.URLError as exc:
        raise OcqError(f"Could not reach HuggingFace: {exc.reason}") from exc


def fetch_model_files(model_id: str, revision: str) -> tuple[dict, list[Tensor], dict[str, str]]:
    """config.json, every tensor's shape and stored type. Only file headers are read."""
    cached = MODEL_CACHE / f"{model_id.replace('/', '--')}.json"
    try:
        saved = json.loads(cached.read_text())
        if saved["revision"] == revision:
            tensors = [(name, tuple(shape)) for name, shape in saved["tensors"]]
            return saved["config"], tensors, saved["dtypes"]
    except (OSError, ValueError, KeyError):
        pass

    from huggingface_hub import get_safetensors_metadata, hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        NotASafetensorsRepoError,
        SafetensorsParsingError,
    )
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    try:
        config = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text("utf-8"))
    except (OSError, ValueError, EntryNotFoundError):
        config = {}

    disable_progress_bars()
    try:
        metadata = get_safetensors_metadata(model_id)
        headers = [
            (name, info)
            for file in metadata.files_metadata.values()
            for name, info in file.tensors.items()
        ]
    except (OSError, NotASafetensorsRepoError, SafetensorsParsingError):
        headers = []
    finally:
        enable_progress_bars()
    tensors = [(name, tuple(info.shape)) for name, info in headers]
    dtypes = {name: info.dtype for name, info in headers}

    if config and tensors:
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        saved = {"revision": revision, "config": config, "tensors": tensors, "dtypes": dtypes}
        cached.write_text(json.dumps(saved))
    return config, tensors, dtypes


def build_snapshot(
    model_id: str,
    meta: dict,
    config: dict,
    tensors: list[Tensor],
    hardware: HardwareProfile | None,
    context: int = DEFAULT_CONTEXT,
    dtypes: dict[str, str] | None = None,
    kv_cache: str = "f16",
) -> ModelSnapshot:
    facts = model_facts(model_id, meta, config, tensors, dtypes or {})
    parameters = facts.parameters.value or 0
    hint = parse_params_from_name(model_id)

    snapshot = ModelSnapshot(
        model_id=model_id,
        provider=model_id.split("/")[0] if "/" in model_id else "unknown",
        parameter_count=(
            format_params(parameters / 1e9)
            if parameters
            else f"unknown (name says {format_params(hint)})"
            if hint
            else "unknown"
        ),
        params_b=parameters / 1e9,
        context_length=facts.context_length.value,
        use_case=str(meta.get("pipeline_tag") or "General"),
        is_moe=bool(experts_in_names(tensors) or expert_settings(config)[0]),
        experts=experts_in_names(tensors) or expert_settings(config)[0],
        experts_per_token=expert_settings(config)[1],
        tensors=tuple(tensors),
        catalog_source="huggingface",
        facts=facts,
        kv_layers=kv_layers_from(config)[0],
        raw=meta,
        **(architecture_from(config) or vocabulary_from_tensors(tensors)),
    )
    if not parameters:
        # No real count, so no fit verdict.
        unknown = replace(snapshot.feasibility, fit_level="unknown", run_mode="unknown")
        return replace(snapshot, feasibility=unknown)
    if hardware is None:
        return snapshot
    return with_fit(snapshot, hardware, context, kv_cache)


def with_fit(
    snapshot: ModelSnapshot, hardware: HardwareProfile, context: int, kv_cache: str = "f16"
) -> ModelSnapshot:
    """The snapshot with the best row of its fit table as its feasibility."""
    best = best_option(quant_options(snapshot, hardware, context, kv_cache))
    return replace(
        snapshot,
        llmfit_best_quant=best.label,
        feasibility=FeasibilityEstimate(
            quant_label=best.label,
            fit_level=best.fit_level,
            run_mode=best.run_mode,
            score=0.0,
            estimated_tps=best.estimated_tps,
            memory_required_gb=best.memory_required_gb,
            memory_available_gb=best.memory_available_gb,
            utilization_pct=best.utilization_pct,
            runtime=runtime_for_row(best),
            estimate_confidence=best.speed_basis,
        ),
    )


def gguf_snapshot(
    model_id: str,
    meta: dict,
    header: GgufHeader,
    files: list[tuple[str, int]],
    file_size: int | None,
    whole: bool,
) -> ModelSnapshot:
    """A GGUF model's facts, read from one file's header. files: (quantization, tensor bytes)
    of every file of the model. whole: the header lists all tensors (not one part of a split).
    The settings become config.json's terms, so the same code sizes the context memory."""
    config = config_from_gguf(header)
    tensors = hf_style_tensors(header) if whole else []
    snapshot = build_snapshot(model_id, meta, config, tensors, None)
    stored = header.data_bytes
    by_type: dict[str, int] = {}
    for tensor in header.tensors:
        by_type[tensor.type_name] = by_type.get(tensor.type_name, 0) + tensor.elements
    ranked = sorted(by_type, key=by_type.get, reverse=True)
    precision = Fact(
        ranked[0] if ranked else None,
        "GGUF tensor table",
        FactKind.EXACT if ranked else FactKind.UNKNOWN,
        ", ".join(f"{name} {format_params(by_type[name] / 1e9)}" for name in ranked),
    )
    stored_fact = Fact()
    if whole and stored is not None:
        matches = file_size == header.data_start + stored
        stored_fact = Fact(
            stored,
            "GGUF tensor table + file size" if matches else "GGUF tensor table",
            FactKind.CROSS_CHECKED if matches else FactKind.EXACT,
            f"{stored / GB:.2f} GiB of tensors"
            + ("; with the header it is the file's exact size" if matches else ""),
        )
    labels = ", ".join(label for label, _ in files)
    quantization = Fact(
        f"GGUF {header.file_type or files[0][0]}",
        "GGUF general.file_type",
        FactKind.EXACT,
        f"{len(files)} file{'s' if len(files) > 1 else ''}: {labels}",
    )
    facts = replace(
        snapshot.facts, precision=precision, stored_bytes=stored_fact, quantization=quantization
    )
    # The shared code names safetensors and config.json as sources; here they are the GGUF's.
    renamed = {"safetensors headers": "GGUF tensor table", "config.json": "GGUF settings"}

    def from_gguf(fact: Fact) -> Fact:
        source, note = fact.source, fact.note
        for old, new in renamed.items():
            source, note = source.replace(old, new), note.replace(old, new)
        return replace(fact, source=source, note=note)

    facts = ModelFacts(
        **{name: from_gguf(getattr(facts, name)) for name in facts.__dataclass_fields__}
    )
    exact_sizes = {}
    if whole:
        for (name, _), tensor in zip(tensors, header.tensors, strict=True):
            exact_sizes[name] = tensor.nbytes or 0
    return replace(
        snapshot,
        facts=facts,
        gguf_files=tuple(files),
        gguf_tensor_bytes=exact_sizes,
        use_case=str(meta.get("pipeline_tag") or "General"),
    )


def gguf_variants(
    names_and_sizes: list[tuple[str, int]], header_bytes: int
) -> list[tuple[str, int]]:
    """One (quantization, tensor bytes) per GGUF variant: split parts added up, mmproj left out.
    The tensor bytes are the file size less the header, which all variants share."""
    variants: dict[str, int] = {}
    for name, size in names_and_sizes:
        base = Path(name).name
        if base.lower().startswith("mmproj") or not base.endswith(".gguf"):
            continue
        key = SPLIT_PART.sub(".gguf", base)
        variants[key] = variants.get(key, 0) + size - header_bytes
    labels: dict[str, int] = {}
    # Largest first, like the other tables: the best fit is the first comfortable row.
    for key, size in sorted(variants.items(), key=lambda item: item[1], reverse=True):
        label = quant_from_filename(key) or Path(key).stem
        if label in labels:  # two files with the same quantization in their names
            label = Path(key).stem
        labels[label] = size
    return list(labels.items())


def pick_gguf_file(names: list[str]) -> str:
    """The file whose header to read: Q4_K_M if there is one, else the first of the smallest
    variant. All files of a model share the settings; only the tensor types differ."""
    models = [
        n for n in names if n.endswith(".gguf") and not Path(n).name.lower().startswith("mmproj")
    ]
    if not models:
        raise OcqError("No GGUF model file found (only mmproj vision projectors).")
    preferred = [n for n in models if quant_from_filename(n) == "Q4_K_M"]
    firsts = [n for n in (preferred or models) if not SPLIT_PART.search(n) or "-00001-of-" in n]
    return sorted(firsts or models)[0]


def local_gguf_snapshot(
    display_id: str, path: Path, hardware: HardwareProfile | None, context: int, kv_cache: str
) -> ModelSnapshot:
    files = [path] if path.is_file() else gguf_files(path)
    if path.is_file() and SPLIT_PART.search(path.name):
        files = sorted(path.parent.glob(SPLIT_PART.sub("-*-of-*.gguf", path.name)))
    # An Ollama blob is a GGUF file without the .gguf name.
    named = path.is_file() and path.suffix.lower() == ".gguf"
    chosen = Path(pick_gguf_file([str(f) for f in files])) if named or path.is_dir() else path
    header = read_local(chosen)
    whole = not SPLIT_PART.search(chosen.name)
    variants = gguf_variants([(str(f), f.stat().st_size) for f in files], header.data_start)
    if not variants:
        variants = [(header.file_type or "GGUF", chosen.stat().st_size - header.data_start)]
    snapshot = gguf_snapshot(
        display_id, {"pipeline_tag": "local"}, header, variants, chosen.stat().st_size, whole
    )
    snapshot = replace(
        snapshot,
        provider="local",
        use_case="Local GGUF",
        catalog_source="local",
        weights_dir=path,
    )
    return (
        with_fit(snapshot, hardware, context, kv_cache)
        if hardware and snapshot.params_b
        else snapshot
    )


def hub_gguf_snapshot(
    model_id: str, meta: dict, hardware: HardwareProfile | None, context: int, kv_cache: str
) -> ModelSnapshot:
    """A Hub repo with only GGUF files: one header read with range requests, the file sizes
    from the Hub's file list."""
    sizes = hub_file_sizes(model_id)
    chosen = pick_gguf_file(list(sizes))
    revision = meta.get("sha") or "main"
    header = read_remote(f"https://huggingface.co/{model_id}/resolve/{revision}/{chosen}")
    whole = not SPLIT_PART.search(chosen)
    variants = gguf_variants(list(sizes.items()), header.data_start)
    snapshot = gguf_snapshot(model_id, meta, header, variants, sizes.get(chosen), whole)
    return (
        with_fit(snapshot, hardware, context, kv_cache)
        if hardware and snapshot.params_b
        else snapshot
    )


def hub_file_sizes(model_id: str) -> dict[str, int]:
    """Every file of a Hub repo and its size in bytes."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": "ocq/0.1"} | ({"Authorization": f"Bearer {token}"} if token else {})
    request = urllib.request.Request(f"{HF_API}/{model_id}?blobs=true", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            siblings = json.loads(response.read().decode()).get("siblings", [])
    except (urllib.error.URLError, ValueError) as exc:
        raise OcqError(f"Could not list the files of {model_id}") from exc
    return {s["rfilename"]: s.get("size") or 0 for s in siblings}


def runtime_for_row(option: QuantOption) -> str:
    if option.row_kind == "gguf":
        return "llama.cpp"
    if option.row_kind == "stored":
        return "transformers (full precision)"
    return "vLLM / transformers"


# Where vision and multimodal models (Gemma 3, Llama 4, Idefics, LLaVA, InternVL) keep the
# settings of their language model.
NESTED_CONFIGS = ("text_config", "llm_config", "language_config")


def text_config(config: dict) -> tuple[dict, str]:
    """The settings of the language model, and the config.json key they were under ("" = top).

    A nested block is merged over the top level, so values only found at the top still count.
    """
    if config.get("num_hidden_layers") or config.get("n_layer"):
        return config, ""
    for key in NESTED_CONFIGS:
        nested = config.get(key)
        if isinstance(nested, dict) and (nested.get("num_hidden_layers") or nested.get("n_layer")):
            return config | {k: v for k, v in nested.items() if v is not None}, key
    return config, ""


def architecture_from(config: dict) -> dict:
    text, _ = text_config(config)
    layers = text.get("num_hidden_layers") or text.get("n_layer")
    hidden = text.get("hidden_size") or text.get("n_embd") or text.get("d_model")
    heads = text.get("num_attention_heads") or text.get("n_head")
    vocab = text.get("vocab_size")
    if not (layers and hidden and heads and vocab):
        return {}
    return {
        "architecture": (config.get("architectures") or [config.get("model_type", "model")])[0],
        "num_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_kv_heads": text.get("num_key_value_heads") or heads,
        "vocab_size": vocab,
        "attention_head_dim": text.get("head_dim") or 0,
    }


def model_facts(
    model_id: str, meta: dict, config: dict, tensors: list[Tensor], dtypes: dict[str, str]
) -> ModelFacts:
    """Everything known about the model, each fact with its source and how sure it is."""
    quantization, bits = quantization_fact(config, tensors, dtypes)
    by_type = parameters_by_type(tensors, dtypes, bits) if tensors else None
    copy = lm_head_copy(config, tensors)
    parameters = parameter_fact(meta, model_id, tensors, by_type, bits, copy)
    return ModelFacts(
        parameters=parameters,
        active_parameters=active_fact(config, tensors, dtypes, bits, parameters),
        precision=precision_fact(meta, by_type, bits),
        stored_bytes=stored_bytes_fact(meta, tensors, dtypes, bits),
        architecture=architecture_fact(config, tensors),
        context_length=context_fact(config, meta),
        kv_cache=kv_cache_fact(config, tensors),
        quantization=quantization,
    )


def quantization_config(config: dict) -> dict:
    text, _ = text_config(config)
    return text.get("quantization_config") or text.get("compression_config") or {}


def quantized_bits(settings: dict) -> int:
    """Bit width of the quantized weights. 0 when quantization_config does not say."""
    if settings.get("load_in_4bit"):
        return 4
    if settings.get("load_in_8bit"):
        return 8
    for key in ("bits", "w_bit", "weight_bits"):
        if isinstance(settings.get(key), int):
            return settings[key]
    for group in (settings.get("config_groups") or {}).values():
        bits = (group.get("weights") or {}).get("num_bits")
        if bits:
            return bits
    return {"fp8": 8, "mxfp4": 4}.get(str(settings.get("quant_method", "")).lower(), 0)


METHOD_NAMES = {
    "modelopt": "ModelOpt",
    "gptq": "GPTQ",
    "awq": "AWQ",
    "bitsandbytes": "BNB",
    "fp8": "FP8",
    "mxfp4": "MXFP4",
    "compressed-tensors": "compressed-tensors",
}


def quantization_label(settings: dict, bits: int) -> str:
    method = str(settings.get("quant_method") or "")
    name = METHOD_NAMES.get(method.lower(), method or "unknown method")
    if name in ("FP8", "MXFP4"):
        return f"{name} {settings.get('fmt', '')}".strip()
    if settings.get("quant_algo"):
        return f"{name} {settings['quant_algo']}"
    if name == "compressed-tensors" and settings.get("format"):
        name += f" {settings['format']}"
    return f"{name} {bits}-bit" if bits else name


def is_quantization_tensor(name: str, dtype: str) -> bool:
    return (
        bool(QUANT_METADATA.search(name))
        or name.endswith((".qweight", ".weight_packed"))
        or dtype.startswith(("F8", "F6", "F4"))
    )


def quantization_fact(
    config: dict, tensors: list[Tensor], dtypes: dict[str, str]
) -> tuple[Fact, int | None]:
    """Whether the weights are already quantized, and their bit width (None when they are not).

    config.json says so in quantization_config, and the tensors show it: packed integers,
    scales and zero points next to the weights, or 8-bit float weights.
    """
    settings = quantization_config(config)
    found = sum(is_quantization_tensor(name, dtypes.get(name, "")) for name, _ in tensors)
    if settings:
        bits = quantized_bits(settings)
        label = quantization_label(settings, bits)
        note = f"group size {settings['group_size']}" if settings.get("group_size") else ""
        if settings.get("weight_block_size"):
            note = "blocks of " + " x ".join(map(str, settings["weight_block_size"])) + " weights"
        source = "config.json quantization_config"
        if found:
            return Fact(label, f"{source} + tensor names", FactKind.CROSS_CHECKED, note), bits
        if not tensors:
            return Fact(label, source, FactKind.EXACT, note), bits
        note = "; ".join(filter(None, [note, "but no quantized tensors in the weight files"]))
        return Fact(label, source, FactKind.ESTIMATE, note), bits
    if found:
        note = f"{found} quantization tensors, but no quantization_config in config.json"
        return Fact("quantized (method not stated)", "tensor names", FactKind.ESTIMATE, note), 0
    if config and tensors:
        source, kind = "config.json + safetensors headers", FactKind.CROSS_CHECKED
    elif config or tensors:
        source, kind = ("config.json" if config else "safetensors headers"), FactKind.EXACT
    else:
        return Fact(), None
    return Fact("none", source, kind, "no quantization_config, plain weights"), None


EXPERT_COUNT = ("num_experts", "num_local_experts", "n_routed_experts")
EXPERTS_PER_TOKEN = ("num_experts_per_tok", "experts_per_token", "moe_topk")


def expert_settings(config: dict) -> tuple[int, int]:
    """(routed experts per layer, experts each token uses) as config.json states them."""
    text, _ = text_config(config)
    experts = next((text[key] for key in EXPERT_COUNT if text.get(key)), 0)
    per_token = next((text[key] for key in EXPERTS_PER_TOKEN if text.get(key)), 0)
    return experts, per_token


def active_fact(
    config: dict, tensors: list[Tensor], dtypes: dict[str, str], bits: int | None, total: Fact
) -> Fact:
    """Weights read for one token. The speed limit depends on this, not on the total.

    A mixture-of-experts model holds every expert in memory but runs only a few per token.
    """
    configured, per_token = expert_settings(config)
    named = experts_in_names(tensors)
    if not tensors:
        note = f"{configured} experts, but no tensor headers to count them" if configured else ""
        return Fact(note=note)
    experts = named or configured
    if experts and not per_token:
        return Fact(note=f"{experts} experts, but config.json does not say how many run per token")
    counted = weights_per_token(tensors, dtypes, bits, experts, per_token)
    if counted is None:
        return Fact(note="packed tensors could not be unpacked")

    read, embedding = counted
    parts = [f"{per_token} of {experts} experts per token" if experts else "dense"]
    if embedding:
        parts.append(f"input embedding left out: {format_params(embedding / 1e9)}, one row read")
    if total.value:
        parts.append(f"{read / total.value:.0%} of the weights")

    source, kind = "tensor names", FactKind.EXACT
    if configured and configured != named:
        kind = FactKind.ESTIMATE
        shown = named or "none"
        parts.append(f"config.json says {configured} experts, the tensor names show {shown}")
    elif config and (configured == named):
        source, kind = "config.json + tensor names", FactKind.CROSS_CHECKED
    return Fact(read, source, kind, ", ".join(parts))


def lm_head_copy(config: dict, tensors: list[Tensor]) -> int:
    """Weights of an lm_head the files store although config.json ties it to the embeddings."""
    text, _ = text_config(config)
    if not text.get("tie_word_embeddings", True):
        return 0
    return sum(math.prod(shape) for name, shape in tensors if name.endswith("lm_head.weight"))


def parameter_fact(
    meta: dict,
    model_id: str,
    tensors: list[Tensor],
    by_type: dict[str, int] | None,
    bits: int | None,
    copy: int = 0,
) -> Fact:
    """Our count from the tensor headers, checked against the HF API and, for a quantized repo,
    against the count of the model it was made from."""
    hf_total = parameters_from_hf(meta)
    base = meta.get("base_model_parameters")
    if not by_type:
        if hf_total:
            return Fact(hf_total, "HF API", FactKind.EXACT, "tensor headers not readable")
        hint = parse_params_from_name(model_id)
        return Fact(note=f"name says {format_params(hint)}, not used" if hint else "")

    counted = sum(by_type.values())
    notes = [f"{len(tensors)} tensors"]
    if bits is not None:
        stored = sum(math.prod(shape) for _, shape in tensors)
        unpacked = any("-bit in " in dtype for dtype in by_type)
        what = "packed weights unpacked, scales left out" if unpacked else "scales left out"
        notes[0] += f", {what} ({stored:,} stored values)"

    sources = []
    if hf_total == counted:
        sources.append("HF API")
        notes.append("matches HF API total")
    elif hf_total:
        notes.append(f"HF API says {hf_total:,}")
    if base and base[1] in (counted, counted - copy):
        sources.append("base model")
        extra = f" plus a stored lm_head copy of {copy:,}" if base[1] != counted else ""
        notes.append(f"matches base model {base[0]}{extra}")
    elif base:
        notes.append(f"base model {base[0]} has {base[1]:,}")

    if sources:
        source = " + ".join(["safetensors headers", *sources])
        return Fact(counted, source, FactKind.CROSS_CHECKED, "; ".join(notes))
    # Unpacking quantized weights is our own reading, so on its own it is an estimate.
    kind = FactKind.EXACT if bits is None else FactKind.ESTIMATE
    return Fact(counted, "safetensors headers", kind, "; ".join(notes))


def precision_fact(meta: dict, by_type: dict[str, int] | None, bits: int | None) -> Fact:
    """The type most weights are stored in, from the headers, checked against the HF API."""
    hf_counts = dict((meta.get("safetensors") or {}).get("parameters") or {})
    if bits is not None:
        # The HF API counts packed weights unpacked, under their integer type.
        hf_counts = {
            f"{bits}-bit in {dtype}" if dtype in INTEGER_TYPES else dtype: count
            for dtype, count in hf_counts.items()
        }
    if by_type and "" not in by_type:
        counts, source = by_type, "safetensors headers"
        kind = FactKind.CROSS_CHECKED if counts == hf_counts else FactKind.EXACT
        if kind == FactKind.CROSS_CHECKED:
            source += " + HF API"
    elif hf_counts:
        counts, source, kind = hf_counts, "HF API", FactKind.EXACT
    else:
        return Fact()
    ranked = sorted(counts, key=counts.get, reverse=True)
    note = ", ".join(f"{dtype} {format_params(counts[dtype] / 1e9)}" for dtype in ranked)
    return Fact(ranked[0], source, kind, note)


def stored_bytes_fact(
    meta: dict, tensors: list[Tensor], dtypes: dict[str, str], bits: int | None
) -> Fact:
    """Bytes of the weight files' tensors. A type we do not know makes it unknown, not a guess."""
    if tensors and dtypes:
        total = stored_bytes(tensors, dtypes)
        if total is not None:
            return Fact(total, "safetensors headers", FactKind.EXACT, f"{total / GB:.2f} GiB")
        unknown = sorted({dtypes.get(name, "?") for name, _ in tensors} - DTYPE_BITS.keys())
        note = f"unknown tensor type {', '.join(unknown)}"
        return Fact(source="safetensors headers", note=note)

    counts = (meta.get("safetensors") or {}).get("parameters") or {}
    if not counts:
        return Fact()
    unknown = sorted(set(counts) - DTYPE_BITS.keys())
    if unknown:
        return Fact(source="HF API", note=f"unknown tensor type {', '.join(unknown)}")
    if bits is not None and any(dtype in INTEGER_TYPES for dtype in counts):
        return Fact(source="HF API", note="HF API counts packed weights unpacked")
    total = -(-sum(count * DTYPE_BITS[dtype] for dtype, count in counts.items()) // 8)
    # The HF API leaves quantization scales out, so for quantized weights this runs short.
    kind = FactKind.EXACT if bits is None else FactKind.ESTIMATE
    return Fact(total, "HF API parameter counts", kind, f"{total / GB:.2f} GiB")


LAYER_NAME = re.compile(r"^(.*?)\.(?:layers|h|blocks)\.(\d+)\.")
NOT_LANGUAGE_MODEL = re.compile(r"vision|visual|audio|image|speech|mm_projector")


def layers_in_names(tensors: list[Tensor]) -> int | None:
    """Layers of the language model, counted from tensor names such as model.layers.27.mlp.

    None when the names show no single language-model stack (vision towers are left out).
    """
    stacks: dict[str, set[int]] = {}
    for name, _ in tensors:
        match = LAYER_NAME.match(name)
        if match and not NOT_LANGUAGE_MODEL.search(match.group(1)):
            stacks.setdefault(match.group(1), set()).add(int(match.group(2)))
    return len(next(iter(stacks.values()))) if len(stacks) == 1 else None


KV_PROJECTION = re.compile(
    r"^(.*?)\.(?:layers|h|blocks)\.\d+\.(?:self_attn|attention|attn|mixer)\.[kv]_proj\.weight(?:_packed)?$"
)


def attention_from_tensors(tensors: list[Tensor]) -> tuple[int, int, int] | None:
    """(K+V values per token over all layers, layers with attention, vocabulary) from shapes.

    The K and V projections write keys and values, so their output width (first dimension,
    also in llm-compressor's weight_packed) is what each layer caches per token. Summed over
    every layer, so models where only some layers have attention come out right too.
    None when the names do not show separate k_proj and v_proj (fused qkv, MLA, Mamba) or
    bitsandbytes flattened them.
    """
    flattened = {name.split(".weight.")[0] + ".weight" for name, _ in tensors if ".weight." in name}
    width, layers, vocab = 0, set(), None
    for name, shape in tensors:
        match = KV_PROJECTION.match(name)
        if match and not NOT_LANGUAGE_MODEL.search(match.group(1)):
            if len(shape) != 2 or name in flattened:
                return None
            width += shape[0]
            layers.add(LAYER_NAME.match(name).group(2))
        elif name.endswith(EMBEDDING) and not NOT_LANGUAGE_MODEL.search(name) and len(shape) == 2:
            vocab = shape[0]
    return (width, len(layers), vocab) if width and vocab else None


def vocabulary_from_tensors(tensors: list[Tensor]) -> dict:
    shapes = attention_from_tensors(tensors)
    return {"vocab_size": shapes[2]} if shapes else {}


# What a layer keeps for the context. Only attention layers keep keys and values for every token;
# Mamba, linear-attention and convolution layers keep a small state of fixed size instead.
FULL, SLIDING, CHUNKED = "full", "sliding", "chunked"
STATE = "state"  # Mamba, linear-attention or convolution layer: a fixed-size state, no cache
NO_CACHE = "no cache"  # a plain MLP layer (Nemotron-H "-"): neither
LAYER_TYPE_KINDS = {
    "full_attention": FULL,
    "attention": FULL,
    "sliding_attention": SLIDING,
    "chunked_attention": CHUNKED,
    "mamba": STATE,
    "mamba2": STATE,
    "linear_attention": STATE,
    "conv": STATE,
    "mlp": NO_CACHE,  # a layer with neither (from GGUF files: no KV heads, no SSM tensors)
}
# Nemotron-H's hybrid_override_pattern: attention, Mamba, MLP, MoE MLP.
PATTERN_KINDS = {"*": FULL, "M": STATE, "-": NO_CACHE, "E": NO_CACHE}


def known_kinds(names: list[str], table: dict[str, str]) -> tuple[list[str], list[str]]:
    """Layer kinds by name, and the names not in the table. An unknown name is counted as full
    attention: if it is not, the memory is overestimated, never underestimated."""
    unknown = sorted({name for name in names if name not in table})
    return [table.get(name, FULL) for name in names], unknown


def layer_kinds(text: dict, layers: int) -> tuple[list[str], str, list[str]]:
    """The kind of every layer and which config.json key said so. Families write it differently:

    layer_types               one name per layer (Gemma 3, gpt-oss, Granite 4.0-H, ...)
    full_attn_idxs            the attention layers by number, the rest convolutions (LFM2)
    full_attention_interval   every Nth layer attention, the rest linear attention (Qwen3-Next)
    hybrid_override_pattern   a letter per layer, * attention, M Mamba, - MLP (Nemotron-H)
    attn_layer_period/offset  attention at offset, offset + period, ... (Jamba)
    no_rope_layers            1 chunked attention, 0 full (Llama 4)
    sliding_window_pattern    every Nth layer full, the rest sliding (Gemma 3, Cohere2)
    Gemma 2 alternates sliding and full layers without saying so (as transformers does).
    The third value lists layer kinds config.json names that QuantPilot does not know.
    """
    types = text.get("layer_types")
    if isinstance(types, list) and len(types) == layers:
        kinds, unknown = known_kinds(types, LAYER_TYPE_KINDS)
        return kinds, "layer_types", unknown
    if isinstance(text.get("full_attn_idxs"), list):
        attention = set(text["full_attn_idxs"])
        kinds = [FULL if index in attention else STATE for index in range(layers)]
        return kinds, "full_attn_idxs", []
    if text.get("full_attention_interval"):
        interval = text["full_attention_interval"]
        kinds = [STATE if (index + 1) % interval else FULL for index in range(layers)]
        return kinds, "full_attention_interval", []
    pattern = text.get("hybrid_override_pattern")
    if isinstance(pattern, str) and len(pattern) == layers:
        kinds, unknown = known_kinds(list(pattern), PATTERN_KINDS)
        return kinds, "hybrid_override_pattern", unknown
    if text.get("attn_layer_period"):
        period, offset = text["attn_layer_period"], text.get("attn_layer_offset", 0)
        kinds = [FULL if index % period == offset else STATE for index in range(layers)]
        return kinds, "attn_layer_period", []
    no_rope = text.get("no_rope_layers")
    if text.get("attention_chunk_size") and isinstance(no_rope, list) and len(no_rope) >= layers:
        kinds = [CHUNKED if no_rope[index] else FULL for index in range(layers)]
        return kinds, "no_rope_layers", []
    if text.get("sliding_window_pattern"):
        every = text["sliding_window_pattern"]
        kinds = [SLIDING if (index + 1) % every else FULL for index in range(layers)]
        return kinds, "sliding_window_pattern", []
    if text.get("model_type") == "gemma2":
        return [SLIDING if index % 2 == 0 else FULL for index in range(layers)], "Gemma 2", []
    if isinstance(types, list):
        # A layer_types list of the wrong length does not say which layer is which.
        return [FULL] * layers, "", [f"layer_types lists {len(types)} of {layers} layers"]
    if text.get("sliding_window"):
        return [SLIDING] * layers, "sliding_window", []
    return [FULL] * layers, "", []


def layer_windows(text: dict, layers: int) -> tuple[list[int | None], list[str], str, list[str]]:
    """Tokens each layer keeps per conversation (0 = all, None = no cache) and a note.

    llama.cpp only shrinks the cache of models that mix windowed and full layers (Gemma 2/3,
    gpt-oss, Cohere2, Llama 4, ...). A window on every layer is counted in full: llama.cpp keeps
    Mistral 7B v0.1's whole cache, and for Phi-3 it sets swa_type NONE ("Phi SWA is currently
    disabled", src/models/phi3.cpp).
    """
    kinds, said_by, unknown = layer_kinds(text, layers)
    window = text.get("sliding_window") or 0
    if text.get("use_sliding_window") is False:
        window = 0
    chunk = text.get("attention_chunk_size") or 0
    windows: list[int | None] = []
    for kind in kinds:
        if kind in (STATE, NO_CACHE):
            windows.append(None)
        elif kind == SLIDING:
            windows.append(window)
        elif kind == CHUNKED:
            windows.append(chunk)
        else:
            windows.append(0)

    notes = []
    attention = [w for w in windows if w is not None]
    if attention and all(attention):
        notes.append(f"a {attention[0]:,}-token window on every layer, counted in full")
        windows = [None if w is None else 0 for w in windows]
    sliding = sum(1 for kind, w in zip(kinds, windows, strict=True) if kind == SLIDING and w)
    chunked = sum(1 for kind, w in zip(kinds, windows, strict=True) if kind == CHUNKED and w)
    if sliding:
        notes.append(f"{sliding} of {layers} layers keep only the last {window:,} tokens")
    if chunked:
        notes.append(f"{chunked} of {layers} layers attend within {chunk:,}-token chunks")
    if said_by and any(kind in (STATE, NO_CACHE) for kind in kinds):
        notes.append(f"attention layers from {said_by}")
    if unknown:
        notes.append(f"unknown layer kind {', '.join(unknown)}, counted as full attention")
    return windows, kinds, "; ".join(notes), unknown


def state_per_layer(text: dict) -> tuple[int, str]:
    """Bytes of the fixed-size state one Mamba or convolution layer keeps per conversation, and
    which kind. (0, "") when config.json does not describe one we know.

    As llama.cpp keeps it, in 32 bits (checked against its "llama_memory_recurrent" log for
    Granite 4.0-H, Falcon-H1, Jamba and LFM2):
      Mamba-2   conv (d_conv - 1) x (d_inner + 2 x groups x d_state) + ssm d_state x d_inner
      Mamba-1   conv (d_conv - 1) x d_inner                          + ssm d_state x d_inner
      LFM2 conv (L_cache - 1) x hidden
    Linear attention (Qwen3-Next) is not covered.
    """

    def first(*keys, default=0):
        return next((text[key] for key in keys if text.get(key)), default)

    if text.get("conv_L_cache"):
        return (text["conv_L_cache"] - 1) * text["hidden_size"] * 4, "convolution"
    d_state = first("mamba_d_state", "ssm_state_size", "mamba_state_dim")
    if not d_state:
        return 0, ""
    d_conv = first("mamba_d_conv", "conv_kernel", default=4)
    heads = first("mamba_n_heads", "mamba_num_heads")
    if heads:
        d_inner = first("mamba_d_ssm") or heads * first("mamba_d_head", "mamba_head_dim")
        groups = first("mamba_n_groups", "n_groups", "mamba_num_groups", default=1)
        conv = (d_conv - 1) * (d_inner + 2 * groups * d_state)
        return (conv + d_state * d_inner) * 4, "Mamba-2"
    d_inner = first("mamba_expand", default=2) * text["hidden_size"]
    return ((d_conv - 1) * d_inner + d_state * d_inner) * 4, "Mamba-1"


def kv_layers_from(config: dict) -> tuple[tuple[KvLayer, ...], str, bool]:
    """What every layer caches per token, from config.json, a note on anything unusual, and
    whether a layer kind had to be guessed."""
    found = architecture_from(config)
    if not found:
        return (), "", False
    text, _ = text_config(config)
    layers_count = found["num_layers"]
    windows, kinds, note, unknown = layer_windows(text, layers_count)
    state, state_kind = state_per_layer(text)
    stateful = [kind == STATE for kind in kinds]
    if state_kind.startswith("Mamba") and not any(stateful):
        # Falcon-H1: attention and Mamba side by side, so every layer has a cache and a state.
        stateful = [True] * layers_count
    states = [state if has_state else 0 for has_state in stateful]
    if any(stateful):
        count = f"{sum(stateful)} of {layers_count} layers"
        if state:
            kept = f"{count} keep a {state_kind} state of {state / 2**20:.2f} MiB per conversation"
        else:
            kept = f"{count} keep a fixed-size state of unknown size instead (not counted)"
        note = "; ".join(filter(None, [note, kept]))
    if text.get("kv_lora_rank"):
        # MLA (DeepSeek V2/V3, Kimi): one compressed vector plus the rotary part per token.
        rope = text.get("qk_rope_head_dim", 0)
        compressed = (text["kv_lora_rank"] + rope) * KV_BYTES_PER_VALUE
        heads = found["num_attention_heads"]
        expanded = heads * (text.get("qk_nope_head_dim", 0) + rope + text.get("v_head_dim", 0))
        expanded *= KV_BYTES_PER_VALUE
        mla = (
            f"MLA: {compressed // KV_BYTES_PER_VALUE} compressed values per layer in llama.cpp "
            f"and vLLM, {expanded // KV_BYTES_PER_VALUE} in transformers"
        )
        layers = tuple(
            KvLayer(0, state_bytes=extra)
            if window is None
            else KvLayer(compressed, window, expanded, extra)
            for window, extra in zip(windows, states, strict=True)
        )
        return layers, "; ".join(filter(None, [note, mla])), bool(unknown)
    head_dim = found["attention_head_dim"] or found["hidden_size"] // found["num_attention_heads"]
    per_layer = 2 * found["num_kv_heads"] * head_dim * KV_BYTES_PER_VALUE  # keys and values
    layers = tuple(
        KvLayer(0, state_bytes=extra)
        if window is None
        else KvLayer(per_layer, window, state_bytes=extra)
        for window, extra in zip(windows, states, strict=True)
    )
    return layers, note, bool(unknown)


MLA_PROJECTION = re.compile(
    r"^(.*?)\.(?:layers|h|blocks)\.\d+\.self_attn\.kv_a_proj_with_mqa\.weight$"
)


def mla_width_from_tensors(tensors: list[Tensor]) -> int:
    """Compressed values cached per token over all layers: kv_a_proj_with_mqa writes them."""
    return sum(
        shape[0]
        for name, shape in tensors
        if MLA_PROJECTION.match(name) and not NOT_LANGUAGE_MODEL.search(name) and len(shape) == 2
    )


def kv_cache_fact(config: dict, tensors: list[Tensor]) -> Fact:
    """Bytes of keys and values per token of context, from config.json and the tensor shapes.

    The value is what every layer adds per token. Sliding-window layers stop adding at their
    window, so the note gives the total at a long context too.
    """
    layers, layout, guessed = kv_layers_from(config)
    from_config = sum(layer.bytes_per_token for layer in layers) or None
    shapes = attention_from_tensors(tensors)
    from_shapes = shapes[0] * 2 if shapes else (mla_width_from_tensors(tensors) * 2 or None)
    _, key = text_config(config)
    config_source = f"config.json {key}".strip()

    notes = [layout] if layout else []
    if layers:
        text, _ = text_config(config)
        longest = text.get("max_position_embeddings") or 0
        windowed = any(layer.window for layer in layers)
        for tokens in sorted({DEFAULT_CONTEXT, longest} - {0}):
            size = sum(layer.cache_bytes(tokens) for layer in layers)
            line = f"{size / 2**20:,.0f} MiB at {tokens:,} tokens"
            alone = sum(layer.cache_bytes(tokens, conversations=1) for layer in layers)
            if windowed and alone != size:
                line += f" ({alone / 2**20:,.0f} MiB for 1 conversation)"
            notes.append(line)
        # Say which numbers come from runtime settings rather than from the model.
        assumed = [f"{KV_BYTES_PER_VALUE * 8}-bit cache (llama.cpp -ctk f16)"]
        if windowed:
            assumed.append(
                f"windows sized for {CONVERSATIONS} conversations (llama-server's default) "
                f"+ {SLIDING_WINDOW_UBATCH}-token micro-batch"
            )
        notes.append("assumes " + ", ".join(assumed))
    if from_config and from_shapes:
        if from_config == from_shapes:
            source = f"{config_source} + tensor shapes"
            return Fact(from_config, source, FactKind.CROSS_CHECKED, "; ".join(notes))
        notes.append(f"tensor shapes give {from_shapes:,} bytes per token")
        return Fact(from_config, config_source, FactKind.ESTIMATE, "; ".join(notes))
    if from_config:
        # A layer kind QuantPilot had to guess, with no tensor shapes to check it, is an estimate.
        kind = FactKind.ESTIMATE if guessed else FactKind.EXACT
        return Fact(from_config, config_source, kind, "; ".join(notes))
    if from_shapes:
        note = "config.json has no layer or head counts; from the K and V projection shapes"
        return Fact(from_shapes, "tensor shapes", FactKind.ESTIMATE, note)
    return Fact(note="no layer or head counts in config.json, no k_proj/v_proj tensors")


def architecture_fact(config: dict, tensors: list[Tensor]) -> Fact:
    text, key = text_config(config)
    found = architecture_from(config)
    source = f"config.json {key}".strip()
    if not found:
        return Fact(config.get("model_type"), source, FactKind.UNKNOWN, "no layer or head counts")

    parts = [found["architecture"]]
    if key:
        parts.append(f"language model ({text.get('model_type')}) read from {key}")
    parts += [
        f"{found['num_layers']} layers",
        f"{found['num_attention_heads']} heads",
        f"{found['num_kv_heads']} KV heads",
        f"hidden {found['hidden_size']}",
    ]
    tied = text.get("tie_word_embeddings", True)
    lm_head = any(name.endswith("lm_head.weight") for name, _ in tensors)
    if tied and tensors and lm_head:
        parts.append("lm_head stored although config.json ties it to the embeddings")
    else:
        parts.append("tied embeddings" if tied else "separate lm_head")

    kind = FactKind.EXACT
    named = layers_in_names(tensors)
    if named == found["num_layers"]:
        kind, source = FactKind.CROSS_CHECKED, f"{source} + tensor names"
    elif named:
        kind = FactKind.ESTIMATE
        parts.append(f"but the tensor names show {named} layers")
    return Fact(config.get("model_type"), source, kind, ", ".join(parts))


def context_fact(config: dict, meta: dict) -> Fact:
    text, key = text_config(config)
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        if text.get(name):
            return Fact(int(text[name]), f"config.json {key}".strip(), FactKind.EXACT, name)
    from_gguf = context_from_hf(meta)
    if from_gguf:
        return Fact(from_gguf, "HF API (GGUF metadata)", FactKind.EXACT, "context_length")
    return Fact()


def quant_options(
    model: ModelSnapshot,
    hardware: HardwareProfile,
    context: int = DEFAULT_CONTEXT,
    kv_cache: str = "f16",
) -> list[QuantOption]:
    """Stored checkpoint, GGUF levels, then AWQ/GPTQ/BNB packed sizes from the same tensors.

    kv_cache is the cache type asked for (--kv-cache); each row uses its runtime's version.
    """
    if model.gguf_files:
        # GGUF files: one row per file, as published. llama.cpp runs them.
        return [
            fit_at(label, size / GB, model, hardware, context, True, "gguf", kv_cache)
            for label, size in model.gguf_files
        ]
    if model.is_quantized:
        # Already quantized: only the files as they are. Re-quantizing would stack the errors.
        if model.original_gb is None:
            raise OcqError(f"Could not read how big the files of {model.model_id} are.")
        label = f"{model.facts.quantization.value} (as stored)"
        return [
            fit_at(
                label,
                model.original_gb,
                model,
                hardware,
                context,
                False,
                row_kind="stored",
                kv_cache=kv_cache,
            )
        ]
    stored = model.stored_precision or "FP16"
    options: list[QuantOption] = [
        fit_at(
            stored,
            gguf_weights_gb(model, stored),
            model,
            hardware,
            context,
            use_measurements=False,
            row_kind="stored",
            kv_cache=kv_cache,
        )
    ]
    for label in QUANTIZED_LEVELS:
        options.append(
            fit_at(
                label,
                gguf_weights_gb(model, label),
                model,
                hardware,
                context,
                use_measurements=True,
                row_kind="gguf",
                kv_cache=kv_cache,
            )
        )
    for algorithm, bits, double_quant in HF_QUANT_ROWS:
        options.append(
            hf_quant_option(model, hardware, context, algorithm, bits, double_quant, kv_cache)
        )
    # NVFP4 is only served on NVIDIA GPUs, so other PCs do not see the row.
    if hardware.has_cuda:
        options.append(
            fit_at(
                hf_label(QuantAlgorithm.NVFP4, 4),
                hf_weights_gb(
                    model,
                    QuantAlgorithm.NVFP4,
                    4,
                    default_group_size(QuantAlgorithm.NVFP4, 4),
                    fp4_activations=hardware.has_fp4_tensor_cores,
                ),
                model,
                hardware,
                context,
                use_measurements=False,
                row_kind=QuantAlgorithm.NVFP4.value,
                kv_cache=kv_cache,
            )
        )
    return options


def hf_quant_option(
    model: ModelSnapshot,
    hardware: HardwareProfile,
    context: int,
    algorithm: QuantAlgorithm,
    bits: int,
    double_quant: bool = False,
    kv_cache: str = "f16",
) -> QuantOption:
    group_size = default_group_size(algorithm, bits)
    if model.tensors and group_size:
        group_size = group_size_that_fits(list(model.tensors), group_size)
    return fit_at(
        hf_label(algorithm, bits, double_quant),
        hf_weights_gb(model, algorithm, bits, group_size, double_quant),
        model,
        hardware,
        context,
        use_measurements=False,
        row_kind=algorithm.value,
        kv_cache=kv_cache,
    )


def predict_for_config(
    model: ModelSnapshot, hardware: HardwareProfile, config: QuantConfig
) -> QuantOption:
    if config.algorithm == QuantAlgorithm.GGUF:
        label = config.gguf_type
        weights_gb = gguf_weights_gb(model, label)
    else:
        label = hf_label(config.algorithm, config.bits, config.double_quant)
        weights_gb = hf_weights_gb(
            model,
            config.algorithm,
            config.bits,
            config.group_size,
            config.double_quant,
            config.fp4_activations,
        )
    return fit_at(
        label,
        weights_gb,
        model,
        hardware,
        config.context,
        use_measurements=config.algorithm == QuantAlgorithm.GGUF,
        row_kind=("gguf" if config.algorithm == QuantAlgorithm.GGUF else config.algorithm.value),
        kv_cache=config.kv_cache,
    )


def hf_label(algorithm: QuantAlgorithm, bits: int, double_quant: bool = False) -> str:
    label = f"{algorithm.value.upper()} {bits}-bit"
    if algorithm == QuantAlgorithm.BNB and bits == 4:
        label += " double" if double_quant else " single"
    return label


def hf_weights_gb(
    model: ModelSnapshot,
    algorithm: QuantAlgorithm,
    bits: int,
    group_size: int,
    double_quant: bool = False,
    fp4_activations: bool = False,
) -> float:
    """Weights of an AWQ, GPTQ, bitsandbytes or NVFP4 artifact, scales included."""
    if algorithm == QuantAlgorithm.BNB:
        if model.tensors:
            return bnb_weights_bytes(list(model.tensors), bits, double_quant) / GB
        scale_bits = bnb_scale_bits(bits, double_quant)
    elif algorithm == QuantAlgorithm.NVFP4:
        if model.tensors:
            return nvfp4_weights_bytes(list(model.tensors), fp4_activations) / GB
        scale_bits = 8 / group_size
    else:
        if model.tensors:
            return packed_weights_bytes(list(model.tensors), bits, group_size) / GB
        scale_bits = 0.5
    return model.params_b * 1e9 * (bits + scale_bits) / 8 / GB


def gguf_weights_gb(model: ModelSnapshot, gguf_type: str) -> float:
    if model.tensors:
        heads_per_kv_head = max(model.num_attention_heads // max(model.num_kv_heads, 1), 1)
        return gguf_weights_bytes(list(model.tensors), gguf_type, heads_per_kv_head) / GB
    return model.params_b * 1e9 * AVERAGE_BITS[gguf_type] / 8 / GB


def best_option(options: list[QuantOption]) -> QuantOption:
    quantized = [option for option in options if option.row_kind != "stored"] or options
    for option in quantized:
        if option.fit_level in (FitLevel.PERFECT, FitLevel.GOOD):
            return option
    for option in quantized:
        if option.runnable:
            return option
    return quantized[-1]


def runtime_of(row_kind: str) -> str:
    """The runtime that serves a row: llama.cpp for GGUF, vLLM for AWQ, GPTQ and NVFP4,
    transformers for the original weights and bitsandbytes."""
    if row_kind == "gguf":
        return LLAMA_CPP
    if row_kind in VLLM_ROWS:
        return VLLM
    return TRANSFORMERS


def fit_at(
    label: str,
    weights_gb: float,
    model: ModelSnapshot,
    hardware: HardwareProfile,
    context: int,
    use_measurements: bool = True,
    row_kind: str = "gguf",
    kv_cache: str = "f16",
) -> QuantOption:
    if model.params_b <= 0:
        hint = model.facts.parameters.note
        raise OcqError(
            f"The parameter count of {model.model_id} is unknown: no readable safetensors "
            f"headers and no count from HuggingFace{f' ({hint})' if hint else ''}. "
            "Its memory needs cannot be computed."
        )
    # transformers expands MLA's compressed cache and keeps 16 bits; llama.cpp and vLLM keep it
    # compressed, and store the cache type asked for in their own format.
    runtime = runtime_of(row_kind)
    value_bytes = kv_value_bytes(kv_cache, runtime, hardware.vllm_fp8_kv_cache)
    kv_bytes = model.kv_cache_bytes(context, runtime, value_bytes=value_bytes)
    context_gb = 0.0
    if kv_bytes is not None:
        context_gb = (kv_bytes + output_buffer_bytes(model.vocab_size)) / GB

    required = weights_gb + context_gb
    # vLLM serves AWQ, GPTQ and NVFP4 with tensor parallel; llama.cpp and transformers split
    # by layers.
    split = TENSOR_PARALLEL if row_kind in VLLM_ROWS else LAYER_SPLIT
    heads = model.num_attention_heads if model.architecture_known else None
    pool = hardware.gpu_pool(split, heads)
    if pool and pool.split == LAYER_SPLIT:
        # Layers on a slower card slow every token down. When the model fits on the best card
        # alone and that card is faster than the split, one card is the better setup.
        single = hardware.gpu_pool(ONE_GPU)
        if required <= single.free_gb and single.read_speed > pool.read_speed:
            pool = single
    vram = pool.free_gb if pool else 0.0
    ram = hardware.available_ram_gb

    experts_ram_layers, experts_ram_gb = 0, 0.0
    offload = None
    if vram and required > vram and not hardware.unified_memory and row_kind == "gguf":
        offload = experts_to_ram(model, label, required - vram, ram, weights_gb)
    if vram and required <= vram:
        run_mode, available = RunMode.GPU, vram
    elif offload:
        run_mode, available = RunMode.GPU_EXPERTS_RAM, vram + ram
        experts_ram_layers, experts_ram_gb = offload
    elif vram and not hardware.unified_memory and required <= vram + ram:
        run_mode, available = RunMode.GPU_OFFLOAD, vram + ram
    else:
        run_mode, available = RunMode.CPU, ram

    utilization = required / available * 100 if available > 0 else 999.0
    if utilization > 100:
        fit_level, run_mode = FitLevel.TOO_TIGHT, RunMode.UNFIT
    elif run_mode == RunMode.GPU_OFFLOAD or utilization > 85:
        fit_level = FitLevel.MARGINAL
    elif run_mode == RunMode.GPU_EXPERTS_RAM:
        # Part of every token is read from RAM, so it is never the roomy fit Perfect promises.
        fit_level = FitLevel.GOOD
    elif utilization > 60:
        fit_level = FitLevel.GOOD
    else:
        fit_level = FitLevel.PERFECT
    if kv_bytes is None and fit_level in (FitLevel.PERFECT, FitLevel.GOOD):
        # The context memory is unknown, so the headroom these levels promise is not known.
        fit_level = FitLevel.MARGINAL

    speed, speed_basis = measurements.predict_speed(
        weights_gb,
        run_mode,
        vram,
        hardware,
        use_measurements,
        gpu_read_speed=pool.read_speed if pool else 0.0,
        read_fraction=model.read_fraction,
        ram_read_gb=experts_ram_gb * model.experts_per_token / model.experts if offload else 0.0,
    )
    return QuantOption(
        label=label,
        bits=weights_gb * GB * 8 / (model.params_b * 1e9),
        weights_gb=weights_gb,
        context_gb=context_gb,
        exact=bool(model.tensors) and kv_bytes is not None,
        context_known=kv_bytes is not None,
        memory_required_gb=required,
        memory_available_gb=available,
        fit_level=fit_level,
        run_mode=run_mode,
        estimated_tps=speed,
        speed_basis=speed_basis,
        row_kind=row_kind,
        gpu_names=tuple(card.name for card in pool.cards)
        if pool and run_mode != RunMode.CPU
        else (),
        gpu_split=pool.split if pool else "",
        experts_in_ram=experts_ram_layers,
    )


def experts_to_ram(
    model: ModelSnapshot,
    gguf_type: str,
    too_much_gb: float,
    ram_gb: float,
    weights_gb: float = 0.0,
) -> tuple[int, float] | None:
    """llama.cpp --n-cpu-moe N: the experts of the first N layers stay in RAM, everything else
    goes to VRAM. The smallest N that frees `too_much_gb` of VRAM, and the GB moved to RAM.

    A token reads only a few experts, so this runs much faster than moving whole layers to
    RAM. None for a dense model, or when even all experts in RAM free too little VRAM, or the
    experts do not fit in RAM.
    """
    if not (model.experts and model.experts_per_token and model.tensors):
        return None
    if model.gguf_tensor_bytes:
        # A GGUF file: its own tensor sizes, scaled when the row is another file of the repo.
        scale = weights_gb * GB / sum(model.gguf_tensor_bytes.values()) if weights_gb else 1.0
        sizes = {name: size * scale for name, size in model.gguf_tensor_bytes.items()}
    else:
        heads_per_kv_head = max(model.num_attention_heads // max(model.num_kv_heads, 1), 1)
        sizes = gguf_tensor_bytes(list(model.tensors), gguf_type, heads_per_kv_head)
    by_layer: dict[int, int] = {}
    for name, shape in model.tensors:
        if routed_expert(name, shape, model.experts):
            by_layer[layer_index(name)] = by_layer.get(layer_index(name), 0) + sizes[name]
    moved = 0.0
    for layer in sorted(by_layer):
        moved += by_layer[layer] / GB
        if moved >= too_much_gb:
            # --n-cpu-moe counts layers from the first one, dense ones too (DeepSeek layer 0).
            return (layer + 1, moved) if moved <= ram_gb else None
    return None


def parameters_from_hf(meta: dict) -> int | None:
    """The count the HuggingFace API reports. The model's name is not a count."""
    for source in ("safetensors", "gguf"):
        total = (meta.get(source) or {}).get("total")
        if total:
            return int(total)
    return None


def parse_params_from_name(model_id: str) -> float | None:
    name = model_id.upper()
    billions = re.search(r"(\d+(?:\.\d+)?)\s*B\b", name)
    if billions:
        return float(billions.group(1))
    millions = re.search(r"(\d+(?:\.\d+)?)\s*M\b", name)
    if millions:
        return float(millions.group(1)) / 1000.0
    return None


def format_params(params_b: float) -> str:
    if params_b < 0.001:
        return f"{params_b * 1e9:,.0f}"
    if params_b >= 1:
        return f"{params_b:.2f}B"
    return f"{params_b * 1000:.0f}M"


def context_from_hf(meta: dict) -> int | None:
    context = (meta.get("gguf") or {}).get("context_length")
    return int(context) if context else None
