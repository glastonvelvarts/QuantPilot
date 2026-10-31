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
from ocq.llmfit_bridge import LlmfitBridge, LlmfitError
from ocq.local_model import fetch_local_model_files, parse_model_ref
from ocq.model_size import (
    QUANT_METADATA,
    Tensor,
    bnb_scale_bits,
    bnb_weights_bytes,
    gguf_weights_bytes,
    group_size_that_fits,
    kv_cache_bytes,
    nvfp4_weights_bytes,
    output_buffer_bytes,
    packed_weights_bytes,
    parameters_by_type,
    stored_bytes,
)
from ocq.types import (
    DTYPE_BITS,
    INTEGER_TYPES,
    Fact,
    FactKind,
    FeasibilityEstimate,
    FitLevel,
    HardwareProfile,
    ModelFacts,
    ModelSnapshot,
    OcqError,
    QuantAlgorithm,
    QuantConfig,
    QuantOption,
    RunMode,
)

HF_API = "https://huggingface.co/api/models"
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
) -> ModelSnapshot:
    """Read a model's facts from its files. Without hardware, the fit is not computed."""
    display_id, local_dir = parse_model_ref(model_ref)
    if local_dir is not None:
        config, tensors, dtypes = fetch_local_model_files(local_dir)
        meta = {"pipeline_tag": "local"}
        snapshot = build_snapshot(display_id, meta, config, tensors, hardware, context, dtypes)
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

    config, tensors, dtypes = fetch_model_files(model_id, meta.get("sha", ""))
    if quantization_config(config):
        meta = meta | {"base_model_parameters": base_model_parameters(meta)}
    snapshot = build_snapshot(model_id, meta, config, tensors, hardware, context, dtypes)
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
        is_moe=is_moe(meta, model_id),
        tensors=tuple(tensors),
        catalog_source="huggingface",
        facts=facts,
        raw=meta,
        **(architecture_from(config) or vocabulary_from_tensors(tensors)),
    )
    if not parameters:
        # No real count, so no fit verdict.
        unknown = replace(snapshot.feasibility, fit_level="unknown", run_mode="unknown")
        return replace(snapshot, feasibility=unknown)
    if hardware is None:
        return snapshot

    best = best_option(quant_options(snapshot, hardware, context))
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
    return ModelFacts(
        parameters=parameter_fact(meta, model_id, tensors, by_type, bits, copy),
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
    r"^(.*?)\.(?:layers|h|blocks)\.\d+\.(?:self_attn|attention|attn)\.[kv]_proj\.weight(?:_packed)?$"
)
EMBEDDING = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")


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


def kv_cache_fact(config: dict, tensors: list[Tensor]) -> Fact:
    """Bytes of keys and values per token of context, from config.json and the tensor shapes."""
    found = architecture_from(config)
    from_config = None
    if found:
        head_dim = (
            found["attention_head_dim"] or found["hidden_size"] // found["num_attention_heads"]
        )
        from_config = kv_cache_bytes(found["num_layers"], found["num_kv_heads"], head_dim, 1)
    shapes = attention_from_tensors(tensors)
    from_shapes = shapes[0] * 2 if shapes else None
    _, key = text_config(config)
    config_source = f"config.json {key}".strip()

    if from_config and from_shapes:
        if from_config == from_shapes:
            note = f"{from_config * DEFAULT_CONTEXT / 2**20:,.0f} MiB at {DEFAULT_CONTEXT:,} tokens"
            return Fact(
                from_config, f"{config_source} + tensor shapes", FactKind.CROSS_CHECKED, note
            )
        note = f"tensor shapes give {from_shapes:,} bytes per token"
        return Fact(from_config, config_source, FactKind.ESTIMATE, note)
    if from_config:
        return Fact(from_config, config_source, FactKind.EXACT, "standard attention assumed")
    if from_shapes:
        note = (
            f"config.json has no layer or head counts; K+V projections of {shapes[1]} layers "
            f"are {shapes[0]:,} values wide"
        )
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
    model: ModelSnapshot, hardware: HardwareProfile, context: int = DEFAULT_CONTEXT
) -> list[QuantOption]:
    """Stored checkpoint, GGUF levels, then AWQ/GPTQ/BNB packed sizes from the same tensors."""
    if model.is_quantized:
        # Already quantized: only the files as they are. Re-quantizing would stack the errors.
        if model.original_gb is None:
            raise OcqError(f"Could not read how big the files of {model.model_id} are.")
        label = f"{model.facts.quantization.value} (as stored)"
        return [
            fit_at(label, model.original_gb, model, hardware, context, False, row_kind="stored")
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
            )
        )
    for algorithm, bits, double_quant in HF_QUANT_ROWS:
        options.append(hf_quant_option(model, hardware, context, algorithm, bits, double_quant))
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


def fit_at(
    label: str,
    weights_gb: float,
    model: ModelSnapshot,
    hardware: HardwareProfile,
    context: int,
    use_measurements: bool = True,
    row_kind: str = "gguf",
) -> QuantOption:
    if model.params_b <= 0:
        hint = model.facts.parameters.note
        raise OcqError(
            f"The parameter count of {model.model_id} is unknown: no readable safetensors "
            f"headers and no count from HuggingFace{f' ({hint})' if hint else ''}. "
            "Its memory needs cannot be computed."
        )
    per_token = model.kv_bytes_per_token
    context_gb = 0.0
    if per_token is not None:
        context_gb = (per_token * context + output_buffer_bytes(model.vocab_size)) / GB

    required = weights_gb + context_gb
    vram = hardware.free_vram_gb
    ram = hardware.available_ram_gb

    if vram and required <= vram:
        run_mode, available = RunMode.GPU, vram
    elif vram and not hardware.unified_memory and required <= vram + ram:
        run_mode, available = RunMode.GPU_OFFLOAD, vram + ram
    else:
        run_mode, available = RunMode.CPU, ram

    utilization = required / available * 100 if available > 0 else 999.0
    if utilization > 100:
        fit_level, run_mode = FitLevel.TOO_TIGHT, RunMode.UNFIT
    elif run_mode == RunMode.GPU_OFFLOAD or utilization > 85:
        fit_level = FitLevel.MARGINAL
    elif utilization > 60:
        fit_level = FitLevel.GOOD
    else:
        fit_level = FitLevel.PERFECT
    if per_token is None and fit_level in (FitLevel.PERFECT, FitLevel.GOOD):
        # The context memory is unknown, so the headroom these levels promise is not known.
        fit_level = FitLevel.MARGINAL

    speed, speed_basis = measurements.predict_speed(
        weights_gb, run_mode, vram, hardware, use_measurements
    )
    return QuantOption(
        label=label,
        bits=weights_gb * GB * 8 / (model.params_b * 1e9),
        weights_gb=weights_gb,
        context_gb=context_gb,
        exact=bool(model.tensors) and per_token is not None,
        context_known=per_token is not None,
        memory_required_gb=required,
        memory_available_gb=available,
        fit_level=fit_level,
        run_mode=run_mode,
        estimated_tps=speed,
        speed_basis=speed_basis,
        row_kind=row_kind,
    )


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


def is_moe(meta: dict, model_id: str) -> bool:
    tags = [tag.lower() for tag in meta.get("tags") or []]
    if any("moe" in tag for tag in tags):
        return True
    return "moe" in model_id.lower() or "mixtral" in model_id.lower()
