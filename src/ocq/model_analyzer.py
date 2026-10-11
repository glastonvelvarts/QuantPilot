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
from ocq.local_model import fetch_local_model_files, local_hf_meta, parse_model_ref
from ocq.model_size import (
    Tensor,
    gguf_weights_bytes,
    group_size_that_fits,
    kv_cache_bytes,
    output_buffer_bytes,
    packed_weights_bytes,
)
from ocq.types import (
    FeasibilityEstimate,
    FitLevel,
    HardwareProfile,
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
HF_QUANT_ROWS = (
    (QuantAlgorithm.AWQ, 4),
    (QuantAlgorithm.GPTQ, 4),
    (QuantAlgorithm.GPTQ, 8),
    (QuantAlgorithm.BNB, 4),
    (QuantAlgorithm.BNB, 8),
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
    hardware: HardwareProfile,
    llmfit_bin: str | None = None,
    context: int = DEFAULT_CONTEXT,
) -> ModelSnapshot:
    display_id, local_dir = parse_model_ref(model_ref)
    if local_dir is not None:
        config, tensors, parameters = fetch_local_model_files(local_dir)
        meta = local_hf_meta(parameters)
        snapshot = build_snapshot(display_id, meta, config, tensors, hardware, context)
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

    config, tensors = fetch_model_files(model_id, meta.get("sha", ""))
    snapshot = build_snapshot(model_id, meta, config, tensors, hardware, context)
    if catalog:
        snapshot = replace(
            snapshot,
            provider=catalog.provider or snapshot.provider,
            use_case=catalog.use_case,
            catalog_source="llmfit",
        )
    return snapshot


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


def fetch_model_files(model_id: str, revision: str) -> tuple[dict, list[Tensor]]:
    """config.json and every tensor's shape. Only file headers are read, never the weights."""
    cached = MODEL_CACHE / f"{model_id.replace('/', '--')}.json"
    try:
        saved = json.loads(cached.read_text())
        if saved["revision"] == revision:
            return saved["config"], [(name, tuple(shape)) for name, shape in saved["tensors"]]
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
        tensors = [
            (name, tuple(info.shape))
            for file in metadata.files_metadata.values()
            for name, info in file.tensors.items()
        ]
    except (OSError, NotASafetensorsRepoError, SafetensorsParsingError):
        tensors = []
    finally:
        enable_progress_bars()

    if config and tensors:
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"revision": revision, "config": config, "tensors": tensors}))
    return config, tensors


def build_snapshot(
    model_id: str,
    meta: dict,
    config: dict,
    tensors: list[Tensor],
    hardware: HardwareProfile,
    context: int = DEFAULT_CONTEXT,
) -> ModelSnapshot:
    if tensors:
        parameters = sum(math.prod(shape) for _, shape in tensors)
    else:
        parameters = parameters_from_hf(meta, model_id)

    snapshot = ModelSnapshot(
        model_id=model_id,
        provider=model_id.split("/")[0] if "/" in model_id else "unknown",
        parameter_count=format_params(parameters / 1e9) if parameters else "unknown (assuming 7B)",
        params_b=(parameters or 7e9) / 1e9,
        context_length=config.get("max_position_embeddings") or context_from_hf(meta),
        use_case=str(meta.get("pipeline_tag") or "General"),
        is_moe=is_moe(meta, model_id),
        tensors=tuple(tensors),
        catalog_source="huggingface",
        raw=meta,
        **architecture_from(config),
    )

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


def architecture_from(config: dict) -> dict:
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    hidden = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    heads = config.get("num_attention_heads") or config.get("n_head")
    vocab = config.get("vocab_size")
    if not (layers and hidden and heads and vocab):
        return {}
    return {
        "architecture": (config.get("architectures") or [config.get("model_type", "model")])[0],
        "num_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_kv_heads": config.get("num_key_value_heads") or heads,
        "vocab_size": vocab,
        "attention_head_dim": config.get("head_dim") or 0,
    }


def quant_options(
    model: ModelSnapshot, hardware: HardwareProfile, context: int = DEFAULT_CONTEXT
) -> list[QuantOption]:
    """Stored checkpoint, GGUF levels, then AWQ/GPTQ/BNB packed sizes from the same tensors."""
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
    for algorithm, bits in HF_QUANT_ROWS:
        options.append(hf_quant_option(model, hardware, context, algorithm, bits))
    return options


def hf_quant_option(
    model: ModelSnapshot,
    hardware: HardwareProfile,
    context: int,
    algorithm: QuantAlgorithm,
    bits: int,
) -> QuantOption:
    group_size = default_group_size(algorithm, bits)
    if model.tensors:
        if group_size:
            group_size = group_size_that_fits(list(model.tensors), group_size)
        weights_gb = packed_weights_bytes(list(model.tensors), bits, group_size) / GB
    else:
        weights_gb = model.params_b * 1e9 * (bits + 0.5) / 8 / GB
    label = f"{algorithm.value.upper()} {bits}-bit"
    return fit_at(
        label,
        weights_gb,
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
        label = f"{config.algorithm.value.upper()} {config.bits}-bit"
        if model.tensors:
            packed = packed_weights_bytes(list(model.tensors), config.bits, config.group_size)
            weights_gb = packed / GB
        else:
            weights_gb = model.params_b * 1e9 * (config.bits + 0.5) / 8 / GB
    return fit_at(
        label,
        weights_gb,
        model,
        hardware,
        config.context,
        use_measurements=config.algorithm == QuantAlgorithm.GGUF,
        row_kind=("gguf" if config.algorithm == QuantAlgorithm.GGUF else config.algorithm.value),
    )


def gguf_weights_gb(model: ModelSnapshot, gguf_type: str) -> float:
    if model.tensors:
        heads_per_kv_head = max(model.num_attention_heads // max(model.num_kv_heads, 1), 1)
        return gguf_weights_bytes(list(model.tensors), gguf_type, heads_per_kv_head) / GB
    return model.params_b * 1e9 * AVERAGE_BITS[gguf_type] / 8 / GB


def best_option(options: list[QuantOption]) -> QuantOption:
    quantized = [option for option in options if option.row_kind != "stored"]
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
    context_gb = 0.0
    if model.architecture_known:
        kv_cache = kv_cache_bytes(model.num_layers, model.num_kv_heads, model.head_dim, context)
        context_gb = (kv_cache + output_buffer_bytes(model.vocab_size)) / GB

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

    speed, speed_basis = measurements.predict_speed(
        weights_gb, run_mode, vram, hardware, use_measurements
    )
    return QuantOption(
        label=label,
        bits=weights_gb * GB * 8 / (model.params_b * 1e9),
        weights_gb=weights_gb,
        context_gb=context_gb,
        exact=bool(model.tensors) and model.architecture_known,
        memory_required_gb=required,
        memory_available_gb=available,
        fit_level=fit_level,
        run_mode=run_mode,
        estimated_tps=speed,
        speed_basis=speed_basis,
        row_kind=row_kind,
    )


def parameters_from_hf(meta: dict, model_id: str) -> float | None:
    for source in ("safetensors", "gguf"):
        total = (meta.get(source) or {}).get("total")
        if total:
            return float(total)
    from_name = parse_params_from_name(model_id)
    return from_name * 1e9 if from_name else None


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
