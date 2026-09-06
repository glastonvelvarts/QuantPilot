"""Resolve model metadata — llmfit catalog when available, HuggingFace API otherwise."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from ocq.llmfit_bridge import LlmfitBridge, LlmfitError
from ocq.types import FeasibilityEstimate, HardwareProfile, ModelSnapshot

HF_API = "https://huggingface.co/api/models"


def resolve_model(
    model_id: str,
    hardware: HardwareProfile,
    bridge: LlmfitBridge,
) -> ModelSnapshot:
    """Return model snapshot; prefer llmfit fit scores when the model is in its catalog."""

    try:
        _, snapshot = bridge.model_fit(model_id)
        return snapshot
    except LlmfitError as exc:
        if "not_found" not in str(exc).lower() and "no models" not in str(exc).lower():
            # Hardware errors etc. should still fail loudly.
            if "llmfit not found" in str(exc).lower() or "binary not executable" in str(exc).lower():
                raise
        # Fall through to HuggingFace for models outside the llmfit catalog.

    meta = fetch_hf_model(model_id)
    return _snapshot_from_hf(model_id, meta, hardware)


def fetch_hf_model(model_id: str) -> dict:
    """Fetch public model metadata from HuggingFace (any repo, not llmfit-hardcoded)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    url = f"{HF_API}/{model_id}"
    headers = {"User-Agent": "ocq/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            msg = (
                f"Model {model_id!r} not found on HuggingFace and not in llmfit catalog. "
                "Check the repo id (org/name)."
            )
            raise LlmfitError(msg) from exc
        raise LlmfitError(f"HuggingFace API error {exc.code} for {model_id}") from exc
    except urllib.error.URLError as exc:
        raise LlmfitError(f"Could not reach HuggingFace API: {exc.reason}") from exc


def _snapshot_from_hf(model_id: str, meta: dict, hardware: HardwareProfile) -> ModelSnapshot:
    params_b = _params_b_from_hf(meta, model_id)
    param_str = _format_params(params_b)
    feasibility = _estimate_feasibility(params_b, hardware)

    return ModelSnapshot(
        model_id=model_id,
        provider=model_id.split("/")[0] if "/" in model_id else "unknown",
        parameter_count=param_str,
        params_b=params_b,
        context_length=_context_from_hf(meta),
        use_case=str(meta.get("pipeline_tag") or "General"),
        is_moe=_is_moe(meta, model_id),
        llmfit_best_quant="Q4_K_M",
        feasibility=feasibility,
        catalog_source="huggingface",
        raw=meta,
    )


def _params_b_from_hf(meta: dict, model_id: str) -> float:
    safetensors = meta.get("safetensors") or {}
    total = safetensors.get("total")
    if total:
        return total / 1e9

    gguf = meta.get("gguf") or {}
    total = gguf.get("total")
    if total:
        return total / 1e9

    parsed = _parse_params_from_name(model_id)
    if parsed:
        return parsed

    return 7.0  # conservative default when metadata is sparse


def _parse_params_from_name(model_id: str) -> float | None:
    upper = model_id.upper()
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*B", upper):
        return float(match.group(1))
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*M", upper):
        return float(match.group(1)) / 1000.0
    return None


def _format_params(params_b: float) -> str:
    if params_b >= 1:
        return f"{params_b:.1f}B".replace(".0B", "B")
    return f"{int(params_b * 1000)}M"


def _context_from_hf(meta: dict) -> int | None:
    gguf = meta.get("gguf") or {}
    ctx = gguf.get("context_length")
    if ctx:
        return int(ctx)
    return None


def _is_moe(meta: dict, model_id: str) -> bool:
    tags = [t.lower() for t in meta.get("tags") or []]
    if any("moe" in t for t in tags):
        return True
    return "moe" in model_id.lower() or "mixtral" in model_id.lower()


def _estimate_feasibility(params_b: float, hardware: HardwareProfile) -> FeasibilityEstimate:
    """Rough memory fit when llmfit has no catalog entry (same formulas as llmfit scraper)."""

    # Q4_K_M weight size: params * 0.5 bytes/param, 1.2 RAM / 1.1 VRAM overhead
    weight_gb = params_b * 0.5
    ram_gb = weight_gb * 1.2
    vram_gb = weight_gb * 1.1

    if hardware.has_gpu and hardware.primary_vram_gb is not None:
        pool = hardware.primary_vram_gb
        required = vram_gb
        run_mode = "gpu"
    else:
        pool = hardware.available_ram_gb
        required = ram_gb
        run_mode = "cpu"

    util = (required / pool * 100) if pool > 0 else 999.0
    if util <= 60:
        fit_level = "Perfect"
    elif util <= 85:
        fit_level = "Good"
    elif util <= 98:
        fit_level = "Marginal"
    else:
        fit_level = "Too Tight"

    # Very rough tok/s placeholder — OCQ benchmark replaces this after quant.
    est_tps = max(1.0, 40.0 / max(params_b, 0.5))

    return FeasibilityEstimate(
        quant_label="Q4_K_M",
        fit_level=fit_level,
        run_mode=run_mode,
        score=50.0,
        estimated_tps=est_tps,
        memory_required_gb=required,
        memory_available_gb=pool,
        utilization_pct=util,
        runtime="unknown",
        estimate_confidence="estimated",
    )
