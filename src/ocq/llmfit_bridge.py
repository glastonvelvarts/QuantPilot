"""Thin wrapper around the llmfit CLI: hardware profile, catalog fit, recommendations."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ocq.hardware_resolver import UniversalHardwareResolver
from ocq.types import (
    FeasibilityEstimate,
    FitLevel,
    GpuProfile,
    HardwareProfile,
    ModelSnapshot,
    OcqError,
    RunMode,
)


class LlmfitError(OcqError):
    """llmfit is missing, failed, or returned something we cannot read."""


class LlmfitBridge:
    def __init__(self, llmfit_bin: str | None = None) -> None:
        self.llmfit_bin = llmfit_bin or os.environ.get("OCQ_LLMFIT_BIN") or find_llmfit()

    def run_json(self, *args: str) -> dict[str, Any]:
        cmd = [self.llmfit_bin, "--no-dashboard", "--json", *args]
        try:
            completed = subprocess.run(
                cmd, check=False, capture_output=True, text=True, encoding="utf-8", timeout=120
            )
        except OSError as exc:
            raise LlmfitError(f"llmfit binary not executable: {self.llmfit_bin}") from exc
        except subprocess.TimeoutExpired as exc:
            raise LlmfitError("llmfit timed out") from exc

        stdout = completed.stdout.strip()
        if not stdout:
            stderr = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise LlmfitError(f"llmfit produced no output: {stderr}")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LlmfitError(f"llmfit returned invalid JSON: {stdout[:200]}") from exc

        if completed.returncode != 0:
            err = payload.get("error", {})
            raise LlmfitError(f"llmfit {err.get('kind', 'error')}: {err.get('message', stdout)}")

        return payload

    def hardware_profile(self) -> HardwareProfile:
        payload = self.run_json("system")
        return parse_hardware(payload.get("system", payload))

    def model_fit(self, model_id: str) -> tuple[HardwareProfile, ModelSnapshot]:
        payload = self.run_json("info", model_id)
        models = payload.get("models", [])
        if not models:
            raise LlmfitError(f"llmfit not_found: no catalog entry for {model_id!r}")
        return parse_hardware(payload.get("system", {})), parse_snapshot(models[0], model_id)

    def recommend(self, limit: int, use_case: str | None = None) -> list[ModelSnapshot]:
        args = ["recommend", "-n", str(limit)]
        if use_case:
            args += ["--use-case", use_case]
        payload = self.run_json(*args)
        return [parse_snapshot(fit, str(fit.get("name", ""))) for fit in payload.get("models", [])]


def find_llmfit() -> str:
    exe = "llmfit.exe" if os.name == "nt" else "llmfit"
    beside_python = Path(sys.executable).parent / exe
    if beside_python.exists():
        return str(beside_python)
    found = shutil.which("llmfit")
    if found:
        return found
    raise LlmfitError(
        "llmfit not found. Install: uv tool install llmfit\n"
        "Or set OCQ_LLMFIT_BIN / pass --llmfit-bin."
    )


def parse_hardware(system: dict[str, Any]) -> HardwareProfile:
    gpus = tuple(
        GpuProfile(
            name=g.get("name", "unknown"),
            vram_gb=g.get("vram_gb"),
            backend=g.get("backend", system.get("backend", "unknown")),
            count=int(g.get("count", 1)),
            unified_memory=bool(g.get("unified_memory", False)),
        )
        for g in system.get("gpus", [])
    )
    raw_profile = HardwareProfile(
        total_ram_gb=float(system.get("total_ram_gb", 0)),
        available_ram_gb=float(system.get("available_ram_gb", 0)),
        cpu_cores=int(system.get("cpu_cores", 0)),
        cpu_name=str(system.get("cpu_name", "")),
        has_gpu=bool(system.get("has_gpu", False)),
        backend=str(system.get("backend", "unknown")),
        unified_memory=bool(system.get("unified_memory", False)),
        gpus=gpus,
        os_name=platform.system(),
        source="llmfit",
        raw=system,
    )
    return UniversalHardwareResolver.resolve_profile(raw_profile)


def parse_snapshot(fit: dict[str, Any], model_id: str) -> ModelSnapshot:
    return ModelSnapshot(
        model_id=str(fit.get("name", model_id)),
        provider=str(fit.get("provider", "")),
        parameter_count=str(fit.get("parameter_count", "")),
        params_b=float(fit.get("params_b", 0)),
        context_length=optional_int(fit.get("context_length")),
        use_case=str(fit.get("use_case", fit.get("category", "General"))),
        is_moe=bool(fit.get("is_moe", False)),
        llmfit_best_quant=str(fit.get("best_quant") or fit.get("quantization") or "Q4_K_M"),
        feasibility=parse_feasibility(fit),
        catalog_source="llmfit",
        raw=fit,
    )


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_feasibility(fit: dict[str, Any]) -> FeasibilityEstimate:
    fit_level_raw = str(fit.get("fit_level") or fit.get("fit_label") or "unknown")
    run_mode_raw = str(fit.get("run_mode") or fit.get("run_mode_label") or "unknown")

    try:
        fit_level = FitLevel(fit_level_raw)
    except ValueError:
        fit_level = fit_level_raw

    try:
        run_mode = RunMode(run_mode_raw.lower())
    except ValueError:
        run_mode = run_mode_raw

    return FeasibilityEstimate(
        quant_label=str(fit.get("best_quant") or fit.get("quantization") or "unknown"),
        fit_level=fit_level,
        run_mode=run_mode,
        score=float(fit.get("score", 0)),
        estimated_tps=float(fit.get("estimated_tps", 0)),
        memory_required_gb=float(fit.get("memory_required_gb", 0)),
        memory_available_gb=float(fit.get("memory_available_gb", 0)),
        utilization_pct=float(fit.get("utilization_pct", 0)),
        runtime=str(fit.get("runtime") or fit.get("runtime_label") or "unknown"),
        estimate_confidence=str(
            fit.get("estimate_confidence") or fit.get("estimate_confidence_label") or "estimated"
        ),
        usable_context=optional_int(fit.get("usable_context")),
        raw=fit,
    )
