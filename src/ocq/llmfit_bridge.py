"""Minimal llmfit CLI wrapper — hardware profile + catalog fit only."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from ocq.hardware_resolver import UniversalHardwareResolver
from ocq.types import (
    FeasibilityEstimate,
    FitLevel,
    GpuProfile,
    HardwareProfile,
    ModelSnapshot,
    RunMode,
)


class LlmfitError(RuntimeError):
    """llmfit subprocess failed or returned unexpected JSON."""


class LlmfitBridge:
    """OCQ only needs two llmfit commands: `system` and `info`."""

    def __init__(self, llmfit_bin: str | None = None) -> None:
        self.llmfit_bin = llmfit_bin or os.environ.get("OCQ_LLMFIT_BIN") or self._discover_bin()

    @staticmethod
    def _discover_bin() -> str:
        found = shutil.which("llmfit")
        if found:
            return found
        msg = (
            "llmfit not found on PATH. Install: uv tool install llmfit\n"
            "Or set OCQ_LLMFIT_BIN / pass --llmfit-bin."
        )
        raise LlmfitError(msg)

    def _run_json(self, *args: str) -> dict[str, Any]:
        cmd = [self.llmfit_bin, "--no-dashboard", "--json", *args]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise LlmfitError(f"llmfit binary not executable: {self.llmfit_bin}") from exc

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
            kind = err.get("kind", "error")
            message = err.get("message", stdout)
            raise LlmfitError(f"llmfit {kind}: {message}")

        return payload

    def hardware_profile(self) -> HardwareProfile:
        payload = self._run_json("system")
        return _parse_hardware(payload.get("system", payload))

    def model_fit(self, model_id: str) -> tuple[HardwareProfile, ModelSnapshot]:
        payload = self._run_json("info", model_id)
        system = payload.get("system", {})
        models = payload.get("models", [])
        if not models:
            raise LlmfitError(f"llmfit not_found: no catalog entry for {model_id!r}")

        fit = models[0]
        hardware = _parse_hardware(system)
        snapshot = ModelSnapshot(
            model_id=str(fit.get("name", model_id)),
            provider=str(fit.get("provider", "")),
            parameter_count=str(fit.get("parameter_count", "")),
            params_b=float(fit.get("params_b", 0)),
            context_length=_optional_int(fit.get("context_length")),
            use_case=str(fit.get("use_case", fit.get("category", "General"))),
            is_moe=bool(fit.get("is_moe", False)),
            llmfit_best_quant=str(fit.get("best_quant") or fit.get("quantization") or "Q4_K_M"),
            feasibility=_parse_feasibility(fit),
            catalog_source="llmfit",
            raw=fit,
        )
        return hardware, snapshot


def _parse_hardware(system: dict[str, Any]) -> HardwareProfile:
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
        raw=system,
    )
    return UniversalHardwareResolver.resolve_profile(raw_profile)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_feasibility(fit: dict[str, Any]) -> FeasibilityEstimate:
    fit_level_raw = str(fit.get("fit_level") or fit.get("fit_label") or "unknown")
    run_mode_raw = str(fit.get("run_mode") or fit.get("run_mode_label") or "unknown")

    try:
        fit_level = FitLevel(fit_level_raw)
    except ValueError:
        fit_level = fit_level_raw

    try:
        run_mode = RunMode(run_mode_raw)
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
        usable_context=_optional_int(fit.get("usable_context")),
        raw=fit,
    )
