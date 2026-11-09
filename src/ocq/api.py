"""QuantPilot for scripts and agents: each function returns plain data (dicts, lists, numbers,
strings: what JSON holds), the same that `ocq <command> --json` prints.

    from ocq import api

    pick = api.recommend(workload="coding")["best"]
    print(pick["model"]["id"], pick["format"]["label"], pick["format"]["tokens_per_second"])

The shape is versioned: every result has "schema_version" (SCHEMA_VERSION). Within one
version, keys are only ever added, never renamed or removed, and a value keeps its type and
unit; anything else is a new version. Units are in the key names (_gb is GiB, _gbs GB/s,
_seconds, tokens_per_second). docs/api.md describes every key.

Functions that read this PC take `hardware` (from api.hardware()) so that several calls reuse
one reading; without it they read the hardware themselves. Failures raise ocq.types.OcqError
with a message for people.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ocq import __version__
from ocq.types import Fact, HardwareProfile, OcqError, QuantOption

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "OcqError",
    "bench",
    "doctor",
    "feasibility",
    "hardware",
    "history",
    "inspect",
    "models",
    "profile",
    "recommend",
]


def result(kind: str, **data: Any) -> dict:
    return {"schema_version": SCHEMA_VERSION, "quantpilot_version": __version__, "kind": kind} | {
        key: plain(value) for key, value in data.items()
    }


def plain(value: Any) -> Any:
    """JSON's types only: enums by value, paths as strings, tuples as lists."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(plain(key)): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, float):
        return round(value, 4)
    return value


# This PC ------------------------------------------------------------------------------------


def read_hardware(llmfit_bin: str | None = None) -> HardwareProfile:
    """This PC as QuantPilot reads it (pass it to the other functions to read it once)."""
    from ocq.hardware import detect_hardware

    return detect_hardware(llmfit_bin)


hardware = read_hardware  # api.hardware(): the public name


def hardware_dict(hw: HardwareProfile) -> dict:
    return {
        "os": hw.os_name,
        "cpu": hw.cpu_name,
        "cpu_cores": hw.cpu_cores,
        "ram_gb": hw.total_ram_gb,
        "ram_free_gb": hw.available_ram_gb,
        "ram_bandwidth_gbs": hw.ram_bandwidth_gbs or None,
        "ram_bandwidth_measured_gbs": hw.measured_ram_bandwidth_gbs or None,
        "backend": hw.backend,
        "unified_memory": hw.unified_memory,
        "gpus": [
            {
                "name": gpu.name,
                "count": gpu.count,
                "vram_gb": gpu.vram_gb,
                "vram_free_gb": gpu.cuda_free_gb or gpu.free_vram_gb,
                "backend": gpu.backend,
                "bandwidth_gbs": gpu.measured_bandwidth_gbs or gpu.bandwidth_gbs or None,
                "compute_capability": gpu.compute_capability,
            }
            for gpu in hw.gpus
        ],
    }


def profile(hardware: HardwareProfile | None = None, llmfit_bin: str | None = None) -> dict:
    """What this PC has: CPU, RAM, GPUs, memory bandwidth, free disk here."""
    from ocq.plan import free_disk_gb

    hw = hardware or read_hardware(llmfit_bin)
    return result("profile", hardware=hardware_dict(hw), disk_free_gb=free_disk_gb(Path.cwd()))


# A model ------------------------------------------------------------------------------------


def fact_dict(fact: Fact) -> dict:
    return {"value": plain(fact.value), "source": fact.source, "kind": plain(fact.kind),
            "note": fact.note}  # fmt: skip


def facts_dict(snapshot) -> dict:
    facts = snapshot.facts
    return {
        name: fact_dict(getattr(facts, name))
        for name in (
            "parameters",
            "active_parameters",
            "precision",
            "stored_bytes",
            "architecture",
            "context_length",
            "kv_cache",
            "quantization",
        )
    }


def inspect(model: str, llmfit_bin: str | None = None) -> dict:
    """Every fact about a model (a Hugging Face id, a local folder or .gguf, ollama:name:tag),
    where it was read from and how sure that is (kind: exact, cross-checked, estimate,
    unknown)."""
    from ocq.model_analyzer import resolve_model

    snapshot = resolve_model(model, None, llmfit_bin)
    return result("inspect", model=snapshot.model_id, facts=facts_dict(snapshot))


def option_dict(option: QuantOption) -> dict:
    """One way to run a model: its format, runtime, memory and speed on this PC."""
    return {
        "label": option.label,
        "kind": option.row_kind,  # gguf, awq, gptq, bnb, nvfp4, stored
        "runtime": option.runtime,
        "bits_per_weight": option.bits,
        "weights_gb": option.weights_gb,
        "context_gb": option.context_gb,
        "memory_required_gb": option.memory_required_gb,
        "memory_free_gb": option.memory_available_gb,
        "fit": plain(option.fit_level),  # Perfect, Good, Marginal, Too Tight
        "runs_on": plain(option.run_mode),  # gpu, gpu_offload, gpu_experts_ram, cpu, unfit
        "gpu_layers": option.gpu_layers or None,
        "experts_in_ram_layers": option.experts_in_ram or None,
        "tokens_per_second": option.estimated_tps,
        # "upper bound": what memory speed allows; "from runs on this PC": measured
        "tokens_per_second_basis": option.speed_basis,
        "prompt_tokens_per_second": option.prompt_tps or None,
        "note": option.note,
    }


def feasibility(
    model: str,
    context: int = 4096,
    kv_cache: str = "f16",
    hardware: HardwareProfile | None = None,
    llmfit_bin: str | None = None,
) -> dict:
    """Every format of a model on this PC, at this context and cache type."""
    from ocq.model_analyzer import quant_options, resolve_model

    hw = hardware or read_hardware(llmfit_bin)
    snapshot = resolve_model(model, hw, llmfit_bin, context, kv_cache)
    options = quant_options(snapshot, hw, context, kv_cache)
    return result(
        "feasibility",
        model=snapshot.model_id,
        context=context,
        kv_cache=kv_cache,
        hardware=hardware_dict(hw),
        formats=[option_dict(option) for option in options],
    )


# The models on this PC ----------------------------------------------------------------------


def models(
    folders: list[Path] | tuple[Path, ...] = (),
    include_partial: bool = False,
    context: int = 4096,
    kv_cache: str = "f16",
    hardware: HardwareProfile | None = None,
    llmfit_bin: str | None = None,
) -> dict:
    """The models already on this PC (Ollama, LM Studio, Hugging Face cache, llama.cpp cache,
    QuantPilot's own, and folders), each with how it fits as it is on disk ("ref" is the
    name the other functions take)."""
    from ocq.discovery import find_models
    from ocq.model_analyzer import quant_options, resolve_model

    hw = hardware or read_hardware(llmfit_bin)
    found = []
    for model in find_models(list(folders) or None):
        if not model.complete and not include_partial:
            continue
        entry = {
            "label": model.label,
            "ref": model.ref,
            "source": model.source,
            "path": str(model.path),
            "format": model.format,
            "quantization": model.quantization,
            "size_gb": round(model.size_bytes / 1024**3, 4) if model.size_bytes else None,
            "complete": model.complete,
            "note": model.note,
            "fit": None,
        }
        if model.complete:
            ref = model.ref if model.ref.startswith("ollama:") else str(model.path)
            try:
                snapshot = resolve_model(ref, hw, None, context, kv_cache)
                entry["fit"] = option_dict(quant_options(snapshot, hw, context, kv_cache)[0])
            except (OcqError, OSError, ValueError, KeyError) as exc:
                entry["note"] = "; ".join(filter(None, [model.note, f"not read: {exc}"]))
        found.append(entry)
    return result("models", context=context, kv_cache=kv_cache, models=found)


# Recommendations ----------------------------------------------------------------------------


def candidate_dict(candidate) -> dict:
    return {
        "model": {
            "ref": candidate.model_ref,
            "id": candidate.model_id,
            "on_this_pc": candidate.local,
            "tags": sorted(candidate.tags),
        },
        "format": option_dict(candidate.option),
        "context": candidate.context,
        "kv_cache": candidate.kv_cache,
        "needs_quantizing": candidate.quantize,
    }


def recommendation_dict(recommendation) -> dict:
    from ocq import recommend as engine
    from ocq.recommend import make_command

    candidate = recommendation.candidate
    entry = candidate_dict(candidate) | {
        "reasons": [{"kind": plain(r.kind), "text": r.text} for r in recommendation.reasons],
        "warnings": list(recommendation.warnings),
        "confidence": {
            "level": plain(recommendation.confidence.level),
            "evidence": [
                {"text": e.text, "known": e.known} for e in recommendation.confidence.evidence
            ],
        },
        "lost_on": recommendation.lost_on or None,
        "command": make_command(candidate) if candidate.quantize else None,
    }
    # What the ranking compared, in plain units.
    values = {plain(objective): value for objective, value in recommendation.values.items()}
    plain_values = {
        "quality_parameters": values.get("quality"),  # what the format is worth, in parameters
        "tokens_per_second": values.get("speed"),
        "memory_gb": round(1 / values["memory"], 4) if values.get("memory") else None,
        "context_room_tokens": values.get("context"),
        "throughput_tokens_per_second": values.get("throughput"),
        "first_token_seconds": round(1 / values["latency"], 4) if values.get("latency") else None,
    }
    kept, measured = engine.kept_of(candidate)
    entry["quality"] = {"kept": round(kept, 4), "measured": measured is not None}
    return entry | {"values": {k: v for k, v in plain_values.items() if v is not None}}


def recommend(
    models: list[str] | tuple[str, ...] = (),
    workload: str = "chat",
    context: int = 0,
    kv_caches: list[str] | tuple[str, ...] = (),
    folders: list[Path] | tuple[Path, ...] = (),
    hardware: HardwareProfile | None = None,
    llmfit_bin: str | None = None,
    limit: int = 8,
) -> dict:
    """Which model, format and runtime to use on this PC for a kind of work (chat, coding, rag,
    agentic, reasoning, long-context, low-latency, throughput), why, and how sure. Without
    models: the ones on this PC and popular ones from the Hub."""
    from ocq import recommend as engine

    if workload not in engine.WORKLOADS:
        raise OcqError(f"Unknown workload {workload!r}. Use one of: {', '.join(engine.WORKLOADS)}")
    hw = hardware or read_hardware(llmfit_bin)
    request = engine.RecommendationRequest(
        engine.WORKLOADS[workload], tuple(models), context, tuple(kv_caches)
    )
    found = engine.recommend(request, hw, list(folders) or None)
    profile_ = request.workload
    return result(
        "recommend",
        workload={
            "key": profile_.key,
            "label": profile_.label,
            "priorities": [plain(p) for p in profile_.priorities],
        },
        context=request.context_tokens,
        hardware=hardware_dict(hw),
        best=recommendation_dict(found.best) if found.best else None,
        alternatives=[recommendation_dict(r) for r in found.alternatives[:limit]],
        considered=len(found.ranked) + len(found.rejected),
        ruled_out=[
            candidate_dict(rejection.candidate)
            | {"failed": [{"check": c.name, "text": c.text} for c in rejection.failed]}
            for rejection in found.rejected
        ],
        unread=[{"model": model, "why": why} for model, why in found.unread],
        too_big=[{"model": model, "why": why} for model, why in found.too_big],
    )


# Benchmarks ---------------------------------------------------------------------------------


def bench(
    model: str,
    quality: str = "",
    context: int = 4096,
    kv_cache: str = "f16",
    hardware: HardwareProfile | None = None,
    llmfit_bin: str | None = None,
) -> dict:
    """The standard benchmark of a model on this PC (speed for GGUF), and with quality="fixed"
    (the first 40 chunks of WikiText-2) or "full" its perplexity and what it keeps of the
    original. Saved to this PC's history, like ocq bench. Takes minutes."""
    from ocq import bench as benchmark
    from ocq import quality as measured

    pieces = {"": None, "fixed": measured.FIXED, "full": measured.FULL}
    if quality not in pieces:
        raise OcqError('quality is "", "fixed" or "full".')
    hw = hardware or read_hardware(llmfit_bin)
    outcome = benchmark.run_benchmark(
        model, hw, piece=pieces[quality], context=context, kv_cache=kv_cache,
        llmfit_bin=llmfit_bin,
    )  # fmt: skip
    speed = None
    if outcome.speed is not None:
        report = outcome.speed
        speed = {
            "tokens_per_second": report.tokens_per_second,
            "tokens_per_second_spread": report.stddev,
            "prompt_tokens_per_second": {str(n): v for n, v in report.prompt_speeds.items()},
            "first_token_seconds": {
                str(n): report.first_token_seconds(n) for n in report.prompt_speeds
            },
            "expected_tokens_per_second": outcome.option.estimated_tps,
            "expected_basis": outcome.option.speed_basis,
            "runtime_version": report.build,
            "runs_of_this_setup": outcome.runs,
        }
    measured_quality = None
    if outcome.quality is not None:
        kept = outcome.kept
        measured_quality = {
            "dataset": measured.DATASET,
            "slice": outcome.piece.name,
            "chunks": outcome.quality.chunks,
            "perplexity": outcome.quality.value,
            "perplexity_error": outcome.quality.error,
            "runtime": outcome.quality.runtime,
            "is_the_original": outcome.note == "original",
            "kept": kept.kept if kept else None,
            "reference": (
                {
                    "label": kept.reference_label,
                    "perplexity": kept.reference,
                    "same_device": kept.same_device,
                }  # fmt: skip
                if kept
                else None
            ),
            "note": "" if outcome.note == "original" else outcome.note,
        }
    return result(
        "bench",
        model=outcome.model_id,
        label=outcome.label,
        placement=outcome.device,
        kv_cache=kv_cache,
        speed=speed,
        quality=measured_quality,
    )


def history(model: str = "") -> dict:
    """This PC's saved speed runs and quality results, newest first (of one model when given:
    a repo id, or local/<file>)."""
    from ocq import measurements
    from ocq import quality as measured

    def of_model(entry: dict) -> bool:
        names = (str(entry.get("model", "")), f"local/{entry.get('file', '')}")
        return not model or model.lower() in (name.lower() for name in names)

    return result(
        "history",
        speed_runs=[run for run in reversed(measurements.load()) if of_model(run)],
        quality_results=[r for r in reversed(measured.load()) if of_model(r)],
    )


# Is QuantPilot ready ------------------------------------------------------------------------


def doctor(
    online: bool = True, hardware: HardwareProfile | None = None, llmfit_bin: str | None = None
) -> dict:
    """What is ready on this PC and what is not, each with the command that fixes it (status:
    ok, info, warn, fail). ready is false when something blocks real work."""
    from ocq import doctor as checks

    hw = hardware or read_hardware(llmfit_bin)
    findings = checks.run_checks(hw, online=online)
    return result(
        "doctor",
        ready=not any(f.status == checks.FAIL for f in findings),
        checks=[
            {"name": f.name, "status": f.status, "detail": f.detail, "fix": f.fix or None}
            for f in findings
        ],
    )
