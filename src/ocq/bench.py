"""The standard benchmark: the same tests, the same way, for every GGUF model on every PC.

llama-bench reads prompts of fixed lengths and writes a fixed number of tokens, each test repeated
and averaged. Two suites:

    quick     a 512 token prompt, then 128 tokens written     after every quantize (under a minute)
    standard  also a 4,096 token prompt                       ocq bench

Reading a prompt takes longer per token as the prompt grows (every token looks back at all the
ones before it), so the time to the first token is measured at a short and a long prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ocq import measurements
from ocq.quantizers.protocol import BENCHMARK_TOKENS, PROMPT_TOKENS
from ocq.runtimes import llama_cache_env
from ocq.types import (
    LLAMA_CPP,
    HardwareProfile,
    ModelSnapshot,
    OcqError,
    QuantOption,
    RunMode,
)

REPEATS = 3
LONG_PROMPT = 4096


@dataclass(frozen=True)
class Suite:
    name: str
    prompts: tuple[int, ...]  # prompt lengths whose reading is timed
    tokens: int = BENCHMARK_TOKENS  # tokens written
    repeats: int = REPEATS


QUICK = Suite("quick", (PROMPT_TOKENS,))
STANDARD = Suite("standard", (PROMPT_TOKENS, LONG_PROMPT))


@dataclass
class BenchReport:
    tokens_per_second: float
    stddev: float
    # Prompt length -> tokens read per second.
    prompt_speeds: dict[int, float] = field(default_factory=dict)
    build: str = ""  # llama.cpp's build, b11429

    def first_token_seconds(self, prompt: int) -> float | None:
        """Time to the first token after a prompt of this length: reading it, then writing one."""
        speed = self.prompt_speeds.get(prompt)
        if not speed or not self.tokens_per_second:
            return None
        return prompt / speed + 1 / self.tokens_per_second


def prompts_for(suite: Suite, model_context: int | None) -> tuple[int, ...]:
    """The suite's prompts that fit, with the tokens written after them, in the model's context."""
    if not model_context:
        return suite.prompts
    kept = tuple(p for p in suite.prompts if p + suite.tokens <= model_context)
    return kept or suite.prompts[:1]


def placement(option: QuantOption) -> tuple[list[str], str]:
    """llama-bench's flags that place the model as QuantPilot's plan does, and what to call that
    in the benchmark history. The plan's layer split is the one llama.cpp's --fit makes."""
    if option.run_mode == RunMode.UNFIT:
        raise OcqError("It does not fit in this PC's memory, so it cannot be benchmarked here.")
    if option.run_mode in (RunMode.CPU, RunMode.CPU_ONLY):
        return ["-ngl", "0", "-dev", "none"], "cpu"
    if option.run_mode == RunMode.GPU_EXPERTS_RAM:
        experts = option.experts_in_ram
        return ["-ngl", "999", "-ncmoe", str(experts)], f"gpu (experts of {experts} layers in ram)"
    if option.run_mode == RunMode.GPU_OFFLOAD and option.gpu_layers:
        layers = option.gpu_layers
        return ["-ngl", str(layers)], f"gpu+cpu ({layers} layers on gpu)"
    return ["-ngl", "999"], "gpu"


def llama_bench(
    tool: Path,
    model: Path,
    suite: Suite,
    *,
    architecture: str,
    kv_cache: str,
    place: list[str],
    log: Path,
    model_context: int | None = None,
) -> BenchReport:
    """Runs the suite with llama-bench and reads its JSON report."""
    from ocq.quantizers.gguf import run_tool  # gguf imports this module

    prompts = prompts_for(suite, model_context)
    cache_flags = [] if kv_cache == "f16" else ["-ctk", kv_cache, "-ctv", kv_cache]
    report, _ = run_tool(
        [
            str(tool),
            "-m",
            str(model),
            "-p",
            ",".join(str(p) for p in prompts),
            "-n",
            str(suite.tokens),
            "-r",
            str(suite.repeats),
            *place,
            "-o",
            "json",
            *cache_flags,
        ],
        log=log,
        env=llama_cache_env(architecture, kv_cache),
    )
    return read_report(report)


def read_report(report: str) -> BenchReport:
    """llama-bench's JSON: one entry per test, n_prompt > 0 for reading, n_gen > 0 for writing."""
    try:
        tests = json.loads(report)
        writing = next(test for test in tests if test.get("n_gen"))
    except (ValueError, StopIteration, AttributeError, TypeError) as exc:
        raise OcqError("llama-bench gave no result for writing tokens.") from exc
    reading = {
        int(test["n_prompt"]): float(test["avg_ts"])
        for test in tests
        if test.get("n_prompt") and not test.get("n_gen")
    }
    build = writing.get("build_number")
    return BenchReport(
        tokens_per_second=float(writing["avg_ts"]),
        stddev=float(writing.get("stddev_ts") or 0.0),
        prompt_speeds=reading,
        build=f"b{build}" if build else "",
    )


def save(
    report: BenchReport,
    suite: Suite,
    *,
    device: str,
    read_gb: float,
    model: str,
    label: str,
    context: int,
    kv_cache: str,
    hardware: HardwareProfile,
    file: str = "",
    active_params_b: float = 0.0,
) -> None:
    """Into this PC's benchmark history (measurements.json)."""
    measurements.record(
        device,
        read_gb,
        report.tokens_per_second,
        model=model,
        label=label,
        runtime=LLAMA_CPP,
        context=context,
        kv_cache=kv_cache,
        hardware=hardware,
        runtime_version=report.build,
        prompt_tps=report.prompt_speeds.get(PROMPT_TOKENS, 0.0),
        prompt_speeds=report.prompt_speeds,
        suite=suite.name,
        file=file,
        active_params_b=active_params_b,
    )


def made_from(path: Path) -> str | None:
    """The repo a file QuantPilot made was quantized from (ocq-output/<org>--<name>/artifact/),
    so its runs count for that repo's recommendations too. None for any other file."""
    if path.parent.name != "artifact" or "--" not in path.parent.parent.name:
        return None
    return path.parent.parent.name.replace("--", "/", 1)


def file_gb(snapshot: ModelSnapshot) -> float:
    return sum(size for _, size in snapshot.gguf_files) / measurements.GB


def make_reference(repo: str, hardware: HardwareProfile, piece, log_dir: Path) -> dict:
    """Measures the reference a repo's formats are compared with: its original weights, from
    the Hugging Face cache, converted to a 16-bit GGUF and run in llama.cpp (whose perplexity
    matches transformers'). When that does not fit in this PC's memory, Q8_0 made from it,
    which loses almost nothing. The files are made in a folder of their own, then deleted.
    Returns the saved result."""
    import shutil
    import sys

    from ocq import quality
    from ocq.local_model import cached_hf_model
    from ocq.model_analyzer import resolve_model
    from ocq.plan import free_disk_gb
    from ocq.quantizers.gguf import intermediate_precision, run_tool
    from ocq.quantizers.llama_cpp import converter_script, tool_path

    folder = cached_hf_model(repo)
    if folder is None:
        raise OcqError(f"The original weights of {repo} are not on this PC to measure.")
    original = resolve_model(str(folder), hardware, None, 4096, "f16", use_llmfit=False)
    precision = intermediate_precision(original.stored_precision)
    size_gb = original.params_b * 1e9 * (4 if precision == "f32" else 2) / measurements.GB
    work = quality.DATASETS.parent / "reference"
    free = free_disk_gb(work.parent)
    if free is not None and free < size_gb * 1.6:  # the 16-bit file, then Q8_0 beside it
        raise OcqError(
            f"Measuring the original of {repo} needs about {size_gb * 1.6:.1f} GB of free disk "
            f"for a while; {free:.1f} GB is free."
        )
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        model = work / f"{repo.replace('/', '--')}-{precision}.gguf"
        print(f"Converting the original to a {precision} GGUF ...", flush=True)
        run_tool(
            [sys.executable, str(converter_script()), str(folder), "--outfile", str(model),
             "--outtype", precision],
            log=log_dir / "reference-convert.log",
            progress=True,
        )  # fmt: skip
        option, snapshot = fitted(model, hardware)
        if option.run_mode == RunMode.UNFIT:
            smaller = work / f"{repo.replace('/', '--')}-Q8_0.gguf"
            print("The original does not fit in memory here: making Q8_0 from it ...", flush=True)
            run_tool(
                [str(tool_path("llama-quantize", hardware)), str(model), str(smaller), "Q8_0"],
                log=log_dir / "reference-quantize.log",
                progress=True,
            )
            model.unlink()
            model = smaller
            option, snapshot = fitted(model, hardware)
        place, device = placement(option)
        print(f"Measuring the reference ({option.label}) ...", flush=True)
        result = quality.llama_perplexity(
            tool_path("llama-perplexity", hardware),
            model,
            quality.wikitext(),
            piece,
            place=place,
            kv_cache="f16",
            env=llama_cache_env(snapshot.architecture, "f16"),
            log=log_dir / "reference-perplexity.log",
        )
        quality.record(
            result, piece, model=repo, label=option.label, hardware=hardware, device=device
        )
        return quality.load()[-1]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def fitted(model: Path, hardware: HardwareProfile):
    """The row llama.cpp would run a GGUF as for llama-perplexity: 4 chunks of 512 at once."""
    from ocq.model_analyzer import quant_options, resolve_model

    snapshot = resolve_model(str(model), hardware, None, 2048, "f16", use_llmfit=False)
    (option,) = quant_options(snapshot, hardware, 2048, "f16")
    return option, snapshot


# The whole benchmark, for the CLI and the app alike -----------------------------------------


class SpeedNotMeasured(OcqError):
    """The model is one only transformers runs: its quality can be measured, not its speed."""


SPEED_STEP = "Measuring the speed"
QUALITY_STEP = "Measuring the quality"
COMPARE_STEP = "Comparing with the original"


@dataclass
class BenchOutcome:
    """What ocq bench (and the app's Benchmark page) measured."""

    model_id: str  # the name it is saved under: the repo a QuantPilot file was made from
    label: str
    option: QuantOption
    device: str = ""
    speed: BenchReport | None = None  # None for a model only transformers runs
    runs: int = 0  # runs of this setup on this PC, this one included
    quality: object | None = None  # quality.Perplexity
    piece: object | None = None  # quality.Slice
    kept: object | None = None  # quality.MeasuredKept
    note: str = ""  # why there is nothing to compare with, or that this is the original


def run_benchmark(
    ref: str,
    hardware: HardwareProfile,
    *,
    piece=None,
    context: int = 4096,
    kv_cache: str = "f16",
    llmfit_bin: str | None = None,
    on_speed=None,
) -> BenchOutcome:
    """Speed (GGUF), then quality when piece (a quality.Slice) is given, then what the format
    keeps against the original. Prints a line as each step starts. on_speed(outcome): called
    once the speed is in, so the CLI can show it before the longer quality measurement."""
    from ocq import quality
    from ocq.local_model import cached_hf_model, parse_model_ref
    from ocq.model_analyzer import active_params_b, quant_options, resolve_model
    from ocq.quantizers.llama_cpp import tool_path

    path = parse_model_ref(ref)[1] or cached_hf_model(ref)
    if path is None:
        raise OcqError(
            f"{ref} is not on this PC. Benchmarks measure a model already here: a .gguf file or "
            "folder, an Ollama model, or a model folder. To measure a Hugging Face model, "
            "quantize it first."
        )
    snapshot = resolve_model(ref, hardware, llmfit_bin, context, kv_cache)
    if not snapshot.gguf_files:
        if piece is None:
            raise SpeedNotMeasured(
                "Its speed can be measured for GGUF models (llama.cpp) only. Measure its "
                "quality instead."
            )
        return transformers_quality(snapshot, path, piece, hardware)

    if path.is_dir():
        path = path / snapshot.gguf_files[0][0]
    model_id = made_from(path) or snapshot.model_id
    suite = STANDARD
    prompts = prompts_for(suite, snapshot.context_length)
    planned = max(context, max(prompts) + suite.tokens)  # llama-bench sizes the context so
    (option,) = quant_options(snapshot, hardware, planned, kv_cache)
    place, device = placement(option)
    print(f"{SPEED_STEP} ({suite.repeats} runs of each test) ...", flush=True)
    report = llama_bench(
        tool_path("llama-bench", hardware),
        path,
        suite,
        architecture=snapshot.architecture,
        kv_cache=kv_cache,
        place=place,
        log=measurements.STORE.parent / "bench.log",
        model_context=snapshot.context_length,
    )
    save(
        report, suite, device=device, read_gb=file_gb(snapshot) * snapshot.read_fraction,
        model=model_id, label=option.label, context=planned, kv_cache=kv_cache,
        hardware=hardware, file=path.name, active_params_b=active_params_b(snapshot),
    )  # fmt: skip
    runs = len(
        measurements.runs_of(
            model_id, option.label, LLAMA_CPP, hardware, measurements.placed(device)
        )
    )
    outcome = BenchOutcome(model_id, option.label, option, device, report, runs, piece=piece)
    if on_speed:
        on_speed(outcome)
    if piece is None:
        return outcome

    text = quality.wikitext()
    reading = report.prompt_speeds.get(min(report.prompt_speeds, default=0))
    chunks = piece.chunks or 600
    minutes = chunks * quality.CHUNK / reading / 60 if reading else None
    about = "a few minutes" if minutes is None else (
        f"about {minutes:.0f} min" if minutes >= 1 else "under a minute"
    )  # fmt: skip
    print(f"{QUALITY_STEP} on {quality.DATASET} ({about}) ...", flush=True)
    outcome.quality = quality.llama_perplexity(
        tool_path("llama-perplexity", hardware),
        path,
        text,
        piece,
        place=place,
        kv_cache=kv_cache,
        env=llama_cache_env(snapshot.architecture, kv_cache),
        log=quality.STORE.parent / "perplexity.log",
        build=report.build,
    )
    quality.record(
        outcome.quality, piece, model=model_id, label=option.label, kv_cache=kv_cache,
        file=path.name, hardware=hardware, device=device,
    )  # fmt: skip
    compare(outcome, hardware)
    return outcome


def transformers_quality(snapshot, path: Path, piece, hardware: HardwareProfile) -> BenchOutcome:
    """Perplexity of a model transformers runs: the original, or AWQ, GPTQ, bitsandbytes."""
    from ocq import quality
    from ocq.model_analyzer import quant_options

    stored = next(
        o for o in quant_options(snapshot, hardware, 4096, "f16") if o.row_kind == "stored"
    )
    label = stored.label.removesuffix(" (as stored)").strip()
    model_id = made_from(path) or snapshot.model_id
    device = "gpu" if hardware.has_cuda else "cpu"
    where = "the NVIDIA GPU" if hardware.has_cuda else "the CPU"
    print(f"{QUALITY_STEP} on {quality.DATASET}, in transformers on {where} ...", flush=True)
    try:
        result = quality.transformers_perplexity(path, quality.wikitext(), piece)
    except OcqError:
        raise
    except Exception as exc:  # the model's own code, CUDA kernels, out of memory
        raise OcqError(f"transformers could not measure it: {type(exc).__name__}: {exc}") from exc
    quality.record(result, piece, model=model_id, label=label, hardware=hardware, device=device)
    outcome = BenchOutcome(model_id, label, stored, device, quality=result, piece=piece)
    compare(outcome, hardware)
    return outcome


def compare(outcome: BenchOutcome, hardware: HardwareProfile) -> None:
    """What the format keeps against the original, measuring the original first (once, from
    the Hugging Face cache) when it has not been."""
    from ocq import quality
    from ocq.local_model import cached_hf_model

    if outcome.label.upper() in quality.FULL_PRECISION:
        outcome.note = "original"
        return
    print(f"{COMPARE_STEP} ...", flush=True)
    model_id = outcome.model_id
    repo = model_id if not model_id.startswith(("local/", "ollama/")) else None
    if (
        repo
        and quality.reference_for(repo, outcome.piece.name) is None
        and cached_hf_model(repo) is not None
    ):
        print(f"Measuring the original of {repo} once, from the Hugging Face cache ...", flush=True)
        try:
            make_reference(repo, hardware, outcome.piece, quality.STORE.parent)
        except OcqError as exc:
            outcome.note = f"Could not measure the original: {exc}"
    outcome.kept = quality.measured_kept(model_id, outcome.label)
    if outcome.kept is None and not outcome.note:
        outcome.note = "The original weights of this model are not on this PC to compare with."
