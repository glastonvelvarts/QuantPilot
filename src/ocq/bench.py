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
