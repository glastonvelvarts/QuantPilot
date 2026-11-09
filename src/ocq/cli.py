"""Terminal CLI for One-Click Quantizer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ocq import measurements
from ocq.engine import GGUF_TYPES
from ocq.goals import goal_label
from ocq.hardware import detect_hardware
from ocq.model_analyzer import DEFAULT_CONTEXT, active_params_b, quant_options, resolve_model
from ocq.pipeline import run_optimize
from ocq.plan import free_disk_gb, generate_plan
from ocq.quantizers import backend_status, get_quantizer
from ocq.runtimes import llama_cache_env
from ocq.types import (
    KV_CACHE_TYPES,
    LLAMA_CPP,
    Fact,
    FactKind,
    FitLevel,
    OcqError,
    OptimizationGoal,
    OptimizationPlan,
    OptimizeResult,
    QuantAlgorithm,
    QuantChoices,
)

app = typer.Typer(
    name="ocq",
    help="One-Click Quantizer: inspect this PC, see which models it can run, quantize them. "
    "Run ocq alone for the full-screen app.",
)
console = Console()


@app.callback(invoke_without_command=True)
def start(ctx: typer.Context) -> None:
    """ocq with no command: the full-screen app in a terminal, the help text anywhere else."""
    from ocq import env_file

    env_file.load()  # the keys saved in Settings (a Hugging Face token), unless the shell has them
    if ctx.invoked_subcommand is not None:
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        cmd_tui()
    else:
        console.print(ctx.get_help())


@app.command("tui")
def cmd_tui() -> None:
    """Open the full-screen app: arrow keys and menus instead of flags."""
    from ocq.tui.app import run_tui

    run_tui()


KIND_COLORS = {
    FactKind.EXACT: "green",
    FactKind.CROSS_CHECKED: "green",
    FactKind.ESTIMATE: "yellow",
    FactKind.UNKNOWN: "red",
}

FIT_COLORS = {
    FitLevel.PERFECT: "green",
    FitLevel.GOOD: "green",
    FitLevel.MARGINAL: "yellow",
    FitLevel.TOO_TIGHT: "red",
}

ModelArg = Annotated[
    str | None,
    typer.Argument(help="HuggingFace id (org/name) or path to a local model folder"),
]
ModelPath = Annotated[
    Path | None,
    typer.Option("--model-path", "-m", help="Local model folder saved with save_pretrained()"),
]
LlmfitBin = Annotated[str | None, typer.Option("--llmfit-bin", help="Path to the llmfit binary")]
Goal = Annotated[str, typer.Option("--goal", "-g", help="balanced, speed, quality or memory")]
Backend = Annotated[
    str | None,
    typer.Option("--backend", "-b", help="gguf, awq, gptq, bnb or nvfp4 (default: auto)"),
]
Bits = Annotated[int | None, typer.Option("--bits", help="Bit width, for example 4 or 8")]
GgufType = Annotated[
    str | None, typer.Option("--gguf-type", help=f"Exact GGUF type: {', '.join(GGUF_TYPES)}")
]
GroupSize = Annotated[
    int | None, typer.Option("--group-size", help="AWQ/GPTQ: weights sharing one scale")
]
Samples = Annotated[
    int | None, typer.Option("--samples", help="AWQ/GPTQ: number of calibration samples")
]
SeqLen = Annotated[
    int | None, typer.Option("--seq-len", help="AWQ/GPTQ: tokens per calibration sample")
]
DoubleQuant = Annotated[
    bool | None,
    typer.Option(
        "--double-quant/--single-quant",
        help="bitsandbytes 4-bit: also quantize the per-block scales, saving about 0.37 bits "
        "per weight (default: on, off for --goal speed). Implies -b bnb --bits 4.",
    ),
]
Fp4Activations = Annotated[
    bool | None,
    typer.Option(
        "--fp4-activations/--fp4-weights-only",
        help="NVFP4: quantize activations to FP4 too (fast on Blackwell GPUs, needs calibration) "
        "or weights only (default: activations on Blackwell, weights only elsewhere). "
        "Implies -b nvfp4.",
    ),
]
Output = Annotated[Path | None, typer.Option("--output", "-o", help="Where to write the result")]
Context = Annotated[
    int, typer.Option("--context", "-c", help="Context size in tokens the model will run with")
]
KvCache = Annotated[
    str,
    typer.Option(
        "--kv-cache",
        help="Cache type for the context: f16 (default), q8_0 (about half the memory, "
        "almost no quality loss) or q4_0 (about a quarter, some loss). llama.cpp uses it as "
        "is, vLLM uses fp8 for both, transformers stays at f16.",
    ),
]


def fail(exc: Exception) -> typer.Exit:
    console.print(f"[red]Error:[/red] {exc}")
    return typer.Exit(1)


def read_goal(value: str) -> OptimizationGoal:
    try:
        return OptimizationGoal(value.lower())
    except ValueError as exc:
        valid = ", ".join(goal.value for goal in OptimizationGoal)
        raise typer.BadParameter(f"Use one of: {valid}") from exc


def read_choices(
    backend: str | None,
    bits: int | None,
    gguf_type: str | None,
    group_size: int | None,
    samples: int | None,
    seq_len: int | None,
    context: int,
    double_quant: bool | None = None,
    fp4_activations: bool | None = None,
    kv_cache: str = "f16",
) -> QuantChoices:
    algorithm = None
    if backend:
        try:
            algorithm = QuantAlgorithm(backend.lower())
        except ValueError as exc:
            valid = ", ".join(a.value for a in QuantAlgorithm)
            raise typer.BadParameter(f"Use one of: {valid}") from exc
    return QuantChoices(
        algorithm=algorithm,
        bits=bits,
        gguf_type=gguf_type,
        group_size=group_size,
        calibration_samples=samples,
        calibration_seq_len=seq_len,
        context=context,
        kv_cache=kv_cache,
        double_quant=double_quant,
        fp4_activations=fp4_activations,
    )


def read_kv_cache(value: str) -> str:
    cache_type = value.lower()
    if cache_type not in KV_CACHE_TYPES:
        raise typer.BadParameter(f"Use one of: {', '.join(KV_CACHE_TYPES)}")
    return cache_type


def model_ref(model: str | None, model_path: Path | None) -> str:
    if model_path is not None:
        if model:
            raise typer.BadParameter("Pass either MODEL or --model-path (-m), not both.")
        return str(model_path)
    if not model:
        raise typer.BadParameter(
            "Pass a HuggingFace model id or --model-path (-m) to a local checkpoint folder."
        )
    return model


def default_output(model_ref: str) -> Path:
    from ocq.local_model import parse_model_ref, slug_local

    display_id, _ = parse_model_ref(model_ref)
    return Path.cwd() / "ocq-output" / slug_local(display_id)


def colored_fit(fit_level) -> str:
    color = FIT_COLORS.get(fit_level, "white")
    return f"[{color}]{fit_level}[/{color}]"


@app.command("profile")
def cmd_profile(llmfit_bin: LlmfitBin = None) -> None:
    """Show what this PC has: CPU, RAM, GPU, memory bandwidth, free disk."""
    hw = detect_hardware(llmfit_bin)

    bandwidth = []
    if hw.ram_bandwidth_gbs > 0:
        bandwidth.append(f"{hw.ram_bandwidth_gbs:.1f} GB/s peak")
    if hw.measured_ram_bandwidth_gbs > 0:
        bandwidth.append(f"{hw.measured_ram_bandwidth_gbs:.1f} GB/s live")
    ram_bandwidth = f" ({', '.join(bandwidth)})" if bandwidth else ""

    table = Table(title="Hardware Profile")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("System", hw.os_name)
    table.add_row("CPU", f"{hw.cpu_name} ({hw.cpu_cores} cores)")
    table.add_row(
        "RAM", f"{hw.available_ram_gb:.1f} free / {hw.total_ram_gb:.1f} GB{ram_bandwidth}"
    )
    table.add_row("Backend", hw.backend)
    table.add_row("GPU", "yes" if hw.has_gpu else "no")
    for index, gpu in enumerate(hw.gpus):
        vram = f"{gpu.vram_gb:.1f} GB" if gpu.vram_gb is not None else "?"
        if gpu.free_vram_gb is not None:
            vram = f"{gpu.free_vram_gb:.2f} free / {vram}"
        speeds = []
        if gpu.bandwidth_gbs > 0:
            speeds.append(f"{gpu.bandwidth_gbs:.1f} GB/s spec")
        if gpu.measured_bandwidth_gbs > 0:
            speeds.append(f"{gpu.measured_bandwidth_gbs:.1f} GB/s live")
        speed = f" ({', '.join(speeds)})" if speeds else ""
        cards = f" x{gpu.count}" if gpu.count > 1 else ""
        table.add_row(f"GPU {index}", f"{gpu.name}{cards} - {vram}{speed}")
    if hw.has_gpu and hw.pcie_bandwidth_gbs > 0 and not hw.unified_memory:
        table.add_row("PCIe Bandwidth", f"{hw.pcie_bandwidth_gbs:.1f} GB/s")
    table.add_row("Free disk", f"{free_disk_gb(Path.cwd()):.0f} GB (current folder)")
    table.add_row("Detected by", "llmfit" if hw.source == "llmfit" else "built-in probe")
    console.print(table)


@app.command("inspect")
def cmd_inspect(
    model: ModelArg = None,
    model_path: ModelPath = None,
    llmfit_bin: LlmfitBin = None,
) -> None:
    """Show what QuantPilot knows about a model, where each fact comes from and how sure it is."""
    ref = model_ref(model, model_path)
    try:
        snapshot = resolve_model(ref, None, llmfit_bin)
    except OcqError as exc:
        raise fail(exc) from exc

    facts = snapshot.facts
    console.print(Panel(f"[bold]{snapshot.model_id}[/bold]\n{fact_summary(snapshot)}"))
    rows = (
        ("Parameters", facts.parameters),
        ("Precision", facts.precision),
        ("Stored bytes", facts.stored_bytes),
        ("Architecture", facts.architecture),
        ("Context length", facts.context_length),
        ("KV bytes/token", facts.kv_cache),
        ("Used per token", facts.active_parameters),
        ("Quantization", facts.quantization),
    )
    # Three columns: long sources and notes wrap inside the last one, the kinds line up.
    grid = Table.grid(padding=(0, 2, 1, 0))
    grid.add_column(no_wrap=True, style="bold")
    grid.add_column(no_wrap=True)
    grid.add_column()
    for name, fact in rows:
        color = KIND_COLORS[fact.kind]
        details = [f"[bold]{escape(fact_value(fact))}[/bold]"]
        details += [f"[dim]from {escape(fact.source)}[/dim]"] if fact.source else []
        details += [f"[dim]{escape(fact.note)}[/dim]"] if fact.note else []
        grid.add_row(name, f"[{color}]{fact.kind}[/{color}]", "\n".join(details))
    console.print(grid, highlight=False)
    if snapshot.catalog_source == "llmfit" and not facts.parameters.known:
        console.print("[dim]HuggingFace could not be reached: only llmfit's catalog entry.[/dim]")


def fact_value(fact: Fact) -> str:
    if fact.value is None:
        return "-"
    if isinstance(fact.value, int):
        return f"{fact.value:,}"
    return str(fact.value)


def fact_summary(snapshot) -> str:
    """One line, for example: GPTQ 4-bit, 1.54B params (cross-checked), 1.07 GB of weights."""
    facts = snapshot.facts
    quantized = facts.quantization.value if snapshot.is_quantized else "not quantized"
    params = "unknown params"
    if facts.parameters.known:
        params = f"{snapshot.parameter_count} params ({facts.parameters.kind})"
    size = f"{snapshot.original_gb:.2f} GB" if snapshot.original_gb is not None else "unknown size"
    return f"{quantized}, {params}, {size} of weights"


@app.command("verify")
def cmd_verify(
    model: Annotated[str, typer.Argument(help="A local .gguf file, or ollama:name:tag")],
    context: Context = DEFAULT_CONTEXT,
    kv_cache: KvCache = "f16",
    n_cpu_moe: Annotated[
        int | None,
        typer.Option(
            "--n-cpu-moe",
            help="Experts of the first N layers in RAM (default: what QuantPilot plans for this "
            "PC)",
        ),
    ] = None,
    save_log: Annotated[
        Path | None, typer.Option("--save-log", help="Also write llama.cpp's whole log here")
    ] = None,
    llmfit_bin: LlmfitBin = None,
) -> None:
    """Load a GGUF in llama.cpp once and compare its memory with QuantPilot's prediction."""
    from ocq.local_model import parse_model_ref
    from ocq.quantizers.llama_cpp import tool_path
    from ocq.verify import verify

    kv_cache = read_kv_cache(kv_cache)
    try:
        hw = detect_hardware(llmfit_bin)
        snapshot = resolve_model(model, hw, llmfit_bin, context, kv_cache)
        path = parse_model_ref(model)[1]
        if not snapshot.gguf_files or path is None or not path.is_file():
            raise OcqError("ocq verify takes one local .gguf file, or ollama:name:tag.")
        if n_cpu_moe is None:
            (row,) = quant_options(snapshot, hw, context, kv_cache)
            n_cpu_moe = row.experts_in_ram
        log, checks, _ = verify(
            snapshot,
            path,
            hw,
            tool_path("llama-server", hw),
            context,
            kv_cache,
            n_cpu_moe,
            save_log,
        )
    except OcqError as exc:
        if save_log and save_log.exists():
            console.print(f"[dim]llama.cpp's log: {save_log}[/dim]")
        raise fail(exc) from exc

    offloaded, total = log.layers_on_gpu
    where = "on the CPU, no GPU" if log.cpu_only else f"{offloaded}/{total} layers on the GPU"
    console.print(
        f"[bold]{snapshot.model_id}[/bold] in llama.cpp: {log.context:,} tokens, "
        f"{log.conversations} conversation(s), {where}"
        + (f", experts of {n_cpu_moe} layers in RAM" if n_cpu_moe else "")
        + (f", cache {log.cache_types}" if log.cache_types else "")
    )
    table = Table(title="QuantPilot's prediction vs llama.cpp's own log (MiB)")
    for column in ("Memory", "QuantPilot", "llama.cpp", "Difference", ""):
        table.add_column(column, justify="right" if column != "Memory" else "left")
    for check in checks:
        mark = "[green]ok[/green]" if check.ok else "[red]differs[/red]"
        number = "{:,.0f}" if check.unit == "layers" else "{:,.2f}"
        table.add_row(
            check.name,
            number.format(check.predicted_mib),
            number.format(check.actual_mib),
            ("{:+,.0f}" if check.unit == "layers" else "{:+,.2f}").format(
                check.predicted_mib - check.actual_mib
            ),
            mark,
        )
    console.print(table)
    console.print(
        f"Working memory llama-server reserved: {log.gpu_compute_mib:,.2f} MiB on the GPU, "
        f"{log.cpu_compute_mib:,.2f} MiB on the CPU, {log.output_mib:,.2f} MiB for the logits. "
        "QuantPilot's estimate (in the context column of ocq feasibility) is an upper bound "
        "of the first and the last together."
    )
    if save_log:
        console.print(f"[dim]llama.cpp's log: {save_log}[/dim]")
    if not offloaded and not log.cpu_only:
        console.print("[dim]No layer is on the GPU, so the weights were not compared.[/dim]")
    if not all(check.ok for check in checks):
        raise typer.Exit(1)


@app.command("bench")
def cmd_bench(
    model: Annotated[
        str,
        typer.Argument(
            help="A local .gguf file or folder, ollama:name:tag, or a model folder or "
            "Hugging Face id already on this PC (ocq models)"
        ),
    ],
    quality: Annotated[
        bool,
        typer.Option(
            "--quality",
            help="Also measure its quality: perplexity on WikiText-2 (a few minutes)",
        ),
    ] = False,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Quality on the whole WikiText-2 test text, not its first 40 chunks (much "
            "longer; comparable to published numbers)",
        ),
    ] = False,
    context: Context = DEFAULT_CONTEXT,
    kv_cache: KvCache = "f16",
    llmfit_bin: LlmfitBin = None,
) -> None:
    """Run the standard benchmark on a model of this PC and save it to its benchmark history."""
    from ocq import bench
    from ocq import quality as measured_quality
    from ocq.quantizers.llama_cpp import tool_path

    kv_cache = read_kv_cache(kv_cache)
    piece = measured_quality.FULL if full else measured_quality.FIXED
    quality = quality or full
    try:
        hw = detect_hardware(llmfit_bin)
        path = model_on_this_pc(model)
        snapshot = resolve_model(model, hw, llmfit_bin, context, kv_cache)
        if not snapshot.gguf_files:
            if not quality:
                raise OcqError(
                    "ocq bench measures the speed of GGUF models (llama.cpp). For this one, "
                    "add --quality to measure its quality."
                )
            bench_transformers_quality(snapshot, path, piece, hw)
            return
        if path.is_dir():
            path = path / snapshot.gguf_files[0][0]
        model_id = bench.made_from(path) or snapshot.model_id
        option, device, place, report = bench_gguf_speed(snapshot, path, model_id, context,
                                                          kv_cache, hw)  # fmt: skip
        if quality:
            text = measured_quality.wikitext()
            reading = report.prompt_speeds.get(min(report.prompt_speeds, default=0))
            chunks = piece.chunks or 600
            if reading:
                minutes = chunks * measured_quality.CHUNK / reading / 60
                about = f"about {minutes:.0f} min" if minutes >= 1 else "under a minute"
            else:
                about = "a few minutes"
            console.print(f"Measuring its quality on {measured_quality.DATASET} ({about}) ...")
            result = measured_quality.llama_perplexity(
                tool_path("llama-perplexity", hw),
                path,
                text,
                piece,
                place=place,
                kv_cache=kv_cache,
                env=llama_cache_env(snapshot.architecture, kv_cache),
                log=Path(measured_quality.STORE).parent / "perplexity.log",
                build=report.build,
            )
            measured_quality.record(
                result, piece, model=model_id, label=option.label, kv_cache=kv_cache,
                file=path.name, hardware=hw, device=device,
            )  # fmt: skip
            print_quality(result, piece)
    except OcqError as exc:
        raise fail(exc) from exc


def model_on_this_pc(model: str) -> Path:
    """The file or folder of a model on this PC: a path, ollama:name:tag, or a Hugging Face id
    whose files are all in the Hugging Face cache already."""
    from ocq.local_model import has_tokenizer, parse_model_ref, weight_files

    path = parse_model_ref(model)[1]
    if path is not None:
        return path
    not_here = OcqError(
        f"{model} is not on this PC. ocq bench measures a model already here: a .gguf file or "
        "folder, ollama:name:tag, or a model folder (ocq models lists them). To measure a "
        "Hugging Face model, quantize it first with ocq optimize."
    )
    from huggingface_hub import try_to_load_from_cache

    try:
        config = try_to_load_from_cache(model, "config.json")
    except Exception as exc:  # not a repo id
        raise not_here from exc
    if not isinstance(config, str):  # never downloaded
        raise not_here
    folder = Path(config).parent  # the cached snapshot: its files QuantPilot downloaded
    # Often only config.json is cached (QuantPilot reads it to size a model): not a model.
    files = weight_files(folder)
    if not files or not all(file.is_file() for file in files) or not has_tokenizer(folder):
        raise not_here
    return folder


def bench_gguf_speed(snapshot, path: Path, model_id: str, context: int, kv_cache: str, hw):
    """The standard suite, saved and printed. Returns the row it ran as, its placement and
    the report."""
    from ocq import bench
    from ocq.quantizers.llama_cpp import tool_path

    suite = bench.STANDARD
    prompts = bench.prompts_for(suite, snapshot.context_length)
    # The fit at the longest test, as llama-bench sizes the context to it.
    planned = max(context, max(prompts) + suite.tokens)
    (option,) = quant_options(snapshot, hw, planned, kv_cache)
    place, device = bench.placement(option)
    console.print(
        f"[bold]{escape(snapshot.model_id)}[/bold] {option.label}: running the standard "
        f"benchmark ({suite.repeats} runs of each test) ...",
        highlight=False,
    )
    report = bench.llama_bench(
        tool_path("llama-bench", hw),
        path,
        suite,
        architecture=snapshot.architecture,
        kv_cache=kv_cache,
        place=place,
        log=Path(measurements.STORE).parent / "bench.log",
        model_context=snapshot.context_length,
    )
    placed = measurements.placed(device)
    earlier = len(measurements.runs_of(model_id, option.label, LLAMA_CPP, hw, placed))
    bench.save(
        report,
        suite,
        device=device,
        read_gb=bench.file_gb(snapshot) * snapshot.read_fraction,
        model=model_id,
        label=option.label,
        context=planned,
        kv_cache=kv_cache,
        hardware=hw,
        file=path.name,
        active_params_b=active_params_b(snapshot),
    )
    print_bench(report, option, device, kv_cache, earlier + 1)
    return option, device, place, report


def bench_transformers_quality(snapshot, path: Path, piece, hw) -> None:
    """Perplexity of a model transformers runs: the original, or AWQ, GPTQ, bitsandbytes."""
    import importlib.util

    from ocq import bench
    from ocq import quality as measured_quality

    if not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")):
        raise OcqError(
            "Measuring this model's quality needs torch and transformers: uv sync --extra gguf"
        )
    stored = next(o for o in quant_options(snapshot, hw, 4096, "f16") if o.row_kind == "stored")
    label = stored.label.removesuffix(" (as stored)").strip()
    model_id = bench.made_from(path) or snapshot.model_id
    where = "the NVIDIA GPU" if hw.has_cuda else "the CPU"
    console.print(
        f"[bold]{escape(snapshot.model_id)}[/bold] {label}: measuring its quality on "
        f"{measured_quality.DATASET} in transformers, on {where} ...",
        highlight=False,
    )
    try:
        result = measured_quality.transformers_perplexity(path, measured_quality.wikitext(), piece)
    except OcqError:
        raise
    except Exception as exc:  # the model's own code, CUDA kernels, out of memory
        raise OcqError(f"transformers could not measure it: {type(exc).__name__}: {exc}") from exc
    device = "gpu" if hw.has_cuda else "cpu"
    measured_quality.record(result, piece, model=model_id, label=label, hardware=hw, device=device)
    print_quality(result, piece)


def print_quality(result, piece) -> None:
    from ocq import quality as measured_quality

    of = f"its first {result.chunks}" if piece.chunks else f"all {result.chunks}"
    console.print(
        f"[bold]Quality[/bold]      perplexity {result.value:.4f} ± {result.error:.4f} "
        f"(lower is better) on {measured_quality.DATASET}, {of} chunks of "
        f"{measured_quality.CHUNK} tokens",
        highlight=False,
    )
    console.print(
        "[dim]Formats of one model read exactly the same tokens, so their perplexities compare "
        "much more closely than the ± says. Saved to this PC's quality history.[/dim]"
    )


def print_bench(report, option, device: str, kv_cache: str, runs: int) -> None:
    where = {"gpu": "all on the GPU", "cpu": "on the CPU"}.get(device, device.replace("gpu", "GPU"))
    console.print(
        f"llama.cpp {report.build or ''}, {where}, {kv_cache} cache".replace("  ", " "),
        style="dim",
        highlight=False,
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    expected = (
        f"QuantPilot expected about {option.estimated_tps:.0f}"
        if option.speed_basis == measurements.MEASURED
        else f"QuantPilot expected up to {option.estimated_tps:.0f} (this PC's memory speed)"
    )
    table.add_row(
        "Writing",
        f"{report.tokens_per_second:,.1f} tok/s (± {report.stddev:.1f})  [dim]{expected}[/dim]",
    )
    lengths = sorted(report.prompt_speeds)
    table.add_row(
        "Reading",
        ", ".join(f"{report.prompt_speeds[n]:,.0f} tok/s for {n:,} tokens" for n in lengths),
    )
    table.add_row(
        "First token",
        ", ".join(f"{report.first_token_seconds(n):.2f} s after {n:,} tokens" for n in lengths),
    )
    console.print(table)
    console.print(
        f"[dim]Saved to this PC's benchmark history ({runs} run{'s' * (runs > 1)} of this "
        "setup).[/dim]"
    )


@app.command("models")
def cmd_models(
    show_all: Annotated[
        bool, typer.Option("--all", help="Also list partial downloads (for example config only)")
    ] = False,
    folders: Annotated[
        list[Path] | None,
        typer.Option("--dir", "-d", help="Also search this folder (can be given more than once)"),
    ] = None,
    context: Context = DEFAULT_CONTEXT,
    kv_cache: KvCache = "f16",
    llmfit_bin: LlmfitBin = None,
) -> None:
    """List the models already on this PC and how each one fits it, as it is on disk."""
    from ocq.discovery import SOURCES, find_models, searched_locations, tilde

    kv_cache = read_kv_cache(kv_cache)
    try:
        models = find_models(folders)
        hw = detect_hardware(llmfit_bin)
    except OcqError as exc:
        raise fail(exc) from exc
    shown = [m for m in models if m.complete or show_all]
    table = Table(title=f"Models on this PC ({len(shown)})")
    table.add_column("Model", overflow="fold")
    table.add_column("Source")
    table.add_column("Format")
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Fit")
    table.add_column("Runs on")
    table.add_column("tok/s", justify="right")
    table.add_column("Use with ocq", overflow="fold")
    for model in sorted(shown, key=lambda m: (m.source, m.label.lower())):
        size = f"{model.size_bytes / 1e9:.2f} GB" if model.size_bytes else "-"
        label = escape(model.label)
        note = model.note
        fit, place, speed = local_fit(model, hw, context, kv_cache)
        if fit.startswith("?") and len(fit) > 1:
            note, fit = "; ".join(filter(None, [note, fit[2:]])), "?"
        if note:
            color = "dim" if model.complete else "yellow"
            label += f"\n[{color}]{escape(note)}[/{color}]"
        kind = f"{model.format} {model.quantization}".strip()
        table.add_row(label, model.source, escape(kind), size, fit, place, speed, escape(model.ref))
    console.print(table)
    console.print(
        f"[dim]Fit as stored, at {context:,} tokens of context, in the memory free right now "
        f"({hw.free_vram_gb:.2f} GB VRAM{gpu_note(hw)}, {hw.available_ram_gb:.2f} GB RAM). "
        "ocq feasibility <model> shows the other formats.[/dim]"
    )
    hidden = len(models) - len(shown)
    if hidden:
        console.print(
            f"[dim]{hidden} partial download(s) hidden, most with only config.json "
            "(fetched to size a model, not to run it): ocq models --all[/dim]"
        )
    searched = searched_locations(folders)
    places = ", ".join(f"{source} {tilde(path)}" for source, path in searched)
    console.print(f"[dim]Searched: {escape(places) or 'no model folders found'}[/dim]")
    missing = [s for s in SOURCES if s not in {source for source, _ in searched}]
    if missing:
        console.print(f"[dim]Not found on this PC: {', '.join(missing)}[/dim]")


def local_fit(model, hw, context: int, kv_cache: str) -> tuple[str, str, str]:
    """(fit, runs on, tok/s) of a found model as it is on disk, read offline from its files.
    "? reason" when it cannot be worked out."""
    from ocq.discovery import GGUF, MLX_LAYOUT, SAFETENSORS

    if not model.complete:
        return "", "", ""
    if model.format not in (GGUF, SAFETENSORS, MLX_LAYOUT):
        return "?", "", ""
    # The local copy, not the Hub: no network, and exactly the files on this PC.
    ref = model.ref if model.ref.startswith("ollama:") else str(model.path)
    try:
        snapshot = resolve_model(ref, hw, None, context, kv_cache)
        option = quant_options(snapshot, hw, context, kv_cache)[0]
    except OcqError as exc:
        return f"? {exc}", "", ""
    except (OSError, ValueError, KeyError) as exc:  # a broken file must not hide the others
        return f"? could not read it ({type(exc).__name__})", "", ""
    if snapshot.is_mlx and not hw.apple_silicon:
        return "[red]needs a Mac[/red]", "", ""
    return colored_fit(option.fit_level), runs_on(option), speed_cell(option)


@app.command("backends")
def cmd_backends(llmfit_bin: LlmfitBin = None) -> None:
    """Show which quantization backends can run on this PC."""
    table = Table(title="Quantization Backends")
    table.add_column("Backend")
    table.add_column("Status")
    table.add_column("Detail")
    for status in backend_status(detect_hardware(llmfit_bin)):
        state = "[green]ready[/green]" if status.ready else "[red]not ready[/red]"
        table.add_row(status.algorithm.value, state, status.detail)
    console.print(table)


MARKS = {"good": "[green]✓[/green]", "tradeoff": "[yellow]△[/yellow]", "bad": "[red]✗[/red]"}
CONFIDENCE_COLORS = {"Low": "red", "Medium": "yellow", "High": "green", "Very High": "green"}


@app.command("recommend")
def cmd_recommend(
    models: Annotated[
        list[str] | None,
        typer.Argument(help="Models to choose between (default: this PC's, plus popular ones)"),
    ] = None,
    workload: Annotated[
        str,
        typer.Option(
            "--workload",
            "-w",
            help="chat, coding, rag, agentic, reasoning, long-context, low-latency, throughput",
        ),
    ] = "chat",
    context: Annotated[
        int, typer.Option("--context", "-c", help="Tokens of context (default: the workload's)")
    ] = 0,
    kv_cache: Annotated[
        list[str] | None,
        typer.Option("--kv-cache", help="Cache types to try (default: f16 and q8_0)"),
    ] = None,
    folders: Annotated[
        list[Path] | None,
        typer.Option("--dir", "-d", help="Also look for models in this folder"),
    ] = None,
    why: Annotated[
        list[str] | None, typer.Option("--why", help="Also explain why not this model")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Alternatives to list")] = 8,
    llmfit_bin: LlmfitBin = None,
) -> None:
    """Which model, format and runtime to use on this PC for a kind of work, and why."""
    from ocq.recommend import (
        WORKLOADS,
        RecommendationRequest,
        explain_model,
        recommend,
        rejected_summary,
    )

    if workload not in WORKLOADS:
        console.print(f"[red]Unknown workload {workload!r}. Use one of: {', '.join(WORKLOADS)}")
        raise typer.Exit(1)
    caches = tuple(read_kv_cache(value) for value in kv_cache or ())
    request = RecommendationRequest(WORKLOADS[workload], tuple(models or ()), context, caches)
    try:
        hw = detect_hardware(llmfit_bin)
        with console.status("Reading the models and sizing every option ..."):
            result = recommend(request, hw, folders)
    except OcqError as exc:
        raise fail(exc) from exc

    profile = request.workload
    console.print(
        f"[bold]{profile.label}[/bold] on this PC, {request.context_tokens:,} tokens of context"
        f" [dim]({hw.available_ram_gb:.1f} GB RAM free{gpu_note(hw)})[/dim]"
    )
    best = result.best
    if best is None:
        console.print("[red]Nothing considered runs on this PC.[/red]")
    else:
        print_pick(best)
        print_alternatives(result, limit)
    for name in why or ():
        console.print(f"\n[bold]Why not {escape(name)}?[/bold]")
        for reason in explain_model(result, name):
            console.print(f"  {MARKS[reason.kind.value]} {escape(reason.text)}")
    summary = [f"{n} options {words}" for n, words in rejected_summary(result)]
    if result.too_big:
        summary.append(f"{len(result.too_big)} models too big to try even at the smallest size")
    ruled_out = ", ".join(summary)
    if ruled_out:
        console.print(f"\n[dim]Ruled out: {ruled_out}. --why <model> explains one.[/dim]")
    for label, reason in result.unread:
        console.print(f"[yellow]Could not read {escape(label)}: {escape(reason)}[/yellow]")
    if best is not None and best.candidate.quantize:
        console.print(f"[dim]Make it: {escape(make_command(best.candidate))}[/dim]")


def print_pick(best) -> None:
    candidate = best.candidate
    level = str(best.confidence.level)
    cache = f" · {candidate.kv_cache} cache" if candidate.kv_cache != "f16" else ""
    lines = [
        f"[bold]{escape(candidate.model_id)}[/bold]",
        f"{escape(candidate.option.label)} · {candidate.runtime}{cache}",
        "",
    ]
    lines += [f"{MARKS[r.kind.value]} {escape(r.text)}" for r in best.reasons]
    lines += [f"[yellow]! {escape(warning)}[/yellow]" for warning in best.warnings]
    lines.append(f"\nConfidence: [{CONFIDENCE_COLORS[level]}]{level}[/]")
    for evidence in best.confidence.evidence:
        mark = "[green]✓[/green]" if evidence.known else "[dim]✗[/dim]"
        lines.append(f"  {mark} [dim]{escape(evidence.text)}[/dim]")
    console.print(Panel("\n".join(lines), title="★ Recommended", title_align="left"))


def print_alternatives(result, limit: int) -> None:
    alternatives = result.alternatives[:limit]
    if not alternatives:
        return
    table = Table(title="Alternatives", title_justify="left")
    table.add_column("Model", overflow="fold")
    for column in ("Format", "Runs on", "Memory", "Speed"):
        table.add_column(column, no_wrap=True)
    table.add_column("Why not", overflow="fold")
    for alternative in alternatives:
        candidate, option = alternative.candidate, alternative.candidate.option
        cache = f" + {candidate.kv_cache}" if candidate.kv_cache != "f16" else ""
        why = next((r.text for r in alternative.reasons if r.kind.value == "bad"), None)
        why = why or next((r.text for r in alternative.reasons), "")
        table.add_row(
            escape(candidate.model_id),
            escape(option.label + cache),
            f"{candidate.runtime}, {runs_on(option)}",
            f"{option.memory_required_gb:.1f} GB",
            speed_cell(option),
            escape(why),
        )
    console.print(table)


def make_command(candidate) -> str:
    """The ocq optimize command that makes a recommended file."""
    option = candidate.option
    parts = ["ocq optimize", candidate.model_ref]
    if option.row_kind == "gguf":
        parts += ["-b gguf", f"--gguf-type {option.label}"]
    else:
        parts.append(f"-b {option.row_kind}")
        words = option.label.split()
        bits = next((w.removesuffix("-bit") for w in words if w.endswith("-bit")), "")
        if bits and option.row_kind != "nvfp4":
            parts[-1] += f" --bits {bits}"
        if option.label.endswith((" double", " single")):
            parts.append(f"--{option.label.split()[-1]}-quant")
    parts += [f"--context {candidate.context}", f"--kv-cache {candidate.kv_cache}", "--execute"]
    return " ".join(parts)


@app.command("feasibility")
def cmd_feasibility(
    model: ModelArg = None,
    model_path: ModelPath = None,
    context: Context = DEFAULT_CONTEXT,
    kv_cache: KvCache = "f16",
    llmfit_bin: LlmfitBin = None,
) -> None:
    """Show whether a model can run on this PC at each quantization level."""
    ref = model_ref(model, model_path)
    try:
        hw = detect_hardware(llmfit_bin)
        kv_cache = read_kv_cache(kv_cache)
        snapshot = resolve_model(ref, hw, llmfit_bin, context, kv_cache)
        options = quant_options(snapshot, hw, context, kv_cache)
    except OcqError as exc:
        raise fail(exc) from exc

    if not snapshot.tensors:
        sizes = "estimated from the parameter count (tensor shapes not readable)"
    elif options[0].context_known:
        sizes = "exact, computed from the model's tensor shapes"
    else:
        sizes = "weights exact from the tensor shapes, context memory unknown"
    if snapshot.gguf_files:
        sizes = (
            f"the {len(snapshot.gguf_files)} GGUF files as published (file sizes; settings and "
            "tensor table from one file's header)"
        )
    elif snapshot.is_quantized:
        sizes = (
            f"already quantized ({snapshot.facts.quantization.value}), shown as stored. "
            "Quantize the original model to compare formats"
        )
    fit = snapshot.feasibility
    console.print(
        Panel(
            f"[bold]{snapshot.model_id}[/bold]\n"
            f"Params: {snapshot.parameter_count} | Use case: {snapshot.use_case}\n"
            f"Sizes: {sizes}\n"
            f"Facts: {snapshot.facts_summary()}\n"
            f"Context: {context} tokens\n"
            f"Free right now: {hw.free_vram_gb:.2f} GB VRAM{gpu_note(hw)}, "
            f"{hw.available_ram_gb:.2f} GB RAM\n"
            f"Best fit: {colored_fit(fit.fit_level)} ({mode_label(fit.run_mode)}) "
            f"@ {fit.quant_label}, {fit.memory_text}",
            title="Feasibility",
        )
    )

    title = (
        "By format, sizes in GB (stored checkpoint, GGUF, AWQ, GPTQ, BNB, NVFP4 — same tensor math)"
    )
    if snapshot.gguf_files:
        title = "The repo's GGUF files, sizes in GB (run with llama.cpp, Ollama or LM Studio)"
    table = Table(title=title)
    table.add_column("Level", no_wrap=True)
    table.add_column("Backend", no_wrap=True)
    for column in ("Bits", "Weights", "Context", "Needs"):
        table.add_column(column, no_wrap=True, justify="right")
    table.add_column("Fit", no_wrap=True)
    table.add_column("Runs on", no_wrap=True)
    table.add_column("tok/s", no_wrap=True, justify="right")
    for option in options:
        table.add_row(
            option.label,
            backend_cell(option, hw),
            f"{option.bits:.2f}",
            f"{option.weights_gb:.2f}",
            f"{option.context_gb:.2f}" if option.context_known else "?",
            f"{option.memory_required_gb:.2f}",
            colored_fit(option.fit_level),
            runs_on(option),
            speed_cell(option),
        )
    console.print(table)
    if not options[0].context_known:
        console.print(
            "[dim]?  context memory unknown: config.json has no layer or head counts and the "
            "tensor names show no separate k_proj/v_proj. Fit and Needs count the weights only, "
            "so no row is rated better than Marginal.[/dim]"
        )
    if snapshot.gguf_files and any(option.experts_in_ram for option in options):
        # ocq plan does not take a GGUF repo, so the flag is given here.
        flags = ", ".join(
            f"{option.label}: --n-cpu-moe {option.experts_in_ram}"
            for option in options
            if option.experts_in_ram
        )
        console.print(
            f"[dim]gpu+ram (experts): run llama.cpp with the experts of the first layers in "
            f"RAM ({flags}).[/dim]"
        )
    elif any(option.experts_in_ram for option in options):
        console.print(
            "[dim]gpu+ram (experts): llama.cpp only (--n-cpu-moe, see ocq plan). The AWQ, GPTQ, "
            "BNB and NVFP4 rows cannot keep experts in RAM: vLLM and transformers split by "
            "whole layers instead.[/dim]"
        )
    split = [option for option in options if option.gpu_layers]
    if split:
        layers = ", ".join(f"{option.label}: {option.gpu_layers}" for option in split)
        console.print(
            f"[dim]gpu+ram: llama.cpp puts whole layers on the GPU, from the last one, and keeps "
            f"1 GB free: {layers} of {snapshot.num_layers + 1} (the output layer counts as one)."
            "[/dim]"
        )
    for note in dict.fromkeys(option.note for option in options if option.note):
        console.print(f"[yellow]{escape(note)}[/yellow]")
    if snapshot.is_mlx and not hw.apple_silicon:
        console.print(
            "[yellow]MLX runs only on Apple Silicon Macs (mlx-lm, LM Studio), not on this PC: "
            "the sizes above are for comparison. Use a GGUF of the original model here.[/yellow]"
        )
    bases = {option.speed_basis for option in options if option.runnable}
    if measurements.MEASURED in bases:
        console.print("[dim]~  expected speed, fitted to benchmark runs on this PC.[/dim]")
    if any(option.row_kind == "bnb" for option in options):
        console.print(
            "[dim]BNB single/double: double quantization also stores the 4-bit scales in 8 bits "
            "instead of 32 (--double-quant / --single-quant).[/dim]"
        )
    if any(option.row_kind == "nvfp4" for option in options):
        if hw.has_fp4_tensor_cores:
            console.print(
                "[dim]NVFP4: this GPU computes in FP4 natively (Blackwell). -b nvfp4[/dim]"
            )
        else:
            console.print(
                "[dim]NVFP4: no FP4 tensor cores on this GPU (Blackwell has them), vLLM runs it "
                "weights only at about AWQ speed.[/dim]"
            )
    if measurements.UPPER_BOUND in bases:
        console.print(
            "[dim]<= limit set by memory bandwidth. The real speed is lower and gets learned "
            "from benchmark runs on this PC (ocq optimize --execute).[/dim]"
        )


def gpu_note(hardware) -> str:
    cards = hardware.cards
    return f" on {len(cards)} GPUs" if len(cards) > 1 and not hardware.unified_memory else ""


def backend_cell(option, hardware) -> str:
    if option.row_kind == "stored":
        return "original"
    if option.row_kind == "gguf":
        ready, _ = get_quantizer(QuantAlgorithm.GGUF).check(hardware)
        return "gguf" if ready else "gguf [dim](install --extra gguf)[/dim]"
    try:
        algorithm = QuantAlgorithm(option.row_kind)
    except ValueError:
        return option.row_kind
    ready, _ = get_quantizer(algorithm).check(hardware)
    name = algorithm.value
    return name if ready else f"{name} [dim](install --extra {name})[/dim]"


def mode_label(run_mode) -> str:
    where = str(run_mode).replace("gpu_offload", "gpu+ram")
    return where.replace("gpu_experts_ram", "gpu+ram (experts)")


def runs_on(option) -> str:
    """gpu, gpu+ram or cpu, with the number of GPUs when it uses more than one."""
    where = mode_label(option.run_mode)
    if option.gpu_count > 1:
        where = where.replace("gpu", f"{option.gpu_count} gpus")
    return where


def speed_cell(option) -> str:
    if not option.runnable:
        return "-"
    prefix = "~" if option.speed_basis == measurements.MEASURED else "<="
    return f"{prefix} {option.estimated_tps:.0f}"


@app.command("plan")
def cmd_plan(
    model: ModelArg = None,
    model_path: ModelPath = None,
    goal: Goal = "balanced",
    backend: Backend = None,
    bits: Bits = None,
    gguf_type: GgufType = None,
    group_size: GroupSize = None,
    samples: Samples = None,
    seq_len: SeqLen = None,
    double_quant: DoubleQuant = None,
    fp4_activations: Fp4Activations = None,
    context: Context = DEFAULT_CONTEXT,
    kv_cache: KvCache = "f16",
    output: Output = None,
    llmfit_bin: LlmfitBin = None,
) -> None:
    """Show the quantization plan without running it."""
    ref = model_ref(model, model_path)
    choices = read_choices(
        backend,
        bits,
        gguf_type,
        group_size,
        samples,
        seq_len,
        context,
        double_quant,
        fp4_activations,
        read_kv_cache(kv_cache),
    )
    try:
        hw = detect_hardware(llmfit_bin)
        snapshot = resolve_model(ref, hw, llmfit_bin, context, choices.kv_cache)
        plan = generate_plan(
            snapshot.model_id,
            hw,
            snapshot,
            read_goal(goal),
            output or default_output(ref),
            choices,
        )
    except OcqError as exc:
        raise fail(exc) from exc

    print_plan(plan)


@app.command("optimize")
def cmd_optimize(
    model: ModelArg = None,
    model_path: ModelPath = None,
    goal: Goal = "balanced",
    backend: Backend = None,
    bits: Bits = None,
    gguf_type: GgufType = None,
    group_size: GroupSize = None,
    samples: Samples = None,
    seq_len: SeqLen = None,
    double_quant: DoubleQuant = None,
    fp4_activations: Fp4Activations = None,
    context: Context = DEFAULT_CONTEXT,
    kv_cache: KvCache = "f16",
    output: Output = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--execute", help="Only show the plan, or really quantize"),
    ] = True,
    llmfit_bin: LlmfitBin = None,
) -> None:
    """One click: analyze -> plan -> quantize -> validate -> benchmark."""
    ref = model_ref(model, model_path)
    choices = read_choices(
        backend,
        bits,
        gguf_type,
        group_size,
        samples,
        seq_len,
        context,
        double_quant,
        fp4_activations,
        read_kv_cache(kv_cache),
    )
    try:
        result = run_optimize(
            ref,
            read_goal(goal),
            output or default_output(ref),
            dry_run=dry_run,
            llmfit_bin=llmfit_bin,
            choices=choices,
        )
    except OcqError as exc:
        raise fail(exc) from exc

    print_plan(result.plan)

    if result.dry_run:
        console.print("\n[dim]Dry run - pass --execute to run quantization.[/dim]")
        return

    print_result(result)
    if not (result.validation and result.validation.passed):
        raise typer.Exit(1)


def print_plan(plan: OptimizationPlan) -> None:
    console.print(Panel(f"[bold]{goal_label(plan.goal)}[/bold]", title="Optimization Plan"))
    for line in plan.rationale:
        console.print(f"  - {line}")
    console.print(f"\n  Output: [cyan]{plan.output_dir}[/cyan]")
    console.print(f"  Backend: [green]{plan.quant_config.summary()}[/green]")
    if plan.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in plan.warnings:
            console.print(f"  ! {warning}")


def print_result(result: OptimizeResult) -> None:
    quantization = result.quantization
    color = "green" if quantization.success else "red"
    console.print(f"\n[{color}]Quantization:[/{color}] {quantization.message}")
    predicted = result.plan.predicted
    if quantization.success and predicted:
        console.print(
            f"  Size: {quantization.artifact_size_gb:.3f} GB on disk, "
            f"{predicted.weights_gb:.3f} GB predicted"
        )

    validation = result.validation
    verdict = "[green]PASS[/green]" if validation.passed else "[red]FAIL[/red]"
    console.print(f"[bold]Validation:[/bold] {verdict}")
    for check in validation.checks:
        console.print(f"  {check}", highlight=False)

    if not quantization.success:
        return

    bench = result.benchmark
    console.print(f"[bold]Benchmark[/bold] ({bench.method} on {bench.device or 'unknown'})")
    if bench.tokens_per_second:
        speed = f"  Speed: {bench.tokens_per_second:.1f} tok/s measured"
        predicted = result.plan.predicted
        if result.plan.quant_config.algorithm == QuantAlgorithm.GGUF and predicted:
            if predicted.speed_basis == measurements.MEASURED:
                speed += f" (expected about {predicted.estimated_tps:.0f})"
            else:
                speed += f" (bandwidth limit {predicted.estimated_tps:.0f})"
        console.print(speed)
    else:
        console.print("  Speed: not measured")
    if bench.time_to_first_token_s is not None:
        console.print(f"  Time to first token: {bench.time_to_first_token_s * 1000:.0f} ms")
    for note in bench.notes:
        console.print(f"  [dim]{note}[/dim]")
    if bench.sample_output:
        console.print(Panel(bench.sample_output, title="Sample output"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
