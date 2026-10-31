"""Terminal CLI for One-Click Quantizer."""

from __future__ import annotations

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
from ocq.model_analyzer import DEFAULT_CONTEXT, quant_options, recommend_models, resolve_model
from ocq.pipeline import run_optimize
from ocq.plan import free_disk_gb, generate_plan
from ocq.quantizers import backend_status, get_quantizer
from ocq.types import (
    KV_CACHE_TYPES,
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
    help="One-Click Quantizer: inspect this PC, see which models it can run, quantize them.",
    no_args_is_help=True,
)
console = Console()

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


@app.command("recommend")
def cmd_recommend(
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many models to list")] = 10,
    use_case: Annotated[
        str | None,
        typer.Option("--use-case", help="general, coding, reasoning, chat, multimodal, embedding"),
    ] = None,
    llmfit_bin: LlmfitBin = None,
) -> None:
    """Suggest models that suit this PC."""
    hw = detect_hardware(llmfit_bin)
    models = recommend_models(hw, limit, use_case, llmfit_bin)

    table = Table(title="Models for this PC")
    table.add_column("Model", overflow="fold")
    for column in ("Params", "Fit", "Runs on", "Quant", "Memory GB", "tok/s"):
        table.add_column(column, no_wrap=True)
    for model in models:
        fit = model.feasibility
        table.add_row(
            model.model_id,
            str(model.parameter_count),
            colored_fit(fit.fit_level),
            str(fit.run_mode),
            fit.quant_label,
            f"{fit.memory_required_gb:.1f} / {fit.memory_available_gb:.1f}",
            f"{fit.estimated_tps:.0f}" if fit.runnable else "-",
        )
    console.print(table)
    console.print("[dim]Check one in detail: ocq feasibility <model>[/dim]")


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
    if snapshot.is_quantized:
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
            f"Context: {context} tokens\n"
            f"Free right now: {hw.free_vram_gb:.2f} GB VRAM{gpu_note(hw)}, "
            f"{hw.available_ram_gb:.2f} GB RAM\n"
            f"Best fit: {colored_fit(fit.fit_level)} ({fit.run_mode}) @ {fit.quant_label}, "
            f"{fit.memory_required_gb:.2f} of {fit.memory_available_gb:.2f} GB "
            f"({fit.utilization_pct:.0f}%)",
            title="Feasibility",
        )
    )

    table = Table(
        title="By format, sizes in GB (stored checkpoint, GGUF, AWQ, GPTQ, BNB, NVFP4 — same "
        "tensor math)"
    )
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


def runs_on(option) -> str:
    """gpu, gpu+ram or cpu, with the number of GPUs when it uses more than one."""
    where = str(option.run_mode).replace("gpu_offload", "gpu+ram")
    where = where.replace("gpu_experts_ram", "gpu+ram (experts)")
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
