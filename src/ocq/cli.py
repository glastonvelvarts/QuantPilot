"""Terminal CLI for One-Click Quantizer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ocq import measurements
from ocq.engine import GGUF_TYPES
from ocq.goals import goal_label
from ocq.hardware import detect_hardware
from ocq.model_analyzer import DEFAULT_CONTEXT, quant_options, recommend_models, resolve_model
from ocq.pipeline import run_optimize
from ocq.plan import free_disk_gb, generate_plan
from ocq.quantizers import backend_status
from ocq.types import (
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

FIT_COLORS = {
    FitLevel.PERFECT: "green",
    FitLevel.GOOD: "green",
    FitLevel.MARGINAL: "yellow",
    FitLevel.TOO_TIGHT: "red",
}

ModelArg = Annotated[
    str, typer.Argument(help="HuggingFace model id, for example Qwen/Qwen2.5-7B-Instruct")
]
LlmfitBin = Annotated[str | None, typer.Option("--llmfit-bin", help="Path to the llmfit binary")]
Goal = Annotated[str, typer.Option("--goal", "-g", help="balanced, speed, quality or memory")]
Backend = Annotated[
    str | None, typer.Option("--backend", "-b", help="gguf, awq, gptq or bnb (default: auto)")
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
Output = Annotated[Path | None, typer.Option("--output", "-o", help="Where to write the result")]
Context = Annotated[
    int, typer.Option("--context", "-c", help="Context size in tokens the model will run with")
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
    )


def default_output(model_id: str) -> Path:
    return Path.cwd() / "ocq-output" / model_id.replace("/", "--")


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
        table.add_row(f"GPU {index}", f"{gpu.name} - {vram}{speed}")
    if hw.has_gpu and hw.pcie_bandwidth_gbs > 0 and not hw.unified_memory:
        table.add_row("PCIe Bandwidth", f"{hw.pcie_bandwidth_gbs:.1f} GB/s")
    table.add_row("Free disk", f"{free_disk_gb(Path.cwd()):.0f} GB (current folder)")
    table.add_row("Detected by", "llmfit" if hw.source == "llmfit" else "built-in probe")
    console.print(table)


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
    model: ModelArg, context: Context = DEFAULT_CONTEXT, llmfit_bin: LlmfitBin = None
) -> None:
    """Show whether a model can run on this PC at each quantization level."""
    try:
        hw = detect_hardware(llmfit_bin)
        snapshot = resolve_model(model, hw, llmfit_bin, context)
    except OcqError as exc:
        raise fail(exc) from exc

    options = quant_options(snapshot, hw, context)
    exact = options[0].exact
    sizes = (
        "exact, computed from the model's tensor shapes"
        if exact
        else "estimated from the parameter count (tensor shapes not readable)"
    )
    fit = snapshot.feasibility
    console.print(
        Panel(
            f"[bold]{snapshot.model_id}[/bold]\n"
            f"Params: {snapshot.parameter_count} | Use case: {snapshot.use_case}\n"
            f"Sizes: {sizes}\n"
            f"Context: {context} tokens\n"
            f"Free right now: {hw.free_vram_gb:.2f} GB VRAM, {hw.available_ram_gb:.2f} GB RAM\n"
            f"Best fit: {colored_fit(fit.fit_level)} ({fit.run_mode}) @ {fit.quant_label}, "
            f"{fit.memory_required_gb:.2f} of {fit.memory_available_gb:.2f} GB "
            f"({fit.utilization_pct:.0f}%)",
            title="Feasibility",
        )
    )

    table = Table(title="By quantization level, sizes in GB (first row is the model as stored)")
    table.add_column("Level", no_wrap=True)
    for column in ("Bits", "Weights", "Context", "Needs"):
        table.add_column(column, no_wrap=True, justify="right")
    table.add_column("Fit", no_wrap=True)
    table.add_column("Runs on", no_wrap=True)
    table.add_column("tok/s", no_wrap=True, justify="right")
    for option in options:
        table.add_row(
            option.label,
            f"{option.bits:.2f}",
            f"{option.weights_gb:.2f}",
            f"{option.context_gb:.2f}",
            f"{option.memory_required_gb:.2f}",
            colored_fit(option.fit_level),
            str(option.run_mode).replace("gpu_offload", "gpu+ram"),
            speed_cell(option),
        )
    console.print(table)
    bases = {option.speed_basis for option in options if option.runnable}
    if measurements.MEASURED in bases:
        console.print("[dim]~  expected speed, fitted to benchmark runs on this PC.[/dim]")
    if measurements.UPPER_BOUND in bases:
        console.print(
            "[dim]<= limit set by memory bandwidth. The real speed is lower and gets learned "
            "from benchmark runs on this PC (ocq optimize --execute).[/dim]"
        )


def speed_cell(option) -> str:
    if not option.runnable:
        return "-"
    prefix = "~" if option.speed_basis == measurements.MEASURED else "<="
    return f"{prefix} {option.estimated_tps:.0f}"


@app.command("plan")
def cmd_plan(
    model: ModelArg,
    goal: Goal = "balanced",
    backend: Backend = None,
    bits: Bits = None,
    gguf_type: GgufType = None,
    group_size: GroupSize = None,
    samples: Samples = None,
    seq_len: SeqLen = None,
    context: Context = DEFAULT_CONTEXT,
    output: Output = None,
    llmfit_bin: LlmfitBin = None,
) -> None:
    """Show the quantization plan without running it."""
    choices = read_choices(backend, bits, gguf_type, group_size, samples, seq_len, context)
    try:
        hw = detect_hardware(llmfit_bin)
        snapshot = resolve_model(model, hw, llmfit_bin, context)
        plan = generate_plan(
            model, hw, snapshot, read_goal(goal), output or default_output(model), choices
        )
    except OcqError as exc:
        raise fail(exc) from exc

    print_plan(plan)


@app.command("optimize")
def cmd_optimize(
    model: ModelArg,
    goal: Goal = "balanced",
    backend: Backend = None,
    bits: Bits = None,
    gguf_type: GgufType = None,
    group_size: GroupSize = None,
    samples: Samples = None,
    seq_len: SeqLen = None,
    context: Context = DEFAULT_CONTEXT,
    output: Output = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--execute", help="Only show the plan, or really quantize"),
    ] = True,
    llmfit_bin: LlmfitBin = None,
) -> None:
    """One click: analyze -> plan -> quantize -> validate -> benchmark."""
    choices = read_choices(backend, bits, gguf_type, group_size, samples, seq_len, context)
    try:
        result = run_optimize(
            model,
            read_goal(goal),
            output or default_output(model),
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
