"""Terminal CLI for One-Click Quantizer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ocq.goals import goal_label
from ocq.llmfit_bridge import LlmfitBridge, LlmfitError
from ocq.model_analyzer import resolve_model
from ocq.pipeline import run_optimize
from ocq.plan import generate_plan
from ocq.quantizers import available_backends
from ocq.types import OptimizationGoal

app = typer.Typer(
    name="ocq",
    help="One-Click Quantizer - llmfit-integrated terminal LLM optimizer.",
    no_args_is_help=True,
)
console = Console()


def _goal(value: str) -> OptimizationGoal:
    try:
        return OptimizationGoal(value.lower())
    except ValueError as exc:
        valid = ", ".join(g.value for g in OptimizationGoal)
        raise typer.BadParameter(f"Use one of: {valid}") from exc


def _default_output(model_id: str) -> Path:
    slug = model_id.replace("/", "--")
    return Path.cwd() / "ocq-output" / slug


def _bridge(llmfit_bin: str | None) -> LlmfitBridge:
    return LlmfitBridge(llmfit_bin)


@app.command("profile")
def cmd_profile(
    llmfit_bin: Annotated[str | None, typer.Option("--llmfit-bin")] = None,
) -> None:
    """Show hardware profile (llmfit `system` only)."""
    try:
        hw = _bridge(llmfit_bin).hardware_profile()
    except LlmfitError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title="Hardware Profile")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("CPU", f"{hw.cpu_name} ({hw.cpu_cores} cores)")
    table.add_row("RAM", f"{hw.available_ram_gb:.1f} / {hw.total_ram_gb:.1f} GB")
    table.add_row("Backend", hw.backend)
    table.add_row("GPU", "yes" if hw.has_gpu else "no")
    for i, g in enumerate(hw.gpus):
        vram = f"{g.vram_gb:.1f} GB" if g.vram_gb is not None else "?"
        table.add_row(f"GPU {i}", f"{g.name} - {vram}")
    console.print(table)


@app.command("feasibility")
def cmd_feasibility(
    model: Annotated[str, typer.Argument(help="HuggingFace model id (any public repo)")],
    llmfit_bin: Annotated[str | None, typer.Option("--llmfit-bin")] = None,
) -> None:
    """Show fit/speed for a model (llmfit catalog or HuggingFace fallback)."""
    try:
        bridge = _bridge(llmfit_bin)
        hw = bridge.hardware_profile()
        snapshot = resolve_model(model, hw, bridge)
    except LlmfitError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    f = snapshot.feasibility
    source = "llmfit catalog" if snapshot.catalog_source == "llmfit" else "HuggingFace estimate"
    console.print(
        Panel(
            f"[bold]{snapshot.model_id}[/bold]\n"
            f"Source: {source}\n"
            f"Params: {snapshot.parameter_count} | Use case: {snapshot.use_case}\n"
            f"Fit: [cyan]{f.fit_level}[/cyan] ({f.run_mode}) @ {f.quant_label}\n"
            f"Score: {f.score:.1f} | Est: {f.estimated_tps:.1f} tok/s ({f.estimate_confidence})\n"
            f"Memory: {f.memory_required_gb:.1f} / {f.memory_available_gb:.1f} GB "
            f"({f.utilization_pct:.0f}%)",
            title="Feasibility (hypothesis - confirm after quant)",
        )
    )


@app.command("plan")
def cmd_plan(
    model: Annotated[str, typer.Argument(help="HuggingFace model id")],
    goal: Annotated[str, typer.Option("--goal", "-g")] = "balanced",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    llmfit_bin: Annotated[str | None, typer.Option("--llmfit-bin")] = None,
) -> None:
    """Generate optimization plan without executing."""
    g = _goal(goal)
    out = output or _default_output(model)
    try:
        bridge = _bridge(llmfit_bin)
        hardware = bridge.hardware_profile()
        snapshot = resolve_model(model, hardware, bridge)
        plan = generate_plan(model, hardware, snapshot, g, out)
    except LlmfitError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    _print_plan(plan)


@app.command("optimize")
def cmd_optimize(
    model: Annotated[str, typer.Argument(help="HuggingFace model id")],
    goal: Annotated[str, typer.Option("--goal", "-g")] = "balanced",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--execute", help="Plan only vs run quant pipeline"),
    ] = True,
    llmfit_bin: Annotated[str | None, typer.Option("--llmfit-bin")] = None,
) -> None:
    """One-click: analyze -> plan -> quantize -> validate -> benchmark."""
    g = _goal(goal)
    out = output or _default_output(model)
    if not available_backends() and not dry_run:
        console.print(
            "[yellow]No quant backends installed - forcing --dry-run.[/yellow]\n"
            "Install: uv sync --extra awq  (or gptq / bnb)"
        )
        dry_run = True

    try:
        result = run_optimize(model, g, out, dry_run=dry_run, llmfit_bin=llmfit_bin)
    except LlmfitError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    _print_plan(result.plan)

    if result.dry_run:
        console.print("\n[dim]Dry run - pass --execute to run quantization.[/dim]")
        return

    if result.quantization:
        q = result.quantization
        color = "green" if q.success else "red"
        console.print(f"\n[{color}]Quantization:[/{color}] {q.message}")

    if result.validation:
        v = result.validation
        console.print(f"[bold]Validation:[/bold] {'PASS' if v.passed else 'FAIL'}")
        for check in v.checks:
            console.print(f"  - {check}")

    if result.benchmark:
        b = result.benchmark
        tps = f"{b.tokens_per_second:.1f} tok/s" if b.tokens_per_second else "not measured"
        console.print(f"[bold]Benchmark (ground truth):[/bold] {tps} [{b.method}]")
        for note in b.notes:
            console.print(f"  [dim]{note}[/dim]")


def _print_plan(plan) -> None:
    console.print(Panel(f"[bold]{goal_label(plan.goal)}[/bold]", title="Optimization Plan"))
    for line in plan.rationale:
        console.print(f"  - {line}")
    console.print(f"\n  Output: [cyan]{plan.output_dir}[/cyan]")
    console.print(f"  Backend: [green]{plan.quant_config.summary()}[/green]")
    if plan.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in plan.warnings:
            console.print(f"  ! {w}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
