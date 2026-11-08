"""The pages of the app. Each one asks QuantPilot for data and shows it as cards and tables.

Slow work (reading the hardware, the Hub, quantizing) runs in a thread; the results come back
to the screen through call_from_thread.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Input,
    Label,
    LoadingIndicator,
    Static,
    Switch,
    TextArea,
)

from ocq.model_analyzer import QUANTIZED_LEVELS, best_option
from ocq.tui import words
from ocq.tui.banner import TAGLINE
from ocq.tui.widgets import (
    Action,
    Card,
    Confirm,
    Dropdown,
    ModelActions,
    ModelField,
    ProgressLog,
    Results,
    RowTable,
    field,
)
from ocq.types import (
    KV_CACHE_TYPES,
    OcqError,
    OptimizationGoal,
    QuantAlgorithm,
    QuantChoices,
)

if TYPE_CHECKING:
    from ocq.tui.app import QuantPilotApp

AUTO = "auto"
CONTEXTS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
KV_LABELS = {"f16": "Full", "q8_0": "Half (q8_0)", "q4_0": "Quarter (q4_0)"}
GOALS = {
    OptimizationGoal.BALANCED: "Balanced",
    OptimizationGoal.SPEED: "Fastest",
    OptimizationGoal.QUALITY: "Best quality",
    OptimizationGoal.MEMORY: "Smallest",
}
USES = ("coding", "reasoning", "chat", "multimodal", "embedding")
# Format dropdown on the Quantize page: "backend:level".
FORMATS = (
    [("Recommended", AUTO)]
    + [(f"GGUF {level}", f"gguf:{level}") for level in QUANTIZED_LEVELS]
    + [
        ("AWQ 4-bit", "awq:4"),
        ("GPTQ 4-bit", "gptq:4"),
        ("GPTQ 8-bit", "gptq:8"),
        ("BNB 4-bit", "bnb:4"),
        ("BNB 8-bit", "bnb:8"),
        ("NVFP4", "nvfp4:4"),
    ]
)
FORMAT_KEYS = {key for _, key in FORMATS}
TOOL_ORDER = (
    QuantAlgorithm.GGUF,
    QuantAlgorithm.AWQ,
    QuantAlgorithm.GPTQ,
    QuantAlgorithm.BNB,
    QuantAlgorithm.NVFP4,
)


def context_options(default: int) -> list[tuple[str, int]]:
    return [(words.context(value), value) for value in sorted({*CONTEXTS, default})]


def goal_options() -> list[tuple[str, str]]:
    return [(label, goal.value) for goal, label in GOALS.items()]


def format_key(option) -> str | None:
    """The Format dropdown entry that makes this row of the fit table, if one does."""
    if option.row_kind == "gguf":
        key = f"gguf:{option.label}"
    else:
        bits = re.search(r"(\d+)-bit", option.label)
        key = f"{option.row_kind}:{bits.group(1)}" if bits else option.row_kind
        if option.row_kind == "nvfp4":
            key = "nvfp4:4"
    return key if key in FORMAT_KEYS else None


def format_choices(key: str, context: int, kv_cache: str) -> QuantChoices:
    algorithm = bits = gguf_type = None
    if key != AUTO:
        backend, level = key.split(":")
        algorithm = QuantAlgorithm(backend)
        if algorithm == QuantAlgorithm.GGUF:
            gguf_type = level
        else:
            bits = int(level)
    return QuantChoices(
        algorithm=algorithm, bits=bits, gguf_type=gguf_type, context=context, kv_cache=kv_cache
    )


def background(group: str):
    """Runs a page method in a thread. If it fails, the page shows why (Page.failed) instead
    of the app closing: a lost connection must not end the session."""

    def wrap(method):
        @functools.wraps(method)
        def run(self, *args, **kwargs):
            try:
                method(self, *args, **kwargs)
            except Exception as exc:
                message = str(exc) if isinstance(exc, OcqError) else f"Something went wrong: {exc}"
                self.app.call_from_thread(self.failed, message)

        return work(thread=True, exclusive=True, group=group)(run)

    return wrap


class Page(Vertical):
    TITLE = ""
    # Keys the focused field does not use itself (see widgets.py).
    BINDINGS = [
        Binding("left", "app.menu", "Menu", show=False),
        Binding("up", "step(-1)", show=False),
        Binding("down,right", "step(1)", show=False),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(classes="page", **kwargs)

    @property
    def qp(self) -> QuantPilotApp:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        yield Static(self.TITLE, classes="page-title")
        yield from self.compose_body()

    def compose_body(self) -> ComposeResult:
        yield from ()

    def opened(self) -> None:
        """Each time the page is shown."""

    def run(self) -> None:
        """Enter in a field, the main button, or Ctrl+R."""

    def action_step(self, delta: int) -> None:
        """Focus the next (+1) or previous (-1) field or button on this page; stops at the ends."""
        chain = [w for w in self.screen.focus_chain if self in w.ancestors]
        if not chain:
            return
        focused = self.screen.focused
        index = chain.index(focused) + delta if focused in chain else 0
        chain[max(0, min(index, len(chain) - 1))].focus()

    def focus_first(self) -> None:
        for widget in self.query("Input, Select, Button, DataTable, Switch, TextArea"):
            if widget.focusable:
                widget.focus()
                return

    def value(self, name: str) -> Any:
        return self.query_one(f"#{name}").value

    def model(self) -> str | None:
        ref = self.query_one(ModelField).value
        if not ref:
            self.notify("Type a model or press Browse.", severity="warning")
        return ref or None

    def set_model(self, ref: str) -> None:
        for model_field in self.query(ModelField):
            model_field.value = ref

    def context_field(self) -> Vertical:
        default = self.qp.settings.context
        return field("Context", Dropdown(context_options(default), value=default, id="context"))

    def failed(self, message: str) -> None:
        """Main thread: a background job stopped with this message."""
        for busy in self.query(".busy"):
            busy.display = False
        for widget in self.query("DataTable, #pc-info"):
            widget.loading = False
        problems = self.query(".problem")
        if problems:
            problems.first(Static).update(Text(words.plain(message)))
            problems.first().display = True
        else:
            self.notify(words.plain(message), severity="error")

    @on(Input.Submitted)
    def submitted(self, event: Input.Submitted) -> None:
        if event.input.has_class("model-input"):
            self.run()


# Home -------------------------------------------------------------------------------------


class HomePage(Page):
    def compose(self) -> ComposeResult:
        with Vertical(id="home-box"):
            yield Static("", id="home-banner")
            yield Static(TAGLINE, id="home-tagline")
            yield Static("", id="home-hw")
            with Horizontal(id="home-actions"):
                yield Action("Check a model", variant="primary", name="check")
                yield Action("Discover models", name="discover")
                yield Action("My models", name="models")

    @on(Button.Pressed)
    def go(self, event: Button.Pressed) -> None:
        if event.button.name:
            self.qp.show_page(event.button.name, focus=True)


# This PC ----------------------------------------------------------------------------------


class PcPage(Page):
    TITLE = "This PC"
    loaded = False

    def compose_body(self) -> ComposeResult:
        with Results(classes="results"):
            yield Static("", id="pc-info")
            yield Static("Quantization tools", classes="section")
            yield Vertical(id="tools")
            yield ProgressLog(id="install-log")
        with Horizontal(classes="actions"):
            yield Action("Refresh", id="refresh")

    def on_mount(self) -> None:
        self.query_one("#install-log").display = False

    def opened(self) -> None:
        if not self.loaded:
            self.run()

    @on(Button.Pressed, "#refresh")
    def run(self) -> None:
        self.loaded = True
        self.query_one("#pc-info").loading = True
        self.load()

    @background("pc")
    def load(self) -> None:
        from ocq.plan import free_disk_gb
        from ocq.tui.install import tool_state

        hw = self.qp.hardware(fresh=True)
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="dim", no_wrap=True)
        grid.add_column()
        grid.add_row("Processor", f"{hw.cpu_name}, {hw.cpu_cores} cores")
        grid.add_row("Memory", f"{hw.total_ram_gb:.0f} GB ({hw.available_ram_gb:.1f} GB free)")
        for gpu in hw.gpus:
            vram = f"{gpu.vram_gb:.0f} GB" if gpu.vram_gb else ""
            if gpu.free_vram_gb is not None and not hw.unified_memory:
                vram += f" ({gpu.free_vram_gb:.1f} GB free)"
            count = f"{gpu.count} x " if gpu.count > 1 else ""
            grid.add_row("Graphics", f"{count}{gpu.name}  {vram}".strip())
        if not hw.gpus:
            grid.add_row("Graphics", "None found")
        speed = hw.measured_ram_bandwidth_gbs or hw.ram_bandwidth_gbs
        if speed:
            grid.add_row("Memory speed", f"{speed:.0f} GB/s")
        grid.add_row("Free disk", f"{free_disk_gb(Path.cwd()):.0f} GB")
        tools = [tool_state(algorithm, hw) for algorithm in TOOL_ORDER]
        self.app.call_from_thread(self.show, grid, tools)

    def show(self, grid: Table, tools: list) -> None:
        from ocq.tui.install import MISSING, READY

        info = self.query_one("#pc-info", Static)
        info.update(grid)
        info.loading = False
        box = self.query_one("#tools", Vertical)
        box.remove_children()
        for tool in tools:
            state = "ok" if tool.state == READY else "off"
            row = Horizontal(
                Label(tool.name, classes="tool-name"),
                Label(tool.message, classes=f"tool-state {state}"),
                classes="tool",
            )
            if tool.state == MISSING:
                row.compose_add_child(Action("Install", name=tool.algorithm.value))
            box.mount(row)

    @on(Button.Pressed, ".tool Button")
    def install_pressed(self, event: Button.Pressed) -> None:
        log = self.query_one("#install-log", ProgressLog)

        def started() -> None:
            log.clear()
            log.display = True

        self.qp.ask_install(QuantAlgorithm(event.button.name), log, started, lambda *_: self.run())


# Model lists ------------------------------------------------------------------------------


class ListPage(Page):
    """A table of models; Enter on a row asks what to do with it."""

    COLUMNS: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loaded = False
        self.rows: dict[str, tuple[str, bool]] = {}  # row key -> (model ref, is GGUF)

    def compose_body(self) -> ComposeResult:
        yield from self.compose_filters()
        yield RowTable(cursor_type="row", zebra_stripes=True)
        yield Static("", classes="empty")

    def compose_filters(self) -> ComposeResult:
        yield from ()

    def opened(self) -> None:
        if not self.loaded:
            self.run()

    def run(self) -> None:
        self.loaded = True
        self.query_one(DataTable).loading = True
        self.load(self.params())

    def params(self) -> dict[str, Any]:
        return {}

    def load(self, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def fill(self, rows: list[tuple[str, str, bool, list]], empty: str) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns(*self.COLUMNS)
        self.rows = {}
        for key, ref, gguf, cells in rows:
            self.rows[key] = (ref, gguf)
            table.add_row(*cells, key=key)
        table.loading = False
        table.display = bool(rows)
        note = self.query_one(".empty", Static)
        note.update(empty)
        note.display = not rows

    @on(DataTable.RowSelected)
    def row_selected(self, event: DataTable.RowSelected) -> None:
        ref, gguf = self.rows.get(event.row_key.value or "", ("", False))
        if not ref:
            return

        def chosen(page: str | None) -> None:
            if page:
                self.qp.open_with_model(page, ref)

        self.app.push_screen(ModelActions(ref, gguf), chosen)


class ModelsPage(ListPage):
    TITLE = "My models"
    COLUMNS = ("Model", "From", "Size", "Fit", "Speed")

    def compose_filters(self) -> ComposeResult:
        with Horizontal(classes="actions"):
            yield Action("Refresh", id="refresh")

    @on(Button.Pressed, "#refresh")
    def refresh_pressed(self) -> None:
        self.run()

    def params(self) -> dict[str, Any]:
        s = self.qp.settings
        return {"context": s.context, "kv_cache": s.kv_cache, "folders": s.folders()}

    @background("models")
    def load(self, params: dict[str, Any]) -> None:
        from ocq.discovery import GGUF, find_models

        try:
            models = find_models(params["folders"])
        except OcqError as exc:
            models = []
            self.app.call_from_thread(self.notify, str(exc), severity="error")
        self.qp.known_models = models
        hw = self.qp.hardware()
        rows = []
        complete = [m for m in models if m.complete]
        for i, model in enumerate(sorted(complete, key=lambda m: m.label.lower())):
            option = local_option(model, hw, params["context"], params["kv_cache"])
            size = words.gb(model.size_bytes / 1e9) if model.size_bytes else "-"
            fit = words.fit(option.fit_level) if option else Text("-", "dim")
            speed = words.speed(option) if option else "-"
            cells = [model.label, model.source, size, fit, speed]
            rows.append((str(i), model.ref, model.format == GGUF, cells))
        empty = (
            "No models found yet.\n\n"
            "Models from Ollama, LM Studio and Hugging Face show up here by themselves. "
            "Keep them somewhere else? Add the folder in Settings."
        )
        self.app.call_from_thread(self.fill, rows, empty)
        self.app.call_from_thread(self.qp.refresh_suggestions)


def local_option(model, hw, context: int, kv_cache: str):
    """How a model on disk runs here, as it is stored; None when it cannot be worked out."""
    from ocq.discovery import GGUF, MLX_LAYOUT, SAFETENSORS
    from ocq.model_analyzer import quant_options, resolve_model

    if model.format not in (GGUF, SAFETENSORS, MLX_LAYOUT):
        return None
    ref = model.ref if model.ref.startswith("ollama:") else str(model.path)
    try:
        snapshot = resolve_model(ref, hw, None, context, kv_cache)
        if snapshot.is_mlx and not hw.apple_silicon:
            return None
        return quant_options(snapshot, hw, context, kv_cache)[0]
    except (OcqError, OSError, ValueError, KeyError):
        return None


class DiscoverPage(ListPage):
    TITLE = "Discover"
    COLUMNS = ("Model", "Size", "Fit", "Speed")

    def compose_filters(self) -> ComposeResult:
        uses = [("Anything", AUTO)] + [(use.capitalize(), use) for use in USES]
        with Grid(classes="form"):
            yield field("Good for", Dropdown(uses, value=AUTO, id="use"))

    @on(Dropdown.Changed, "#use")
    def use_changed(self) -> None:
        if self.loaded:
            self.run()

    def params(self) -> dict[str, Any]:
        return {"use": self.value("use")}

    @background("discover")
    def load(self, params: dict[str, Any]) -> None:
        from ocq.model_analyzer import recommend_models

        use = params["use"]
        hw = self.qp.hardware()
        try:
            models = recommend_models(hw, 15, None if use == AUTO else use, self.qp.llmfit_bin)
        except OcqError:
            models = []
        rows = []
        for i, model in enumerate(models):
            fit = model.feasibility
            size = f"{model.parameter_count}" if model.parameter_count else "-"
            cells = [model.model_id, size, words.fit(fit.fit_level), words.speed(fit)]
            rows.append((str(i), model.model_id, False, cells))
        self.app.call_from_thread(self.fill, rows, "Nothing found. Try another choice above.")


# Check a model ----------------------------------------------------------------------------


class CheckPage(Page):
    TITLE = "Check a model"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.format_rows: dict[str, str] = {}  # row key -> Format entry on the Quantize page
        self.recommended = AUTO
        self.ref = ""

    def compose_body(self) -> ComposeResult:
        with Grid(classes="form"):
            yield field("Model", ModelField(), "wide")
            yield self.context_field()
        with Horizontal(classes="actions"):
            yield Action("Check", variant="primary", id="check")
        yield LoadingIndicator(classes="busy")
        with Results(classes="results"):
            with Card(id="pick", classes="card"):
                yield Static("", id="pick-text")
                yield Action("Quantize to this", variant="primary", id="quantize-pick")
            yield RowTable(cursor_type="row", zebra_stripes=True, id="options")
            with Collapsible(title="About this model", collapsed=True, id="about"):
                yield Static("", id="facts")
            yield Static("", classes="problem")

    def on_mount(self) -> None:
        self.show_results(False)

    def show_results(self, shown: bool) -> None:
        for selector in ("#pick", "#options", "#about"):
            self.query_one(selector).display = shown
        self.query_one(".problem").display = False

    @on(Button.Pressed, "#check")
    def run(self) -> None:
        ref = self.model()
        if not ref:
            return
        self.ref = ref
        self.qp.remember(ref)
        self.show_results(False)
        self.query_one(".busy").display = True
        self.load(ref, self.value("context"), self.qp.settings.kv_cache, self.qp.settings.goal)

    @background("check")
    def load(self, ref: str, context: int, kv_cache: str, goal: str) -> None:
        from ocq.cli import default_output
        from ocq.model_analyzer import quant_options, resolve_model
        from ocq.plan import generate_plan

        hw = self.qp.hardware()
        snapshot = resolve_model(ref, hw, self.qp.llmfit_bin, context, kv_cache)
        options = quant_options(snapshot, hw, context, kv_cache)
        plan = None
        if not snapshot.is_quantized and not snapshot.gguf_files:
            try:
                choices = QuantChoices(context=context, kv_cache=kv_cache)
                plan = generate_plan(
                    snapshot.model_id, hw, snapshot, OptimizationGoal(goal),
                    default_output(ref), choices,
                )  # fmt: skip
            except OcqError:
                plan = None
        self.app.call_from_thread(self.show, snapshot, options, plan)

    def show(self, snapshot, options, plan) -> None:
        self.query_one(".busy").display = False
        quantizable = plan is not None
        name = None
        if plan and plan.predicted:
            best = plan.predicted
            title = "Recommended for this PC"
            name = plan.quant_config.summary()
        else:
            best = best_option(options)  # the CLI's best fit
            title = "Best fit for this PC" if best.runnable else "Too big for this PC"
        self.recommended = (format_key(best) or AUTO) if quantizable else AUTO

        text = Text()
        text.append(f"{title}\n", "dim")
        text.append(f"{name or best.label}\n", "bold")
        text.append_text(words.fit(best.fit_level))
        details = [
            words.gb(best.memory_required_gb),
            words.place(best),
            words.layers(best, getattr(snapshot, "num_layers", 0)),
            words.speed(best),
        ]
        text.append("  " + "  |  ".join(d for d in details if d != "-"))
        if plan:
            for warning in plan.warnings:
                text.append(f"\n{words.warning(warning)}", "yellow")
        self.query_one("#pick-text", Static).update(text)
        self.query_one("#quantize-pick").display = quantizable

        table = self.query_one("#options", DataTable)
        table.clear(columns=True)
        table.add_columns("Format", "Memory", "Fit", "Runs on", "Speed")
        self.format_rows = {}
        for i, option in enumerate(options):
            label = Text(option.label)
            if option.label == best.label:
                label.append("  recommended" if quantizable else "  best", "dim")
            cells = [label, words.gb(option.memory_required_gb), words.fit(option.fit_level),
                     words.place(option), words.speed(option)]  # fmt: skip
            table.add_row(*cells, key=str(i))
            key = format_key(option)
            if quantizable and key:
                self.format_rows[str(i)] = key
        self.query_one("#facts", Static).update(facts_grid(snapshot))
        self.show_results(True)

    @on(Button.Pressed, "#quantize-pick")
    def quantize_pick(self) -> None:
        self.qp.open_quantize(self.ref, self.recommended)

    @on(DataTable.RowSelected, "#options")
    def row_selected(self, event: DataTable.RowSelected) -> None:
        key = self.format_rows.get(event.row_key.value or "")
        if key:
            self.qp.open_quantize(self.ref, key)


def facts_grid(snapshot) -> Table:
    facts = snapshot.facts
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    grid.add_column(style="dim")
    rows = (
        ("Parameters", facts.parameters),
        ("Stored as", facts.precision),
        ("Architecture", facts.architecture),
        ("Longest context", facts.context_length),
        ("Quantization", facts.quantization),
    )
    sure = {"exact": "exact", "cross-checked": "checked", "estimate": "estimate"}
    for name, fact in rows:
        if fact.value is None:
            continue
        value = f"{fact.value:,}" if isinstance(fact.value, int) else str(fact.value)
        grid.add_row(name, value, sure.get(str(fact.kind), ""))
    if snapshot.original_gb:
        grid.add_row("Download", words.gb(snapshot.original_gb), "")
    return grid


# Quantize ---------------------------------------------------------------------------------


class QuantizePage(Page):
    TITLE = "Quantize"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.plan_for: tuple | None = None  # the form values the shown plan was made for
        self.tool = QuantAlgorithm.GGUF  # what the Install button on the plan installs

    def compose_body(self) -> ComposeResult:
        s = self.qp.settings
        with Grid(classes="form"):
            yield field("Model", ModelField(), "wide")
            yield field("Format", Dropdown(FORMATS, value=AUTO, id="format"))
            yield field("Aim for", Dropdown(goal_options(), value=s.goal, id="goal"))
            yield self.context_field()
        with Horizontal(classes="actions"):
            yield Action("Review", variant="primary", id="review")
        yield LoadingIndicator(classes="busy")
        with Results(classes="results"):
            with Card(id="plan", classes="card"):
                yield Static("", id="plan-text")
                yield Action("Start", variant="primary", id="start")
                yield Action("Install", variant="primary", id="install")
            with Vertical(id="running"):
                yield Static("", id="stage")
                yield ProgressLog(id="progress")
            yield Card("", id="result", classes="card")
            yield Static("", classes="problem")

    def on_mount(self) -> None:
        for selector in ("#plan", "#running", "#result", ".problem"):
            self.query_one(selector).display = False

    def set_format(self, key: str) -> None:
        self.query_one("#format", Dropdown).value = key if key in FORMAT_KEYS else AUTO

    def form(self) -> tuple | None:
        ref = self.model()
        if not ref:
            return None
        return ref, self.value("format"), self.value("goal"), self.value("context")

    @on(Button.Pressed, "#review")
    def run(self) -> None:
        form = self.form()
        if form is None:
            return
        self.qp.remember(form[0])
        self.on_mount()
        self.query_one(".busy").display = True
        self.review(*form)

    @background("quantize-plan")
    def review(self, ref: str, key: str, goal: str, context: int) -> None:
        from ocq.cli import default_output
        from ocq.model_analyzer import resolve_model
        from ocq.plan import generate_plan

        kv_cache = self.qp.settings.kv_cache
        hw = self.qp.hardware()
        snapshot = resolve_model(ref, hw, self.qp.llmfit_bin, context, kv_cache)
        plan = generate_plan(
            snapshot.model_id, hw, snapshot, OptimizationGoal(goal), default_output(ref),
            format_choices(key, context, kv_cache),
        )  # fmt: skip
        self.app.call_from_thread(
            self.show_plan, plan, (ref, key, goal, context), snapshot.num_layers
        )

    def show_plan(self, plan, form: tuple, num_layers: int = 0) -> None:
        from ocq.tui.install import MISSING, READY, tool_state

        self.query_one(".busy").display = False
        self.plan_for = form
        predicted = plan.predicted
        text = Text()
        text.append(f"{plan.model_id}\n", "dim")
        text.append(f"{plan.quant_config.summary()}\n", "bold")
        if predicted:
            text.append_text(words.fit(predicted.fit_level))
            details = [words.gb(predicted.memory_required_gb), words.place(predicted),
                       words.layers(predicted, num_layers), words.speed(predicted)]  # fmt: skip
            text.append("  " + "  |  ".join(d for d in details if d != "-"))
        text.append(f"\nSaves to {plan.output_dir}", "dim")
        tool = tool_state(plan.quant_config.algorithm, self.qp.hardware())
        for warning in plan.warnings:
            if tool.state == MISSING and words.warning(warning).endswith(
                "not installed on this PC."
            ):
                continue  # the Install button says it
            text.append(f"\n{words.warning(warning)}", "yellow")
        self.query_one("#plan-text", Static).update(text)
        self.query_one("#start").display = tool.state == READY
        install = self.query_one("#install", Button)
        install.display = tool.state == MISSING
        install.label = f"Install {tool.name} tools"
        self.tool = tool.algorithm
        self.query_one("#plan").display = True

    @on(Button.Pressed, "#install")
    def install(self) -> None:
        log = self.query_one(ProgressLog)

        def started() -> None:
            self.query_one("#plan").display = False
            self.query_one("#stage", Static).update(Text("Installing ...", "bold"))
            log.clear()
            self.query_one("#running").display = True

        def done(state, problem: str) -> None:
            self.query_one("#running").display = False
            if problem:
                self.failed(problem)
            else:
                self.run()  # review again, now with the tools in place

        self.qp.ask_install(self.tool, log, started, done)

    @on(Button.Pressed, "#start")
    def start(self) -> None:
        form = self.form()
        if form is None or form != self.plan_for:
            self.notify("The choices changed. Review again first.", severity="warning")
            return

        def confirmed(yes: bool | None) -> None:
            if yes:
                self.begin(*form)

        self.app.push_screen(
            Confirm(
                "Start quantizing?",
                "This downloads the model if needed and can take a while.",
                "Start",
            ),
            confirmed,
        )

    def begin(self, ref: str, key: str, goal: str, context: int) -> None:
        if self.qp.busy:
            self.notify("Another quantization is running.", severity="warning")
            return
        self.query_one("#plan").display = False
        self.query_one("#result").display = False
        self.query_one("#running").display = True
        stage = self.query_one("#stage", Static)
        stage.update(Text(f"Quantizing {ref} ...", "bold"))
        progress = self.query_one(ProgressLog)
        progress.clear()

        def step(line: str) -> None:  # the latest progress line, under the title
            stage.update(Text.assemble((f"Quantizing {ref}\n", "bold"), (line[:120], "dim")))

        progress.on_line = step
        self.qp.job_started("Quantizing")
        self.app.begin_capture_print(progress)
        self.quantize(ref, key, goal, context)

    @background("quantize")
    def quantize(self, ref: str, key: str, goal: str, context: int) -> None:
        from ocq.cli import default_output
        from ocq.pipeline import run_optimize

        try:
            result = run_optimize(
                ref, OptimizationGoal(goal), default_output(ref), dry_run=False,
                llmfit_bin=self.qp.llmfit_bin,
                choices=format_choices(key, context, self.qp.settings.kv_cache),
            )  # fmt: skip
        except Exception as exc:  # the app must stay up whatever happens
            self.app.call_from_thread(self.finished, None, f"{type(exc).__name__}: {exc}")
            return
        self.app.call_from_thread(self.finished, result, "")

    def finished(self, result, problem: str) -> None:
        progress = self.query_one(ProgressLog)
        self.app.end_capture_print(progress)
        progress.flush()
        ok = bool(result and result.validation and result.validation.passed)
        self.qp.job_finished("Quantized" if ok else "Quantizing failed", ok)
        self.query_one("#running").display = False
        card = self.query_one("#result", Card)
        card.update(result_text(result, problem))
        card.set_class(not ok, "bad")
        card.display = True


def result_text(result, problem: str) -> Text:
    text = Text()
    if result is None or not result.quantization:
        text.append("Quantizing failed\n", "bold red")
        text.append(problem or "Something went wrong.")
        return text
    quantization, validation, bench = result.quantization, result.validation, result.benchmark
    if not quantization.success:
        text.append("Quantizing failed\n", "bold red")
        text.append(quantization.message)
        return text
    passed = validation is not None and validation.passed
    text.append(
        "Done\n" if passed else "Made, but the check failed\n",
        "bold green" if passed else "bold yellow",
    )
    text.append(
        f"{result.plan.quant_config.summary()}  |  {words.gb(quantization.artifact_size_gb)}"
    )
    if bench and bench.tokens_per_second:
        text.append(f"  |  {bench.tokens_per_second:.0f} tok/s measured")
    text.append(f"\nSaved to {quantization.artifact_path or result.plan.output_dir}", "dim")
    if not passed and validation:
        for check in validation.checks:
            text.append(f"\n{check}", "yellow")
    if bench and bench.sample_output:
        text.append("\n\nSample answer\n", "dim")
        text.append(bench.sample_output.strip()[:400])
    return text


# Verify -----------------------------------------------------------------------------------


class VerifyPage(Page):
    TITLE = "Verify"

    def compose_body(self) -> ComposeResult:
        with Grid(classes="form"):
            yield field("GGUF model", ModelField(gguf_only=True), "wide")
            yield self.context_field()
        with Horizontal(classes="actions"):
            yield Action("Verify", variant="primary", id="verify")
        yield LoadingIndicator(classes="busy")
        with Results(classes="results"):
            yield Static("", id="verdict")
            yield RowTable(cursor_type="row", id="checks")
            yield Static("", classes="problem")

    def on_mount(self) -> None:
        for selector in ("#verdict", "#checks", ".problem"):
            self.query_one(selector).display = False

    @on(Button.Pressed, "#verify")
    def run(self) -> None:
        ref = self.model()
        if not ref:
            return
        self.on_mount()
        self.query_one(".busy").display = True
        self.load(ref, self.value("context"), self.qp.settings.kv_cache)

    @background("verify")
    def load(self, ref: str, context: int, kv_cache: str) -> None:
        from ocq.local_model import parse_model_ref
        from ocq.model_analyzer import quant_options, resolve_model
        from ocq.quantizers.llama_cpp import tool_path
        from ocq.verify import verify

        hw = self.qp.hardware()
        snapshot = resolve_model(ref, hw, None, context, kv_cache)
        path = parse_model_ref(ref)[1]
        if not snapshot.gguf_files or path is None or not path.is_file():
            raise OcqError("Choose a .gguf file on this PC, or an Ollama model.")
        (row,) = quant_options(snapshot, hw, context, kv_cache)
        # llama-server, as the CLI uses: what QuantPilot plans for.
        tool = tool_path("llama-server", hw)
        log, checks, _ = verify(snapshot, path, hw, tool, context, kv_cache, row.experts_in_ram)
        self.app.call_from_thread(self.show, log, checks)

    def show(self, log, checks) -> None:
        self.query_one(".busy").display = False
        differ = [check for check in checks if not check.ok]
        verdict = Text()
        if differ:
            verdict.append(f"{len(differ)} of {len(checks)} differ", "bold yellow")
        else:
            verdict.append("Prediction matches llama.cpp", "bold green")
        offloaded, total = log.layers_on_gpu
        where = "Ran on the CPU" if log.cpu_only else f"{offloaded} of {total} layers on the GPU"
        verdict.append(f"\n{where}", "dim")
        self.query_one("#verdict", Static).update(verdict)
        table = self.query_one("#checks", DataTable)
        table.clear(columns=True)
        table.add_columns("", "Predicted", "Measured", "")
        for check in checks:
            if not check.ok:
                mark = Text("differs", "yellow")
            else:
                mark = Text("within" if check.upper_bound else "match", "green")
            table.add_row(
                CHECK_NAMES.get(check.name, check.name.capitalize()),
                check_value(check.predicted_mib, check.unit),
                check_value(check.actual_mib, check.unit),
                mark,
            )
        for selector in ("#verdict", "#checks"):
            self.query_one(selector).display = True


# What ocq verify compares, in the app's words.
CHECK_NAMES = {
    "context cache (KV)": "Context memory",
    "Mamba / convolution state": "Model state",
    "layers on the GPU": "Layers on the GPU",
    "weights on the GPU": "Weights on the GPU",
    "weights on the CPU (file range)": "Weights in RAM",
    "weights in RAM (file range)": "Weights in RAM",
    "weights repacked for this CPU": "Weights copied for this CPU",
    "working memory (upper bound)": "Working memory",
}


def check_value(value: float, unit: str) -> str:
    if unit == "layers":
        return f"{value:,.0f}"
    return f"{value:,.1f} MB" if value < 10 else f"{value:,.0f} MB"


# Settings ---------------------------------------------------------------------------------


class SettingsPage(Page):
    TITLE = "Settings"

    def compose_body(self) -> ComposeResult:
        s = self.qp.settings
        themes = [(name, name) for name in sorted(self.app.available_themes)]
        kv = [(KV_LABELS.get(kind, kind), kind) for kind in KV_CACHE_TYPES]
        with Results(classes="results"):
            with Grid(classes="form"):
                yield field("Theme", Dropdown(themes, value=self.app.theme, id="theme"))
                yield field(
                    "Context",
                    Dropdown(context_options(s.context), value=s.context, id="context"),
                )
                yield field("Context memory", Dropdown(kv, value=s.kv_cache, id="kv_cache"))
                yield field("Aim for", Dropdown(goal_options(), value=s.goal, id="goal"))
                yield field("Big banner", Switch(value=s.show_banner, id="show_banner"))
                yield field("Simple borders", Switch(value=s.ascii_borders, id="ascii_borders"))
                yield field(
                    "Model folders",
                    TextArea("\n".join(s.model_folders), id="model_folders"),
                    "wide",
                )

    @on(Dropdown.Changed)
    def dropdown_changed(self, event: Dropdown.Changed) -> None:
        if event.select.id == "theme":
            self.app.theme = str(event.value)  # the app saves it
        elif event.select.id:
            self.qp.update_setting(event.select.id, event.value)

    @on(Switch.Changed)
    def switch_changed(self, event: Switch.Changed) -> None:
        self.qp.update_setting(event.switch.id or "", event.value)

    @on(TextArea.Changed)
    def folders_changed(self, event: TextArea.Changed) -> None:
        folders = [line.strip() for line in event.text_area.text.splitlines() if line.strip()]
        self.qp.update_setting("model_folders", folders)
