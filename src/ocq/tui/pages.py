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
from textual.color import Gradient
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Input,
    Label,
    LoadingIndicator,
    ProgressBar,
    Static,
    Switch,
    TextArea,
)

from ocq import env_file
from ocq.model_analyzer import QUANTIZED_LEVELS
from ocq.tui import words
from ocq.tui.banner import TAGLINE
from ocq.tui.widgets import (
    BENCH_STEPS,
    Action,
    Card,
    Confirm,
    Dropdown,
    ModelActions,
    ModelField,
    ProgressLog,
    Results,
    RowTable,
    StepTracker,
    TextField,
    field,
    meter_progress,
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
BAR_GRADIENT = Gradient.from_colors("#22d3ee", "#3b82f6", "#a78bfa")
CONTEXTS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
KV_LABELS = {"f16": "Full", "q8_0": "Half (q8_0)", "q4_0": "Quarter (q4_0)"}
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
    """Every model considered for a kind of work: the ones on this PC and the current ones on
    the Hub. Those that run here come first, best first, with their best option; then the ones
    that do not, with why."""

    TITLE = "Discover"
    COLUMNS = ("Model", "Best option", "Fit", "Speed", "Where", "Why not")

    def compose_filters(self) -> ComposeResult:
        with Grid(classes="form"):
            yield workload_field(self.qp.settings.workload)

    @on(Dropdown.Changed, "#workload")
    def workload_changed(self, event: Dropdown.Changed) -> None:
        if event.value != self.qp.settings.workload:
            self.qp.settings.workload = str(event.value)
            self.qp.save_settings()
        if self.loaded:
            self.run()

    def params(self) -> dict[str, Any]:
        return {"workload": self.value("workload")}

    @background("discover")
    def load(self, params: dict[str, Any]) -> None:
        from ocq import recommend as engine

        hw = self.qp.hardware()
        request = engine.RecommendationRequest(engine.WORKLOADS[params["workload"]])
        result = engine.recommend(request, hw, self.qp.settings.folders())
        self.app.call_from_thread(self.fill, discover_rows(result), "No models found.")


def discover_rows(result) -> list[tuple[str, str, bool, list]]:
    """Rows for the Discover table: (key, model ref, is a GGUF file, cells)."""
    from ocq import recommend as engine

    rows = []
    shown: set[str] = set()
    for i, best in enumerate(engine.best_per_model(result)):
        candidate, option = best.candidate, best.candidate.option
        name = Text(candidate.model_id)
        if i == 0:
            name.append("  recommended", "dim")
        where = "On this PC" if candidate.local else "Download"
        cells = [name, option.label, words.fit(option.fit_level), words.speed(option), where, ""]
        gguf = option.row_kind == "gguf" and not candidate.quantize
        rows.append((f"r{i}", candidate.model_ref, gguf, cells))
        shown.add(candidate.model_ref)
    # Models with no option that runs here: the limit their closest option broke.
    closest: dict[str, Any] = {}
    for rejection in result.rejected:
        ref = rejection.candidate.model_ref
        if ref in shown:
            continue
        key = (rejection.failed[0].name == engine.RUNTIME, len(rejection.failed),
               rejection.candidate.option.memory_required_gb)  # fmt: skip
        if ref not in closest or key < closest[ref][0]:
            closest[ref] = (key, rejection)
    for i, (_, rejection) in enumerate(sorted(closest.values(), key=lambda item: item[0][2])):
        candidate = rejection.candidate
        fit = Text("Too big" if rejection.failed[0].name == engine.MEMORY else "Not here", "red")
        where = "On this PC" if candidate.local else "Download"
        why = Text(words.plain(rejection.failed[0].text), "dim")
        cells = [Text(candidate.model_id, "dim"), "-", fit, "-", where, why]
        rows.append((f"x{i}", candidate.model_ref, False, cells))
    for i, (repo, reason) in enumerate(result.too_big):
        why = Text(words.plain(reason.removeprefix("Too big for this PC: ")), "dim")
        cells = [Text(repo, "dim"), "-", Text("Too big", "red"), "-", "Download", why]
        rows.append((f"t{i}", repo, False, cells))
    return rows


# Check a model ----------------------------------------------------------------------------


class CheckPage(Page):
    TITLE = "Check a model"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.format_rows: dict[str, tuple[str, int, str]] = {}  # row -> (format, context, cache)
        self.pick: tuple[str, int, str] | None = None  # what Quantize to this opens with
        self.ref = ""
        self.chart = None  # the chart last shown (FormatChart)

    def compose_body(self) -> ComposeResult:
        with Grid(classes="form"):
            yield field("Model", ModelField(), "wide")
            yield workload_field(self.qp.settings.workload)
            yield field(
                "Context",
                Dropdown(
                    [(FOR_WORKLOAD, 0), *((words.context(c), c) for c in CONTEXTS)],
                    value=0,
                    id="context",
                ),
            )
        with Horizontal(classes="actions"):
            yield Action("Check", variant="primary", id="check")
        yield LoadingIndicator(classes="busy")
        with Results(classes="results"):
            with Card(id="pick", classes="card"):
                yield Static("", id="pick-text")
                yield Action("Quantize to this", variant="primary", id="quantize-pick")
            with Collapsible(title="Why this pick", collapsed=True, id="why"):
                yield Static("", id="why-text")
            yield Static("", id="chart")
            yield RowTable(cursor_type="row", zebra_stripes=True, id="options")
            with Collapsible(title="About this model", collapsed=True, id="about"):
                yield Static("", id="facts")
            yield Static("", classes="problem")

    def on_mount(self) -> None:
        self.show_results(False)

    def show_results(self, shown: bool) -> None:
        for selector in ("#pick", "#why", "#chart", "#options", "#about"):
            self.query_one(selector).display = shown
        self.query_one(".problem").display = False

    @on(Dropdown.Changed, "#workload")
    def workload_changed(self, event: Dropdown.Changed) -> None:
        if event.value != self.qp.settings.workload:
            self.qp.settings.workload = str(event.value)
            self.qp.save_settings()

    @on(Button.Pressed, "#check")
    def run(self) -> None:
        ref = self.model()
        if not ref:
            return
        self.ref = ref
        self.qp.remember(ref)
        self.show_results(False)
        self.query_one(".busy").display = True
        self.load(ref, self.value("workload"), self.value("context"))

    @background("check")
    def load(self, ref: str, workload: str, context: int) -> None:
        from ocq import recommend as engine
        from ocq.cli import default_output
        from ocq.plan import generate_plan

        hw = self.qp.hardware()
        request = engine.RecommendationRequest(engine.WORKLOADS[workload], (ref,), context)
        result = engine.recommend(request, hw, self.qp.settings.folders())
        if not result.ranked and not result.rejected:
            why = result.unread[0][1] if result.unread else "could not read this model."
            raise OcqError(why[:1].upper() + why[1:])
        warnings: list[str] = []
        best = result.best
        key = format_key(best.candidate.option) if best and best.candidate.quantize else None
        if best and key:
            # The plan's checks of quantizing itself: tools installed, disk, calibration.
            candidate = best.candidate
            try:
                plan = generate_plan(
                    candidate.model.model_id, hw, candidate.model, OptimizationGoal.BALANCED,
                    default_output(ref), format_choices(key, candidate.context, candidate.kv_cache),
                )  # fmt: skip
                warnings = list(plan.warnings)
            except OcqError as exc:
                warnings = [str(exc)]
        self.app.call_from_thread(self.show, result, warnings)

    def show(self, result, warnings: list[str]) -> None:
        from ocq import recommend as engine

        self.query_one(".busy").display = False
        best = result.best
        rows = engine.compare_rows(result)
        model = (best.candidate if best else rows[0].candidate).model
        self.pick = None
        text = Text()
        if best is None:
            text.append("Not for this PC\n", "dim")
            closest = engine.explain_model(result, rows[0].candidate.model_ref)
            text.append(words.plain(closest[-1].text), "bold")
            why = Text()
        else:
            candidate, option = best.candidate, best.candidate.option
            text.append(f"Recommended for {result.request.workload.label}\n", "dim")
            cache = f" · {words.cache(candidate.kv_cache)}" if candidate.kv_cache != "f16" else ""
            text.append(f"{option.label} · {candidate.runtime}{cache}\n", "bold")
            text.append_text(words.fit(option.fit_level))
            details = [
                words.gb(option.memory_required_gb),
                words.place(option),
                words.layers(option, getattr(model, "num_layers", 0)),
                words.speed(option),
                words.context(candidate.context),
            ]
            text.append("  " + "  |  ".join(d for d in details if d != "-"))
            text.append("\nConfidence: ")
            text.append_text(words.confidence(best.confidence.level))
            for warning in [*best.warnings, *warnings]:
                text.append(f"\n{words.warning(warning)}", "yellow")
            why = words.reasons(best.reasons, best.confidence)
            key = format_key(option) if candidate.quantize else None
            if key:
                self.pick = (key, candidate.context, candidate.kv_cache)
        self.query_one("#pick-text", Static).update(text)
        self.query_one("#why-text", Static).update(why)
        self.query_one("#quantize-pick").display = self.pick is not None

        table = self.query_one("#options", DataTable)
        table.clear(columns=True)
        table.add_columns("Format", "Runs with", "Memory", "Fit", "Speed", "Why")
        self.format_rows = {}
        for i, row in enumerate(rows):
            candidate, option = row.candidate, row.candidate.option
            label = Text(option.label)
            if candidate.kv_cache != "f16":
                label.append(f" + {words.cache(candidate.kv_cache)}", "dim")
            fit = words.fit(option.fit_level)
            if row.status == engine.REJECTED and row.limit != engine.MEMORY:
                fit = Text("Not here", style="red")  # a runtime, context, disk... limit
            why = Text(words.plain(row.why), "bold" if row.status == engine.PICK else "dim")
            speed = words.speed(option) if row.status != engine.REJECTED else "-"
            memory = words.gb(option.memory_required_gb)
            cells = [label, candidate.runtime, memory, fit, speed, why]
            table.add_row(*cells, key=str(i))
            key = format_key(option)
            if candidate.quantize and key and row.status != engine.REJECTED:
                self.format_rows[str(i)] = (key, candidate.context, candidate.kv_cache)
        self.query_one("#facts", Static).update(facts_grid(model))
        self.chart = format_chart(result, rows)
        self.query_one("#chart", Static).update(self.chart)
        self.show_results(True)
        self.query_one("#why").display = best is not None

    @on(Button.Pressed, "#quantize-pick")
    def quantize_pick(self) -> None:
        if self.pick:
            self.qp.open_quantize(self.ref, *self.pick)

    @on(DataTable.RowSelected, "#options")
    def row_selected(self, event: DataTable.RowSelected) -> None:
        chosen = self.format_rows.get(event.row_key.value or "")
        if chosen:
            self.qp.open_quantize(self.ref, *chosen)


ROLE_WORDS = {"read": "read access", "write": "read and write access",
              "fineGrained": "fine-grained access"}  # fmt: skip
CHART_ROWS = 12


def short_runtime_reason(why: str) -> str:
    """'bitsandbytes needs an NVIDIA GPU, and this PC has none.' -> 'needs an NVIDIA GPU'."""
    for need in ("needs an NVIDIA GPU", "Apple Silicon"):
        if need in why:
            return need if need.startswith("needs") else "needs a Mac"
    if "Windows" in why:
        return "not on Windows"
    return why.split(",")[0].rstrip(".")


SHORT_LIMITS = {
    "memory": "too big",
    "context": "holds too little context",
    "architecture": "cannot be converted yet",
    "quantization": "cannot be made here",
    "disk": "not enough disk",
}


def format_chart(result, rows):
    """Memory, speed and quality of each format of the model, as bars."""
    from ocq import recommend as engine
    from ocq.tui.charts import ChartRow, FormatChart

    chart = []
    for row in rows[:CHART_ROWS]:
        candidate, option = row.candidate, row.candidate.option
        label = option.label
        if candidate.kv_cache != "f16":
            label += f" + {words.cache(candidate.kv_cache).removesuffix(' cache')}"
        runs = row.status != engine.REJECTED
        note = ""
        if not runs:
            note = SHORT_LIMITS.get(row.limit) or short_runtime_reason(row.why)
        quality = engine.kept_of(candidate)[0]  # measured on this PC when it was
        speed = option.estimated_tps if runs else 0.0
        chart.append(
            ChartRow(label, option.memory_required_gb, speed, quality, row.status == engine.PICK,
                     runs, note)
        )  # fmt: skip
    best = result.best
    free = (
        best.candidate.option.memory_available_gb
        if best
        else rows[0].candidate.option.memory_available_gb
    )
    return FormatChart(chart, free)


FOR_WORKLOAD = "For the workload"


def workload_field(value: str) -> Vertical:
    from ocq.recommend import WORKLOADS

    options = [(profile.label, key) for key, profile in WORKLOADS.items()]
    return field("Workload", Dropdown(options, value=value, id="workload"))


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
        # Check's pick: (model, format, context) -> the cache type it was recommended with.
        self.presets: dict[tuple, str] = {}
        # Form values with the Recommended format -> the (format, cache type) review chose.
        self.resolved: dict[tuple, tuple[str, str]] = {}
        self.made: Path | None = None  # the file the last run made, or the plan finds made

    def compose_body(self) -> ComposeResult:
        s = self.qp.settings
        with Grid(classes="form"):
            yield field("Model", ModelField(), "wide")
            yield field("Format", Dropdown(FORMATS, value=AUTO, id="format"))
            yield workload_field(s.workload)
            yield self.context_field()
        with Horizontal(classes="actions"):
            yield Action("Review", variant="primary", id="review")
        yield LoadingIndicator(classes="busy")
        with Results(classes="results"):
            with Card(id="plan", classes="card"):
                yield Static("", id="plan-text")
                with Horizontal(classes="card-actions"):
                    yield Action("Check it", variant="primary", id="check-existing")
                    yield Action("Start", variant="primary", id="start")
                    yield Action("Install", variant="primary", id="install")
            with Vertical(id="running"):
                yield Static("", id="stage")
                yield StepTracker(id="steps")
                yield ProgressBar(
                    id="step-bar", show_eta=False, show_percentage=False, gradient=BAR_GRADIENT
                )
                yield Static("", id="step-detail")
                yield ProgressLog(id="progress")
            with Card(id="result", classes="card"):
                yield Static("", id="result-text")
                with Horizontal(classes="card-actions"):
                    yield Action("Check it", variant="primary", id="check-made")
                    yield Action("Benchmark it", id="bench-made")
                    yield Action("Verify it", id="verify-made")
            yield Static("", classes="problem")

    def on_mount(self) -> None:
        for selector in ("#plan", "#running", "#result", ".problem"):
            self.query_one(selector).display = False

    def set_format(self, key: str) -> None:
        self.query_one("#format", Dropdown).value = key if key in FORMAT_KEYS else AUTO

    def show_running(self, what: str) -> None:
        """The progress area: "quantize" (title, steps, bar, log), "install" (title, bar, log),
        "steps" (only the finished steps), or "" (hidden)."""
        shown = {
            "quantize": ("#stage", "#steps", "#step-bar", "#step-detail", "#progress"),
            "install": ("#stage", "#step-bar", "#progress"),
            "steps": ("#steps",),
        }.get(what, ())
        for selector in ("#stage", "#steps", "#step-bar", "#step-detail", "#progress"):
            self.query_one(selector).display = selector in shown
        self.query_one("#running").display = bool(shown)

    def preset(self, ref: str, key: str, context: int, kv_cache: str) -> None:
        """What Check recommended: its context goes in the field, and its cache type is used
        while the form still asks for that model, format and context."""
        self.presets = {}
        if context:
            box = self.query_one("#context", Dropdown)
            box.set_options(context_options(context))  # every usual size, and this one
            box.value = context
        if kv_cache:
            self.presets[(ref, key, context or self.value("context"))] = kv_cache

    def kv_cache_for(self, ref: str, key: str, context: int) -> str:
        return self.presets.get((ref, key, context)) or self.qp.settings.kv_cache

    def form(self) -> tuple | None:
        ref = self.model()
        if not ref:
            return None
        return ref, self.value("format"), self.value("workload"), self.value("context")

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
    def review(self, ref: str, key: str, workload: str, context: int) -> None:
        from ocq.cli import default_output
        from ocq.model_analyzer import resolve_model
        from ocq.plan import generate_plan

        hw = self.qp.hardware()
        form = (ref, key, workload, context)
        made_key, kv_cache = self.resolve_format(form, hw)
        snapshot = resolve_model(ref, hw, self.qp.llmfit_bin, context, kv_cache)
        plan = generate_plan(
            snapshot.model_id, hw, snapshot, OptimizationGoal.BALANCED, default_output(ref),
            format_choices(made_key, context, kv_cache),
        )  # fmt: skip
        made = made_file(plan)
        self.app.call_from_thread(self.show_plan, plan, form, snapshot.num_layers, made)

    def resolve_format(self, form: tuple, hw) -> tuple[str, str]:
        """(format, cache type) to make. Recommended: the engine's best option for this model
        and workload that QuantPilot makes, as Check shows it. Called from a worker thread."""
        ref, key, workload, context = form
        if key != AUTO:
            return key, self.kv_cache_for(ref, key, context)
        from ocq import recommend as engine

        request = engine.RecommendationRequest(engine.WORKLOADS[workload], (ref,), context)
        result = engine.recommend(request, hw, self.qp.settings.folders())
        for recommendation in result.ranked:
            candidate = recommendation.candidate
            made = format_key(candidate.option) if candidate.quantize else None
            if made:
                self.resolved[form] = (made, candidate.kv_cache)
                return made, candidate.kv_cache
        if result.best is not None:
            raise OcqError(
                f"Nothing to quantize: {result.best.candidate.option.label} runs here as it is."
            )
        reasons = engine.explain_model(result, ref)
        raise OcqError(f"Nothing fits this PC. {reasons[-1].text}")

    def show_plan(self, plan, form: tuple, num_layers: int = 0, made: Path | None = None) -> None:
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
        if made:
            text.append("\nAlready made: ", "bold green")
            text.append(str(made), "dim")
        else:
            text.append(f"\nSaves to {plan.output_dir}", "dim")
        tool = tool_state(plan.quant_config.algorithm, self.qp.hardware())
        for warning in plan.warnings:
            if tool.state == MISSING and words.warning(warning).endswith(
                "not installed on this PC."
            ):
                continue  # the Install button says it
            text.append(f"\n{words.warning(warning)}", "yellow")
        self.query_one("#plan-text", Static).update(text)
        self.made = made
        start = self.query_one("#start", Button)
        start.display = tool.state == READY
        start.label = "Make it again" if made else "Start"
        start.variant = "default" if made else "primary"
        self.query_one("#check-existing").display = made is not None
        install = self.query_one("#install", Button)
        install.display = tool.state == MISSING
        install.label = f"Install {tool.name} tools"
        self.tool = tool.algorithm
        self.query_one("#plan").display = True

    @on(Button.Pressed, "#check-made, #check-existing")
    def check_made(self) -> None:
        if self.made:
            self.qp.open_with_model("check", str(self.made))

    @on(Button.Pressed, "#bench-made")
    def bench_made(self) -> None:
        if self.made:  # speed and quality: what the new file keeps of the original
            self.qp.open_bench(str(self.made), "fixed")

    @on(Button.Pressed, "#verify-made")
    def verify_made(self) -> None:
        if self.made:
            self.qp.open_with_model("verify", str(self.made))

    @on(Button.Pressed, "#install")
    def install(self) -> None:
        log = self.query_one(ProgressLog)

        def started() -> None:
            self.query_one("#plan").display = False
            self.query_one("#stage", Static).update(Text("Installing ...", "bold"))
            log.clear()
            self.query_one("#step-bar", ProgressBar).update(total=None, progress=0)
            self.show_running("install")

        def done(state, problem: str) -> None:
            self.show_running("")
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

    def begin(self, ref: str, key: str, workload: str, context: int) -> None:
        if self.qp.busy:
            self.notify("Another quantization is running.", severity="warning")
            return
        self.query_one("#plan").display = False
        self.query_one("#result").display = False
        self.show_running("quantize")
        self.query_one("#review", Button).focus()  # Start is hidden now: keep the keys working
        stage = self.query_one("#stage", Static)
        stage.update(Text(f"Quantizing {ref} ...", "bold"))
        progress = self.query_one(ProgressLog)
        progress.clear()
        steps = self.query_one(StepTracker)
        bar = self.query_one("#step-bar", ProgressBar)
        detail = self.query_one("#step-detail", Static)
        steps.start()
        bar.update(total=None, progress=0)  # pulses until a step says how far it is
        detail.update("")

        def step(line: str) -> None:  # the latest progress line, under the title
            if steps.heard(line):
                bar.update(total=None, progress=0)
                detail.update("")
            stage.update(Text.assemble((f"Quantizing {ref}\n", "bold"), (line[:120], "dim")))

        def meter(line: str) -> None:  # a download or conversion meter as it moves
            percent, done = meter_progress(line)
            if percent is not None:
                bar.update(total=100, progress=percent)
            shown = [f"{percent:.0f}%" if percent is not None else "", done]
            if any(shown):
                detail.update("  ·  ".join(part for part in shown if part))

        progress.on_line = step
        progress.on_meter = meter
        self.qp.job_started("Quantizing")
        self.app.begin_capture_print(progress)
        self.quantize(ref, key, workload, context)

    @background("quantize")
    def quantize(self, ref: str, key: str, workload: str, context: int) -> None:
        from ocq.cli import default_output
        from ocq.pipeline import run_optimize

        made, kv_cache = self.resolved.get((ref, key, workload, context)) or (
            key,
            self.kv_cache_for(ref, key, context),
        )
        try:
            result = run_optimize(
                ref, OptimizationGoal.BALANCED, default_output(ref), dry_run=False,
                llmfit_bin=self.qp.llmfit_bin, choices=format_choices(made, context, kv_cache),
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
        self.query_one(StepTracker).finish(ok)
        self.qp.job_finished("Quantized" if ok else "Quantizing failed", ok)
        self.show_running("steps")  # the steps and their times stay above the result
        card = self.query_one("#result", Card)
        self.query_one("#result-text", Static).update(result_text(result, problem))
        card.set_class(not ok, "bad")
        card.display = True
        made = result.quantization.artifact_path if result and result.quantization else ""
        made_ok = bool(made) and result.quantization.success and Path(made).exists()
        self.made = Path(made) if made_ok else None
        self.query_one("#check-made").display = made_ok
        self.query_one("#verify-made").display = made_ok and made.endswith(".gguf")
        self.query_one("#bench-made").display = made_ok and made.endswith(".gguf")
        # The keys need a visible place to be: the result's first action, else Review.
        self.query_one("#check-made" if made_ok else "#review", Button).focus()


def made_file(plan) -> Path | None:
    """The GGUF file this plan makes, when it is there already (ocq optimize names it
    <model>-<type>.gguf). Other formats share one folder per model, so they are not told apart."""
    from ocq.quantizers.base import model_slug

    if plan.quant_config.algorithm != QuantAlgorithm.GGUF:
        return None
    path = (
        plan.output_dir
        / "artifact"
        / f"{model_slug(plan.model_id)}-{plan.quant_config.gguf_type}.gguf"
    )
    return path if path.is_file() else None


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


# Benchmark --------------------------------------------------------------------------------

# What to measure: speed only, or speed and quality (on the fixed slice, or the whole text).
MEASURES = (
    ("Speed", ""),
    ("Speed and quality", "fixed"),  # the first 40 chunks: a few minutes
    ("Quality, whole text", "full"),  # every chunk: long
)
PLACED = {
    "gpu": "GPU",
    "cpu": "CPU",
    "gpu+cpu": "GPU and CPU",
    "gpu+experts": "GPU, experts in RAM",
}
HISTORY_ROWS = 12


class BenchPage(Page):
    TITLE = "Benchmark"

    def compose_body(self) -> ComposeResult:
        with Grid(classes="form"):
            yield field("Model", ModelField(gguf_only=True), "wide")
            yield field("Measure", Dropdown(MEASURES, value="", id="measure"))
        with Horizontal(classes="actions"):
            yield Action("Benchmark", variant="primary", id="bench")
        with Results(classes="results"):
            with Vertical(id="running"):
                yield Static("", id="stage")
                yield StepTracker(BENCH_STEPS, id="steps")
                yield ProgressBar(
                    id="step-bar", show_eta=False, show_percentage=False, gradient=BAR_GRADIENT
                )
                yield Static("", id="step-detail")
                yield ProgressLog(id="progress")
            yield Static("", id="verdict")
            yield RowTable(cursor_type="row", id="measured")
            yield Static("", classes="problem")
            yield Static("", id="history-title", classes="section-title")
            yield RowTable(cursor_type="row", id="speed-history")
            yield RowTable(cursor_type="row", id="quality-history")

    def on_mount(self) -> None:
        for selector in ("#running", "#verdict", "#measured", ".problem"):
            self.query_one(selector).display = False

    def opened(self) -> None:
        self.show_history("")

    def set_measure(self, value: str) -> None:
        self.query_one("#measure", Dropdown).value = value

    @on(Button.Pressed, "#bench")
    def run(self) -> None:
        ref = self.model()
        if not ref:
            return
        if self.qp.busy:
            self.notify("Another job is running.", severity="warning")
            return
        self.on_mount()
        measure = self.value("measure") or ""
        steps = self.query_one(StepTracker)
        # Only the steps this benchmark goes through.
        steps.steps = BENCH_STEPS if measure else BENCH_STEPS[:1]
        self.query_one("#running").display = True
        for selector in ("#stage", "#steps", "#step-bar", "#step-detail", "#progress"):
            self.query_one(selector).display = True
        stage = self.query_one("#stage", Static)
        stage.update(Text(f"Benchmarking {ref} ...", "bold"))
        progress = self.query_one(ProgressLog)
        progress.clear()
        bar = self.query_one("#step-bar", ProgressBar)
        detail = self.query_one("#step-detail", Static)
        bar.update(total=None, progress=0)
        detail.update("")
        steps.start()

        def step(line: str) -> None:
            if steps.heard(line):
                bar.update(total=None, progress=0)
                detail.update("")
            stage.update(Text.assemble((f"Benchmarking {ref}\n", "bold"), (line[:120], "dim")))

        def meter(line: str) -> None:
            percent, done = meter_progress(line)
            if percent is not None:
                bar.update(total=100, progress=percent)
            shown = [f"{percent:.0f}%" if percent is not None else "", done]
            if any(shown):
                detail.update("  ·  ".join(part for part in shown if part))

        progress.on_line = step
        progress.on_meter = meter
        self.qp.job_started("Benchmarking")
        self.app.begin_capture_print(progress)
        self.measure(ref, measure)

    @background("bench")
    def measure(self, ref: str, measure: str) -> None:
        from ocq import bench, quality

        piece = {"fixed": quality.FIXED, "full": quality.FULL}.get(measure)
        settings = self.qp.settings
        try:
            outcome = bench.run_benchmark(
                ref, self.qp.hardware(), piece=piece, context=settings.context,
                kv_cache=settings.kv_cache, llmfit_bin=self.qp.llmfit_bin,
            )  # fmt: skip
        except Exception as exc:  # the app must stay up whatever happens
            message = str(exc) if isinstance(exc, OcqError) else f"{type(exc).__name__}: {exc}"
            self.app.call_from_thread(self.finished, None, message)
            return
        self.app.call_from_thread(self.finished, outcome, "")

    def finished(self, outcome, problem: str) -> None:
        progress = self.query_one(ProgressLog)
        self.app.end_capture_print(progress)
        progress.flush()
        ok = outcome is not None
        self.query_one(StepTracker).finish(ok)
        self.qp.job_finished("Benchmarked" if ok else "Benchmark failed", ok)
        for selector in ("#stage", "#step-bar", "#step-detail", "#progress"):
            self.query_one(selector).display = False  # the steps and their times stay
        if not ok:
            self.failed(problem)
            return
        self.query_one("#verdict", Static).update(bench_verdict(outcome))
        table = self.query_one("#measured", DataTable)
        table.clear(columns=True)
        table.add_columns("", "Measured", "Expected")
        for row in bench_rows(outcome):
            table.add_row(*row)
        for selector in ("#verdict", "#measured"):
            self.query_one(selector).display = True
        self.show_history(outcome.model_id)
        table.focus()

    def show_history(self, model_id: str) -> None:
        """This PC's runs, newest first: of one model after a benchmark, else of all."""
        speed = self.query_one("#speed-history", DataTable)
        kept = self.query_one("#quality-history", DataTable)
        speeds, qualities = history_rows(model_id)
        title = self.query_one("#history-title", Static)
        if not speeds and not qualities:
            title.update(Text("Nothing benchmarked on this PC yet.", "dim"))
        else:
            title.update(Text("History on this PC" + (", this model" if model_id else ""), "bold"))
        speed.clear(columns=True)
        speed.add_columns("When", "Model", "Format", "Where", "Writing", "First token")
        for row in speeds:
            speed.add_row(*row)
        speed.display = bool(speeds)
        kept.clear(columns=True)
        kept.add_columns("When", "Model", "Format", "Perplexity", "Quality kept")
        for row in qualities:
            kept.add_row(*row)
        kept.display = bool(qualities)


def bench_verdict(outcome) -> Text:
    from ocq import measurements

    text = Text()
    text.append(f"{outcome.label}", "bold")
    if outcome.speed is not None:
        where = PLACED.get(measurements.placed(outcome.device), outcome.device)
        build = f" {outcome.speed.build}" if outcome.speed.build else ""
        text.append(f"  ·  {where}  ·  llama.cpp{build}", "dim")
        runs = outcome.runs
        text.append(f"\nSaved: {runs} run{'s' * (runs != 1)} of this setup on this PC", "dim")
    else:
        text.append("  ·  transformers", "dim")
    if outcome.note and outcome.note != "original":
        text.append(f"\n{words.plain(outcome.note)}", "yellow")
    return text


def bench_rows(outcome) -> list[tuple]:
    """Measured against what QuantPilot expected, in the app's words."""
    from ocq import measurements
    from ocq.recommend import GGUF_PPL_INCREASE, quality_kept

    option, rows = outcome.option, []
    speed = outcome.speed
    if speed is not None:
        expected = (
            f"about {option.estimated_tps:.0f}"
            if option.speed_basis == measurements.MEASURED
            else f"up to {option.estimated_tps:.0f}"
        )
        rows.append(("Writing", f"{speed.tokens_per_second:,.1f} tok/s", expected))
        for length in sorted(speed.prompt_speeds):
            rows.append(
                (f"Reading {length:,} tokens", f"{speed.prompt_speeds[length]:,.0f} tok/s", "")
            )
        for length in sorted(speed.prompt_speeds):
            guess = ""
            if length == min(speed.prompt_speeds) and option.prompt_tps and option.estimated_tps:
                guess = f"about {length / option.prompt_tps + 1 / option.estimated_tps:.2f} s"
            seconds = speed.first_token_seconds(length)
            rows.append((f"First token after {length:,}", f"{seconds:.2f} s", guess))
    if outcome.quality is not None:
        rows.append(("Perplexity", f"{outcome.quality.value:.2f}", "lower is better"))
        label = outcome.label.upper()
        estimate = f"about {quality_kept(label):.0%}" if label in GGUF_PPL_INCREASE else ""
        if outcome.note == "original":
            rows.append(("Quality kept", "the original", ""))
        elif outcome.kept is not None:
            rows.append(("Quality kept", f"{outcome.kept.kept:.0%}", estimate))
    return rows


def history_rows(model_id: str) -> tuple[list[tuple], list[tuple]]:
    from ocq import measurements, quality

    def matches(entry: dict) -> bool:
        if not model_id:
            return True
        names = (str(entry.get("model", "")), f"local/{entry.get('file', '')}")
        return model_id.lower() in (name.lower() for name in names)

    def when(entry: dict) -> str:  # 11-09 18:28: the month and day are enough here
        return str(entry.get("date", ""))[5:16].replace("T", " ") or "earlier"

    def name(entry: dict) -> str:  # SmolLM2-135M-Instruct, without who published it
        return str(entry.get("model", "")).split("/")[-1] or "?"

    speeds = []
    for run in reversed(measurements.load()):
        if not matches(run) or not run.get("label"):
            continue
        reading, writing = run.get("prompt_tps") or 0, run.get("tps") or 0
        first = f"{512 / reading + 1 / writing:.2f} s" if reading and writing else ""
        where = PLACED.get(measurements.placed(str(run.get("device", ""))), "")
        speeds.append((when(run), name(run), run["label"], where, f"{writing:,.1f} tok/s", first))
    qualities = []
    for result in reversed(quality.load()):
        if not matches(result):
            continue
        label = str(result.get("label", ""))
        if label.upper() in quality.FULL_PRECISION:
            kept = "the original"
        else:
            measured = quality.measured_kept(str(result.get("model", "")), label)
            kept = f"{measured.kept:.0%}" if measured else ""
        qualities.append((when(result), name(result), label, f"{result['ppl']:.2f}", kept))
    return speeds[:HISTORY_ROWS], qualities[:HISTORY_ROWS]


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
                yield workload_field(s.workload)
                yield field("Big banner", Switch(value=s.show_banner, id="show_banner"))
                yield field("Simple borders", Switch(value=s.ascii_borders, id="ascii_borders"))
                yield field(
                    "Model folders",
                    TextArea("\n".join(s.model_folders), id="model_folders"),
                    "wide",
                )
            yield Static("Hugging Face token", classes="section")
            yield Static("", id="token-state")
            with Horizontal(id="token-entry"):
                yield TextField(placeholder="Paste a read token (hf_...)", password=True,
                                id="hf-token")  # fmt: skip
                yield Action("Save", variant="primary", id="token-save")
                yield Action("Cancel", id="token-cancel")
            with Horizontal(classes="actions", id="token-actions"):
                yield Action("Add a token", variant="primary", id="token-add")
                yield Action("Change", id="token-change")
                yield Action("Remove", id="token-remove")

    def on_mount(self) -> None:
        self.account = None  # who the saved token belongs to, once checked
        self.token_note: Text | None = None  # the outcome of the last save, shown once
        self.query_one("#token-entry").display = False
        self.show_token()

    def opened(self) -> None:
        token, where = env_file.hf_token()
        if token and self.account is None:
            self.check_token(token, save=False)  # who it belongs to, once per session

    def show_token(self) -> None:
        """The token as it is now: masked, where it comes from, and what it may do."""
        token, where = env_file.hf_token()
        text = Text()
        if not token:
            text.append("None yet. Gated models such as Llama and Gemma need a read token: ", "dim")
            text.append("huggingface.co/settings/tokens", "underline")
        else:
            text.append(env_file.mask(token), "bold")
            if self.account and self.account.name:
                access = ROLE_WORDS.get(self.account.role, self.account.role)
                text.append(f"   Signed in as {self.account.name}", "green")
                text.append(f", {access}" if access else "", "green")
            if where == "shell":
                text.append("\nFrom your shell, which wins over a token saved here.", "dim")
        if self.token_note is not None:
            text.append("\n")
            text.append_text(self.token_note)
        self.query_one("#token-state", Static).update(text)
        editing = self.query_one("#token-entry").display
        self.query_one("#token-actions").display = not editing
        self.query_one("#token-add").display = not token
        self.query_one("#token-change").display = where == "file"
        self.query_one("#token-remove").display = where == "file"

    def edit_token(self, editing: bool) -> None:
        self.query_one("#token-entry").display = editing
        self.query_one("#hf-token", Input).value = ""
        self.token_note = None
        self.show_token()
        if editing:
            self.query_one("#hf-token", Input).focus()
        else:
            self.focus_token_actions()

    def focus_token_actions(self) -> None:
        """The first button shown: a hidden one would leave the keys nowhere to go."""
        for button in self.query("#token-actions Button"):
            if button.display:
                button.focus()
                return

    @on(Button.Pressed, "#token-add, #token-change")
    def add_token(self) -> None:
        self.edit_token(True)

    @on(Button.Pressed, "#token-cancel")
    def cancel_token(self) -> None:
        self.edit_token(False)

    @on(Button.Pressed, "#token-save")
    @on(Input.Submitted, "#hf-token")
    def save_token(self) -> None:
        token = self.query_one("#hf-token", Input).value.strip()
        if not token.startswith("hf_"):
            self.token_note = Text("Hugging Face tokens start with hf_.", "red")
            self.show_token()
            return
        self.token_note = Text("Checking the token with Hugging Face ...", "dim")
        self.show_token()
        self.check_token(token, save=True)

    @background("token")
    def check_token(self, token: str, save: bool) -> None:
        try:
            account, note = env_file.check_hf_token(token), None
        except OcqError as exc:  # Hugging Face says no: nothing is saved
            self.app.call_from_thread(self.token_checked, None, Text(str(exc), "red"), False)
            return
        except ConnectionError:
            account = None
            note = Text("Saved, not checked: Hugging Face could not be reached.", "yellow")
        if save:
            env_file.save(env_file.HF_TOKEN, token)
            if account and account.role == "write":
                note = Text("Saved. It can write too: a read token is enough.", "yellow")
            elif account:
                note = Text("Saved.", "green")
        self.app.call_from_thread(self.token_checked, account, note, save)

    def token_checked(self, account, note: Text | None, saved: bool) -> None:
        self.account = account or self.account
        if saved:
            self.query_one("#token-entry").display = False
        self.token_note = note
        self.show_token()
        if saved:
            self.focus_token_actions()

    @on(Button.Pressed, "#token-remove")
    def remove_token(self) -> None:
        def confirmed(yes: bool | None) -> None:
            if yes:
                env_file.remove(env_file.HF_TOKEN)
                self.account, self.token_note = None, Text("Removed.", "dim")
                self.show_token()

        self.app.push_screen(
            Confirm("Remove the token?", "Gated models cannot be read without one.", "Remove"),
            confirmed,
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
