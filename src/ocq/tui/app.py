"""QuantPilot's full-screen terminal app: `ocq` with no arguments, or `ocq tui`.

Arrow keys move through the menu on the left, Enter opens a page, Tab moves between fields,
and every choice that has a fixed set of answers is a dropdown. Each page runs the same
function as the matching `ocq` command (see runner.py) and shows the command line it stands
for, so what works here also works in scripts.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    DataTable,
    Footer,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    OptionList,
    RichLog,
    Select,
    Static,
    Switch,
    TextArea,
)
from textual.widgets.option_list import Option

from ocq import cli
from ocq.engine import GGUF_TYPES
from ocq.tui import runner
from ocq.tui.banner import COMPACT, TAGLINE, banner_text
from ocq.tui.settings import SETTINGS_FILE, Settings, load_settings, save_settings
from ocq.types import KV_CACHE_TYPES, OcqError, OptimizationGoal, QuantAlgorithm

AUTO = "auto"  # dropdown value for "let QuantPilot choose", what leaving out the flag does
CONTEXTS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
KV_LABELS = {
    "f16": "f16 - full size",
    "q8_0": "q8_0 - about half, almost no loss",
    "q4_0": "q4_0 - about a quarter, some loss",
}
USE_CASES = ("general", "coding", "reasoning", "chat", "multimodal", "embedding")
ON_OFF = [("Auto", AUTO), ("On", "on"), ("Off", "off")]

# (page id, menu label, section it sits under). The order is the menu's order.
PAGES = (
    ("home", "Home", ""),
    ("pc", "This PC", "Your PC"),
    ("models", "My models", "Models"),
    ("recommend", "Suggestions", "Models"),
    ("inspect", "Inspect", "One model"),
    ("feasibility", "What fits", "One model"),
    ("plan", "Plan", "One model"),
    ("quantize", "Quantize", "One model"),
    ("verify", "Verify GGUF", "One model"),
    ("settings", "Settings", ""),
)
PAGE_LABELS = {page: label for page, label, _ in PAGES}


# Small helpers ----------------------------------------------------------------------------


def quote(arg: str) -> str:
    """An argument as it would be typed in a shell, for the "Same as" line."""
    if arg and not any(char in arg for char in " \t\"'&|;<>()$`*?"):
        return arg
    return '"' + arg.replace('"', '\\"') + '"'


def context_options(default: int) -> list[tuple[str, int]]:
    return [(f"{value:,} tokens", value) for value in sorted({*CONTEXTS, default})]


def kv_options() -> list[tuple[str, str]]:
    return [(KV_LABELS.get(kind, kind), kind) for kind in KV_CACHE_TYPES]


GOAL_LABELS = {
    OptimizationGoal.BALANCED: "Balanced",
    OptimizationGoal.SPEED: "Speed - fastest",
    OptimizationGoal.QUALITY: "Quality - best that fits",
    OptimizationGoal.MEMORY: "Memory - smallest",
}


def goal_options() -> list[tuple[str, str]]:
    return [(GOAL_LABELS[goal], goal.value) for goal in OptimizationGoal]


def on_off(value: str) -> bool | None:
    return None if value == AUTO else value == "on"


def field(label: str, widget: Widget, classes: str = "") -> Vertical:
    """A form field: a short label above its input or dropdown."""
    return Vertical(Label(label, classes="field-label"), widget, classes=f"field {classes}".strip())


def hardware_summary(hw) -> str:
    """One line, for example: Apple M1, 8 cores | 16.0 GB RAM | Apple M1 16.0 GB (Metal)."""
    parts = [f"{hw.cpu_name}, {hw.cpu_cores} cores", f"{hw.total_ram_gb:.1f} GB RAM"]
    for gpu in hw.gpus:
        vram = f" {gpu.vram_gb:.1f} GB" if gpu.vram_gb else ""
        count = f"{gpu.count} x " if gpu.count > 1 else ""
        parts.append(f"{count}{gpu.name}{vram}")
    if not hw.gpus:
        parts.append("no GPU")
    return " | ".join(parts) + f" ({hw.backend})"


def this_pc(llmfit_bin: str | None = None) -> None:
    """What `ocq profile` and `ocq backends` print, one after the other."""
    cli.cmd_profile(llmfit_bin=llmfit_bin)
    cli.console.print()
    cli.cmd_backends(llmfit_bin=llmfit_bin)


# Widgets ----------------------------------------------------------------------------------


class Dropdown(Select):
    """A one-line dropdown that always has a value."""

    def __init__(self, options: Iterable[tuple[str, Any]], **kwargs: Any) -> None:
        super().__init__(options, allow_blank=False, compact=True, **kwargs)


class TextField(Input):
    """A one-line text field."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, compact=True, **kwargs)


class Action(Button):
    """A one-line button."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, compact=True, **kwargs)


class OutputLog(RichLog):
    """Where a command's output goes. It also receives plain print() lines, such as the
    progress lines a quantization run prints, while that command runs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(min_width=20, wrap=False, highlight=False, markup=False, **kwargs)
        self._partial = ""

    def on_print(self, event: events.Print) -> None:
        *lines, self._partial = (self._partial + event.text).split("\n")
        for line in lines:
            # Progress bars redraw one line with \r: keep what it shows last.
            self.write(Text.from_ansi(line.split("\r")[-1]))

    def flush_partial(self) -> None:
        if self._partial.strip():
            self.write(Text.from_ansi(self._partial.split("\r")[-1]))
        self._partial = ""

    def render_width(self) -> int:
        width = self.scrollable_content_region.width or self.app.size.width - 30
        return max(width - 1, 40)


class NavList(OptionList):
    """The menu on the left. Up/Down move and show the page; Enter or Right go into it."""

    BINDINGS = [Binding("right", "select", "Open", show=False)]


class ModelField(Horizontal):
    """Which model: a Hugging Face id, a local folder, a .gguf file or ollama:name:tag.
    Typing suggests models already on this PC and recent ones; Pick shows them as a list."""

    def __init__(self, placeholder: str, gguf_only: bool = False, **kwargs: Any) -> None:
        super().__init__(classes="model-field", **kwargs)
        self.placeholder = placeholder
        self.gguf_only = gguf_only

    def compose(self) -> ComposeResult:
        yield TextField(placeholder=self.placeholder, classes="model-input")
        yield Action("Pick", classes="pick", tooltip="Choose from the models on this PC")

    @property
    def input(self) -> Input:
        return self.query_one(Input)

    @property
    def value(self) -> str:
        return self.input.value.strip()

    @value.setter
    def value(self, ref: str) -> None:
        self.input.value = ref

    def set_suggestions(self, refs: list[str]) -> None:
        self.input.suggester = SuggestFromList(refs, case_sensitive=False)

    @on(Button.Pressed, ".pick")
    def pick(self, event: Button.Pressed) -> None:
        event.stop()

        def chosen(ref: str | None) -> None:
            if ref:
                self.value = ref
                self.input.focus()

        self.app.push_screen(ModelPicker(self.app, self.gguf_only), chosen)


class StatusBar(Horizontal):
    def compose(self) -> ComposeResult:
        yield LoadingIndicator(id="spinner")
        yield Static("Ready", id="status-text")
        yield Static("", id="status-hw")

    def running(self, text: str) -> None:
        self.add_class("running")
        self.query_one("#status-text", Static).update(text)

    def finished(self, text: str) -> None:
        self.remove_class("running")
        self.query_one("#status-text", Static).update(text)

    def hardware(self, text: str) -> None:
        self.query_one("#status-hw", Static).update(text)


# Pop-up screens ---------------------------------------------------------------------------


class ModelPicker(ModalScreen[str | None]):
    """A list of recent models and the models found on this PC."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, app: QuantPilotApp, gguf_only: bool) -> None:
        super().__init__()
        self.options: list[Option | None] = []
        recent = [] if gguf_only else app.settings.recent_models
        if recent:
            self.options.append(Option("[b]Recent[/b]", disabled=True))
            self.options += [Option(escape(ref), id=f"r{i}:{ref}") for i, ref in enumerate(recent)]
        models = [m for m in app.known_models if m.complete]
        if gguf_only:
            models = [m for m in models if m.format == "GGUF"]
        if models:
            self.options.append(Option("[b]On this PC[/b]", disabled=True))
        for i, model in enumerate(sorted(models, key=lambda m: m.label.lower())):
            size = f"{model.size_bytes / 1e9:.1f} GB" if model.size_bytes else ""
            kind = f"{model.format} {model.quantization}".strip()
            details = " | ".join(filter(None, [model.source, kind, size]))
            self.options.append(
                Option(
                    f"{escape(model.label)}\n  [dim]{escape(details)}[/dim]", id=f"m{i}:{model.ref}"
                )
            )
        if not self.options:
            what = "GGUF models" if gguf_only else "models"
            self.options.append(
                Option(
                    f"No {what} found on this PC yet. Type a Hugging Face id instead, "
                    "or add your model folders in Settings.",
                    disabled=True,
                )
            )

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Pick a model", classes="dialog-title")
            yield OptionList(*self.options, id="picker")
            yield Label("Enter picks, Esc closes", classes="dialog-hint")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id.split(":", 1)[1] if event.option.id else None)


class ModelActions(ModalScreen[str | None]):
    """What to do with a model chosen from a table: each entry opens that page with it."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, title: str, gguf: bool) -> None:
        super().__init__()
        self.title_text = title
        self.gguf = gguf

    def compose(self) -> ComposeResult:
        actions = [
            Option("What fits on this PC", id="feasibility"),
            Option("Inspect what QuantPilot knows", id="inspect"),
            Option("Plan a quantization", id="plan"),
        ]
        if self.gguf:
            actions.append(Option("Verify against llama.cpp", id="verify"))
        with Vertical(classes="dialog"):
            yield Label(escape(self.title_text), classes="dialog-title")
            yield OptionList(*actions, id="actions")
            yield Label("Enter opens, Esc closes", classes="dialog-hint")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class Confirm(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel")]

    def __init__(self, title: str, message: str, yes: str) -> None:
        super().__init__()
        self.title_text, self.message, self.yes = title, message, yes

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Static(self.message, classes="dialog-body")
            with Horizontal(classes="dialog-buttons"):
                yield Button(self.yes, variant="warning", id="yes")
                yield Button("Cancel", id="no")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


HELP = """\
## Moving around

| Key | Does |
|---|---|
| Up / Down | Move in the menu, lists and tables |
| Enter or Right | Open the page under the cursor |
| Tab / Shift+Tab | Next / previous field |
| Enter on a dropdown | Open it; Up / Down and Enter choose |
| Enter in a text field | Run the page |
| Ctrl+R | Run the page |
| Esc | Back to the menu |
| Ctrl+P | Command palette: jump to any page, change theme |
| ? | This help |
| q | Quit (outside text fields), Ctrl+Q anywhere |

## Pages

- **This PC**: CPU, RAM, GPU and memory speed, and which quantization backends are ready.
- **My models**: models already downloaded by Ollama, LM Studio, Hugging Face or llama.cpp,
  and how each one fits. Enter on a row for what to do with it.
- **Suggestions**: models that suit this PC.
- **Inspect**: what QuantPilot knows about a model and how sure it is.
- **What fits**: memory and speed for every format and size of one model.
- **Plan**: what quantizing would do, without doing it.
- **Quantize**: quantize, check and benchmark a model for real.
- **Verify GGUF**: load a GGUF in llama.cpp once and compare its memory with the prediction.

Every page shows the `ocq` command it ran, for use in scripts.
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog help"):
            yield Label("QuantPilot help", classes="dialog-title")
            yield Markdown(HELP)
            yield Label("Esc closes", classes="dialog-hint")


# Pages ------------------------------------------------------------------------------------


class Page(Vertical):
    TITLE = ""
    HELP = ""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(classes="page", **kwargs)

    @property
    def qp(self) -> QuantPilotApp:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        yield Static(self.TITLE, classes="page-title")
        yield Static(self.help_text(), classes="page-help")
        yield from self.compose_body()

    def help_text(self) -> str:
        return self.HELP

    def compose_body(self) -> ComposeResult:
        yield from ()

    def opened(self) -> None:
        """Called each time the page is shown."""

    def run(self) -> None:
        """What Ctrl+R does on this page."""

    def focus_first(self) -> None:
        for widget in self.query("Input, Select, Button, DataTable, Switch, TextArea"):
            if widget.focusable:
                widget.focus()
                return

    def select(self, name: str) -> Any:
        return self.query_one(f"#{name}", Select).value

    def context_field(self) -> Vertical:
        default = self.qp.settings.context
        return field(
            "Context",
            Dropdown(context_options(default), value=default, id="context"),
        )

    def kv_field(self) -> Vertical:
        return field(
            "Context cache",
            Dropdown(kv_options(), value=self.qp.settings.kv_cache, id="kv"),
        )

    def number(self, name: str, label: str) -> int | None:
        """A whole number typed in an optional field. None when empty; ValueError if not."""
        text = self.query_one(f"#{name}", Input).value.strip()
        if not text:
            return None
        if not text.isdigit():
            raise ValueError(f"{label} must be a whole number.")
        return int(text)


class HomePage(Page):
    TITLE = "Welcome to QuantPilot"
    HELP = TAGLINE

    def compose_body(self) -> ComposeResult:
        yield Static("Looking at this PC ...", id="home-hw")
        yield Markdown(
            "1. **This PC** shows what QuantPilot sees: memory, GPU and memory speed.\n"
            "2. **My models** lists what you already downloaded; **Suggestions** finds more.\n"
            "3. **What fits** shows every format and size of a model with its memory and "
            "speed.\n"
            "4. **Plan** and **Quantize** make the version that fits.\n\n"
            "Use the arrow keys in the menu on the left, Enter to go into a page, and `?` for "
            "all the keys.",
            id="home-steps",
        )
        with Horizontal(classes="actions"):
            yield Action("Check this PC", variant="primary", name="pc")
            yield Action("My models", name="models")
            yield Action("Suggest models", name="recommend")
            yield Action("Check a model", name="feasibility")

    @on(Button.Pressed)
    def go(self, event: Button.Pressed) -> None:
        if event.button.name:
            self.qp.show_page(event.button.name, focus=True)


class CommandPage(Page):
    """A page with a form, a Run button and the command's output below."""

    AUTO_RUN = False  # run once the first time the page is shown
    RUN_LABEL = "Run"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.has_run = False

    def compose_body(self) -> ComposeResult:
        with Grid(classes="form"):
            yield from self.compose_form()
        yield from self.compose_extra()
        with Horizontal(classes="actions"):
            yield from self.compose_buttons()
        yield OutputLog(classes="output")

    def compose_form(self) -> ComposeResult:
        yield from ()

    def compose_extra(self) -> ComposeResult:
        yield from ()

    def compose_buttons(self) -> ComposeResult:
        yield Action(self.RUN_LABEL, variant="primary", classes="run")
        yield Action("Clear", classes="clear")

    def opened(self) -> None:
        if self.AUTO_RUN and not self.has_run:
            self.run()

    def build(self) -> tuple[Callable[..., Any], dict[str, Any], list[str]] | None:
        """(function, its options, the same as a command line) or None when a field is wrong."""
        return None

    def run(self) -> None:
        try:
            built = self.build()
        except ValueError as exc:
            self.notify(str(exc), title="Check the form", severity="warning")
            return
        if built is not None:
            self.has_run = True
            self.qp.run_job(self, *built)

    def job_done(self, ok: bool) -> None:
        """Called on the main thread when the command has finished."""

    @property
    def output(self) -> OutputLog:
        return self.query_one(OutputLog)

    @on(Button.Pressed, ".run")
    def run_pressed(self) -> None:
        self.run()

    @on(Button.Pressed, ".clear")
    def clear_pressed(self) -> None:
        self.output.clear()

    @on(Input.Submitted)
    def submitted(self) -> None:
        self.run()

    # Fields most pages share.

    def model(self) -> str:
        ref = self.query_one(ModelField).value
        if not ref:
            raise ValueError("Type a model, or press Pick to choose one.")
        return ref

    def set_model(self, ref: str) -> None:
        for model_field in self.query(ModelField):
            model_field.value = ref

    def context_args(self) -> tuple[dict[str, Any], list[str]]:
        context, kv_cache = self.select("context"), self.select("kv")
        args = ["-c", str(context)] + (["--kv-cache", kv_cache] if kv_cache != "f16" else [])
        return {"context": context, "kv_cache": kv_cache}, args


class PcPage(CommandPage):
    TITLE = "This PC"
    HELP = "What QuantPilot sees on this computer, and which quantization backends can run here."
    AUTO_RUN = True
    RUN_LABEL = "Check again"

    def build(self):
        return this_pc, {"llmfit_bin": self.qp.llmfit_bin}, ["profile", "&&", "ocq", "backends"]


class InspectPage(CommandPage):
    TITLE = "Inspect a model"
    HELP = (
        "Everything QuantPilot knows about a model, where each fact comes from and how sure it is."
    )

    def compose_form(self) -> ComposeResult:
        yield field(
            "Model", ModelField("Qwen/Qwen3-0.6B, a folder, a .gguf or ollama:name:tag"), "wide"
        )

    def build(self):
        ref = self.model()
        options = {"model": ref, "model_path": None, "llmfit_bin": self.qp.llmfit_bin}
        return cli.cmd_inspect, options, ["inspect", quote(ref)]


class FeasibilityPage(CommandPage):
    TITLE = "What fits"
    HELP = (
        "Memory and speed of a model on this PC, for the stored weights and each quantized format."
    )

    def compose_form(self) -> ComposeResult:
        yield field(
            "Model", ModelField("Qwen/Qwen3-0.6B, a folder, a .gguf or ollama:name:tag"), "wide"
        )
        yield self.context_field()
        yield self.kv_field()

    def build(self):
        ref = self.model()
        context, args = self.context_args()
        options = {"model": ref, "model_path": None, "llmfit_bin": self.qp.llmfit_bin, **context}
        return cli.cmd_feasibility, options, ["feasibility", quote(ref), *args]


class PlanPage(CommandPage):
    TITLE = "Plan a quantization"
    HELP = "What quantizing would do: backend, settings, output and warnings. Nothing is run."

    def compose_form(self) -> ComposeResult:
        yield field("Model", ModelField("Qwen/Qwen3-0.6B or a local model folder"), "wide")
        yield field(
            "Goal",
            Dropdown(goal_options(), value=self.qp.settings.goal, id="goal"),
        )
        backends = [("Auto - QuantPilot picks", AUTO)] + [
            (a.value, a.value) for a in QuantAlgorithm
        ]
        yield field("Backend", Dropdown(backends, value=AUTO, id="backend"))
        bits = [("Auto", AUTO)] + [(f"{b}-bit", b) for b in (2, 3, 4, 5, 6, 8)]
        yield field("Bits", Dropdown(bits, value=AUTO, id="bits"))
        gguf = [("Auto", AUTO)] + [(name, name) for name in GGUF_TYPES]
        yield field("GGUF type", Dropdown(gguf, value=AUTO, id="gguf-type"))
        yield self.context_field()
        yield self.kv_field()

    def compose_extra(self) -> ComposeResult:
        with Collapsible(title="More options", collapsed=True, classes="more"):
            with Grid(classes="form"):
                yield field(
                    "Output folder",
                    TextField(placeholder="./ocq-output/<model>", id="output"),
                    "wide",
                )
                yield field("AWQ/GPTQ group size", TextField(placeholder="auto", id="group-size"))
                yield field("Calibration samples", TextField(placeholder="auto", id="samples"))
                yield field("Tokens per sample", TextField(placeholder="auto", id="seq-len"))
                yield field("BNB double quant", Dropdown(ON_OFF, value=AUTO, id="double"))
                yield field("NVFP4 activations", Dropdown(ON_OFF, value=AUTO, id="fp4"))

    def plan_options(self) -> tuple[dict[str, Any], list[str]]:
        ref = self.model()
        args = [quote(ref)]
        goal = self.select("goal")
        if goal != OptimizationGoal.BALANCED.value:
            args += ["-g", goal]
        backend, bits, gguf_type = (
            self.select("backend"),
            self.select("bits"),
            self.select("gguf-type"),
        )
        args += ["-b", backend] if backend != AUTO else []
        args += ["--bits", str(bits)] if bits != AUTO else []
        args += ["--gguf-type", gguf_type] if gguf_type != AUTO else []
        group_size = self.number("group-size", "Group size")
        samples = self.number("samples", "Calibration samples")
        seq_len = self.number("seq-len", "Tokens per sample")
        args += ["--group-size", str(group_size)] if group_size else []
        args += ["--samples", str(samples)] if samples else []
        args += ["--seq-len", str(seq_len)] if seq_len else []
        double, fp4 = on_off(self.select("double")), on_off(self.select("fp4"))
        if double is not None:
            args.append("--double-quant" if double else "--single-quant")
        if fp4 is not None:
            args.append("--fp4-activations" if fp4 else "--fp4-weights-only")
        output = self.query_one("#output", Input).value.strip()
        args += ["-o", quote(output)] if output else []
        context, context_args = self.context_args()
        options = {
            "model": ref,
            "model_path": None,
            "goal": goal,
            "backend": None if backend == AUTO else backend,
            "bits": None if bits == AUTO else bits,
            "gguf_type": None if gguf_type == AUTO else gguf_type,
            "group_size": group_size,
            "samples": samples,
            "seq_len": seq_len,
            "double_quant": double,
            "fp4_activations": fp4,
            "output": Path(output).expanduser() if output else None,
            "llmfit_bin": self.qp.llmfit_bin,
            **context,
        }
        return options, args + context_args

    def build(self):
        options, args = self.plan_options()
        return cli.cmd_plan, options, ["plan", *args]


class QuantizePage(PlanPage):
    TITLE = "Quantize"
    HELP = (
        "Downloads the model, quantizes it, checks the result and measures its speed. "
        "Preview shows the plan first."
    )

    def compose_buttons(self) -> ComposeResult:
        yield Action("Preview plan", classes="run")
        yield Action("Quantize now", variant="warning", id="execute")
        yield Action("Clear", classes="clear")

    def build(self):
        options, args = self.plan_options()
        return cli.cmd_optimize, {**options, "dry_run": True}, ["optimize", *args]

    @on(Button.Pressed, "#execute")
    def execute(self) -> None:
        try:
            options, args = self.plan_options()
        except ValueError as exc:
            self.notify(str(exc), title="Check the form", severity="warning")
            return

        def confirmed(yes: bool | None) -> None:
            if yes:
                self.has_run = True
                command = ["optimize", *args, "--execute"]
                self.qp.run_job(self, cli.cmd_optimize, {**options, "dry_run": False}, command)

        self.app.push_screen(
            Confirm(
                "Quantize for real?",
                f"This downloads {options['model']} if needed and quantizes it. It can take a "
                "long time and needs free disk space for the original and the result. It keeps "
                "running until it finishes; quitting QuantPilot stops it.",
                "Quantize",
            ),
            confirmed,
        )


class VerifyPage(CommandPage):
    TITLE = "Verify a GGUF"
    HELP = (
        "Loads a GGUF in llama.cpp once and compares its memory with QuantPilot's prediction. "
        "Downloads llama.cpp the first time."
    )

    def compose_form(self) -> ComposeResult:
        yield field(
            "GGUF file", ModelField("path/to/model.gguf or ollama:name:tag", gguf_only=True), "wide"
        )
        yield self.context_field()
        yield self.kv_field()
        yield field(
            "Expert layers in RAM", TextField(placeholder="auto (what fits)", id="n-cpu-moe")
        )

    def build(self):
        ref = self.model()
        context, args = self.context_args()
        n_cpu_moe = self.number("n-cpu-moe", "Expert layers in RAM")
        args += ["--n-cpu-moe", str(n_cpu_moe)] if n_cpu_moe is not None else []
        options = {
            "model": ref,
            "n_cpu_moe": n_cpu_moe,
            "llmfit_bin": self.qp.llmfit_bin,
            **context,
        }
        return cli.cmd_verify, options, ["verify", quote(ref), *args]


class TablePage(Page):
    """A page whose result is a table of models; Enter on a row offers what to do with it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loaded = False
        self.refs: dict[str, tuple[str, bool]] = {}  # row key -> (model ref, is GGUF)

    def compose_body(self) -> ComposeResult:
        with Grid(classes="form"):
            yield from self.compose_form()
        with Horizontal(classes="actions"):
            yield Action(self.RUN_LABEL, variant="primary", classes="run")
        yield DataTable(cursor_type="row", zebra_stripes=True)
        yield Static("", classes="table-note")

    RUN_LABEL = "Refresh"

    def compose_form(self) -> ComposeResult:
        yield from ()

    def opened(self) -> None:
        if not self.loaded:
            self.run()

    @on(Button.Pressed, ".run")
    def run_pressed(self) -> None:
        self.run()

    def run(self) -> None:
        self.loaded = True
        self.query_one(DataTable).loading = True
        self.load(self.params())  # read the form here: widgets belong to the main thread

    def params(self) -> dict[str, Any]:
        return {}

    def fill(
        self, columns: Iterable[str], rows: list[tuple[str, str, bool, list]], note: str
    ) -> None:
        """Main thread: rows are (key, model ref, is GGUF, cells)."""
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns(*columns)
        self.refs = {}
        for key, ref, gguf, cells in rows:
            self.refs[key] = (ref, gguf)
            table.add_row(*cells, key=key)
        table.loading = False
        self.query_one(".table-note", Static).update(note)

    def failed(self, message: str) -> None:
        self.query_one(DataTable).loading = False
        self.query_one(".table-note", Static).update(f"[red]{escape(message)}[/red]")

    def load(self, params: dict[str, Any]) -> None:
        """Runs in a thread; hands its rows to fill() or its error to failed()."""
        raise NotImplementedError

    @on(DataTable.RowSelected)
    def row_selected(self, event: DataTable.RowSelected) -> None:
        ref, gguf = self.refs.get(event.row_key.value or "", ("", False))
        if not ref:
            return

        def chosen(page: str | None) -> None:
            if page:
                self.qp.open_with_model(page, ref)

        self.app.push_screen(ModelActions(ref, gguf), chosen)


class ModelsPage(TablePage):
    TITLE = "My models"
    HELP = (
        "Models already on this PC (Ollama, LM Studio, Hugging Face cache, llama.cpp, your "
        "folders) and how each fits as it is. Enter on a row for more."
    )

    def compose_form(self) -> ComposeResult:
        yield self.context_field()
        yield self.kv_field()
        yield field("Partial downloads", Switch(value=False, id="partial"))

    @on(Select.Changed)
    @on(Switch.Changed)
    def changed(self) -> None:
        if self.loaded:
            self.run()

    def params(self) -> dict[str, Any]:
        return {
            "context": self.select("context"),
            "kv_cache": self.select("kv"),
            "show_all": self.query_one("#partial", Switch).value,
            "folders": self.qp.settings.folders(),
        }

    @work(thread=True, exclusive=True, group="models")
    def load(self, params: dict[str, Any]) -> None:
        from ocq.discovery import GGUF, find_models, searched_locations, tilde
        from ocq.hardware import detect_hardware

        context, kv_cache = params["context"], params["kv_cache"]
        show_all, folders = params["show_all"], params["folders"]
        try:
            models = find_models(folders)
            hw = detect_hardware(self.qp.llmfit_bin)
        except OcqError as exc:
            self.app.call_from_thread(self.failed, str(exc))
            return
        self.qp.known_models = models
        rows = []
        for i, model in enumerate(sorted(models, key=lambda m: (m.source, m.label.lower()))):
            if not (model.complete or show_all):
                continue
            fit, place, speed = cli.local_fit(model, hw, context, kv_cache)
            note = model.note
            if fit.startswith("?") and len(fit) > 1:
                note, fit = "; ".join(filter(None, [note, fit[2:]])), "?"
            label = Text(model.label)
            if note:
                label.append(f"  {note}", style="dim" if model.complete else "yellow")
            size = f"{model.size_bytes / 1e9:.2f} GB" if model.size_bytes else "-"
            kind = f"{model.format} {model.quantization}".strip()
            cells = [label, model.source, kind, size, Text.from_markup(fit), place, speed]
            rows.append((f"{i}", model.ref, model.format == GGUF, cells))
        searched = ", ".join(f"{s} {tilde(p)}" for s, p in searched_locations(folders))
        note = (
            f"[dim]{len(rows)} model(s), fit at {context:,} tokens in the memory free now. "
            f"Searched: {escape(searched) or 'no model folders found'}. "
            "Add folders in Settings.[/dim]"
        )
        if not rows:
            note = (
                "[b]No models found on this PC yet.[/b] Use Suggestions to find one, or add the "
                "folders where you keep models in Settings.\n" + note
            )
        columns = ("Model", "Source", "Format", "Size", "Fit", "Runs on", "tok/s")
        self.app.call_from_thread(self.fill, columns, rows, note)
        self.app.call_from_thread(self.qp.refresh_suggestions)


class RecommendPage(TablePage):
    TITLE = "Suggestions"
    HELP = "Models that suit this PC, from llmfit's catalog. Enter on a row for more."
    RUN_LABEL = "Find models"

    def compose_form(self) -> ComposeResult:
        uses = [("Any", AUTO)] + [(use.capitalize(), use) for use in USE_CASES]
        yield field("Use", Dropdown(uses, value=AUTO, id="use"))
        counts = [(str(n), n) for n in (5, 10, 20, 50)]
        yield field("How many", Dropdown(counts, value=10, id="limit"))

    def params(self) -> dict[str, Any]:
        return {"use": self.select("use"), "limit": self.select("limit")}

    @work(thread=True, exclusive=True, group="recommend")
    def load(self, params: dict[str, Any]) -> None:
        from ocq.hardware import detect_hardware
        from ocq.model_analyzer import recommend_models

        use, limit = params["use"], params["limit"]
        try:
            hw = detect_hardware(self.qp.llmfit_bin)
            models = recommend_models(hw, limit, None if use == AUTO else use, self.qp.llmfit_bin)
        except OcqError as exc:
            self.app.call_from_thread(self.failed, str(exc))
            return
        rows = []
        for i, model in enumerate(models):
            fit = model.feasibility
            cells = [
                model.model_id,
                str(model.parameter_count),
                Text.from_markup(cli.colored_fit(fit.fit_level)),
                cli.mode_label(fit.run_mode),
                fit.quant_label,
                f"{fit.memory_required_gb:.1f} / {fit.memory_available_gb:.1f}",
                f"{fit.estimated_tps:.0f}" if fit.runnable else "-",
            ]
            rows.append((f"{i}", model.model_id, False, cells))
        command = "ocq recommend -n " + str(limit) + ("" if use == AUTO else f" --use-case {use}")
        note = f"[dim]Same as: {command}. Memory GB is needed / free.[/dim]"
        columns = ("Model", "Params", "Fit", "Runs on", "Quant", "Memory GB", "tok/s")
        self.app.call_from_thread(self.fill, columns, rows, note)


class SettingsPage(Page):
    TITLE = "Settings"

    def help_text(self) -> str:
        return f"Saved as you change them, in {escape(str(self.qp.settings_path))}"

    def compose_body(self) -> ComposeResult:
        s = self.qp.settings
        themes = []
        for name in sorted(self.app.available_themes):
            label = f"{name} (your terminal's own colors)" if name.startswith("ansi-") else name
            themes.append((label, name))
        with Grid(classes="form"):
            yield field("Theme", Dropdown(themes, value=self.app.theme, id="theme"))
            yield field("Big banner", Switch(value=s.show_banner, id="show_banner"))
            yield field("Plain ASCII borders", Switch(value=s.ascii_borders, id="ascii_borders"))
            yield field(
                "Default context",
                Dropdown(context_options(s.context), value=s.context, id="context"),
            )
            yield field(
                "Default context cache",
                Dropdown(kv_options(), value=s.kv_cache, id="kv_cache"),
            )
            yield field("Default goal", Dropdown(goal_options(), value=s.goal, id="goal"))
            yield field(
                "Extra model folders, one per line",
                TextArea("\n".join(s.model_folders), id="model_folders"),
                "wide",
            )
            yield field(
                "llmfit program (empty: the one that came with QuantPilot)",
                TextField(s.llmfit_bin, placeholder="llmfit", id="llmfit_bin"),
                "wide",
            )
        yield Static(
            "[dim]Plain ASCII borders: for old consoles that show boxes or question marks "
            "instead of lines. Defaults apply to pages the next time QuantPilot starts.[/dim]",
            classes="table-note",
        )

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "theme":
            self.app.theme = str(event.value)  # saved by the app's theme watcher
        elif event.select.id in ("context", "kv_cache", "goal"):
            self.qp.update_setting(event.select.id, event.value)

    @on(Switch.Changed)
    def switch_changed(self, event: Switch.Changed) -> None:
        self.qp.update_setting(event.switch.id or "", event.value)

    @on(TextArea.Changed)
    def folders_changed(self, event: TextArea.Changed) -> None:
        folders = [line.strip() for line in event.text_area.text.splitlines() if line.strip()]
        self.qp.update_setting("model_folders", folders)

    @on(Input.Changed, "#llmfit_bin")
    def llmfit_changed(self, event: Input.Changed) -> None:
        self.qp.update_setting("llmfit_bin", event.value.strip())


# The app ----------------------------------------------------------------------------------


def nav_options() -> list[Option | None]:
    options: list[Option | None] = []
    section = None
    for page, label, heading in PAGES:
        if heading != section:
            options.append(None)
            if heading:
                options.append(Option(f"[b]{heading}[/b]", disabled=True))
            section = heading
        options.append(Option(f" {label}", id=page))
    return options[1:]


class QuantPilotApp(App):
    CSS_PATH = "styles.tcss"
    TITLE = "QuantPilot"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (100, "-wide")]
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help"),
        Binding("f1", "help", "Help", show=False),
        Binding("ctrl+r", "run", "Run"),
        Binding("escape", "menu", "Menu"),
    ]

    def __init__(self, settings_path: Path = SETTINGS_FILE, look_on_start: bool = True) -> None:
        super().__init__()
        self.settings_path = settings_path
        self.settings: Settings = load_settings(settings_path)
        self.look_on_start = look_on_start  # hardware and models in the background (off in tests)
        self.known_models: list = []
        self.current_model = ""

    @property
    def llmfit_bin(self) -> str | None:
        return self.settings.llmfit_bin or None

    def compose(self) -> ComposeResult:
        with Vertical(id="masthead"):
            yield Static(id="banner")
            yield Static(TAGLINE, id="tagline")
        with Horizontal(id="main"):
            yield NavList(*nav_options(), id="nav")
            with ContentSwitcher(initial="home", id="pages"):
                yield HomePage(id="home")
                yield PcPage(id="pc")
                yield ModelsPage(id="models")
                yield RecommendPage(id="recommend")
                yield InspectPage(id="inspect")
                yield FeasibilityPage(id="feasibility")
                yield PlanPage(id="plan")
                yield QuantizePage(id="quantize")
                yield VerifyPage(id="verify")
                yield SettingsPage(id="settings")
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        if self.settings.theme in self.available_themes:
            self.theme = self.settings.theme
        self.theme_changed_signal.subscribe(self, self.theme_changed)
        self.apply_look()
        nav = self.query_one(NavList)
        nav.highlighted = 0
        nav.focus()
        if self.look_on_start:
            self.look_around()

    def on_resize(self, event: events.Resize) -> None:
        self.apply_look()

    def apply_look(self) -> None:
        """The big banner on the home page (if it is on in Settings and fits), one line on the
        other pages so their results get the room."""
        on_home = self.query_one(ContentSwitcher).current == "home"
        big = self.settings.show_banner and on_home
        text = banner_text(self.size.width, self.size.height) if big else COMPACT
        self.query_one("#banner", Static).update(text)
        self.query_one("#tagline").display = big
        self.query_one("#masthead").set_class(text == COMPACT, "slim")
        self.set_class(self.settings.ascii_borders, "-ascii")

    def theme_changed(self, theme) -> None:
        self.update_setting("theme", theme.name)
        select = self.query_one("#settings #theme", Select)
        if select.value != theme.name:
            select.value = theme.name

    def update_setting(self, name: str, value: Any) -> None:
        if getattr(self.settings, name, None) == value:
            return
        setattr(self.settings, name, value)
        self.save_settings()
        if name in ("show_banner", "ascii_borders"):
            self.apply_look()

    def save_settings(self) -> None:
        try:
            save_settings(self.settings, self.settings_path)
        except OSError as exc:
            self.notify(f"Could not save settings: {exc}", severity="error")

    # Background look at the PC, so the status bar and the model suggestions are ready.

    @work(thread=True, group="startup")
    def look_around(self) -> None:
        from ocq.discovery import find_models
        from ocq.hardware import detect_hardware

        try:
            hw = detect_hardware(self.llmfit_bin)
            summary = hardware_summary(hw)
        except Exception as exc:  # the app works without it
            summary = f"Could not read the hardware: {exc}"
        self.call_from_thread(self.show_hardware, summary)
        try:
            self.known_models = find_models(self.settings.folders())
        except Exception:
            self.known_models = []
        self.call_from_thread(self.refresh_suggestions)

    def show_hardware(self, summary: str) -> None:
        self.query_one(StatusBar).hardware(summary)
        self.query_one("#home-hw", Static).update(f"[b]This PC:[/b] {escape(summary)}")

    def refresh_suggestions(self) -> None:
        refs = list(self.settings.recent_models)
        refs += [m.ref for m in self.known_models if m.complete and m.ref not in refs]
        for model_field in self.query(ModelField):
            model_field.set_suggestions(refs)

    # Pages.

    @property
    def page(self) -> Page:
        switcher = self.query_one(ContentSwitcher)
        return switcher.query_one(f"#{switcher.current}", Page)

    def show_page(self, page_id: str, focus: bool = False) -> None:
        switcher = self.query_one(ContentSwitcher)
        if switcher.current != page_id:
            switcher.current = page_id
            nav = self.query_one(NavList)
            index = nav.get_option_index(page_id)
            if nav.highlighted != index:
                nav.highlighted = index
            page = self.page
            if self.current_model:
                for model_field in page.query(ModelField):
                    if not model_field.value and not model_field.gguf_only:
                        model_field.value = self.current_model
            self.apply_look()
            page.opened()
        if focus:
            self.page.focus_first()

    def open_with_model(self, page_id: str, ref: str) -> None:
        self.show_page(page_id, focus=True)
        page = self.page
        if isinstance(page, CommandPage):
            page.set_model(ref)
            page.run()

    @on(OptionList.OptionHighlighted, "#nav")
    def nav_moved(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id:
            self.show_page(event.option.id)

    @on(OptionList.OptionSelected, "#nav")
    def nav_chosen(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.show_page(event.option.id, focus=True)

    # Running commands.

    def run_job(
        self, page: CommandPage, function: Callable[..., Any], options: dict, args: list[str]
    ) -> None:
        if runner.busy():
            self.notify("Wait for the running command to finish.", severity="warning")
            return
        model = options.get("model")
        if isinstance(model, str) and model:
            self.current_model = model
            self.settings.remember_model(model)
            self.save_settings()
            self.refresh_suggestions()
        log = page.output
        log.clear()
        log.write(Text.assemble(("Same as: ", "dim"), ("ocq " + " ".join(args), "bold")))
        self.begin_capture_print(log)
        self.query_one(StatusBar).running(f"Running ocq {args[0]} ...")
        self.job(page, log, function, options, log.render_width(), args[0])

    @work(thread=True, group="job")
    def job(
        self,
        page: CommandPage,
        log: OutputLog,
        function: Callable[..., Any],
        options: dict,
        width: int,
        name: str,
    ) -> None:
        started = time.monotonic()

        def emit(text: Text) -> None:
            self.call_from_thread(log.write, text)

        ok = runner.run_command(function, emit, width, self.settings.ascii_borders, **options)
        self.call_from_thread(self.job_done, page, log, ok, name, time.monotonic() - started)

    def job_done(
        self, page: CommandPage, log: OutputLog, ok: bool, name: str, seconds: float
    ) -> None:
        self.end_capture_print(log)
        log.flush_partial()
        if name != "optimize":  # results read from the top; a quantization log from the end
            log.call_after_refresh(log.scroll_home, animate=False)  # after its own scroll
        state = "[green]Done[/green]" if ok else "[red]Finished with errors[/red]"
        self.query_one(StatusBar).finished(f"{state}: ocq {name} ({seconds:.1f} s)")
        page.job_done(ok)

    # Keys and the command palette.

    def action_help(self) -> None:
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen())

    def action_run(self) -> None:
        self.page.run()

    def action_menu(self) -> None:
        self.query_one(NavList).focus()

    async def action_quit(self) -> None:
        if not runner.busy():
            self.exit()
            return

        def answer(yes: bool | None) -> None:
            if yes:
                self.exit()

        self.push_screen(
            Confirm("Quit QuantPilot?", "A command is still running. Quitting stops it.", "Quit"),
            answer,
        )

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        for page, label, _ in PAGES:
            yield SystemCommand(
                f"Go to {label}", f"Open the {label} page", partial(self.show_page, page, True)
            )
        yield SystemCommand("Run this page", "Same as Ctrl+R", self.action_run)


def run_tui() -> None:
    app = QuantPilotApp()
    app.run()
    if runner.busy():
        # A command thread cannot be stopped from outside; leave without waiting for it.
        os._exit(0)
