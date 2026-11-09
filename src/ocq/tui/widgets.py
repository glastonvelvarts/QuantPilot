"""Building blocks the pages share: one-line fields, the model field and its picker, pop-ups."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, OptionList, RichLog, Select, Static
from textual.widgets.option_list import Option

from ocq.tui import words

# Keyboard: on a page, Up/Down (and Right) move between fields and buttons, and Left goes back
# to the menu. The widgets below give those keys back to the page where they would otherwise
# keep them: a closed dropdown, a text field with the cursor at its start, a table at its top.


def step_focus(widget: Widget, delta: int) -> None:
    """Moves the focus to the next (+1) or previous (-1) field of the page holding widget."""
    for node in widget.ancestors_with_self:
        if hasattr(node, "action_step"):
            node.action_step(delta)
            return


class Dropdown(Select, inherit_bindings=False):
    """A one-line dropdown that always has a value. Enter or Space opens it; Up/Down move
    between fields like everywhere else on the page."""

    BINDINGS = [Binding("enter,space", "show_overlay", "Open", show=False)]

    def __init__(self, options: Iterable[tuple[str, Any]], **kwargs: Any) -> None:
        super().__init__(options, allow_blank=False, compact=True, **kwargs)


class TextField(Input):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, compact=True, **kwargs)

    def action_cursor_left(self, select: bool = False) -> None:
        if self.cursor_position == 0 and not select and self.selection.is_empty:
            self.app.action_menu()  # type: ignore[attr-defined]
        else:
            super().action_cursor_left(select)


class RowTable(DataTable):
    """Up/Down move through the rows; Up on the first row leaves the table, Left goes back to
    the menu unless the table is scrolled sideways."""

    def action_cursor_up(self) -> None:
        if self.cursor_row <= 0:
            step_focus(self, -1)
        else:
            super().action_cursor_up()

    def action_cursor_left(self) -> None:
        if self.scroll_x <= 0:
            self.app.action_menu()  # type: ignore[attr-defined]
        else:
            super().action_cursor_left()


class Results(VerticalScroll, can_focus=False, inherit_bindings=False):
    """Where a page shows its result. It scrolls to whatever is focused inside it but takes
    no focus itself, and no keys: once it is taller than the screen, its scrolling keys would
    take Up and Down from the fields inside it."""


class Action(Button):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, compact=True, **kwargs)


def field(label: str, widget: Widget, classes: str = "") -> Vertical:
    """A short label above its field."""
    return Vertical(Label(label, classes="field-label"), widget, classes=f"field {classes}".strip())


class ModelField(Horizontal):
    """A Hugging Face id, a local folder, a .gguf file or an Ollama model, typed or picked."""

    def __init__(self, gguf_only: bool = False, **kwargs: Any) -> None:
        super().__init__(classes="model-field", **kwargs)
        self.gguf_only = gguf_only

    def compose(self) -> ComposeResult:
        hint = "a .gguf file or ollama:name" if self.gguf_only else "e.g. Qwen/Qwen3-4B"
        yield TextField(placeholder=hint, classes="model-input")
        yield Action("Browse", classes="pick")

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


class Card(Static):
    """A bordered box for one result, such as the recommendation."""


class ProgressLog(RichLog, can_focus=False):
    """Progress lines printed by a long job (download, convert, quantize...).

    Progress bars redraw their line, with \\r or with the "erase line" code that Rich uses;
    only what such a line shows last is kept. `on_line` hears every finished line, so a page
    can show the latest step above the box.
    """

    REDRAW = re.compile(r"\r|\x1b\[2K")
    CURSOR = re.compile(r"\x1b\[\?25[hl]|\x1b\[\d*[ABCDG]")  # hide/show, moves
    # Download and progress meters from libraries (Hugging Face, Rich): the steps QuantPilot
    # prints say the same in words.
    METER = re.compile(r"\|.*\||━|it/s|\d+(?:\.\d+)?\s?[kKMG]?i?B\b")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(min_width=20, wrap=True, markup=False, **kwargs)
        self._partial = ""
        self.on_line: Callable[[str], None] | None = None

    def on_print(self, event: events.Print) -> None:
        *lines, self._partial = (self._partial + event.text).split("\n")
        self._partial = self._last(self._partial, keep_empty=True)  # do not let redraws pile up
        for line in lines:
            self._add(self._last(line))

    def flush(self) -> None:
        self._add(self._last(self._partial))
        self._partial = ""

    def _last(self, line: str, keep_empty: bool = False) -> str:
        line = self.CURSOR.sub("", line)
        parts = [part for part in self.REDRAW.split(line) if part.strip() or keep_empty]
        return parts[-1] if parts else ""

    def _add(self, line: str) -> None:
        text = Text.from_ansi(line.strip())
        if text.plain and not self.METER.search(text.plain):
            self.write(text)
            if self.on_line:
                self.on_line(text.plain)


# Pop-ups ----------------------------------------------------------------------------------


class ModelPicker(ModalScreen[str | None]):
    """Recent models and the models found on this PC."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, app: Any, gguf_only: bool) -> None:
        super().__init__()
        self.options: list[Option | None] = []
        recent = [] if gguf_only else app.settings.recent_models
        if recent:
            self.options.append(Option("[dim]Recent[/dim]", disabled=True))
            self.options += [Option(escape(ref), id=f"r{i}:{ref}") for i, ref in enumerate(recent)]
        models = [m for m in app.known_models if m.complete]
        if gguf_only:
            models = [m for m in models if m.format == "GGUF"]
        if models:
            self.options.append(Option("[dim]On this PC[/dim]", disabled=True))
        for i, model in enumerate(sorted(models, key=lambda m: m.label.lower())):
            size = words.gb(model.size_bytes / 1e9) if model.size_bytes else ""
            details = escape(" | ".join(filter(None, [model.source, size])))
            label = f"{escape(model.label)}  [dim]{details}[/dim]"
            self.options.append(Option(label, id=f"m{i}:{model.ref}"))
        if not self.options:
            self.options.append(Option("No models on this PC yet", disabled=True))

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Choose a model", classes="dialog-title")
            yield OptionList(*self.options)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id.split(":", 1)[1] if event.option.id else None)


class ModelActions(ModalScreen[str | None]):
    """What to do with a model picked from a list."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, title: str, gguf: bool) -> None:
        super().__init__()
        self.title_text = title
        self.gguf = gguf

    def compose(self) -> ComposeResult:
        actions = [Option("Check how it runs", id="check"), Option("Quantize", id="quantize")]
        if self.gguf:
            actions.append(Option("Verify", id="verify"))
        with Vertical(classes="dialog"):
            yield Label(escape(self.title_text), classes="dialog-title")
            yield OptionList(*actions)

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
                yield Action(self.yes, variant="primary", id="yes")
                yield Action("Cancel", id="no")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


KEYS = (
    ("Up / Down", "Move"),
    ("Right", "Into a page, next field"),
    ("Left", "Back to the menu"),
    ("Enter", "Open, choose, run"),
    ("Esc", "Back to the menu"),
    ("Ctrl+P", "Go anywhere, theme"),
    ("Ctrl+C", "Quit"),
)


class KeysScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog keys"):
            yield Label("Keys", classes="dialog-title")
            for key, does in KEYS:
                yield Static(f"[b]{key:<10}[/b] {does}")
