"""QuantPilot's full-screen app: `ocq` with no arguments, or `ocq tui`.

The menu on the left opens the pages (pages.py). The app keeps what the pages share: the
settings, the hardware reading, the models found on this PC and the last model used.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import ContentSwitcher, Footer, LoadingIndicator, OptionList, Static
from textual.widgets.option_list import Option

from ocq.tui import words
from ocq.tui.banner import COMPACT, banner_text
from ocq.tui.pages import (
    CheckPage,
    DiscoverPage,
    HomePage,
    ModelsPage,
    Page,
    PcPage,
    QuantizePage,
    SettingsPage,
    VerifyPage,
)
from ocq.tui.settings import SETTINGS_FILE, Settings, load_settings, save_settings
from ocq.tui.widgets import Confirm, KeysScreen, ModelField, ProgressLog
from ocq.types import QuantAlgorithm

# (page id, menu label). None draws a line between groups.
MENU = (
    ("home", "Home"),
    None,
    ("check", "Check a model"),
    ("quantize", "Quantize"),
    None,
    ("discover", "Discover"),
    ("models", "My models"),
    ("pc", "This PC"),
    None,
    ("verify", "Verify"),
    ("settings", "Settings"),
)
PAGES = {
    "home": HomePage,
    "check": CheckPage,
    "quantize": QuantizePage,
    "discover": DiscoverPage,
    "models": ModelsPage,
    "pc": PcPage,
    "verify": VerifyPage,
    "settings": SettingsPage,
}
HARDWARE_FRESH_S = 30  # free memory changes; read it again after this long


class NavList(OptionList):
    """Up/Down move and show the page; Enter or Right go into it."""

    BINDINGS = [Binding("right", "select", "Open", show=False)]


class StatusBar(Horizontal):
    def compose(self) -> ComposeResult:
        yield LoadingIndicator(id="spinner")
        yield Static("", id="status-text")
        yield Static("", id="status-hw")

    def set(self, text: str | Text, running: bool = False) -> None:
        self.set_class(running, "running")
        self.query_one("#status-text", Static).update(text)

    def hardware(self, text: str) -> None:
        self.query_one("#status-hw", Static).update(text)


class QuantPilotApp(App):
    CSS_PATH = "styles.tcss"
    TITLE = "QuantPilot"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (100, "-wide")]
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("question_mark", "keys", "Keys"),
        Binding("escape", "menu", "Menu", show=False),
        Binding("ctrl+r", "run", "Run", show=False),
    ]

    def __init__(self, settings_path: Path = SETTINGS_FILE, look_on_start: bool = True) -> None:
        super().__init__()
        self.settings_path = settings_path
        self.settings: Settings = load_settings(settings_path)
        self.look_on_start = look_on_start  # read the PC and its models at start (off in tests)
        self.known_models: list = []
        self.current_model = ""
        self.busy = False  # a quantization is running
        self._hardware = None
        self._hardware_at = 0.0
        self._hardware_lock = threading.Lock()

    @property
    def llmfit_bin(self) -> str | None:
        return self.settings.llmfit_bin or None

    def compose(self) -> ComposeResult:
        with Vertical(id="masthead"):
            yield Static(COMPACT, id="banner")
        with Horizontal(id="main"):
            options = [Option(f" {item[1]}", id=item[0]) if item else None for item in MENU]
            yield NavList(*options, id="nav")
            with ContentSwitcher(initial="home", id="pages"):
                for page_id, page in PAGES.items():
                    yield page(id=page_id)
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
        self.refresh_suggestions()
        if self.look_on_start:
            self.look_around()

    def on_resize(self, event: events.Resize) -> None:
        self.apply_look()

    def apply_look(self) -> None:
        """Home shows the big banner in its middle (when it fits and is on in Settings); the
        other pages a one-line name at the top, so their results get the room."""
        on_home = self.query_one(ContentSwitcher).current == "home"
        big = banner_text(self.size.width, self.size.height) if self.settings.show_banner else ""
        big = "" if big == COMPACT else big
        self.query_one("#home-banner", Static).update(big or COMPACT)
        self.query_one("#home-tagline").display = bool(big)
        self.query_one("#masthead").display = not on_home
        self.set_class(self.settings.ascii_borders, "-ascii")

    # Settings.

    def theme_changed(self, theme) -> None:
        self.update_setting("theme", theme.name)
        select = self.query_one("#settings #theme")
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

    def remember(self, ref: str) -> None:
        """The model the user just used: kept for the other pages and the suggestions."""
        self.current_model = ref
        self.settings.remember_model(ref)
        self.save_settings()
        self.refresh_suggestions()

    # What the pages share.

    def hardware(self, fresh: bool = False):
        """This PC, read again when the last reading is older than HARDWARE_FRESH_S.
        Called from worker threads."""
        from ocq.hardware import detect_hardware

        with self._hardware_lock:
            old = time.monotonic() - self._hardware_at > HARDWARE_FRESH_S
            if self._hardware is None or fresh or old:
                self._hardware = detect_hardware(self.llmfit_bin)
                self._hardware_at = time.monotonic()
            return self._hardware

    @work(thread=True, group="startup")
    def look_around(self) -> None:
        from ocq.discovery import find_models

        try:
            summary = words.hardware(self.hardware())
        except Exception:  # the app works without it
            summary = ""
        self.call_from_thread(self.show_hardware, summary)
        try:
            self.known_models = find_models(self.settings.folders())
        except Exception:
            self.known_models = []
        self.call_from_thread(self.refresh_suggestions)

    def show_hardware(self, summary: str) -> None:
        self.query_one(StatusBar).hardware(summary)
        self.query_one("#home-hw", Static).update(summary)

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
        self.page.set_model(ref)
        self.page.run()

    def open_quantize(self, ref: str, format_key: str) -> None:
        self.show_page("quantize", focus=True)
        page = self.query_one(QuantizePage)
        page.set_model(ref)
        page.set_format(format_key)
        page.run()

    @on(OptionList.OptionHighlighted, "#nav")
    def nav_moved(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id:
            self.show_page(event.option.id)

    @on(OptionList.OptionSelected, "#nav")
    def nav_chosen(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.show_page(event.option.id, focus=True)

    # Long jobs.

    def job_started(self, what: str) -> None:
        self.busy = True
        self.query_one(StatusBar).set(f"{what} ...", running=True)

    def job_finished(self, what: str, ok: bool) -> None:
        self.busy = False
        self.query_one(StatusBar).set(Text(what, "green" if ok else "red"))

    # Installing a quantization tool (This PC and the Quantize page both use this).

    INSTALLS = {
        QuantAlgorithm.GGUF: "PyTorch, Transformers and llama.cpp",
        QuantAlgorithm.AWQ: "PyTorch and LLM Compressor",
        QuantAlgorithm.GPTQ: "PyTorch and LLM Compressor",
        QuantAlgorithm.NVFP4: "PyTorch and LLM Compressor",
        QuantAlgorithm.BNB: "PyTorch and bitsandbytes",
    }

    def ask_install(
        self,
        algorithm: QuantAlgorithm,
        log: ProgressLog,
        started: Callable[[], None],
        done: Callable[[Any, str], None],
    ) -> None:
        """Asks first, then installs in the background. `done(state, problem)` afterwards."""
        from ocq.tui.install import NAMES

        if self.busy:
            self.notify("Wait for the running job to finish.", severity="warning")
            return
        name = NAMES[algorithm]

        def answer(yes: bool | None) -> None:
            if yes:
                started()
                self.job_started(f"Installing the {name} tools")
                self.install_worker(algorithm, log, done)

        self.push_screen(
            Confirm(
                f"Install the {name} tools?",
                f"Downloads {self.INSTALLS[algorithm]}, a few GB. It can take several minutes.",
                "Install",
            ),
            answer,
        )

    @work(thread=True, group="install")
    def install_worker(
        self, algorithm: QuantAlgorithm, log: ProgressLog, done: Callable[[Any, str], None]
    ) -> None:
        from ocq.tui.install import install

        def say(line: str) -> None:
            self.call_from_thread(log.write, Text(line))

        try:
            state, problem = install(algorithm, self.hardware(), say), ""
        except Exception as exc:  # the app must stay up
            state, problem = None, str(exc) or type(exc).__name__
        self.call_from_thread(self.installed, state, problem, done)

    def installed(self, state, problem: str, done: Callable[[Any, str], None]) -> None:
        from ocq.tui.install import READY

        ok = state is not None and state.state == READY
        if ok:
            self.job_finished(f"{state.name} tools installed", True)
        else:
            problem = problem or (state.message if state else "The install did not finish.")
            self.job_finished("Install failed", False)
        done(state, problem)

    # Keys and the command palette.

    def action_keys(self) -> None:
        if not isinstance(self.screen, KeysScreen):
            self.push_screen(KeysScreen())

    def action_run(self) -> None:
        self.page.run()

    def action_menu(self) -> None:
        self.query_one(NavList).focus()

    async def action_quit(self) -> None:
        if not self.busy:
            self.exit()
            return

        def answer(yes: bool | None) -> None:
            if yes:
                self.exit()

        self.push_screen(
            Confirm("Quit?", "A quantization is still running. Quitting stops it.", "Quit"),
            answer,
        )

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        for item in MENU:
            if item:
                page_id, label = item
                yield SystemCommand(label, f"Open {label}", partial(self.show_page, page_id, True))


def run_tui() -> None:
    app = QuantPilotApp()
    app.run()
    if app.busy:
        # A quantization thread cannot be stopped from outside; leave without waiting for it.
        os._exit(0)
