"""Runs the ocq commands for the TUI.

The TUI does not have its own copy of feasibility, plan and the rest: it calls the same
functions the flags call (cli.cmd_feasibility and so on) and shows what they print. For that the
CLI's console is swapped, for the length of one command, for one that writes into the TUI.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.measure import Measurement
from rich.table import Table
from rich.text import Text

from ocq import cli
from ocq.types import OcqError

# One command at a time: they all print through the one cli.console.
_lock = threading.Lock()


class _Sink(io.StringIO):
    """Hands each piece of output to the TUI as soon as the command prints it."""

    def __init__(self, emit: Callable[[Text], None], ascii_only: bool) -> None:
        super().__init__()
        self._emit = emit
        self._ascii_only = ascii_only

    @property
    def encoding(self) -> str:  # Rich draws tables with +-| when the output is not UTF-8
        return "ascii" if self._ascii_only else "utf-8"

    def write(self, text: str) -> int:
        if text:
            self._emit(Text.from_ansi(text.rstrip("\n")))
        return len(text)


class _TuiConsole(Console):
    """Text and panels wrap to the TUI's output box; tables keep their full width, so no column
    is cut to "..." when the box is narrow (the box scrolls sideways instead)."""

    def __init__(self, box_width: int, **kwargs: Any) -> None:
        super().__init__(width=MAX_TABLE_WIDTH, **kwargs)
        self.box_width = box_width

    def print(self, *objects: Any, **kwargs: Any) -> None:
        if kwargs.get("width") is None:
            tables = [o for o in objects if isinstance(o, Table) and o.box is not None]
            width = self.box_width
            for table in tables:  # Table.grid() has no box: it wraps like text
                width = max(width, Measurement.get(self, self.options, table).maximum)
            kwargs["width"] = width
        super().print(*objects, **kwargs)


MAX_TABLE_WIDTH = 300


def busy() -> bool:
    return _lock.locked()


def run_command(
    command: Callable[..., Any],
    emit: Callable[[Text], None],
    width: int,
    ascii_only: bool = False,
    **options: Any,
) -> bool:
    """Calls `command` with the options the flags would give it. Everything it prints goes to
    `emit`, rendered `width` columns wide. True when it finished without an error."""
    if not _lock.acquire(blocking=False):
        emit(Text("Another command is still running.", style="yellow"))
        return False
    console = _TuiConsole(
        max(width, 40),
        file=_Sink(emit, ascii_only),
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
        highlight=False,
    )
    saved = cli.console
    cli.console = console
    try:
        command(**options)
        return True
    except typer.Exit as exc:
        return getattr(exc, "exit_code", 1) == 0
    except typer.BadParameter as exc:
        console.print(f"[red]Error:[/red] {escape(exc.format_message())}")
    except OcqError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
    except Exception as exc:  # the TUI must stay up, whatever a command does
        console.print(f"[red]Unexpected error ({type(exc).__name__}):[/red] {escape(str(exc))}")
    finally:
        cli.console = saved
        _lock.release()
    return False
