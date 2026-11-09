"""Bar charts drawn with text: gradient bars on a grid, sized to the space they get.

Each bar is made of block characters with eighth-of-a-cell steps (▏▎▍▌▋▊▉█), coloured
along a gradient, over a faint grid at the scale's ticks.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderResult
from rich.style import Style
from rich.text import Text

EIGHTHS = " ▏▎▍▌▋▊▉█"
GRID = "┊"
LIMIT = "┃"
# For consoles that cannot draw these (the app's simple borders): whole cells only.
ASCII = {"█": "#", GRID: ":", LIMIT: "|", "▸": ">", "★": "*", "…": "."}

# Gradients, low end first.
MEMORY_COLORS = ("#3ddc84", "#f6c445", "#ff5f56")  # fits easily -> tight -> too big
SPEED_COLORS = ("#22d3ee", "#3b82f6")
QUALITY_COLORS = ("#a78bfa", "#f472b6")
DIM = "#5c6370"
GRID_COLOR = "#3a3f4b"
LIMIT_COLOR = "#f6c445"


@dataclass(frozen=True)
class ChartRow:
    label: str
    memory_gb: float
    speed_tps: float  # 0: does not run here
    quality: float  # share of the model's quality kept, 0 to 1
    pick: bool = False
    runs: bool = True  # False: ruled out here (drawn grey)
    note: str = ""  # instead of the speed bar when it does not run, for example "too big"


def blend(colors: tuple[str, ...], at: float) -> Color:
    """The colour at `at` (0 to 1) along evenly spaced colours."""
    at = min(max(at, 0.0), 1.0)
    rgb = [Color.parse(color).get_truecolor() for color in colors]
    position = at * (len(rgb) - 1)
    index = min(int(position), len(rgb) - 2)
    part = position - index
    low, high = rgb[index], rgb[index + 1]
    return Color.from_rgb(*(round(a + (b - a) * part) for a, b in zip(low, high, strict=True)))


def nice_step(top: float, ticks: int = 4) -> float:
    """A round step (1, 2 or 5 x 10^n) that splits 0..top into about `ticks` parts."""
    if top <= 0:
        return 1.0
    raw = top / ticks
    power = 10 ** math.floor(math.log10(raw))
    return next(step * power for step in (1, 2, 5, 10) if step * power >= raw)


def number(value: float) -> str:
    return f"{value:.0f}" if value >= 10 or value == int(value) else f"{value:.1f}"


@dataclass(frozen=True)
class Panel:
    """One column of bars: its title, scale and colours."""

    title: str
    width: int
    top: float  # the value at the right edge
    step: float  # between grid lines
    colors: tuple[str, ...]
    unit: str = ""
    percent: bool = False

    def cells(self, value: float) -> float:
        return min(value / self.top, 1.0) * self.width if self.top else 0.0

    def grid_columns(self) -> set[int]:
        count = int(self.top / self.step) if self.step else 0
        return {round(i * self.step / self.top * self.width) for i in range(1, count + 1)}

    def scale(self) -> Text:
        """Tick labels along the top of the panel."""
        line = [" "] * self.width
        value = 0.0
        while value <= self.top + 1e-9:
            label = f"{value * 100:.0f}%" if self.percent else number(value)
            column = min(round(value / self.top * self.width), self.width - len(label))
            if all(c == " " for c in line[max(column - 1, 0) : column + len(label) + 1]):
                line[column : column + len(label)] = label
            value += self.step
        return Text("".join(line), style=DIM)


def bar(
    panel: Panel,
    value: float,
    color_at,
    grey: bool = False,
    limit: float | None = None,
) -> Text:
    """One gradient bar over the panel's grid. color_at(cell) gives the gradient position;
    limit draws a marker at that value (free memory)."""
    text = Text()
    filled = panel.cells(value)
    full = int(filled)
    grid = panel.grid_columns()
    limit_column = (
        min(int(panel.cells(limit)), panel.width - 1) if limit is not None and panel.top else None
    )
    for column in range(panel.width):
        if column == limit_column:  # over the bar too: where the free memory ends
            text.append(LIMIT, Style(color=LIMIT_COLOR, bold=True))
            continue
        color = Color.parse(DIM) if grey else color_at(column)
        if column < full:
            text.append("█", Style(color=color))
        elif column == full and filled - full > 0.06:
            text.append(EIGHTHS[max(1, round((filled - full) * 8))], Style(color=color))
        elif column in grid:
            text.append(GRID, Style(color=GRID_COLOR))
        else:
            text.append(" ")
    if value > panel.top * 1.001:
        text = text[:-1]
        text.append("▸", Style(color=Color.parse(DIM) if grey else color_at(panel.width - 1)))
    return text


class FormatChart:
    """Memory (against the free memory), speed and quality of each format, side by side."""

    GAP = 3

    def __init__(self, rows: list[ChartRow], free_gb: float, ascii_only: bool = False) -> None:
        self.rows = rows
        self.free_gb = free_gb
        self.ascii_only = ascii_only

    def panels(self, width: int) -> list[Panel]:
        label = self.label_width()
        room = max(width - label - self.GAP * 2, 30)
        show_quality = room >= 60
        shares = (0.42, 0.33, 0.25) if show_quality else (0.55, 0.45)
        widths = [
            max(int(room * share) - (self.GAP if i else 0), 8) for i, share in enumerate(shares)
        ]
        # Memory: up to the biggest bar, but no wider than twice the free memory (bigger ones
        # end in an arrow), so the ones that fit stay readable.
        biggest = max((row.memory_gb for row in self.rows), default=1.0)
        memory_top = max(min(biggest, self.free_gb * 2), self.free_gb) * 1.05 or 1.0
        memory_step = nice_step(memory_top)
        memory_top = math.ceil(memory_top / memory_step) * memory_step
        fastest = max((row.speed_tps for row in self.rows), default=1.0) or 1.0
        speed_step = nice_step(fastest)
        speed_top = math.ceil(fastest * 1.05 / speed_step) * speed_step
        panels = [
            Panel("Memory", widths[0], memory_top, memory_step, MEMORY_COLORS, " GB"),
            Panel("Speed", widths[1], speed_top, speed_step, SPEED_COLORS, " tok/s"),
        ]
        if show_quality:
            panels.append(Panel("Quality kept", widths[2], 1.0, 0.25, QUALITY_COLORS, percent=True))
        return panels

    def label_width(self) -> int:
        return max((len(row.label) for row in self.rows), default=6) + 3

    def lines(self, width: int):
        panels = self.panels(width)
        label = self.label_width()
        gap = " " * self.GAP

        title = Text(" " * label)
        scale = Text(" " * label)
        for i, panel in enumerate(panels):
            if i:
                title.append(gap)
                scale.append(gap)
            heading = panel.title + (f" ({panel.unit.strip()})" if panel.unit else "")
            title.append(heading.ljust(panel.width), Style(bold=True))
            scale.append_text(panel.scale())
        yield title
        yield scale

        memory = panels[0]
        for row in self.rows:
            line = Text()
            mark = "★ " if row.pick else "  "
            style = Style(bold=True) if row.pick else Style(color=DIM) if not row.runs else Style()
            line.append(mark, Style(color=LIMIT_COLOR, bold=True))
            line.append(row.label.ljust(label - 2), style)
            # Memory: green at the start, yellow near the free memory, red past it.
            free_cells = memory.cells(self.free_gb) or 1.0
            line.append_text(
                bar(
                    memory,
                    row.memory_gb,
                    lambda column, f=free_cells: blend(MEMORY_COLORS, column / f / 1.1),
                    not row.runs,
                    self.free_gb,
                )
            )
            for panel in panels[1:]:
                line.append(gap)
                if panel.title == "Speed" and not row.runs:
                    note = (
                        row.note
                        if len(row.note) <= panel.width
                        else row.note[: panel.width - 1] + "…"
                    )
                    line.append(note.ljust(panel.width), Style(color=DIM))
                    continue
                value = row.speed_tps if panel.title == "Speed" else row.quality
                width = panel.width or 1
                line.append_text(
                    bar(panel, value, lambda column, p=panel, w=width: blend(p.colors, column / w),
                        not row.runs)
                )  # fmt: skip
            yield line

        legend = Text(" " * label)
        legend.append(LIMIT, Style(color=LIMIT_COLOR, bold=True))
        legend.append(f" free memory {self.free_gb:.1f} GB", Style(color=DIM))
        yield legend

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for line in self.lines(options.max_width):
            yield plain_ascii(line) if self.ascii_only else line

    def __rich_measure__(self, console, options):
        from rich.measure import Measurement

        return Measurement(60, options.max_width)


def plain_ascii(line: Text) -> Text:
    """The line with ASCII in place of block and line characters; partial cells become whole."""
    out = Text()
    for index, char in enumerate(line.plain):
        style = line.get_style_at_offset(Console(), index)
        if char in EIGHTHS[1:-1]:
            char = "#"
        out.append(ASCII.get(char, char), style)
    return out


def lines_of(renderable, width: int) -> list[str]:
    """The renderable as plain lines at a width (for tests)."""
    console = Console(width=width, file=io.StringIO(), color_system=None)
    console.print(renderable)
    return console.file.getvalue().rstrip("\n").splitlines()
