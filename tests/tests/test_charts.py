"""The TUI's bar charts and progress pieces, rendered as text."""

from rich.color import Color

from ocq.tui.charts import ChartRow, FormatChart, blend, lines_of, nice_step
from ocq.tui.widgets import STEPS, StepTracker, meter_progress

ROWS = [
    ChartRow("Q4_K_M", 6.5, 12.4, 0.854, pick=True),
    ChartRow("Q2_K", 4.6, 18.3, 0.079),
    ChartRow("BF16", 19.5, 0, 1.0, runs=False, note="too big"),
    ChartRow("AWQ 4-bit", 9.0, 0, 0.787, runs=False, note="vLLM needs an NVIDIA GPU"),
]


def blocks(line: str) -> float:
    """Cells a bar fills: whole blocks, plus eighths."""
    eighths = " ▏▎▍▌▋▊▉█"
    return sum(1 for c in line if c == "█") + sum(
        eighths.index(c) / 8 for c in line if c in eighths[1:-1]
    )


def test_scale_steps_are_round_numbers():
    assert [nice_step(top) for top in (7.5, 20, 1, 160)] == [2, 5, 0.5, 50]


def test_gradient_runs_from_the_first_colour_to_the_last():
    colors = ("#000000", "#ffffff")
    assert blend(colors, 0).get_truecolor() == Color.parse("#000000").get_truecolor()
    assert blend(colors, 1).get_truecolor() == Color.parse("#ffffff").get_truecolor()
    assert blend(colors, 0.5).get_truecolor().red == 128


def test_every_format_gets_a_row_with_the_pick_starred_and_free_memory_marked():
    lines = lines_of(FormatChart(ROWS, 7.0), 130)
    title, scale, *rows, legend = lines
    assert "Memory (GB)" in title and "Speed (tok/s)" in title and "Quality kept" in title
    assert scale.split()[0] == "0"
    assert [row.split()[0] for row in rows] == ["★", "Q2_K", "BF16", "AWQ"]
    assert all("┃" in row for row in rows)  # where the free memory ends, on every row
    assert legend.strip() == "┃ free memory 7.0 GB"
    # Formats that do not run say why where their speed would be, and have no speed bar.
    assert "too big" in rows[2] and "vLLM needs an NVIDIA GPU" in rows[3]
    # Bigger memory, longer bar; one past the scale ends in an arrow.
    memory = [row[11:60] for row in rows]
    assert blocks(memory[1]) < blocks(memory[0])
    assert "▸" in rows[2]


def test_a_narrow_screen_drops_the_quality_column():
    lines = lines_of(FormatChart(ROWS, 7.0), 70)
    assert "Quality kept" not in lines[0] and "Speed (tok/s)" in lines[0]
    assert max(len(line) for line in lines) <= 70


def test_simple_borders_draw_the_chart_in_ascii():
    text = "\n".join(lines_of(FormatChart(ROWS, 7.0, ascii_only=True), 120))
    assert all(ord(c) < 128 for c in text)
    assert "#" in text and "|" in text and "* Q4_K_M" in text


def test_meters_give_a_percentage_and_what_is_done():
    assert meter_progress("Fetching 14 files:  50%|█████     | 7/14 [00:01<00:01]") == (
        50.0,
        "7 of 14",
    )
    assert meter_progress("Downloading:  45%|████▌     | 812M/1.80G [00:12<00:15]") == (
        45.0,
        "812MB of 1.80GB",
    )
    percent, done = meter_progress("[  12/ 335]  token_embd.weight - converting to q8_0 ..")
    assert round(percent, 1) == 3.6 and done == "12 of 335"
    assert meter_progress("Download complete: :  |  0.00B") == (None, "")


def test_the_step_tracker_follows_the_pipelines_lines():
    steps = StepTracker()
    assert not steps.heard("Fetching 14 files")
    assert steps.heard("Downloading org/model ...") and steps.current == 0
    assert steps.heard("Converting to GGUF (bf16, ...) ...") and steps.current == 1
    assert not steps.heard("Downloading org/model ...")  # never back
    assert steps.heard("Measuring the speed ...") and steps.current == 3
    assert len(steps.times) == 3  # Quantize, not printed by this backend, counts as done
    text = steps.render_steps().plain
    assert "✓ Download" in text and "✓ Quantize" in text and "Measure speed" in text
    assert "○ Check" in text
    steps.failed = True
    assert "✗ Measure speed" in steps.render_steps().plain
    assert [name for name, _ in STEPS][-1] == "Check"


def test_tool_progress_is_shown_as_one_redrawn_line_and_the_output_kept(tmp_path, capsys):
    import sys

    from ocq.quantizers.gguf import run_tool

    script = (
        "import sys\n"
        "for i in (25, 50, 100):\n"
        "    sys.stderr.write(f'\\rWriting: {i:3d}%|####| {i}M/100M')\n"
        "    sys.stderr.flush()\n"
        "print('[  1/  2] blk.0.attn_q.weight')\n"
        "print('done')\n"
    )
    out, err = run_tool([sys.executable, "-c", script], tmp_path / "tool.log", progress=True)
    assert out == "[  1/  2] blk.0.attn_q.weight\ndone\n"
    assert err.endswith("Writing: 100%|####| 100M/100M")
    shown = capsys.readouterr().out
    assert "\rWriting: " in shown and "%|####| " in shown  # the newest state of each read
    assert "\rWriting: 100%|####| 100M/100M" in shown  # the last one too, with no line end
    assert "\r[  1/  2] blk.0.attn_q.weight" in shown
    assert shown.endswith("\r\x1b[2K")  # the meter line is cleared afterwards
    assert "done" not in shown  # only progress lines are shown
    assert (tmp_path / "tool.log").read_text().startswith(out)


def test_a_message_glued_to_a_meter_still_moves_the_steps_on():
    from types import SimpleNamespace

    from ocq.tui.widgets import ProgressLog

    class Log(ProgressLog):
        def write(self, text, *args, **kwargs):
            written.append(text.plain)

    written, meters, lines = [], [], []
    log = Log()
    log.on_line, log.on_meter = lines.append, meters.append
    steps = StepTracker()
    # What the real run sent: the Hub's meter ends without a line end, the next message follows.
    for chunk in (
        "Downloading org/model ...\n",
        "\rDownload complete: :  |  0.00B",
        "Converting to GGUF (bf16, the precision the model is stored in) ...\n",
        "\rWriting:  46%|████▌     | 124M/269M",
    ):
        log.on_print(SimpleNamespace(text=chunk))
        for line in lines:
            steps.heard(line)
    assert written == [
        "Downloading org/model ...",
        "Converting to GGUF (bf16, the precision the model is stored in) ...",
    ]
    assert steps.current == 1  # Prepare: the convert message was heard
    assert meter_progress(meters[-1]) == (46.0, "124MB of 269MB")
