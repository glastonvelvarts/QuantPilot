"""The full-screen app: banner, saved settings, running commands, and the pages themselves."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from typer.testing import CliRunner

from ocq import cli
from ocq.discovery import LocalModel
from ocq.tui import runner
from ocq.tui.app import FeasibilityPage, ModelPicker, PlanPage, QuantPilotApp, quote
from ocq.tui.banner import BANNER, BANNER_WIDTH, COMPACT, banner_text
from ocq.tui.settings import Settings, load_settings, save_settings
from ocq.types import OcqError

# Banner ------------------------------------------------------------------------------------


def test_banner_is_plain_ascii_so_every_console_can_show_it():
    for line in BANNER:
        assert all(32 <= ord(char) <= 126 for char in line), line


def test_banner_falls_back_to_one_line_when_the_terminal_is_small():
    assert banner_text(120, 40).splitlines()[0].rstrip() == BANNER[0].rstrip()
    assert banner_text(BANNER_WIDTH, 40) == COMPACT
    assert banner_text(120, 20) == COMPACT


def test_banner_lines_have_one_length_so_centering_keeps_the_letters_together():
    assert {len(line) for line in banner_text(120, 40).splitlines()} == {BANNER_WIDTH}


# Settings ----------------------------------------------------------------------------------


def test_settings_round_trip(tmp_path: Path):
    path = tmp_path / "ocq" / "tui.json"
    settings = Settings(theme="nord", context=8192, kv_cache="q8_0", model_folders=["~/models"])
    save_settings(settings, path)
    assert load_settings(path) == settings


def test_missing_or_broken_settings_give_the_defaults(tmp_path: Path):
    assert load_settings(tmp_path / "nothing.json") == Settings()
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_settings(broken) == Settings()


def test_a_wrong_value_is_dropped_without_resetting_the_others(tmp_path: Path):
    path = tmp_path / "tui.json"
    values = {"theme": "nord", "context": "big", "show_banner": 1, "kv_cache": "q9", "x": 1}
    path.write_text(json.dumps(values), encoding="utf-8")
    settings = load_settings(path)
    assert settings.theme == "nord"
    assert settings.context == Settings().context  # not a number
    assert settings.show_banner is True  # 1 is not true/false
    assert settings.kv_cache == "f16"  # not a cache type


def test_recent_models_are_newest_first_without_repeats():
    settings = Settings()
    for ref in ("a/one", "b/two", "a/one"):
        settings.remember_model(ref)
    assert settings.recent_models == ["a/one", "b/two"]


# Runner ------------------------------------------------------------------------------------


def test_runner_shows_what_a_command_prints_and_puts_the_console_back():
    saved = cli.console
    out = []

    def command(name: str) -> None:
        cli.console.print(f"hello {name}")

    assert runner.run_command(command, out.append, 80, name="tui")
    assert [text.plain for text in out] == ["hello tui"]
    assert cli.console is saved


def test_runner_turns_errors_into_a_message_and_false():
    out = []

    def broken() -> None:
        raise OcqError("no such model")

    assert not runner.run_command(broken, out.append, 80)
    assert out[-1].plain == "Error: no such model"
    assert not runner.busy()


def test_runner_tables_keep_their_width_and_can_be_plain_ascii():
    from rich.table import Table

    def wide_table() -> None:
        table = Table("Level", "Backend", "Weights")
        table.add_row("Q4_K_M", "gguf " + "x" * 80, "0.10")
        cli.console.print(table)
        cli.console.print("word " * 40)

    out = []
    runner.run_command(wide_table, out.append, 60, ascii_only=True)
    table, text = (t.plain.splitlines() for t in out)
    assert max(map(len, table)) > 60  # not squeezed into the box
    assert max(map(len, text)) <= 60  # text still wraps
    assert table[0].startswith("+") and all(ord(c) < 128 for c in "".join(table))


# The app -----------------------------------------------------------------------------------


def run_app(tmp_path: Path, steps, size=(130, 42)) -> QuantPilotApp:
    async def go() -> QuantPilotApp:
        app = QuantPilotApp(settings_path=tmp_path / "tui.json", look_on_start=False)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await steps(app, pilot)
        return app

    return asyncio.run(go())


def test_arrow_keys_in_the_menu_open_the_pages(tmp_path: Path):
    seen = []

    async def steps(app, pilot):
        seen.append(app.query_one("ContentSwitcher").current)
        await pilot.press("down")
        seen.append(app.query_one("ContentSwitcher").current)
        await pilot.press("down", "down")
        seen.append(app.query_one("ContentSwitcher").current)

    run_app(tmp_path, steps)
    assert seen == ["home", "pc", "recommend"]


def test_feasibility_form_gives_the_same_options_as_the_flags(tmp_path: Path):
    built = {}

    async def steps(app, pilot):
        app.show_page("feasibility", focus=True)
        page = app.query_one(FeasibilityPage)
        await pilot.press(*"org/model")
        page.query_one("#context").value = 8192
        page.query_one("#kv").value = "q8_0"
        await pilot.pause()
        built["result"] = page.build()

    run_app(tmp_path, steps)
    function, options, args = built["result"]
    assert function is cli.cmd_feasibility
    assert options == {
        "model": "org/model",
        "model_path": None,
        "llmfit_bin": None,
        "context": 8192,
        "kv_cache": "q8_0",
    }
    assert args == ["feasibility", "org/model", "-c", "8192", "--kv-cache", "q8_0"]


def test_plan_form_leaves_auto_choices_to_quantpilot(tmp_path: Path):
    built = {}

    async def steps(app, pilot):
        app.show_page("plan")
        page = app.query_one(PlanPage)
        page.set_model("org/model")
        page.query_one("#backend").value = "gguf"
        page.query_one("#double").value = "off"
        page.query_one("#samples").value = "64"
        await pilot.pause()
        built["result"] = page.build()

    run_app(tmp_path, steps)
    function, options, args = built["result"]
    assert function is cli.cmd_plan
    assert options["backend"] == "gguf"
    assert options["bits"] is None and options["gguf_type"] is None
    assert options["double_quant"] is False and options["samples"] == 64
    assert args == ["plan", "org/model", "-b", "gguf", "--samples", "64", "--single-quant", "-c",
                    "4096"]  # fmt: skip


def test_a_wrong_number_is_reported_not_run(tmp_path: Path):
    async def steps(app, pilot):
        app.show_page("plan")
        page = app.query_one(PlanPage)
        page.set_model("org/model")
        page.query_one("#samples").value = "lots"
        page.run()
        await pilot.pause()
        assert not page.has_run

    run_app(tmp_path, steps)


def test_running_a_page_shows_the_command_line_and_the_output(tmp_path: Path, monkeypatch):
    def fake_feasibility(**options) -> None:
        cli.console.print(f"fits: {options['model']}")
        print("a plain progress line")

    monkeypatch.setattr(cli, "cmd_feasibility", fake_feasibility)
    lines = []

    async def steps(app, pilot):
        app.show_page("feasibility", focus=True)
        await pilot.press(*"org/model", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        lines.extend(line.text for line in app.query_one(FeasibilityPage).output.lines)

    app = run_app(tmp_path, steps)
    assert lines[0] == "Same as: ocq feasibility org/model -c 4096"
    assert "fits: org/model" in lines and "a plain progress line" in lines
    assert app.settings.recent_models == ["org/model"]
    assert json.loads((tmp_path / "tui.json").read_text())["recent_models"] == ["org/model"]


def test_picker_lists_models_on_this_pc_and_fills_the_field(tmp_path: Path):
    model = LocalModel(
        label="qwen3:0.6b",
        ref="ollama:qwen3:0.6b",
        source="Ollama",
        path=tmp_path,
        format="GGUF",
        quantization="Q4_K_M",
        size_bytes=500_000_000,
        complete=True,
    )
    picked = {}

    async def steps(app, pilot):
        app.known_models = [model]
        app.show_page("verify", focus=True)
        page = app.page
        await pilot.pause()  # let the page lay out before clicking on it
        await pilot.click(page.query_one(".pick"))  # every page has one: this page's
        await pilot.pause()
        assert isinstance(app.screen, ModelPicker)
        await pilot.press("down", "enter")
        await pilot.pause()
        picked["value"] = page.query_one(".model-input").value

    run_app(tmp_path, steps)
    assert picked["value"] == "ollama:qwen3:0.6b"


def test_settings_page_saves_and_applies_at_once(tmp_path: Path):
    async def steps(app, pilot):
        app.show_page("settings")
        app.query_one("#theme").value = "nord"
        app.query_one("#ascii_borders").value = True
        await pilot.pause()
        assert app.theme == "nord"
        assert app.has_class("-ascii")

    run_app(tmp_path, steps)
    saved = load_settings(tmp_path / "tui.json")
    assert saved.theme == "nord" and saved.ascii_borders


def test_small_terminal_gets_the_one_line_banner(tmp_path: Path):
    async def steps(app, pilot):
        assert str(app.query_one("#banner").render()) == COMPACT

    run_app(tmp_path, steps, size=(80, 24))


def test_quote_only_quotes_when_a_shell_needs_it():
    assert quote("org/model") == "org/model"
    assert quote("C:\\My Models\\a.gguf") == '"C:\\My Models\\a.gguf"'


def test_ocq_without_a_terminal_prints_the_help():
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 0
    assert "ocq tui" in result.output or "tui" in result.output
