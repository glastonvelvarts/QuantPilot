"""The full-screen app: banner, words, saved settings, and the pages with stand-in data."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from ocq import cli, model_analyzer, pipeline, plan
from ocq.discovery import LocalModel
from ocq.tui import pages, words
from ocq.tui.app import MENU, QuantPilotApp
from ocq.tui.banner import BANNER, BANNER_WIDTH, COMPACT, banner_text
from ocq.tui.charts import lines_of
from ocq.tui.pages import AUTO, CheckPage, QuantizePage, format_choices, format_key
from ocq.tui.settings import Settings, load_settings, save_settings
from ocq.tui.widgets import ModelPicker, ProgressLog
from ocq.types import (
    Fact,
    FactKind,
    FitLevel,
    HardwareProfile,
    ModelFacts,
    QuantAlgorithm,
    QuantConfig,
    QuantOption,
    RunMode,
)

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


# Words -------------------------------------------------------------------------------------


def option(label="Q4_K_M", kind="gguf", fit=FitLevel.PERFECT, tps=120.0, gb=0.6, bits=4.9):
    return QuantOption(
        label=label,
        bits=bits,
        weights_gb=gb - 0.1,
        context_gb=0.1,
        exact=True,
        memory_required_gb=gb,
        memory_available_gb=8.0,
        fit_level=fit,
        run_mode=RunMode.GPU,
        estimated_tps=tps,
        row_kind=kind,
    )


def test_words_for_fit_speed_and_size():
    assert words.fit(FitLevel.MARGINAL).plain == "Tight"
    assert words.speed(option(tps=120)) == "up to 120 tok/s"
    assert words.speed(option(fit=FitLevel.TOO_TIGHT)) == "-"
    assert words.gb(0.25) == "250 MB" and words.gb(4.26) == "4.3 GB"
    assert words.context(8192) == "8K tokens"


def test_a_split_model_says_how_many_layers_are_on_the_gpu():
    split = replace(option(), run_mode=RunMode.GPU_OFFLOAD, gpu_layers=14)
    assert words.place(split) == "GPU + RAM"
    assert words.layers(split, 28) == "14 of 29 layers on GPU"
    assert words.layers(option(), 28) == "-"


def test_warnings_lose_their_command_lines():
    assert words.warning("GGUF backend not installed. Run: uv sync --extra gguf") == (
        "GGUF tools are not installed on this PC."
    )
    assert words.warning("serve it in WSL2, or quantize to GGUF (-b gguf) to run it here.") == (
        "serve it in WSL2, or quantize to GGUF to run it here."
    )


def test_fit_table_rows_map_to_the_format_dropdown():
    assert format_key(option("Q4_K_M")) == "gguf:Q4_K_M"
    assert format_key(option("GPTQ 8-bit", kind="gptq")) == "gptq:8"
    assert format_key(option("BNB 4-bit double", kind="bnb")) == "bnb:4"
    assert format_key(option("BF16", kind="stored")) is None
    choices = format_choices("gguf:Q5_K_M", 8192, "q8_0")
    assert (choices.algorithm, choices.gguf_type, choices.context) == (
        QuantAlgorithm.GGUF,
        "Q5_K_M",
        8192,
    )
    assert format_choices(AUTO, 4096, "f16").algorithm is None


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
    assert settings.context == Settings().context
    assert settings.show_banner is True
    assert settings.kv_cache == "f16"


def test_recent_models_are_newest_first_without_repeats():
    settings = Settings()
    for ref in ("a/one", "b/two", "a/one"):
        settings.remember_model(ref)
    assert settings.recent_models == ["a/one", "b/two"]


# The app -----------------------------------------------------------------------------------

HARDWARE = HardwareProfile(
    total_ram_gb=16.0,
    available_ram_gb=8.0,
    cpu_cores=8,
    cpu_name="Test CPU",
    has_gpu=False,
    backend="CPU",
    unified_memory=False,
    os_name="Linux",
    ram_bandwidth_gbs=50.0,
)


def run_app(tmp_path: Path, steps, size=(130, 42)) -> QuantPilotApp:
    async def go() -> QuantPilotApp:
        app = QuantPilotApp(settings_path=tmp_path / "tui.json", look_on_start=False)
        app.hardware = lambda fresh=False: HARDWARE  # no probing in tests
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await steps(app, pilot)
        return app

    return asyncio.run(go())


async def settle(app, pilot) -> None:
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def offline_engine(monkeypatch, catalog=()):
    """The recommendation engine without the network: no models on this PC, Hub repos read
    from the saved headers in data/, the popular list replaced by `catalog`."""
    from conftest import real_model

    from ocq import discovery, recommend

    monkeypatch.setattr(discovery, "find_models", lambda *a, **k: [])
    monkeypatch.setattr(recommend, "CATALOG", tuple(recommend.entry(m) for m in catalog))
    monkeypatch.setattr(model_analyzer, "fetch_hf_model", lambda repo: real_model(repo)[0])
    monkeypatch.setattr(model_analyzer, "fetch_model_files", lambda r, sha: real_model(r)[1:])
    monkeypatch.setattr(model_analyzer, "catalog_entry", lambda *a: None)
    monkeypatch.setattr(recommend, "free_disk_gb", lambda path: 1000.0)
    monkeypatch.setattr(
        recommend, "converter_architectures", lambda: frozenset({"Qwen2ForCausalLM"})
    )


def fake_model(monkeypatch, warnings=()):
    """resolve_model, quant_options and generate_plan for a model that is not quantized."""
    options = [
        option("BF16", kind="stored", bits=16, gb=1.2, tps=60),
        option("Q8_0", bits=8.5, gb=0.7, tps=100),
        option("Q4_K_M", bits=4.9, gb=0.5, tps=150),
    ]
    facts = ModelFacts(parameters=Fact(1_000_000_000, "config", FactKind.EXACT))
    snapshot = SimpleNamespace(
        model_id="org/model",
        is_quantized=False,
        gguf_files=(),
        facts=facts,
        original_gb=2.0,
        num_layers=24,
    )
    config = QuantConfig(algorithm=QuantAlgorithm.GGUF, bits=8, group_size=0, gguf_type="Q8_0")
    made = SimpleNamespace(
        model_id="org/model",
        predicted=options[1],
        warnings=list(warnings),
        quant_config=config,
        output_dir=Path("/out/org--model"),
    )
    monkeypatch.setattr(model_analyzer, "resolve_model", lambda *a, **k: snapshot)
    monkeypatch.setattr(model_analyzer, "quant_options", lambda *a, **k: options)
    monkeypatch.setattr(plan, "generate_plan", lambda *a, **k: made)
    offline_engine(monkeypatch)
    return made


def text_of(app, selector: str) -> str:
    return str(app.query_one(selector).render())


def test_menu_arrow_keys_open_the_pages_in_order(tmp_path: Path, monkeypatch):
    offline_engine(monkeypatch)  # Discover opens on the way
    seen = []

    async def steps(app, pilot):
        for _ in range(4):
            seen.append(app.query_one("ContentSwitcher").current)
            await pilot.press("down")

    run_app(tmp_path, steps)
    order = [item[0] for item in MENU if item]
    assert seen == order[:4]


def test_check_shows_the_engines_pick_in_plain_words(tmp_path: Path, monkeypatch):
    from conftest import QWEN

    from ocq import recommend

    offline_engine(monkeypatch)
    made = SimpleNamespace(warnings=["GGUF backend not installed. Run: uv sync --extra gguf"])
    monkeypatch.setattr(plan, "generate_plan", lambda *a, **k: made)
    request = recommend.RecommendationRequest(recommend.WORKLOADS["coding"], (QWEN,))
    expected = recommend.recommend(request, HARDWARE)  # what ocq recommend shows
    seen = {}

    async def steps(app, pilot):
        app.show_page("check", focus=True)
        app.query_one("#check #workload").value = "coding"
        await pilot.press(*QWEN, "enter")
        await settle(app, pilot)
        seen["card"] = text_of(app, "#pick-text")
        seen["why"] = text_of(app, "#why-text")
        seen["chart"] = "\n".join(lines_of(app.query_one(CheckPage).chart, 130))
        seen["rows"] = app.query_one("#options").row_count
        await pilot.click(app.query_one("#quantize-pick"))
        await settle(app, pilot)
        seen["page"] = app.query_one("ContentSwitcher").current
        seen["format"] = app.query_one("#quantize #format").value
        seen["context"] = app.query_one("#quantize #context").value
        seen["saved"] = app.settings.workload

    run_app(tmp_path, steps)
    best = expected.best.candidate
    card = seen["card"].splitlines()
    assert card[:2] == ["Recommended for Coding", f"{best.option.label} · llama.cpp"]
    assert words.speed(best.option) in seen["card"] and "32K tokens" in seen["card"]
    assert f"Confidence: {expected.best.confidence.level}" in seen["card"]
    assert "GGUF tools are not installed on this PC." in seen["card"]
    assert "uv sync" not in seen["card"]
    assert expected.best.reasons[0].text in seen["why"] and "✓" in seen["why"]
    assert seen["rows"] == len(recommend.compare_rows(expected))
    assert "Memory (GB)" in seen["chart"] and f"★ {best.option.label}" in seen["chart"]
    assert "free memory" in seen["chart"]
    assert (seen["page"], seen["format"]) == ("quantize", f"gguf:{best.option.label}")
    assert seen["context"] == 32768 and seen["saved"] == "coding"


def test_a_gguf_file_is_recommended_as_it_is_not_quantized_again(tmp_path: Path, monkeypatch):
    from test_discovery import smollm2_store

    offline_engine(monkeypatch)
    smollm2_store(tmp_path / "ollama")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama"))
    seen = {}

    async def steps(app, pilot):
        app.show_page("check", focus=True)
        app.query_one("#check #context").value = 2048  # what the small test file holds
        await pilot.press(*"ollama:smollm2:135m", "enter")
        await settle(app, pilot)
        seen["card"] = text_of(app, "#pick-text")
        seen["quantize"] = app.query_one("#quantize-pick").display

    run_app(tmp_path, steps)
    assert seen["card"].splitlines()[1] == "Q8_0 · llama.cpp"
    assert not seen["quantize"]


def test_discover_lists_each_models_best_option(tmp_path: Path, monkeypatch):
    from conftest import QWEN, QWEN_AWQ

    offline_engine(monkeypatch, catalog=(QWEN, QWEN_AWQ))
    seen = {}

    async def steps(app, pilot):
        app.show_page("discover", focus=True)
        await settle(app, pilot)
        table = app.query_one("#discover DataTable")
        seen["rows"] = [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]

    run_app(tmp_path, steps)
    ranked, not_here = seen["rows"]  # every model, also the ones that cannot run here
    assert ranked[0] == f"{QWEN}  recommended" and ranked[-2:] == ["Download", ""]
    # The stored AWQ repo needs vLLM, which needs an NVIDIA GPU.
    assert (not_here[0], not_here[1], not_here[2]) == (QWEN_AWQ, "-", "Not here")
    assert not_here[-1].startswith("vLLM needs an NVIDIA GPU")  # and why


def test_quantize_recommended_makes_the_engines_pick(tmp_path: Path, monkeypatch):
    from conftest import QWEN

    from ocq import recommend

    offline_engine(monkeypatch)
    made = []
    real_plan = plan.generate_plan
    monkeypatch.setattr(
        plan, "generate_plan", lambda *a, **k: made.append(a[5]) or real_plan(*a, **k)
    )
    request = recommend.RecommendationRequest(recommend.WORKLOADS["chat"], (QWEN,), 8192)
    best = recommend.recommend(request, HARDWARE).best.candidate
    seen = {}

    async def steps(app, pilot):
        app.show_page("quantize", focus=True)
        app.query_one("#quantize #context").value = 8192
        await pilot.press(*QWEN, "enter")  # the format stays on Recommended
        await settle(app, pilot)
        seen["plan"] = text_of(app, "#plan-text")
        seen["problem"] = app.query_one("#quantize .problem").display

    run_app(tmp_path, steps)
    assert not seen["problem"]
    (choices,) = made
    assert (choices.gguf_type, choices.kv_cache) == (best.option.label, best.kv_cache)
    assert f"GGUF {best.option.label}" in seen["plan"]


def test_quantize_shows_the_plan_then_the_result(tmp_path: Path, monkeypatch):
    made = fake_model(monkeypatch)

    def fake_run(ref, goal, output, **kwargs):
        print("Converting to GGUF ...")
        print("Quantizing to Q8_0 ...")
        quantization = SimpleNamespace(
            success=True, message="", artifact_size_gb=0.7, artifact_path=Path("/out/m.gguf")
        )
        validation = SimpleNamespace(passed=True, checks=[])
        bench = SimpleNamespace(tokens_per_second=95.0, sample_output="Hello there.")
        return SimpleNamespace(
            plan=made, quantization=quantization, validation=validation, benchmark=bench
        )

    monkeypatch.setattr(pipeline, "run_optimize", fake_run)
    seen = {}

    async def steps(app, pilot):
        app.show_page("quantize", focus=True)
        app.query_one(QuantizePage).set_format("gguf:Q8_0")
        await pilot.press(*"org/model", "enter")
        await settle(app, pilot)
        seen["plan"] = text_of(app, "#plan-text")
        page = app.query_one(QuantizePage)
        page.begin(*page.plan_for)
        await settle(app, pilot)
        seen["progress"] = [line.text for line in app.query_one("#progress", ProgressLog).lines]
        seen["result"] = text_of(app, "#result")
        seen["steps"] = text_of(app, "#steps")
        seen["bar shown"] = app.query_one("#step-bar").display
        seen["busy"] = app.busy

    run_app(tmp_path, steps)
    assert "GGUF Q8_0 (8-bit)" in seen["plan"] and "Saves to" in seen["plan"]
    assert seen["progress"] == ["Converting to GGUF ...", "Quantizing to Q8_0 ..."]
    assert seen["result"].startswith("Done") and "95 tok/s measured" in seen["result"]
    # The steps stay above the result, all ticked; the bar and the log are put away.
    assert seen["steps"].count("✓") == 5 and "✓ Quantize" in seen["steps"]
    assert not seen["bar shown"]
    assert not seen["busy"]


def test_no_page_shows_flags_commands_or_llmfit(tmp_path: Path, monkeypatch):
    fake_model(monkeypatch)
    monkeypatch.setattr(pages, "local_option", lambda *a: None)
    shown = []

    async def steps(app, pilot):
        for item in MENU:
            if item:
                app.show_page(item[0])
                await settle(app, pilot)
        app.query_one(CheckPage).set_model("org/model")
        app.query_one(CheckPage).run()
        await settle(app, pilot)
        for widget in app.query("Static, Label, Button"):
            shown.append(str(widget.render()))

    run_app(tmp_path, steps)
    text = "\n".join(shown)
    for word in (" --", "ocq ", "llmfit", "Same as", "uv sync"):
        assert word not in text, word


def test_browse_lists_models_on_this_pc_and_fills_the_field(tmp_path: Path):
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
        await pilot.pause()
        await pilot.click(page.query_one(".pick"))  # every page has one: this page's
        await pilot.pause()
        assert isinstance(app.screen, ModelPicker)
        await pilot.press("down", "enter")
        await pilot.pause()
        picked["value"] = page.query_one(".model-input").value

    run_app(tmp_path, steps)
    assert picked["value"] == "ollama:qwen3:0.6b"


def test_settings_save_and_apply_at_once(tmp_path: Path):
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
        assert text_of(app, "#home-banner") == COMPACT

    run_app(tmp_path, steps, size=(80, 24))


def test_home_shows_the_big_banner_when_there_is_room(tmp_path: Path):
    async def steps(app, pilot):
        assert text_of(app, "#home-banner").splitlines()[0].rstrip() == BANNER[0].rstrip()
        assert not app.query_one("#masthead").display  # no second name at the top

    run_app(tmp_path, steps, size=(120, 40))


def test_ocq_without_a_terminal_prints_the_help():
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 0
    assert "tui" in result.output


def test_a_failing_job_shows_a_message_and_the_app_stays_up(tmp_path: Path, monkeypatch):
    def offline(*args, **kwargs):
        raise ConnectionError("network is unreachable")

    offline_engine(monkeypatch)
    monkeypatch.setattr(model_analyzer, "fetch_hf_model", offline)
    seen = {}

    async def steps(app, pilot):
        app.show_page("check", focus=True)
        await pilot.press(*"org/model", "enter")
        await settle(app, pilot)
        seen["problem"] = text_of(app, "#check .problem")
        seen["running"] = app.is_running

    run_app(tmp_path, steps)
    assert seen["problem"] == "Could not read it: network is unreachable"
    assert seen["running"]


def test_verify_runs_llama_server_and_shows_layers_as_a_count(tmp_path: Path, monkeypatch):
    from ocq import local_model, verify
    from ocq.quantizers import llama_cpp
    from ocq.verify import Check, LlamaLog

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    snapshot = SimpleNamespace(gguf_files=("m.gguf",))
    log = LlamaLog(4096, 4, (14, 29), 2172.17, 2288.28, 448.0, 0.0, "K f16, V f16", gpus=1)
    checks = [
        Check("layers on the GPU", 14, 14, unit="layers"),
        Check("weights on the CPU (file range)", 2288.28, 2288.28),
        Check("working memory (upper bound)", 463.3, 426.0, upper_bound=True),
    ]
    tools = []
    monkeypatch.setattr(model_analyzer, "resolve_model", lambda *a, **k: snapshot)
    monkeypatch.setattr(model_analyzer, "quant_options", lambda *a, **k: [option()])
    monkeypatch.setattr(local_model, "parse_model_ref", lambda ref: (ref, gguf))
    monkeypatch.setattr(llama_cpp, "tool_path", lambda name, hw: tools.append(name) or Path(name))
    monkeypatch.setattr(verify, "verify", lambda *a, **k: (log, checks, "log text"))
    seen = {}

    async def steps(app, pilot):
        app.show_page("verify", focus=True)
        await pilot.press(*"m.gguf", "enter")
        await settle(app, pilot)
        seen["verdict"] = text_of(app, "#verify #verdict")
        table = app.query_one("#verify #checks")
        seen["rows"] = [[str(cell) for cell in table.get_row(key)] for key in table.rows]

    run_app(tmp_path, steps)
    assert tools == ["llama-server"]
    assert seen["verdict"] == "Prediction matches llama.cpp\n14 of 29 layers on the GPU"
    assert seen["rows"] == [
        ["Layers on the GPU", "14", "14", "match"],
        ["Weights in RAM", "2,288 MB", "2,288 MB", "match"],
        ["Working memory", "463 MB", "426 MB", "within"],
    ]


def test_keyboard_only_moves_into_pages_between_fields_and_back(tmp_path: Path, monkeypatch):
    from conftest import QWEN

    fake_model(monkeypatch)
    seen = []

    def where(app) -> str:
        focused = app.focused
        return focused.id or type(focused).__name__

    async def steps(app, pilot):
        await pilot.press("down")  # menu: Home -> Check a model (lines are skipped)
        await pilot.press("right")  # into the page: the model field
        seen.append(where(app))
        await pilot.press("left")  # empty field: back to the menu
        seen.append(where(app))
        await pilot.press("right", *QWEN)
        await pilot.press("left")  # moves the cursor inside the text, stays
        seen.append(where(app))
        await pilot.press("home", "left")  # at the start: back to the menu
        seen.append(where(app))
        await pilot.press("right", "down", "down")  # model -> Browse -> Workload
        seen.append(where(app))
        await pilot.press("down")  # a closed dropdown does not open on Down: -> Context
        seen.append(where(app))
        await pilot.press("down")
        seen.append(where(app))
        await pilot.press("enter")  # Check
        await settle(app, pilot)
        await pilot.press("down", "down", "down")  # -> Quantize to this -> Why -> the table
        seen.append(where(app))
        await pilot.press("up")  # first row: leaves the table
        seen.append(where(app))
        await pilot.press("left")
        seen.append(where(app))

    run_app(tmp_path, steps)
    assert seen == [
        "TextField",
        "nav",
        "TextField",
        "nav",
        "workload",
        "context",
        "check",
        "options",
        "CollapsibleTitle",
        "nav",
    ]


def test_core_messages_lose_cli_advice():
    assert words.plain("Ollama has no model x:1. ocq models lists the ones on this PC.") == (
        "Ollama has no model x:1."
    )
    assert words.plain("llama.cpp needs the OpenMP runtime. Run: sudo apt install libgomp1") == (
        "llama.cpp needs the OpenMP runtime."
    )


def test_tool_state_says_ready_missing_or_not_for_this_pc(monkeypatch):
    from ocq.quantizers import gguf
    from ocq.tui import install

    monkeypatch.setattr(gguf.GGUFQuantizer, "is_available", classmethod(lambda cls: False))
    monkeypatch.setattr(gguf, "has_openmp_runtime", lambda: True)
    assert install.tool_state(QuantAlgorithm.GGUF, HARDWARE).state == install.MISSING
    assert install.tool_state(QuantAlgorithm.BNB, HARDWARE).message == "Needs an NVIDIA GPU"
    monkeypatch.setattr(gguf.GGUFQuantizer, "is_available", classmethod(lambda cls: True))
    assert install.tool_state(QuantAlgorithm.GGUF, HARDWARE).state == install.READY


def test_install_reads_the_extra_from_quantpilots_metadata():
    from ocq.tui import install

    assert any(req.startswith("torch") for req in install.requirements("gguf"))
    command, folder = install.install_command("gguf")
    assert "gguf" in " ".join(command)


def test_this_pc_installs_a_missing_tool(tmp_path: Path, monkeypatch):
    from ocq.tui import install

    states = {"gguf": install.MISSING}

    def fake_state(algorithm, hardware):
        state = states.get(algorithm.value, install.UNSUPPORTED)
        message = {install.MISSING: "Not installed", install.READY: "Ready"}.get(state, "-")
        return install.ToolState(algorithm, state, message)

    def fake_install(algorithm, hardware, say):
        say("Collecting torch")
        states[algorithm.value] = install.READY
        return fake_state(algorithm, hardware)

    monkeypatch.setattr(install, "tool_state", fake_state)
    monkeypatch.setattr(install, "install", fake_install)
    seen = {}

    async def steps(app, pilot):
        app.show_page("pc")
        await settle(app, pilot)
        button = app.query_one("#pc .tool Button")
        seen["before"] = button.name
        await pilot.click(button)
        await pilot.pause()
        await pilot.click(app.screen.query_one("#yes"))  # the "Install?" question
        await settle(app, pilot)
        seen["log"] = [line.text for line in app.query_one("#install-log", ProgressLog).lines]
        seen["status"] = text_of(app, "#status-text")
        seen["after"] = [str(label.render()) for label in app.query("#pc .tool-state")]
        seen["buttons"] = len(app.query("#pc .tool Button"))

    run_app(tmp_path, steps)
    assert seen["before"] == "gguf"
    assert seen["log"] == ["Collecting torch"]
    assert seen["status"] == "GGUF tools installed"
    assert seen["after"][0] == "Ready" and seen["buttons"] == 0


def test_quantize_offers_to_install_missing_tools(tmp_path: Path, monkeypatch):
    from ocq.tui import install

    fake_model(monkeypatch, ["GGUF backend not installed. Run: uv sync --extra gguf"])
    monkeypatch.setattr(
        install, "tool_state", lambda a, hw: install.ToolState(a, install.MISSING, "Not installed")
    )
    seen = {}

    async def steps(app, pilot):
        app.show_page("quantize", focus=True)
        app.query_one(QuantizePage).set_format("gguf:Q8_0")
        await pilot.press(*"org/model", "enter")
        await settle(app, pilot)
        seen["shown"] = app.query_one("#plan").display
        seen["problem"] = app.query_one("#quantize .problem").display
        seen["card"] = text_of(app, "#plan-text")
        seen["start"] = app.query_one("#start").display
        seen["install"] = str(app.query_one("#install").label)

    run_app(tmp_path, steps)
    assert seen["shown"] and not seen["problem"]
    assert not seen["start"]
    assert seen["install"] == "Install GGUF tools"
    assert "not installed" not in seen["card"]  # the button says it


def test_ctrl_c_quits(tmp_path: Path):
    async def steps(app, pilot):
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running

    run_app(tmp_path, steps)


def test_progress_box_keeps_only_the_last_state_of_a_progress_bar():
    from textual import events

    log, lines = ProgressLog(), []
    log.write = lambda text, **kwargs: lines.append(text.plain)
    redraws = "\x1b[?25l\x1b[2KStarting 0%\x1b[2KStarting 50%\x1b[2KStarting done\n"
    meters = "Fetching 14 files: 100%|####| 14/14\nDownload complete: :  |  0.00B\n"
    for chunk in (redraws[:15], redraws[15:40], redraws[40:], meters, "\x1b[?25hConverting ...\n"):
        log.on_print(events.Print(chunk))
    log.flush()
    assert lines == ["Starting done", "Converting ..."]


def test_arrow_keys_move_between_the_buttons_of_a_question(tmp_path: Path):
    from ocq.tui.widgets import Confirm

    seen, answers = [], []

    async def steps(app, pilot):
        app.push_screen(Confirm("Install the GPTQ tools?", "A few GB.", "Install"), answers.append)
        await pilot.pause()
        seen.append(app.focused.id)  # Cancel first: nothing happens by accident
        for key in ("left", "right", "up", "down", "left"):
            await pilot.press(key)
            seen.append(app.focused.id)
        await pilot.press("enter")
        await pilot.pause()

    run_app(tmp_path, steps)
    assert seen == ["no", "yes", "no", "yes", "no", "yes"]
    assert answers == [True]
