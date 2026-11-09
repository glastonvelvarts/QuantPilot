"""How QuantPilot was installed, and the command that adds a tool the same way."""

import sys
from dataclasses import replace

import pytest
from test_model_analyzer import make_hardware

from ocq import installation
from ocq.installation import CHECKOUT, ENVIRONMENT, UV_TOOL, Installation

# What uv 0.11 writes in a tool's environment (uv tool install "quantpilot[gguf] @ <wheel>").
RECEIPT = (
    "[tool]\n"
    'requirements = [{ name = "quantpilot", extras = ["gguf"], '
    'path = "/tmp/dist/quantpilot-0.1.0-py3-none-any.whl" }]\n'
    "entrypoints = [\n"
    '    { name = "ocq", install-path = "/home/me/.local/bin/ocq", from = "quantpilot" },\n'
    "]\n"
)
FROM_PYPI = (
    "[tool]\n"
    'requirements = [{ name = "quantpilot", specifier = "==0.1.0" }, '
    '{ name = "llmfit", specifier = ">=1.2" }]\n'
    'python = "3.12"\n'
    "entrypoints = []\n"
)
ONLY_QUANTPILOT = """[tool]
requirements = [{ name = "quantpilot", specifier = "==0.1.0" }]
"""
NVIDIA = make_hardware(vram_gb=8)
MAC = make_hardware()


@pytest.fixture(params=["tomllib", "by hand"])
def reader(request, monkeypatch):
    """Both receipt readers: tomllib (Python 3.11+) and the one for Python 3.10."""
    if request.param == "by hand":
        monkeypatch.setitem(sys.modules, "tomllib", None)  # import tomllib now fails


def test_a_uv_tool_is_known_by_uvs_receipt(tmp_path, reader):
    (tmp_path / installation.RECEIPT).write_text(RECEIPT)
    found = installation.installation(tmp_path)
    assert found.kind == UV_TOOL
    assert found.extras == ("gguf",)
    assert found.source == " @ /tmp/dist/quantpilot-0.1.0-py3-none-any.whl"

    (tmp_path / installation.RECEIPT).write_text(FROM_PYPI)
    found = installation.installation(tmp_path)
    assert (found.extras, found.source, found.specifier) == ((), "", "==0.1.0")
    assert (found.others, found.python) == (("llmfit>=1.2",), "3.12")


def test_a_checkout_and_any_other_environment(tmp_path, monkeypatch):
    assert installation.installation(tmp_path).kind == CHECKOUT  # the tests run from the repo
    monkeypatch.setattr(installation, "project_root", lambda: None)
    assert installation.installation(tmp_path).kind == ENVIRONMENT


def test_a_uv_tool_gets_the_extra_with_the_ones_it_has_from_where_it_came(monkeypatch):
    monkeypatch.setattr(installation.shutil, "which", lambda name: f"/bin/{name}")
    tool = installation.from_receipt(RECEIPT)
    command, folder = installation.install_command("awq", MAC, tool)
    assert command == [
        "/bin/uv", "tool", "install",
        "quantpilot[awq,gguf] @ /tmp/dist/quantpilot-0.1.0-py3-none-any.whl",
    ]  # fmt: skip
    assert folder is None

    # On an NVIDIA PC uv picks PyTorch for its CUDA driver; --with and the Python are kept.
    command, _ = installation.install_command("bnb", NVIDIA, installation.from_receipt(FROM_PYPI))
    assert command[3:] == [
        "quantpilot[bnb]==0.1.0", "--with", "llmfit>=1.2", "--python", "3.12",
        "--torch-backend", "auto",
    ]  # fmt: skip

    # An editable tool (uv tool install -e .): the same folder.
    command, _ = installation.install_command("gguf", MAC, replace(tool, editable="/src/qp"))
    assert command[3:] == ["--editable", "/src/qp[gguf]"]

    # In the app on Windows: into the running tool's environment (ocq.exe cannot be replaced).
    command, _ = installation.install_command("gguf", NVIDIA, tool, in_place=True)
    assert command[1:4] == ["pip", "install", "--python"] and command[-2:] == [
        "--torch-backend", "auto",
    ]  # fmt: skip


def test_a_checkout_syncs_and_an_environment_installs_the_extras_packages(monkeypatch):
    monkeypatch.setattr(installation.shutil, "which", lambda name: f"/bin/{name}")
    checkout = Installation(CHECKOUT, root=installation.project_root())
    command, folder = installation.install_command("gguf", NVIDIA, checkout)
    assert command == ["/bin/uv", "sync", "--inexact", "--extra", "gguf"]  # the repo's sources
    assert folder == checkout.root

    command, _ = installation.install_command("gguf", MAC, Installation(ENVIRONMENT))
    assert command[:5] == ["/bin/uv", "pip", "install", "--python", sys.executable]
    assert any(package.startswith("torch") for package in command[5:])

    monkeypatch.setattr(installation.shutil, "which", lambda name: None)  # no uv: pip
    command, _ = installation.install_command("gguf", MAC, Installation(ENVIRONMENT))
    assert command[:4] == [sys.executable, "-m", "pip", "install"]


def test_the_hint_is_what_to_type(tmp_path, monkeypatch):
    monkeypatch.setattr(installation.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(installation.sys, "prefix", str(tmp_path))
    assert installation.install_hint("gguf") == "uv sync --extra gguf"  # this checkout
    (tmp_path / installation.RECEIPT).write_text(ONLY_QUANTPILOT)
    assert installation.install_hint("gguf", NVIDIA) == (
        "uv tool install 'quantpilot[gguf]==0.1.0' --torch-backend auto"
    )
    (tmp_path / installation.RECEIPT).unlink()
    monkeypatch.setattr(installation, "project_root", lambda: None)
    assert installation.install_hint("awq", NVIDIA) == (
        'uv pip install "quantpilot[awq]" --torch-backend auto'
    )
    monkeypatch.setattr(installation.shutil, "which", lambda name: None)
    assert installation.install_hint("awq") == 'pip install "quantpilot[awq]"'


def test_the_cli_shows_the_extras_in_brackets(monkeypatch):
    # The table read [awq,gguf] as styling and printed "uv tool install 'quantpilot @ ...".
    from typer.testing import CliRunner

    from ocq import cli

    monkeypatch.setattr(
        installation, "installation", lambda prefix=None: installation.from_receipt(RECEIPT)
    )
    monkeypatch.setattr(cli, "detect_hardware", lambda _bin: MAC)
    result = CliRunner().invoke(cli.app, ["backends"], env={"COLUMNS": "250"})
    assert "uv tool install 'quantpilot[awq,gguf] @ /tmp/dist/" in result.output
