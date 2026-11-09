"""QuantPilot's .env file (the Hugging Face token saved in Settings)."""

import io
import json
import os
import stat
import urllib.error
from types import SimpleNamespace

import pytest

from ocq import env_file
from ocq.types import OcqError

TOKEN = "hf_abcdefghijklmnopqrstuvwxyz1234"


def test_saving_makes_a_private_file_and_keeps_its_other_lines(tmp_path):
    path = tmp_path / "ocq" / ".env"
    env_file.save("HF_TOKEN", TOKEN, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600  # its owner only
    assert path.read_text().startswith("# Keys for QuantPilot")
    path.write_text(path.read_text() + "# mine\nexport OTHER_KEY='kept'\n")
    env_file.save("HF_TOKEN", "hf_newtokenvalue0000", path)
    assert env_file.read(path) == {"OTHER_KEY": "kept", "HF_TOKEN": "hf_newtokenvalue0000"}
    assert path.read_text().count("HF_TOKEN=") == 1 and "# mine" in path.read_text()
    assert os.environ["HF_TOKEN"] == "hf_newtokenvalue0000"


def test_a_token_from_the_shell_wins_over_the_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(f"HF_TOKEN={TOKEN}\n")
    monkeypatch.setenv("HF_TOKEN", "hf_fromtheshell000000")
    assert env_file.load(path) == []
    assert env_file.hf_token() == ("hf_fromtheshell000000", "shell")
    monkeypatch.delenv("HF_TOKEN")
    assert env_file.load(path) == ["HF_TOKEN"]
    assert env_file.hf_token() == (TOKEN, "file")


def test_removing_takes_out_only_that_key(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(f"HF_TOKEN={TOKEN}\nOTHER_KEY=1\n")
    monkeypatch.setenv("OTHER_KEY", "from the shell")  # the shell's, before QuantPilot starts
    assert env_file.load(path) == ["HF_TOKEN"]
    env_file.remove("HF_TOKEN", path)
    assert env_file.read(path) == {"OTHER_KEY": "1"}
    assert "HF_TOKEN" not in os.environ
    env_file.remove("OTHER_KEY", path)
    assert os.environ["OTHER_KEY"] == "from the shell"  # not the file's to take away


def test_a_token_is_shown_masked():
    assert env_file.mask(TOKEN) == "hf_••••••••1234"
    assert env_file.mask("short") == "•••••"


def test_hugging_face_says_who_the_token_belongs_to(monkeypatch):
    body = {"name": "glaston", "auth": {"accessToken": {"role": "read", "displayName": "ocq"}}}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    seen = {}

    def urlopen(request, timeout):
        seen["auth"] = request.get_header("Authorization")
        return Response(json.dumps(body).encode())

    monkeypatch.setattr(env_file.urllib.request, "urlopen", urlopen)
    assert env_file.check_hf_token(TOKEN) == env_file.HfAccount("glaston", "read")
    assert seen["auth"] == f"Bearer {TOKEN}"


def test_a_rejected_token_and_no_network_are_told_apart(monkeypatch):
    def rejected(request, timeout):
        raise urllib.error.HTTPError(env_file.WHOAMI, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(env_file.urllib.request, "urlopen", rejected)
    with pytest.raises(OcqError, match="does not accept this token"):
        env_file.check_hf_token(TOKEN)

    def offline(request, timeout):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(env_file.urllib.request, "urlopen", offline)
    with pytest.raises(ConnectionError):
        env_file.check_hf_token(TOKEN)


def test_ocq_loads_the_file_for_every_command(monkeypatch):
    from ocq import cli

    env_file.save("HF_TOKEN", TOKEN)
    monkeypatch.delenv("HF_TOKEN")
    env_file.from_file.clear()
    cli.start(SimpleNamespace(invoked_subcommand="models"))
    assert os.environ["HF_TOKEN"] == TOKEN
