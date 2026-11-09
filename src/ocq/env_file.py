"""Keys QuantPilot keeps for the user, such as their Hugging Face token, in a .env file.

The file is in QuantPilot's own settings folder (next to the app's tui.json), not in the folder
QuantPilot runs from, where it could end up in a git commit. It is readable by its owner only.
A key already set in the environment (the shell) always wins over the file.

    macOS    ~/Library/Application Support/ocq/.env
    Linux    ~/.config/ocq/.env
    Windows  %LOCALAPPDATA%\\ocq\\.env
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_dir

from ocq.types import OcqError

ENV_FILE = Path(user_config_dir("ocq", appauthor=False)) / ".env"
HF_TOKEN = "HF_TOKEN"
HF_TOKEN_NAMES = (HF_TOKEN, "HUGGING_FACE_HUB_TOKEN")  # the second is the older name
WHOAMI = "https://huggingface.co/api/whoami-v2"
LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

# Keys this process took from the file or saved itself: the others in os.environ came from
# the shell.
from_file: set[str] = set()


def read(path: Path | None = None) -> dict[str, str]:
    """KEY=value lines of the file (comments, blank lines and quotes left out)."""
    try:
        text = (path or ENV_FILE).read_text(encoding="utf-8")
    except OSError:
        return {}
    values = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if match and not line.lstrip().startswith("#"):
            value = match.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[match.group(1)] = value
    return values


def load(path: Path | None = None) -> list[str]:
    """Puts the file's keys into the environment, those the shell has not set. Returns them."""
    loaded = []
    for key, value in read(path).items():
        if key not in os.environ or key in from_file:
            os.environ[key] = value
            from_file.add(key)
            loaded.append(key)
    return loaded


def save(key: str, value: str, path: Path | None = None) -> Path:
    """Sets key=value in the file (made if missing, other lines kept) and in this process."""
    path = path or ENV_FILE
    if not LINE.match(f"{key}={value}") or "\n" in value:
        raise OcqError(f"{key} cannot be saved: it has a line break or is not a valid name.")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = ["# Keys for QuantPilot (ocq). Readable by you only."]
    kept = [line for line in lines if not (LINE.match(line) and LINE.match(line).group(1) == key)]
    write_private(path, "\n".join([*kept, f"{key}={value}"]) + "\n")
    os.environ[key] = value
    from_file.add(key)
    return path


def remove(key: str, path: Path | None = None) -> None:
    """Takes key out of the file, and out of this process when it came from the file."""
    path = path or ENV_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    kept = [line for line in lines if not (LINE.match(line) and LINE.match(line).group(1) == key)]
    if kept != lines:
        write_private(path, "\n".join(kept) + "\n")
    if key in from_file:
        os.environ.pop(key, None)
        from_file.discard(key)


def write_private(path: Path, text: str) -> None:
    """Writes the file so that only its owner can read it (0600 where the system has modes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(text)
    try:
        os.chmod(path, 0o600)  # also when the file was there before with wider modes
    except OSError:
        pass


def hf_token() -> tuple[str, str]:
    """(token, where from): "shell", "file" or "" when there is none."""
    for name in HF_TOKEN_NAMES:
        value = os.environ.get(name)
        if value:
            return value, "file" if name in from_file else "shell"
    return "", ""


def mask(value: str) -> str:
    """hf_••••••••x7Qa: enough to recognise it, not enough to use it."""
    if len(value) <= 10:
        return "•" * len(value)
    return f"{value[:3]}{'•' * 8}{value[-4:]}"


@dataclass(frozen=True)
class HfAccount:
    name: str
    role: str  # read, write or fineGrained (Hugging Face's names)


def check_hf_token(token: str) -> HfAccount:
    """Who the token belongs to and what it may do, from Hugging Face."""
    request = urllib.request.Request(
        WHOAMI, headers={"Authorization": f"Bearer {token}", "User-Agent": "ocq/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise OcqError("Hugging Face does not accept this token.") from exc
        raise OcqError(f"Hugging Face answered with error {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ConnectionError("Could not reach Hugging Face to check the token.") from exc
    token_info = (data.get("auth") or {}).get("accessToken") or {}
    return HfAccount(str(data.get("name", "")), str(token_info.get("role", "")))
