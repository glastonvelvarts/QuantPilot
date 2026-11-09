"""How QuantPilot was installed, and the command that adds one of its tools (an extra: gguf,
awq, gptq, bnb, nvfp4) the same way:

    uv tool      uv tool install quantpilot     uv tool install "quantpilot[gguf]"  (with the
                                                extras it has already, from where it came)
    checkout     a clone of the repo, with uv   uv sync --extra gguf
    environment  pip or uv pip, into a venv     uv pip install "quantpilot[gguf]"

On a PC with an NVIDIA GPU, uv is told to pick the PyTorch build for its CUDA driver
(--torch-backend auto): PyPI's own build for Windows runs on the CPU only.
"""

from __future__ import annotations

import re
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from ocq import DISTRIBUTION, __version__
from ocq.types import HardwareProfile

UV_TOOL = "uv tool"
CHECKOUT = "checkout"
ENVIRONMENT = "environment"
RECEIPT = "uv-receipt.toml"  # what uv writes in every tool's environment


@dataclass(frozen=True)
class Installation:
    kind: str
    root: Path | None = None  # the repo, for a checkout
    # uv tool: what it was installed with, from uv's receipt.
    extras: tuple[str, ...] = ()
    source: str = ""  # " @ <path or URL>" when not from PyPI
    specifier: str = ""
    editable: str = ""  # the folder of an editable install (uv tool install -e)
    others: tuple[str, ...] = field(default_factory=tuple)  # --with requirements
    python: str = ""


def installation(prefix: Path | None = None) -> Installation:
    """How the QuantPilot that is running was installed."""
    receipt = (prefix or Path(sys.prefix)) / RECEIPT
    if receipt.is_file():
        try:
            return from_receipt(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Installation(UV_TOOL)
    root = project_root()
    if root is not None:
        return Installation(CHECKOUT, root=root)
    return Installation(ENVIRONMENT)


def project_root() -> Path | None:
    """The repo folder when QuantPilot runs from a checkout (an editable install), else None."""
    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    try:
        named = rf'^name = "{DISTRIBUTION}"'
        if pyproject.is_file() and re.search(named, pyproject.read_text(), re.M):
            return root
    except OSError:
        pass
    return None


def from_receipt(text: str) -> Installation:
    """uv's receipt: [tool] requirements = [{ name = "quantpilot", extras = [...], ... }, ...]."""
    requirements = parse_requirements(text)
    ours = next((r for r in requirements if r.get("name") == DISTRIBUTION), {})
    source = ""
    for key, prefix in (("path", ""), ("directory", ""), ("url", ""), ("git", "git+")):
        if ours.get(key):
            source = f" @ {prefix}{ours[key]}"
            break
    others = tuple(
        f"{r['name']}{r.get('specifier', '')}"
        for r in requirements
        if r.get("name") != DISTRIBUTION
    )
    python = re.search(r'^python\s*=\s*"([^"]+)"', text, re.M)
    return Installation(
        UV_TOOL,
        extras=tuple(ours.get("extras", ())),
        source=source,
        specifier=str(ours.get("specifier", "")),
        editable=str(ours.get("editable", "")),
        others=others,
        python=python.group(1) if python else "",
    )


def parse_requirements(text: str) -> list[dict]:
    try:
        import tomllib
    except ImportError:  # Python 3.10: the receipt's inline tables, read by hand
        tomllib = None
    if tomllib is not None:
        requirements = tomllib.loads(text).get("tool", {}).get("requirements", [])
        return [r for r in requirements if isinstance(r, dict)]
    found = []
    for table in re.findall(r"\{([^{}]*)\}", text.split("entrypoints", 1)[0]):
        entry: dict = {}
        for key, value in re.findall(r'([\w-]+)\s*=\s*(\[[^\]]*\]|"[^"]*")', table):
            if value.startswith("["):
                entry[key] = re.findall(r'"([^"]*)"', value)
            else:
                entry[key] = value.strip('"')
        found.append(entry)
    return found


def requirements(extra: str) -> list[str]:
    """The packages an extra adds, read from QuantPilot's installed metadata."""
    found = []
    for requirement in metadata.requires(DISTRIBUTION) or []:
        spec, _, marker = requirement.partition(";")
        if re.search(rf"""extra\s*==\s*["']{re.escape(extra)}["']""", marker):
            found.append(spec.strip())
    return found


def torch_backend(hardware: HardwareProfile | None) -> list[str]:
    """PyTorch for the CUDA driver on a PC with an NVIDIA GPU (every extra uses PyTorch)."""
    return ["--torch-backend", "auto"] if hardware is not None and hardware.has_cuda else []


def install_command(
    extra: str,
    hardware: HardwareProfile | None = None,
    found: Installation | None = None,
    in_place: bool = False,
) -> tuple[list[str], Path | None]:
    """(command, folder to run it in) that adds the extra the way QuantPilot was installed.
    in_place: for a uv tool on Windows, where the running ocq.exe cannot be replaced, the
    packages go into its environment directly (the app does this; uv tool install keeps them
    over upgrades)."""
    found = found or installation()
    uv = shutil.which("uv") or "uv"
    if found.kind == UV_TOOL and not in_place:
        extras = ",".join(sorted({*found.extras, extra}))
        if found.editable:
            target = ["--editable", f"{found.editable}[{extras}]"]
        else:
            target = [f"{DISTRIBUTION}[{extras}]{found.specifier}{found.source}"]
        command = [uv, "tool", "install", *target]
        command += [part for other in found.others for part in ("--with", other)]
        command += ["--python", found.python] if found.python else []
        return command + torch_backend(hardware), None
    if found.kind == CHECKOUT:
        # --inexact keeps the extras already there; the project's own sources pick PyTorch.
        return [uv, "sync", "--inexact", "--extra", extra], found.root
    packages = requirements(extra) or [f"{DISTRIBUTION}[{extra}]=={__version__}"]
    if shutil.which("uv"):
        return [uv, "pip", "install", "--python", sys.executable, *packages, *torch_backend(
            hardware
        )], None  # fmt: skip
    return [sys.executable, "-m", "pip", "install", *packages], None


def install_hint(extra: str, hardware: HardwareProfile | None = None) -> str:
    """What to type to add the extra, for messages: the command, short where it can be."""
    found = installation()
    if found.kind == ENVIRONMENT:
        uv = "uv pip install" if shutil.which("uv") else "pip install"
        backend = " --torch-backend auto" if uv.startswith("uv") and torch_backend(hardware) else ""
        return f'{uv} "{DISTRIBUTION}[{extra}]"{backend}'
    if found.kind == CHECKOUT:
        return f"uv sync --extra {extra}"
    command, _ = install_command(extra, hardware, found)
    command[0] = "uv"
    return shlex.join(command)
