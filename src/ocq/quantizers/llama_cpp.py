"""Locate the llama.cpp tools, downloading the official release when the PC has none."""

from __future__ import annotations

import os
import platform
import re
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from rich.progress import Progress

from ocq.types import HardwareProfile, OcqError

# Pinned so everyone converts and quantizes with the build this code was tested against.
TAG = "b11429"
GITHUB = "https://github.com/ggml-org/llama.cpp"
CACHE = Path.home() / ".cache" / "ocq" / "llama.cpp" / TAG

# The Linux releases link against the system's OpenMP runtime, which minimal installs (servers,
# containers, a fresh WSL) leave out.
LIBRARY_PACKAGES = {
    "libgomp.so.1": "libgomp1 (Debian, Ubuntu), libgomp (Fedora) or gcc-libs (Arch)",
}
# Windows' exit code when a program needs a DLL it cannot find (STATUS_DLL_NOT_FOUND).
DLL_NOT_FOUND = 0xC0000135


def missing_library(output: str, returncode: int | None = None) -> str | None:
    """What to tell the user when a llama.cpp tool could not start for want of a library:
    Linux ("error while loading shared libraries"), macOS ("Library not loaded"), Windows (no
    message, only the exit code). None when that is not why it stopped."""
    found = re.search(
        r"error while loading shared libraries: ([^:\s]+)|Library not loaded: (\S+)", output
    )
    if found:
        library = found.group(1) or found.group(2)
        package = LIBRARY_PACKAGES.get(library)
        install = f" Install {package}." if package else ""
        return f"llama.cpp cannot start: this system lacks {library}.{install}"
    if returncode is not None and returncode & 0xFFFFFFFF == DLL_NOT_FOUND:
        return (
            "llama.cpp cannot start: Windows did not find a DLL it needs (for the CUDA build, "
            "the CUDA runtime next to it; else the Microsoft Visual C++ runtime)."
        )
    return None


def tool_path(name: str, hardware: HardwareProfile) -> Path:
    """Order: OCQ_LLAMA_CPP_BIN, then PATH, then our own download."""
    exe = name + (".exe" if os.name == "nt" else "")

    own_build = os.environ.get("OCQ_LLAMA_CPP_BIN")
    if own_build:
        tool = Path(own_build).expanduser() / exe
        if not tool.exists():
            raise OcqError(f"OCQ_LLAMA_CPP_BIN is set but {tool} does not exist.")
        return tool

    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)

    return release_folder(hardware) / exe


def converter_script() -> Path:
    own_source = os.environ.get("OCQ_LLAMA_CPP")
    if own_source:
        script = Path(own_source).expanduser() / "convert_hf_to_gguf.py"
        if not script.exists():
            raise OcqError(f"OCQ_LLAMA_CPP is set but {script} does not exist.")
        return script

    folder = CACHE / "source"
    if not (folder / "ready").exists():
        archive = download(f"{GITHUB}/archive/refs/tags/{TAG}.tar.gz", CACHE / "source.tar.gz")
        wanted = ("convert_hf_to_gguf.py", "conversion/", "gguf-py/gguf/")
        with tarfile.open(archive) as tar:
            for member in tar:
                inside_repo = member.name.partition("/")[2]
                if member.isfile() and inside_repo.startswith(wanted):
                    target = folder / inside_repo
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
        archive.unlink()
        (folder / "ready").touch()
    return folder / "convert_hf_to_gguf.py"


REGISTER = re.compile(r"@ModelBase\.register\((.*?)\)", re.S)


def converter_architectures() -> frozenset[str] | None:
    """The model classes (config.json "architectures") llama.cpp's converter turns into GGUF,
    read from its @ModelBase.register lines. None when the converter is not on this PC yet:
    this never downloads it."""
    own_source = os.environ.get("OCQ_LLAMA_CPP")
    folder = Path(own_source).expanduser() if own_source else CACHE / "source"
    script = folder / "convert_hf_to_gguf.py"
    if not script.is_file() or (not own_source and not (folder / "ready").exists()):
        return None
    names: set[str] = set()
    for path in [script, *sorted((folder / "conversion").glob("*.py"))]:
        for args in REGISTER.findall(path.read_text("utf-8", errors="replace")):
            names.update(re.findall(r'"([^"]+)"', args))
    return frozenset(names)


def release_folder(hardware: HardwareProfile) -> Path:
    build, archives = release_build(hardware)
    folder = CACHE / build
    if not (folder / "ready").exists():
        folder.mkdir(parents=True, exist_ok=True)
        for name in archives:
            archive = download(f"{GITHUB}/releases/download/{TAG}/{name}", CACHE / name)
            unpack(archive, folder)
            archive.unlink()
        (folder / "ready").touch()
    return folder


def release_build(hardware: HardwareProfile) -> tuple[str, list[str]]:
    system = platform.system()
    arm = platform.machine().lower() in ("arm64", "aarch64")
    arch = "arm64" if arm else "x64"
    nvidia = hardware.has_cuda and not arm

    if system == "Windows" and nvidia:
        return "win-cuda", [
            f"llama-{TAG}-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        ]
    if system == "Windows":
        return "win-cpu", [f"llama-{TAG}-bin-win-cpu-{arch}.zip"]
    if system == "Darwin":
        return "macos", [f"llama-{TAG}-bin-macos-{arch}.tar.gz"]
    if system == "Linux" and nvidia:
        return "linux-cuda", [
            f"llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        ]
    if system == "Linux":
        return "linux-cpu", [f"llama-{TAG}-bin-ubuntu-{arch}.tar.gz"]
    raise OcqError(
        f"No llama.cpp release for {system}. Build it yourself and set OCQ_LLAMA_CPP_BIN."
    )


def amount(size: int) -> str:
    return f"{size / 2**20:,.0f} MB" if size >= 2**20 else f"{size:,} bytes"


DOWNLOAD_TRIES = 3  # a connection that closes early is tried again, from where it stopped
CONTENT_RANGE = re.compile(r"bytes \d+-\d+/(\d+)")


def download(url: str, target: Path) -> Path:
    """Downloads url to target. A connection can close early without an error (the CUDA
    runtime zip stopped at 247 of 391 MB on a laptop): then the rest is asked for (an HTTP
    Range request) up to DOWNLOAD_TRIES times, and the file is only kept when it is whole."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    got, size = 0, 0
    with Progress() as progress:
        bar = progress.add_task(f"Downloading {target.name}", total=None)
        for attempt in range(1, DOWNLOAD_TRIES + 1):
            headers = {"User-Agent": "ocq/0.1"}
            if got:
                headers["Range"] = f"bytes={got}-"
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=60
                ) as response:
                    resumed = got > 0 and getattr(response, "status", 200) == 206
                    if not resumed:  # the whole file, from the start
                        got = 0
                        size = int(response.headers.get("Content-Length") or 0)
                        progress.update(bar, total=size or None, completed=0)
                    elif not size:
                        whole = CONTENT_RANGE.match(response.headers.get("Content-Range") or "")
                        size = int(whole.group(1)) if whole else 0
                    with open(partial, "ab" if resumed else "wb") as out:
                        while chunk := response.read(1024 * 1024):
                            out.write(chunk)
                            got += len(chunk)
                            progress.advance(bar, len(chunk))
            except urllib.error.HTTPError as exc:
                if exc.code < 500:  # not found, refused: another try would get the same
                    partial.unlink(missing_ok=True)
                    raise OcqError(f"Could not download {url}: {exc}") from exc
                error = exc
            except OSError as exc:  # a dropped connection, a timeout
                error = exc
            else:
                error = None
                if not size or got >= size:
                    break
            if attempt < DOWNLOAD_TRIES:
                where = f" at {amount(got)} of {amount(size)}" if size else ""
                next_try = f"try {attempt + 1} of {DOWNLOAD_TRIES}"
                print(f"Download cut off{where}, resuming ({next_try}) ...", flush=True)
        else:  # every try ended early
            if error is not None:
                partial.unlink(missing_ok=True)
                raise OcqError(
                    f"Could not download {url} in {DOWNLOAD_TRIES} tries: {error}. "
                    "Check the internet connection and try again."
                ) from error
    if size and got != size:
        partial.unlink(missing_ok=True)
        raise OcqError(
            f"The download of {target.name} stopped after {got:,} of {size:,} bytes, "
            f"{DOWNLOAD_TRIES} times. Check the internet connection and try again."
        )
    partial.replace(target)
    return target


def unpack(archive: Path, folder: Path) -> None:
    """Extract everything straight into one folder so the tools sit next to their libraries."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(folder)
    else:
        with tarfile.open(archive) as bundle:
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(folder, filter="data")
            else:
                bundle.extractall(folder)

    for path in list(folder.rglob("*")):
        if path.parent != folder and (path.is_file() or path.is_symlink()):
            path.replace(folder / path.name)
