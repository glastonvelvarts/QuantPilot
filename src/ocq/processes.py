"""The programs QuantPilot starts (llama.cpp's tools, the converter, an installer), so that
quitting the app stops them instead of leaving them running on their own."""

from __future__ import annotations

import subprocess
import threading

_running: set[subprocess.Popen] = set()
_lock = threading.Lock()


def started(process: subprocess.Popen) -> subprocess.Popen:
    with _lock:
        _running.add(process)
    return process


def ended(process: subprocess.Popen) -> None:
    with _lock:
        _running.discard(process)


def stop_all(grace: float = 2.0) -> int:
    """Asks every program still running to stop, then ends the ones that do not within
    grace seconds. Returns how many there were."""
    with _lock:
        running = [process for process in _running if process.poll() is None]
        _running.clear()
    for process in running:
        try:
            process.terminate()
        except OSError:
            pass
    for process in running:
        try:
            process.wait(timeout=grace)
        except (subprocess.TimeoutExpired, OSError):
            try:
                process.kill()
            except OSError:
                pass
    return len(running)
