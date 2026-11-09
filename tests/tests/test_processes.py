"""Quitting the app stops the programs it started and does not wait for its background work."""

import subprocess
import sys
import threading
import time

from ocq import processes
from ocq.tui import app as tui_app


def test_stop_all_ends_a_running_program_quickly():
    process = processes.started(
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    )
    start = time.monotonic()
    assert processes.stop_all() == 1
    assert process.poll() is not None
    assert time.monotonic() - start < 5


def test_an_ended_program_is_not_stopped_again():
    process = processes.started(subprocess.Popen([sys.executable, "-c", "pass"]))
    process.wait()
    processes.ended(process)
    assert processes.stop_all() == 0


def test_leaving_does_not_wait_for_background_work(monkeypatch):
    left: list[int] = []
    monkeypatch.setattr(tui_app.os, "_exit", left.append)
    done = threading.Event()
    worker = threading.Thread(target=done.wait, args=(30,))  # a lookup still running
    worker.start()
    try:
        tui_app.leave_now(0)
    finally:
        done.set()
        worker.join()
    assert left == [0]


def test_the_app_runs_on_its_own_loop(monkeypatch):
    """asyncio.run() would wait for the background lookups after the app has closed."""
    seen: dict = {}

    class FakeApp:
        return_code = None

        def run(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(tui_app, "QuantPilotApp", FakeApp)
    monkeypatch.setattr(tui_app, "leave_now", lambda code=0: seen.setdefault("code", code))
    tui_app.run_tui()
    assert seen["loop"] is not None
    assert seen["code"] == 0
