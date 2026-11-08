"""Every check of the Phase 4 test plan in a row, on any PC (Windows, Linux, Mac; CUDA, Metal
or CPU). The logs land in results/<os>-<gpu>/, so one folder holds what a machine measured.

uv run python scripts/gpu_check.py ~/models            # every step, every GGUF in the folder
uv run python scripts/gpu_check.py ~/models/x.gguf     # one file
uv run python scripts/gpu_check.py ~/models verify     # one of: verify server moe

1. ocq verify on each GGUF at 4k, 16k and 64k tokens (and 16k with a q8_0 cache), with
   llama.cpp's whole log: its working memory on the GPU, and where the input embedding table goes
2. llama-server's own start-up log at 4k and 16k
3. llama-bench of each mixture of experts: experts in RAM (--n-cpu-moe) against whole layers in
   RAM (-ngl)
The llama.cpp tools are the ones QuantPilot uses on this PC (its own download, PATH or
OCQ_LLAMA_CPP_BIN).
"""

import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ocq.gguf_file import read_local
from ocq.hardware import detect_hardware
from ocq.quantizers.llama_cpp import tool_path

MODELS = Path(sys.argv[1]).expanduser()
CONTEXTS = (4096, 16384, 65536)
SERVER_CONTEXTS = (4096, 16384)
ENV = os.environ | {"COLUMNS": "150", "PYTHONIOENCODING": "utf-8"}

HW = detect_hardware()
GPU = HW.gpus[0].name if HW.gpus else "cpu"
MACHINE = re.sub(r"[^a-z0-9]+", "-", f"{platform.system()} {GPU}".lower()).strip("-")
OUT = Path(__file__).parent.parent / "results" / MACHINE
FILES = (
    [MODELS] if MODELS.is_file() else sorted(MODELS.glob("*.gguf"), key=lambda f: f.stat().st_size)
)


def ocq(*args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "ocq.cli", *args], capture_output=True, env=ENV, timeout=1800
    )
    return (done.stdout + done.stderr).decode("utf-8", errors="replace")


def verify_runs() -> None:
    out = OUT / "verify"
    out.mkdir(parents=True, exist_ok=True)
    for file in FILES:
        runs = [(c, "f16") for c in CONTEXTS] + [(16384, "q8_0")]
        for context, cache in runs:
            name = f"{file.stem}-{context // 1024}k" + ("" if cache == "f16" else f"-{cache}")
            print(name, flush=True)
            text = ocq(
                "verify",
                str(file),
                "-c",
                str(context),
                "--kv-cache",
                cache,
                "--save-log",
                str(out / f"{name}.log"),
            )
            (out / f"{name}.txt").write_text(text, encoding="utf-8")


def server_log(file: Path, context: int) -> str:
    """llama-server's log (-v, for the memory lines) until it listens, then it is stopped."""
    command = [
        str(tool_path("llama-server", HW)),
        "-m",
        str(file),
        "-c",
        str(context),
        "--port",
        "8199",
        "--no-webui",
        "-v",
    ]
    log = OUT / "server" / f"{file.stem}-{context // 1024}k.log"
    with open(log, "wb") as handle:
        server = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline and server.poll() is None:
            # "listening on http://..." in recent builds, "server is listening" before.
            if re.search(rb"listening on|server is listening", log.read_bytes()):
                break
            time.sleep(1)
        server.kill()
        server.wait()
    return " ".join(command)


def server_runs() -> None:
    (OUT / "server").mkdir(parents=True, exist_ok=True)
    commands = []
    for file in FILES:
        for context in SERVER_CONTEXTS:
            print(f"server {file.stem} {context}", flush=True)
            commands.append(server_log(file, context))
    (OUT / "server" / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def bench(file: Path, *flags: str) -> dict:
    command = [
        str(tool_path("llama-bench", HW)),
        "-m",
        str(file),
        "-p",
        "512",
        "-n",
        "128",
        "-r",
        "3",
        "-o",
        "json",
        *flags,
    ]
    done = subprocess.run(command, capture_output=True, timeout=3600)
    try:
        rows = json.loads(done.stdout.decode("utf-8", errors="replace"))  # prompt, then generate
    except ValueError:
        rows = []
    if not rows:
        tail = done.stderr.decode("utf-8", errors="replace").strip().splitlines()[-3:]
        return {"flags": " ".join(flags), "error": " | ".join(tail)}
    return {"flags": " ".join(flags), "rows": rows}


def moe_runs() -> None:
    """A MoE that fits: what each layer of experts in RAM costs. One that does not: QuantPilot's
    --n-cpu-moe plan and a few around it, against whole layers in RAM."""
    out = OUT / "moe"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for file in FILES:
        header = read_local(file)
        if not any("_exps" in t.name for t in header.tensors):
            continue
        layers = int(header.get("block_count") or 0)
        planned = re.search(r"experts of (\d+) layers in RAM", ocq("verify", str(file)))
        n = int(planned.group(1)) if planned else 0
        print(f"{file.name}: {layers} layers, QuantPilot plans --n-cpu-moe {n}", flush=True)
        if n:
            runs = [("-ngl", "99", "-ncmoe", str(k)) for k in sorted({n, n + 2, n + 4, layers})]
            runs += [("-ngl", str(k)) for k in range(layers // 4, layers, max(1, layers // 8))]
        else:
            runs = [
                ("-ngl", "99", "-ncmoe", str(k))
                for k in sorted({0, layers // 4, layers // 2, layers})
            ]
        for flags in runs:
            print(f"  llama-bench {' '.join(flags)}", flush=True)
            results.append(
                {
                    "model": file.name,
                    "layers": layers,
                    "planned_n_cpu_moe": n,
                    **bench(file, *flags),
                }
            )
    (out / "bench.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "machine.json").write_text(
        json.dumps(
            asdict(HW) | {"raw": None, "platform": platform.platform()}, indent=1, default=str
        ),
        encoding="utf-8",
    )
    print(f"Results: {OUT}", flush=True)
    for step in sys.argv[2:] or ["verify", "server", "moe"]:
        {"verify": verify_runs, "server": server_runs, "moe": moe_runs}[step]()
