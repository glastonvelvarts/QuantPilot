"""GGUF: convert with llama.cpp's script, then quantize with llama-quantize."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from ocq.quantizers.base import download_model, model_slug
from ocq.quantizers.llama_cpp import converter_script, tool_path
from ocq.quantizers.protocol import BENCHMARK_PROMPT, BENCHMARK_TOKENS, Quantizer
from ocq.types import (
    BenchmarkResult,
    HardwareProfile,
    OcqError,
    OptimizationPlan,
    QuantAlgorithm,
)

BENCHMARK_REPEATS = 3


class GGUFQuantizer(Quantizer):
    algorithm = QuantAlgorithm.GGUF
    summary = "2 to 8 bit. Runs on CPU and any GPU with llama.cpp or Ollama."
    install_hint = "uv sync --extra gguf"

    def __init__(self) -> None:
        self.model_dir: Path | None = None
        self.converter: Path | None = None
        self.quantize_tool: Path | None = None
        self.artifact: Path | None = None

    @classmethod
    def is_available(cls) -> bool:
        # The converter script needs these. The llama.cpp binaries are downloaded on first use.
        return all(importlib.util.find_spec(module) for module in ("torch", "transformers"))

    def check(self, hardware: HardwareProfile) -> tuple[bool, str]:
        if platform.system() == "Linux" and not has_openmp_runtime():
            return False, "llama.cpp needs the OpenMP runtime. Run: sudo apt install libgomp1"
        return super().check(hardware)

    def prepare(self, plan: OptimizationPlan) -> None:
        self.converter = converter_script()
        self.quantize_tool = tool_path("llama-quantize", plan.hardware)
        self.model_dir = download_model(plan.model_id)

    def calibrate(self, plan: OptimizationPlan) -> None:
        return

    def quantize(self, plan: OptimizationPlan) -> None:
        gguf_type = plan.quant_config.gguf_type
        workspace = plan.output_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        full_precision = workspace / "model-f16.gguf"

        print("Converting to GGUF (f16) ...", flush=True)
        run_tool(
            [
                sys.executable,
                str(self.converter),
                str(self.model_dir),
                "--outfile",
                str(full_precision),
                "--outtype",
                "f16",
            ],
            log=workspace / "convert.log",
        )

        artifact_dir = plan.output_dir / "artifact"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact = artifact_dir / f"{model_slug(plan.model_id)}-{gguf_type}.gguf"

        print(f"Quantizing to {gguf_type} ...", flush=True)
        run_tool(
            [str(self.quantize_tool), str(full_precision), str(self.artifact), gguf_type],
            log=workspace / "quantize.log",
        )
        full_precision.unlink()

    def save(self, plan: OptimizationPlan) -> str:
        return str(self.artifact)

    def benchmark(self, plan: OptimizationPlan, artifact_path: str) -> BenchmarkResult:
        workspace = plan.output_dir / "workspace"

        # One real answer: the sample text, the time to first token, and where the layers ran.
        text, log = run_tool(
            [
                str(tool_path("llama-completion", plan.hardware)),
                "-m",
                artifact_path,
                "-p",
                BENCHMARK_PROMPT,
                "-n",
                str(BENCHMARK_TOKENS),
                "-c",
                str(plan.quant_config.context),
                "--temp",
                "0",
                "--single-turn",
                "--no-display-prompt",
                "--log-verbosity",
                "4",
            ],
            log=workspace / "benchmark.log",
        )
        prompt = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", log)
        on_gpu = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
        gpu_layers = int(on_gpu.group(1)) if on_gpu else 0

        device = "cpu"
        if gpu_layers:
            device = "gpu"
            if on_gpu.group(1) != on_gpu.group(2):
                device = f"gpu+cpu ({on_gpu.group(1)}/{on_gpu.group(2)} layers on gpu)"

        # Speed comes from llama-bench: a fixed number of tokens, repeated, then averaged.
        report, _ = run_tool(
            [
                str(tool_path("llama-bench", plan.hardware)),
                "-m",
                artifact_path,
                "-p",
                "0",
                "-n",
                str(BENCHMARK_TOKENS),
                "-r",
                str(BENCHMARK_REPEATS),
                "-ngl",
                str(gpu_layers),
                "-o",
                "json",
            ],
            log=workspace / "speed.log",
        )
        speed = json.loads(report)[0]

        notes = [
            f"Speed is the average of {BENCHMARK_REPEATS} runs of {BENCHMARK_TOKENS} tokens "
            f"(+/- {speed['stddev_ts']:.1f} tok/s)."
        ]
        if prompt:
            notes.append(f"Time to first token is for a {prompt.group(2)} token prompt.")

        return BenchmarkResult(
            tokens_per_second=speed["avg_ts"],
            time_to_first_token_s=float(prompt.group(1)) / 1000 if prompt else None,
            method="llama.cpp",
            device=device,
            sample_output=text.strip(),
            notes=notes,
        )


def has_openmp_runtime() -> bool:
    try:
        ctypes.CDLL("libgomp.so.1")
        return True
    except OSError:
        return False


def run_tool(command: list[str], log: Path) -> tuple[str, str]:
    """Run a command quietly, keep its output in a log file, and fail with the last lines of it."""
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        last_lines = "\n".join((completed.stderr or completed.stdout).strip().splitlines()[-15:])
        raise OcqError(f"{log.stem} step failed, full log in {log}\n{last_lines}")
    return completed.stdout, completed.stderr
