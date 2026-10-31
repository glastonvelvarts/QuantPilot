"""ocq verify: load a GGUF in llama.cpp once and compare its own memory numbers with QuantPilot's.

llama.cpp logs (with -v) what it allocates: the model buffers per device, the KV cache, the
recurrent state of Mamba and convolution layers. Before the real load it runs a trial pass that
allocates nothing, so only the lines after the last "loading model tensors" count.

How llama.cpp lays out the weights:
  CPU side  the input embedding table (the token lookup runs on the CPU), and the experts of the
            layers --n-cpu-moe keeps in RAM
  GPU side  everything else. A model whose embeddings are tied also needs the table there, as its
            output layer.
A memory-mapped buffer (llama.cpp's default) is the range of the file from its first tensor to
its last, so it can include tensors of the other side stored in between: on the CPU side always,
on the GPU side with unified memory (Apple). A separate GPU gets copies: the sum of its tensors.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ocq.gguf_file import GgufHeader, GgufTensor, read_local
from ocq.model_size import kv_value_bytes
from ocq.types import LLAMA_CPP, HardwareProfile, ModelSnapshot, OcqError

MIB = 2**20
# llama.cpp prints sizes in MiB with two decimals.
ROUNDING_MIB = 0.01


@dataclass(frozen=True)
class LlamaLog:
    context: int
    conversations: int
    layers_on_gpu: tuple[int, int]  # (offloaded, total)
    gpu_mib: float
    cpu_mib: float
    kv_mib: float
    state_mib: float
    cache_types: str


@dataclass(frozen=True)
class Check:
    name: str
    predicted_mib: float
    actual_mib: float

    @property
    def ok(self) -> bool:
        # Every log line is rounded to 0.01 MiB; a sum of several lines a little more.
        return abs(self.predicted_mib - self.actual_mib) <= 3 * ROUNDING_MIB


def parse_log(log: str) -> LlamaLog:
    """llama.cpp's memory numbers from the real load (after its trial pass)."""
    starts = [m.start() for m in re.finditer(r"load_tensors: loading model tensors", log)]
    if not starts:
        raise OcqError("llama.cpp did not load the model; see its log.")
    real = log[starts[-1] :]

    def number(pattern: str, default: float = 0.0) -> float:
        match = re.search(pattern, real)
        return float(match.group(1)) if match else default

    gpu = cpu = 0.0
    for device, size in re.findall(
        r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB", real
    ):
        if device.startswith("CPU"):
            cpu += float(size)
        else:
            gpu += float(size)
    layers = re.search(r"offloaded (\d+)/(\d+) layers to GPU", real)
    types = re.search(r"llama_kv_cache: size = .*K \((\w+)\).*V \((\w+)\)", real)
    return LlamaLog(
        context=int(number(r"llama_context: n_ctx\s+=\s+(\d+)")),
        conversations=int(number(r"llama_context: n_seq_max\s+=\s+(\d+)", 1)),
        layers_on_gpu=(int(layers.group(1)), int(layers.group(2))) if layers else (0, 0),
        gpu_mib=gpu,
        cpu_mib=cpu,
        kv_mib=sum(float(s) for s in re.findall(r"llama_kv_cache: size =\s+([\d.]+) MiB", real)),
        state_mib=sum(
            float(s) for s in re.findall(r"llama_memory_recurrent: size =\s+([\d.]+) MiB", real)
        ),
        cache_types=f"K {types.group(1)}, V {types.group(2)}" if types else "",
    )


def file_range(tensors: list[GgufTensor]) -> int:
    """Bytes a memory-mapped buffer of these tensors spans in the file."""
    if not tensors:
        return 0
    return max(t.offset + (t.nbytes or 0) for t in tensors) - min(t.offset for t in tensors)


def on_cpu(tensor: GgufTensor, experts_in_ram: int) -> bool:
    if tensor.name == "token_embd.weight":
        return True
    parts = tensor.name.split(".")
    return (
        "_exps" in tensor.name
        and parts[0] == "blk"
        and parts[1].isdigit()
        and int(parts[1]) < experts_in_ram
    )


def predict(
    model: ModelSnapshot,
    header: GgufHeader,
    log: LlamaLog,
    kv_cache: str,
    experts_in_ram: int,
    unified_memory: bool,
) -> list[Check]:
    """QuantPilot's numbers for what this llama.cpp run allocated, next to llama.cpp's."""
    value_bytes = kv_value_bytes(kv_cache, LLAMA_CPP)
    with_state = model.kv_cache_bytes(log.context, LLAMA_CPP, log.conversations, value_bytes)
    state = sum(layer.state_bytes for layer in model.kv_layers) * log.conversations
    checks = [
        Check("context cache (KV)", ((with_state or 0) - state) / MIB, log.kv_mib),
    ]
    if state or log.state_mib:
        checks.append(Check("Mamba / convolution state", state / MIB, log.state_mib))

    offloaded, total = log.layers_on_gpu
    if header.tensors and total and offloaded == total:
        tied = not any(t.name == "output.weight" for t in header.tensors)
        cpu = [t for t in header.tensors if on_cpu(t, experts_in_ram)]
        gpu = [
            t
            for t in header.tensors
            if not on_cpu(t, experts_in_ram) or (tied and t.name == "token_embd.weight")
        ]
        gpu_bytes = file_range(gpu) if unified_memory else sum(t.nbytes or 0 for t in gpu)
        checks.append(Check("weights on the GPU", gpu_bytes / MIB, log.gpu_mib))
        checks.append(Check("weights on the CPU (file range)", file_range(cpu) / MIB, log.cpu_mib))
    return checks


def run_llama(
    tool: Path, model_path: Path, context: int, kv_cache: str, experts_in_ram: int
) -> str:
    command = [
        str(tool),
        "-m",
        str(model_path),
        "-c",
        str(context),
        "-n",
        "1",
        "-p",
        "Hi",
        "--no-warmup",
        "--single-turn",
        "-v",
    ]
    if kv_cache != "f16":
        command += ["-ctk", kv_cache, "-ctv", kv_cache]
    if experts_in_ram:
        command += ["--n-cpu-moe", str(experts_in_ram)]
    try:
        completed = subprocess.run(
            command, capture_output=True, stdin=subprocess.DEVNULL, timeout=900
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcqError(f"Could not run {tool.name}: {exc}") from exc
    # The sample text can hold half a UTF-8 character; only the log lines matter here.
    return (completed.stdout + completed.stderr).decode("utf-8", errors="replace")


def verify(
    model: ModelSnapshot,
    model_path: Path,
    hardware: HardwareProfile,
    tool: Path,
    context: int,
    kv_cache: str = "f16",
    experts_in_ram: int = 0,
) -> tuple[LlamaLog, list[Check]]:
    log = parse_log(run_llama(tool, model_path, context, kv_cache, experts_in_ram))
    header = read_local(model_path)
    return log, predict(model, header, log, kv_cache, experts_in_ram, hardware.unified_memory)
