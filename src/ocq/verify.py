"""ocq verify: load a GGUF in llama.cpp once and compare its own memory numbers with QuantPilot's.

llama.cpp logs (with -v) what it allocates: the model buffers per device, the KV cache, the
recurrent state of Mamba and convolution layers. Before the real load it runs a trial pass that
allocates nothing, so only the lines after the last "loading model tensors" count.

How llama.cpp lays out the weights:
  CPU side  the input embedding table (the token lookup runs on the CPU), the experts of the
            layers --n-cpu-moe keeps in RAM, and the first blocks when not every layer fits
            ("offloaded 14/29 layers": the output layer and the last 13 blocks are on the GPU)
  GPU side  everything else. A model whose embeddings are tied also needs the table there, as its
            output layer.
A memory-mapped buffer (llama.cpp's default) is the range of the file from its first tensor to
its last, so it can include tensors of the other side stored in between: on the CPU side always,
on the GPU side with unified memory (Apple). A separate GPU gets copies: the sum of its tensors,
each as CUDA allocates it (cuda_bytes).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ocq.gguf_file import TENSOR_TYPES, GgufHeader, GgufTensor, read_local
from ocq.model_analyzer import GB, kv_copies, llama_layers_on_gpu, working_bytes
from ocq.model_size import kv_value_bytes
from ocq.quantizers.llama_cpp import missing_library
from ocq.runtimes import LLAMA_FIT_TARGET_GB, llama_cache_env, llama_device_env
from ocq.types import LLAMA_CPP, HardwareProfile, ModelSnapshot, OcqError

MIB = 2**20
# llama.cpp prints sizes in MiB with two decimals.
ROUNDING_MIB = 0.01

# llama.cpp's CUDA buffers (ggml-cuda.cu, b11429): a quantized tensor whose rows are not a
# multiple of 512 values gets the rest of one such row as padding, and every tensor starts at a
# multiple of 128 bytes. Only Gemma (1152 and 2304 wide) and gpt-oss (2880) show it: +0.02-0.04 MiB.
CUDA_ROW_PADDING = 512
CUDA_ALIGNMENT = 128

# Without a GPU, llama.cpp's CPU backend copies the weights of each matrix multiply into its own
# layout (a buffer of its own, CPU_REPACK) when it has kernels for the type on this CPU; the
# copy is as large as the tensor. ggml-cpu/repack.cpp, b11429: (type, CPU features it needs,
# the tensor's rows a multiple of). The features are the ones llama.cpp logs in system_info.
REPACK_RULES = (
    ("Q4_0", ("AVX2",), 8),
    ("Q4_0", ("SVE", "MATMUL_INT8", "SVE_CNT=32"), 8),
    ("Q4_0", ("NEON", "MATMUL_INT8"), 4),
    ("Q4_0", ("NEON", "DOTPROD"), 4),
    ("Q4_0", ("VXE",), 4),
    ("Q4_K", ("AVX2",), 8),
    ("Q4_K", ("NEON", "MATMUL_INT8"), 8),
    ("Q4_K", ("NEON", "DOTPROD"), 8),
    ("Q2_K", ("AVX512",), 8),
    ("Q5_K", ("NEON", "MATMUL_INT8"), 8),
    ("Q5_K", ("NEON", "DOTPROD"), 8),
    ("Q6_K", ("NEON", "MATMUL_INT8"), 8),
    ("Q6_K", ("NEON", "DOTPROD"), 8),
    ("IQ4_NL", ("AVX2",), 8),
    ("IQ4_NL", ("NEON", "DOTPROD"), 4),
    ("MXFP4", ("AVX2",), 8),
    ("MXFP4", ("NEON", "DOTPROD"), 4),
    ("Q8_0", ("NEON", "MATMUL_INT8"), 4),
    ("Q8_0", ("NEON", "DOTPROD"), 4),
)
CPU_ALIGNMENT = 32


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
    # Working memory llama.cpp reserves for one batch, and the logits buffer.
    gpu_compute_mib: float = 0.0
    cpu_compute_mib: float = 0.0
    output_mib: float = 0.0
    # What llama.cpp saw free on its GPUs once it had started CUDA, MiB, and how many GPUs.
    free_mib: float = 0.0
    gpus: int = 0
    server: bool = False  # llama-server's log (what QuantPilot plans for), not llama-completion's
    # The CPU backend's copy of the weights in its own layout (CPU_REPACK), and the CPU features
    # it logged ("AVX2", "NEON", "SVE_CNT=32", ...).
    repack_mib: float = 0.0
    cpu_features: frozenset[str] = frozenset()

    @property
    def cpu_only(self) -> bool:
        """No GPU took part. llama.cpp's CPU build still logs "offloaded 17/17 layers to GPU"."""
        return not self.gpus and not self.gpu_mib


@dataclass(frozen=True)
class Check:
    name: str
    predicted_mib: float
    actual_mib: float
    unit: str = "MiB"  # or "layers": a count, compared exactly
    upper_bound: bool = False  # QuantPilot's number only has to be at or above llama.cpp's

    @property
    def ok(self) -> bool:
        if self.upper_bound:
            return self.predicted_mib >= self.actual_mib - ROUNDING_MIB
        # Every log line is rounded to 0.01 MiB; a sum of several lines a little more.
        return abs(self.predicted_mib - self.actual_mib) <= 3 * ROUNDING_MIB


def in_ram(device: str) -> bool:
    """CPU buffers, and the pinned RAM a GPU backend copies through (CUDA_Host)."""
    return device.startswith("CPU") or device.endswith("_Host")


def parse_log(log: str) -> LlamaLog:
    """llama.cpp's memory numbers from the real load (after its trial pass)."""
    starts = [m.start() for m in re.finditer(r"load_tensors: loading model tensors", log)]
    if not starts:
        raise OcqError("llama.cpp did not load the model; see its log.")
    real = log[starts[-1] :]
    crash = re.search(r"^.*(?:fatal error|GGML_ASSERT).*$", real, re.MULTILINE)
    if crash:
        raise OcqError(f"llama.cpp stopped with an error: {crash.group(0).strip()}")

    def number(pattern: str, default: float = 0.0) -> float:
        match = re.search(pattern, real)
        return float(match.group(1)) if match else default

    gpu = cpu = repack = 0.0
    for device, size in re.findall(
        r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB", real
    ):
        if device == "CPU_REPACK":
            repack += float(size)
        elif in_ram(device):
            cpu += float(size)
        else:
            gpu += float(size)
    gpu_compute = cpu_compute = 0.0
    for device, size in re.findall(
        r"sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB", real
    ):
        if in_ram(device):
            cpu_compute += float(size)
        else:
            gpu_compute += float(size)
    layers = re.search(r"offloaded (\d+)/(\d+) layers to GPU", real)
    # "using device CUDA0 (NVIDIA ...) (0000:01:00.0) - 7106 MiB free", once per load pass.
    free = dict(re.findall(r"using device (\S+) .*- (\d+) MiB free", log))
    types = re.search(r"llama_kv_cache: size = .*K \((\w+)\).*V \((\w+)\)", real)
    # "system_info: ... | CPU : SSE3 = 1 | AVX2 = 1 | ... | SVE_CNT = 32 | REPACK = 1 |"
    system = re.search(r"system_info:.*?CPU : (.*)", log)
    features = frozenset(
        name if name != "SVE_CNT" else f"SVE_CNT={value}"
        for name, value in re.findall(r"(\w+) = (\d+)", system.group(1) if system else "")
        if value != "0"
    )
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
        gpu_compute_mib=gpu_compute,
        cpu_compute_mib=cpu_compute,
        output_mib=sum(float(s) for s in re.findall(r"output buffer size =\s+([\d.]+) MiB", real)),
        free_mib=sum(float(mib) for mib in free.values()),
        gpus=len(free),
        server="srv  llama_server" in log,
        repack_mib=repack,
        cpu_features=features,
    )


def file_range(tensors: list[GgufTensor]) -> int:
    """Bytes a memory-mapped buffer of these tensors spans in the file."""
    if not tensors:
        return 0
    return max(t.offset + (t.nbytes or 0) for t in tensors) - min(t.offset for t in tensors)


def cuda_bytes(tensor: GgufTensor) -> int:
    """Bytes llama.cpp's CUDA backend allocates for this tensor."""
    size = tensor.nbytes or 0
    block = next((t for t in TENSOR_TYPES.values() if t[0] == tensor.type_name), None)
    if block and block[2] > 1 and tensor.shape and tensor.shape[0] % CUDA_ROW_PADDING:
        size += (CUDA_ROW_PADDING - tensor.shape[0] % CUDA_ROW_PADDING) // block[2] * block[1]
    return -(-size // CUDA_ALIGNMENT) * CUDA_ALIGNMENT


def repacked(tensor: GgufTensor, features: frozenset[str], as_output: bool = False) -> bool:
    """The CPU backend keeps this tensor in its own layout (REPACK_RULES). The input embedding
    table is looked up, not multiplied, so it never is; as_output: the same table loaded again
    as the output layer of a model without output.weight (llama-model-loader.cpp, duplicated
    tensors), which is multiplied, so it can be (Q6_K on Arm: LFM2-350M on an M1)."""
    if "REPACK" not in features or len(tensor.shape) < 2:
        return False
    if tensor.name == "token_embd.weight" and not as_output:
        return False
    return any(
        tensor.type_name == type_name
        and all(feature in features for feature in needs)
        and tensor.shape[1] % rows == 0
        for type_name, needs, rows in REPACK_RULES
    )


def on_cpu(
    tensor: GgufTensor, experts_in_ram: int, first_gpu_block: int = 0, experts_above: int = 0
) -> bool:
    """experts_in_ram: --n-cpu-moe, the experts of the first N blocks are in RAM.
    first_gpu_block: blocks before it are in RAM, when not every layer is on the GPU.
    experts_above: --fit keeps the experts of the blocks from this one on in RAM (0: none)."""
    if tensor.name == "token_embd.weight":
        return True
    parts = tensor.name.split(".")
    if parts[0] != "blk" or not parts[1].isdigit():
        return False
    block = int(parts[1])
    expert = "_exps" in tensor.name
    return (
        block < first_gpu_block
        or (expert and block < experts_in_ram)
        or (expert and 0 < experts_above <= block)
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
    if log.cpu_only and header.tensors:
        # Every weight is in RAM: the repacked ones in a copy, the rest mapped from the file.
        copies = [t for t in header.tensors if repacked(t, log.cpu_features)]
        mapped = [t for t in header.tensors if not repacked(t, log.cpu_features)]
        if not any(t.name == "output.weight" for t in header.tensors):
            # Tied: the embedding table also feeds the output layer, from a copy when repacked.
            copies += [
                t
                for t in header.tensors
                if t.name == "token_embd.weight" and repacked(t, log.cpu_features, as_output=True)
            ]
        copy_bytes = sum(-(-(t.nbytes or 0) // CPU_ALIGNMENT) * CPU_ALIGNMENT for t in copies)
        checks.append(Check("weights in RAM (file range)", file_range(mapped) / MIB, log.cpu_mib))
        if copies or log.repack_mib:
            checks.append(Check("weights repacked for this CPU", copy_bytes / MIB, log.repack_mib))
        return checks
    if not unified_memory and not experts_in_ram and log.free_mib and model.gguf_tensor_bytes:
        # llama.cpp's --fit, from the free memory it saw itself and --fit-target per GPU.
        fits_gb = (log.free_mib - LLAMA_FIT_TARGET_GB * 1024 * max(log.gpus, 1)) * MIB / GB
        weights_gb = sum(model.gguf_tensor_bytes.values()) / GB
        # With llama.cpp's own working memory, so this checks the layout on its own.
        working = log.gpu_compute_mib * MIB if log.gpu_compute_mib else None
        layout = llama_layers_on_gpu(
            model, "", weights_gb, log.context, value_bytes, fits_gb, working, log.conversations
        )
        if layout and total:
            checks.append(Check("layers on the GPU", layout[0], offloaded, unit="layers"))
    if header.tensors and total and offloaded:
        # The output layer counts as one of llama.cpp's layers, and goes to the GPU first.
        first_gpu_block = total - offloaded
        # --fit keeps the experts in RAM when the layers do not fit even without them, but those
        # of the lowest block on the GPU.
        experts_above = (
            first_gpu_block + 1 if model.experts and offloaded < total and not experts_in_ram else 0
        )
        tied = not any(t.name == "output.weight" for t in header.tensors)

        def cpu_side(tensor: GgufTensor) -> bool:
            return on_cpu(tensor, experts_in_ram, first_gpu_block, experts_above)

        cpu = [t for t in header.tensors if cpu_side(t)]
        gpu = [
            t for t in header.tensors if not cpu_side(t) or (tied and t.name == "token_embd.weight")
        ]
        gpu_bytes = file_range(gpu) if unified_memory else sum(cuda_bytes(t) for t in gpu)
        checks.append(Check("weights on the GPU", gpu_bytes / MIB, log.gpu_mib))
        checks.append(
            Check(
                "weights on the CPU (file range)",
                file_range(cpu) / MIB,
                log.cpu_mib + log.repack_mib,
            )
        )
    if log.server and offloaded:
        # llama-server's working memory, which QuantPilot estimates as an upper bound.
        metal = unified_memory
        in_ram = offloaded < total and not experts_in_ram
        copies = kv_copies(value_bytes, not unified_memory, metal, in_ram)
        working = working_bytes(model, log.context, LLAMA_CPP, copies, metal) / MIB
        actual = log.gpu_compute_mib + log.output_mib
        checks.append(Check("working memory (upper bound)", working, actual, upper_bound=True))
    return checks


# llama-server prints this once the model is loaded and its buffers are reserved.
SERVER_READY = b"srv  llama_server: model loaded"
SERVER_TIMEOUT_S = 900


def run_llama(
    tool: Path,
    model_path: Path,
    context: int,
    kv_cache: str,
    experts_in_ram: int,
    env: dict[str, str] | None = None,
) -> str:
    """Start llama-server (what QuantPilot plans for: 4 conversations), wait until it has loaded
    the model, stop it, and return its log."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        str(tool),
        "-m",
        str(model_path),
        "-c",
        str(context),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-warmup",
        "-v",
    ]
    if kv_cache != "f16":
        command += ["-ctk", kv_cache, "-ctv", kv_cache]
    if experts_in_ram:
        command += ["--n-cpu-moe", str(experts_in_ram)]
    # llama-server writes through its own handle and the log is read through another: a handle
    # it inherits shares the file position, so seeking it to read would make it write over lines.
    with tempfile.TemporaryDirectory() as folder:
        log = Path(folder) / "llama-server.log"
        with open(log, "wb") as out, open(log, "rb") as reader:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env={**os.environ, **(env or {})},
                )
            except OSError as exc:
                raise OcqError(f"Could not run {tool.name}: {exc}") from exc
            deadline, tail = time.monotonic() + SERVER_TIMEOUT_S, b""
            try:
                while process.poll() is None and time.monotonic() < deadline:
                    chunk = reader.read()
                    if SERVER_READY in tail + chunk:
                        break
                    tail = chunk[-len(SERVER_READY) :] if chunk else tail
                    time.sleep(0.2)
                # Windows says a DLL is missing only with the exit code.
                library = missing_library("", process.poll())
                if library:
                    raise OcqError(library)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        # The log can hold half a UTF-8 character; only its memory lines matter here.
        return log.read_bytes().decode("utf-8", errors="replace")


def verify(
    model: ModelSnapshot,
    model_path: Path,
    hardware: HardwareProfile,
    tool: Path,
    context: int,
    kv_cache: str = "f16",
    experts_in_ram: int = 0,
    save_log: Path | None = None,
) -> tuple[LlamaLog, list[Check], str]:
    """llama.cpp's numbers, QuantPilot's next to them, and llama.cpp's whole log (also written
    to save_log, before anything can fail)."""
    env = {**llama_cache_env(model.architecture, kv_cache), **llama_device_env(hardware.has_gpu)}
    text = run_llama(tool, model_path, context, kv_cache, experts_in_ram, env)
    if save_log:
        save_log.write_text(text, encoding="utf-8")
    library = missing_library(text)
    if library:
        raise OcqError(library)
    log = parse_log(text)
    header = read_local(model_path)
    checks = predict(model, header, log, kv_cache, experts_in_ram, hardware.unified_memory)
    return log, checks, text
