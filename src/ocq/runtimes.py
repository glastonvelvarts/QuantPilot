"""How each runtime budgets memory, beyond the weights and the context cache.

vLLM (vllm/v1/worker/utils.py request_memory, vllm/config/cache.py)
  Takes gpu_memory_utilization x the card's TOTAL memory, whatever else runs on it, and
  refuses to start when less than that is free ("Free memory on device ... on startup is less
  than desired GPU memory utilization"). Inside that budget go the weights, the working memory
  of a profile run, CUDA graphs and the context cache, which takes everything left over and
  must hold at least one sequence of max_model_len. The default was 0.9 up to v0.19.x and is
  0.92 since v0.20.0 (still in v0.32.0).
  The free memory it checks is measured after it has created its own CUDA context.

llama.cpp (common/fit.cpp, build b11429)
  With --fit on (the default) it fits the settings the user did not give into the free
  memory and keeps --fit-target (1024 MiB) free on every GPU: when the model and the context
  would leave less, it moves whole layers to RAM (a context given with -c is kept). Given
  --n-cpu-moe (or -ot), it does not fit anything ("tensor_buft_overrides already set by user,
  abort") and the plan has to fit by itself.

Free memory has two figures on NVIDIA:
  CUDA's    what cuMemGetInfo reports once a context exists. llama.cpp's --fit and vLLM's
            start-up check work from it, so it predicts what they do (usable_gb). QuantPilot
            reads it from the driver; without it, nvidia-smi's free memory less 0.55 GB.
  physical  what is really left. With --n-cpu-moe llama.cpp fits nothing itself, so a plan is
            limited by this (physical_gb).
On Linux they are one: CUDA's figure is nvidia-smi's less the context. Windows' driver model
(WDDM) reports a fixed figure instead: 7,106 MiB on an RTX 4060 Laptop in 127 loads, while
nvidia-smi showed 7,420-7,792 free, and also with another process holding 2 GB. A context
really takes about 90 MiB there (torch's 2,048 MiB took 2,136 in nvidia-smi, llama-server's
7,127 took 7,219), and going over CUDA's figure did not slow llama.cpp down: gpt-oss with
--n-cpu-moe 9 used 7,309 MiB of weights and was the fastest; only 8, over nvidia-smi's free
memory, fell.
"""

from __future__ import annotations

import importlib.util
import math
import re
from dataclasses import dataclass
from pathlib import Path

from ocq.types import GpuProfile

# gpu_memory_utilization of the newest vLLM release checked (v0.32.0), used when vLLM is not
# installed. With vLLM installed, its own default is read from its source.
VLLM_LATEST_UTILIZATION = 0.92
VLLM_LATEST = "v0.32.0"
CUDA_CONTEXT_GB = 0.55  # nvidia-smi's free less CUDA's figure, RTX 4060 Laptop (Windows)
WDDM_CONTEXT_GB = 90 / 1024  # what a CUDA context really takes under WDDM

# llama.cpp b11429 rotates the keys and values of a quantized cache (q8_0, q4_0) with a Hadamard
# matrix, which adds nodes to every layer's graph. A graph may have 8 x the model's tensors in
# nodes; Granite 3.0/3.1 MoE, with many steps for few tensors, then goes over and llama.cpp stops
# ("ggml-impl.h:330: fatal error", RTX 4060, q8_0 at 16k). Without the rotation its graph is the
# f16 one, which runs.
ROTATION_OVER_NODE_LIMIT = frozenset({"granitemoe"})
ROTATION_OFF = {"LLAMA_ATTN_ROT_DISABLE": "1"}


def llama_cache_env(architecture: str, kv_cache: str) -> dict[str, str]:
    """Environment llama.cpp needs to run this model with this cache type."""
    name = architecture.split()[0] if architecture else ""
    if kv_cache != "f16" and name in ROTATION_OVER_NODE_LIMIT:
        return dict(ROTATION_OFF)
    return {}


def llama_device_env(has_gpu: bool) -> dict[str, str]:
    """Without a GPU (or with OCQ_CPU_ONLY) llama.cpp must not offload to one, even when its build
    has a backend for it (macOS's Metal, a CUDA build on PATH): --device none for llama-server
    and llama-completion. llama-bench takes -dev none instead."""
    return {} if has_gpu else {"LLAMA_ARG_DEVICE": "none"}


def llama_cache_warning(architecture: str, kv_cache: str) -> str:
    if not llama_cache_env(architecture, kv_cache):
        return ""
    return (
        f"llama.cpp stops with an error on this model with a {kv_cache} cache unless "
        "LLAMA_ATTN_ROT_DISABLE=1 is set (its cache rotation takes the graph over llama.cpp's "
        "node limit)."
    )


# When vLLM's default share is not free, rate at the highest lower share in these steps
# (0.85, 0.80 ...): round values a user types, with some room under what is free.
VLLM_SHARE_STEP = 0.05
LLAMA_FIT_TARGET_GB = 1.0
UTILIZATION_DEFAULT = re.compile(r"gpu_memory_utilization: float = (?:Field\(default=)?([0-9.]+)")


@dataclass(frozen=True)
class VllmBudget:
    utilization: float
    source: str  # where the value comes from, for the notes


def vllm_budget() -> VllmBudget:
    """vLLM's default gpu_memory_utilization: the installed vLLM's, else the newest release's."""
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        spec = None
    if spec and spec.submodule_search_locations:
        for folder in spec.submodule_search_locations:
            source = Path(folder) / "config" / "cache.py"
            try:
                match = UTILIZATION_DEFAULT.search(source.read_text(encoding="utf-8"))
            except OSError:
                continue
            if match:
                return VllmBudget(float(match.group(1)), "the installed vLLM")
    return VllmBudget(VLLM_LATEST_UTILIZATION, f"vLLM {VLLM_LATEST}")


def cuda_context_gb(card: GpuProfile) -> float:
    """Memory the runtime's CUDA context takes on this card. 0 for other GPUs: not measured."""
    return CUDA_CONTEXT_GB if "cuda" in card.backend.lower() and not card.unified_memory else 0.0


def usable_gb(card: GpuProfile, free_gb: float) -> float:
    """The free memory a runtime sees once it has started: CUDA's figure when read (what
    --fit and vLLM's start-up check use), else the free memory less the CUDA context."""
    if card.cuda_free_gb is not None:
        return card.cuda_free_gb
    return max(free_gb - cuda_context_gb(card), 0.0)


def physical_gb(card: GpuProfile, free_gb: float) -> float:
    """What is really left once a runtime has started, for a plan llama.cpp does not fit
    itself (--n-cpu-moe). Under WDDM, more than CUDA's figure."""
    if card.wddm and "cuda" in card.backend.lower():
        return max(free_gb - WDDM_CONTEXT_GB, 0.0)
    return usable_gb(card, free_gb)


def vllm_card_budget_gb(card: GpuProfile, free_gb: float, utilization: float) -> float:
    """GB vLLM works with on this card: utilization x its total memory, or 0 when that much
    is not free once vLLM has created its CUDA context, and vLLM would refuse to start."""
    if card.vram_gb is None:
        return usable_gb(card, free_gb)  # total unknown: the free memory is all to go by
    requested = card.vram_gb * utilization
    return requested if usable_gb(card, free_gb) >= requested else 0.0


def vllm_highest_share(card: GpuProfile, free_gb: float, default: float) -> float:
    """The highest gpu_memory_utilization, in VLLM_SHARE_STEP steps and at most the default,
    that vLLM starts with on this card. 0 when not even one step is free."""
    if not card.vram_gb:
        return default
    steps = math.floor(round(usable_gb(card, free_gb) / card.vram_gb / VLLM_SHARE_STEP, 6))
    return round(min(steps * VLLM_SHARE_STEP, default), 2)
