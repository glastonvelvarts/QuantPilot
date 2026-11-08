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

Every CUDA runtime first creates a CUDA context, which takes memory nvidia-smi still shows as
free. Measured on an RTX 4060 Laptop with 7,610 MiB free: vLLM 0.33.0 then saw 6.88 GiB,
llama.cpp b11429 7,078-7,106 MiB, three starts each.
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
CUDA_CONTEXT_GB = 0.55
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
    """What a runtime can still use of the free memory once it has started."""
    return max(free_gb - cuda_context_gb(card), 0.0)


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
