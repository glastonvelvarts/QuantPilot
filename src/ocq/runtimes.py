"""How each runtime budgets memory, beyond the weights and the context cache.

vLLM (vllm/v1/worker/utils.py request_memory, vllm/config/cache.py)
  Takes gpu_memory_utilization x the card's TOTAL memory, whatever else runs on it, and
  refuses to start when less than that is free ("Free memory on device ... on startup is less
  than desired GPU memory utilization"). Inside that budget go the weights, the working memory
  of a profile run, CUDA graphs and the context cache, which takes everything left over and
  must hold at least one sequence of max_model_len. The default was 0.9 up to v0.19.x and is
  0.92 since v0.20.0 (still in v0.32.0).
  The free memory it checks is measured after it has created its own CUDA context, so that
  context comes off what nvidia-smi shows as free.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

from ocq.types import GpuProfile

# gpu_memory_utilization of the newest vLLM release checked (v0.32.0), used when vLLM is not
# installed. With vLLM installed, its own default is read from its source.
VLLM_LATEST_UTILIZATION = 0.92
VLLM_LATEST = "v0.32.0"
# vLLM's CUDA context: free memory it measured at start-up against nvidia-smi just before.
# RTX 4060 Laptop, vLLM 0.33.0 in WSL2: 6.88 GiB against 7,610 MiB (7.43 GiB), three starts.
VLLM_CUDA_CONTEXT_GB = 0.55
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
                return VllmBudget(float(match.group(1)), "the installed vLLM's default")
    return VllmBudget(VLLM_LATEST_UTILIZATION, f"vLLM's default ({VLLM_LATEST})")


def vllm_card_budget_gb(card: GpuProfile, free_gb: float, utilization: float) -> float:
    """GB vLLM works with on this card: utilization x its total memory, or 0 when that much
    is not free once vLLM has created its CUDA context, and vLLM would refuse to start."""
    if card.vram_gb is None:
        return free_gb  # total unknown: the free memory is all there is to go by
    requested = card.vram_gb * utilization
    return requested if free_gb - VLLM_CUDA_CONTEXT_GB >= requested else 0.0
