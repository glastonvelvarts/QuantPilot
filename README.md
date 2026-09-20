# QuantPilot — An OCQ (One-Click Quantizer)

Terminal LLM optimizer: **model + hardware telemetry + goal -> optimal quantization plan -> quantized artifact**.

QuantPilot integrates [llmfit](https://github.com/AlexsJones/llmfit) with a **universal hardware resolver**, live memory micro-probing, and real quantization backends (GGUF, AWQ, GPTQ, BNB) to deliver non-hallucinated fitment scores, accurate tokens/sec predictions, and one-click model quantization.

---

## Key Features

- **Dynamic Hardware & Interconnect Resolution**:
  - **Live DRAM Micro-Probe**: Sub-5ms stream copy benchmark (`ctypes.memmove`) measuring actual, empirical RAM bandwidth on any CPU (x86, ARM64, RISC-V) with zero `sudo` or dependencies.
  - **Hardware Bus Driver Telemetry**: Mathematically calculates dedicated GPU bandwidth from bus width and clock frequencies directly from the driver (`(bus_width × clock × 2) / 8000`).
  - **Unified Memory Architecture (UMA)**: Native awareness for Apple Silicon (Metal) and integrated APUs (Intel Iris Xe, AMD Radeon Vega) with zero-copy PCIe overhead.
  - **PCIe Offload Telemetry**: Dynamically checks PCIe generation (Gen 1–5) and lane width to model host-to-device layer offloading bottlenecks.
  - **Zero-Hallucination Predictions**: Throughput and fit levels are constrained by physical hardware transfer rates, not arbitrary estimates.
- **Pluggable Quantization Backends**:
  - **GGUF**: Full `llama.cpp` quantization for CPU, Apple Silicon (Metal), and cross-platform deployment.
  - **AWQ**: Activation-aware 4-bit weight quantization optimized for NVIDIA CUDA throughput.
  - **GPTQ & BitsAndBytes**: 4-bit / 8-bit weight quantization fallbacks for constrained memory or rapid testing.
- **Universal Model Support**:
  - Native integration with the ~13k model catalog from `llmfit`.
  - Automatic HuggingFace API fallback for any public or gated repo.

---

## Setup

```sh
# Clone and install dependencies with uv
uv sync --group dev

# Install llmfit (external system detection dependency)
uv tool install llmfit
```

Optional quantization extras:
```sh
# Install specific quantization backends
uv sync --group dev --extra awq      # AutoAWQ for CUDA
uv sync --group dev --extra gguf     # GGUF / llama.cpp tooling
```

---

## Commands

```sh
# 1. Hardware profile with live empirical and peak bandwidths
uv run ocq profile

# 2. Feasibility assessment (llmfit catalog or HuggingFace fallback)
uv run ocq feasibility meta-llama/Llama-3.2-1B-Instruct

# 3. Generate optimization plan without executing
uv run ocq plan meta-llama/Llama-3.1-8B-Instruct --goal speed

# 4. Execute end-to-end quantization pipeline (--dry-run by default)
uv run ocq optimize Qwen/Qwen2.5-7B-Instruct --goal balanced --dry-run
```

| Command | Description |
|---------|-------------|
| `profile` | Inspect hardware (CPU, RAM capacity, peak & live bandwidth, GPUs, VRAM bandwidth, PCIe rate). |
| `feasibility` | Evaluates model fit, run mode (`gpu`, `gpu_offload`, `cpu`), and estimated tokens/sec. |
| `plan` | Generates a tailored quantization configuration (`algorithm`, `bits`, `group_size`, `calibration`). |
| `optimize` | Full pipeline: analyze -> plan -> quantize -> validate -> benchmark. |

### Optimization Goals

- `balanced` (default): Optimal trade-off between perplexity retention and throughput.
- `speed`: Maximizes tokens/second (e.g. 4-bit AWQ or GGUF Q4_K_M).
- `quality`: Retains higher precision (e.g. 8-bit or higher-tier GGUF quants like Q6_K / Q8_0).
- `memory`: Prioritizes minimum memory footprint to fit tightly constrained VRAM/RAM.

---

## Hardware Telemetry Example

Running `ocq profile` provides a comprehensive, non-hallucinated hardware breakdown:

```text
                      Hardware Profile                      
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field   ┃ Value                                          ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ CPU     │ Apple M1 (8 cores)                             │
│ RAM     │ 7.0 / 16.0 GB (68.2 GB/s peak, 49.0 GB/s live) │
│ Backend │ Metal                                          │
│ GPU     │ yes                                            │
│ GPU 0   │ Apple M1 - 16.0 GB @ 68.2 GB/s                 │
└─────────┴────────────────────────────────────────────────┘
```

On dedicated NVIDIA / AMD systems, `profile` additionally reports dedicated VRAM bandwidth and discrete PCIe link speeds (e.g. `31.5 GB/s` for PCIe 4.0 x16).

---

## Project Layout

```
src/ocq/
  cli.py                  Rich terminal commands (profile, feasibility, plan, optimize)
  hardware_resolver.py    Universal resolver: live micro-probe, bus telemetry & UMA
  llmfit_bridge.py        llmfit CLI JSON wrapper enriched with hardware resolver
  model_analyzer.py       HuggingFace metadata resolver for non-catalog models
  types.py                Shared domain models (HardwareProfile, GpuProfile, ModelSnapshot)
  engine.py               Optimization engine: selects backend, bit width, and calibration
  plan.py                 Generates optimization plans and technical rationales
  pipeline.py             End-to-end execution orchestrator
  quantizers/             Pluggable quantization engines
    gguf.py               GGUF / llama.cpp quantization engine
    awq.py                AWQ (AutoAWQ) CUDA quantization engine
    bnb.py                BitsAndBytes quantization backend
    gptq.py               GPTQ quantization backend
    protocol.py           Quantizer protocol interface
tests/
  tests/
    test_hardware_resolver.py  Unit tests for live probe, Apple Silicon, NVIDIA & APUs
    test_engine.py             Unit tests for decision engine
    test_llmfit_bridge.py      Unit tests for llmfit bridge
    test_model_analyzer.py     Unit tests for HuggingFace fallback
    test_plan.py               Unit tests for optimization plan generation
```

---

## Development & Testing

```sh
# Run tests with pytest
uv run pytest

# Format and lint check
uv run ruff check .
```

---

## Environment Variables

- `OCQ_LLMFIT_BIN`: Path to the `llmfit` binary if not available on system `PATH`.
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`: Optional HuggingFace token for gated models (e.g. LLaMA / Gemma).
