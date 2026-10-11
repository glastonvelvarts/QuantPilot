# QuantPilot — One-Click Quantizer (OCQ)

Terminal LLM optimizer: **inspect the PC -> see which models it can run -> pick a quantization -> get a quantized, validated, benchmarked artifact**.

QuantPilot combines [llmfit](https://github.com/AlexsJones/llmfit) with a **universal hardware resolver**, live memory micro-probing, and four working quantization backends (GGUF, AWQ, GPTQ, bitsandbytes). Every run ends by loading the artifact, generating text with it, and measuring real tokens/sec and time to first token.

---

## Key Features

- **Works on any PC**: hardware comes from llmfit when it is available and from a built-in probe (psutil, `nvidia-smi`, OS queries) when it is not. Nothing hard-fails because a tool is missing.
- **Tells you what can and cannot run**: `feasibility` shows, for every quantization level from FP16 down to Q2, exactly how much memory the model needs, whether it fits in the VRAM that is free right now, spills into RAM, or does not fit at all.
- **No invented numbers**: sizes are computed from the model's real tensor shapes with llama.cpp's own rules, hardware figures are read or measured on the machine, and speed is either a physical limit or a fit to benchmarks this PC has already run. See [How the numbers are calculated](#how-the-numbers-are-calculated).
- **Quantization control**: choose the backend, bit width, exact GGUF type, group size and calibration size, or leave any of them on auto.
- **Dynamic Hardware & Interconnect Resolution**:
  - **Live DRAM Micro-Probe**: Sub-5ms stream copy benchmark (`ctypes.memmove`) measuring actual RAM bandwidth on any CPU with zero `sudo` or dependencies.
  - **Live VRAM Probe**: the same copy test inside GPU memory (needs PyTorch with CUDA), measured once per GPU and remembered.
  - **Real memory module telemetry**: peak RAM bandwidth from the configured speed and channel count of the installed modules.
  - **Unified Memory Architecture (UMA)**: Native awareness for Apple Silicon (Metal) and integrated APUs.
  - **PCIe Offload Telemetry**: Checks PCIe generation and lane width to model host-to-device offloading.
- **Checks the quantization job too**: the plan warns when the PC lacks the disk, RAM or VRAM to *produce* the quantized model, which is a different question from running it.

---

## Backends

| Backend | Bits | Needs | Output runs with |
|---------|------|-------|------------------|
| `gguf` | 2, 3, 4, 5, 6, 8 | Any CPU. GPU optional. | llama.cpp, Ollama, LM Studio |
| `awq` | 4 | NVIDIA GPU | vLLM, transformers |
| `gptq` | 4, 8 | NVIDIA GPU, or CPU (slow) | vLLM, transformers |
| `bnb` | 4, 8 | NVIDIA GPU | transformers |

- **GGUF** uses llama.cpp. The official llama.cpp release for your OS is downloaded on first use into `~/.cache/ocq/llama.cpp` (CUDA build on NVIDIA PCs), so no compiler or manual build is needed.
- **AWQ and GPTQ** use [llm-compressor](https://github.com/vllm-project/llm-compressor) and save in the compressed-tensors format that vLLM loads directly.
- When no backend is chosen, OCQ picks AWQ/GPTQ on a Linux NVIDIA machine where the model fits in VRAM, and GGUF everywhere else.

---

## Setup

```sh
cd quantPilot
uv sync --group dev                 # CLI only: profile, recommend, feasibility, plan
uv sync --group dev --all-extras    # plus every quantization backend
```

Or install only the backends you want:

```sh
uv sync --group dev --extra gguf
uv sync --group dev --extra awq --extra gptq
uv sync --group dev --extra bnb
```

The backends pull in PyTorch (several GB). On Windows the CUDA build of PyTorch is installed so an NVIDIA card is used when present. If the project lives in a synced folder such as OneDrive, keep the environment outside it:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\quantpilot"
```

Linux needs the OpenMP runtime for llama.cpp: `sudo apt install libgomp1` (already there if `gcc` is installed).

---

## Commands

```sh
# 1. What does this PC have?
uv run ocq profile

# 2. Which backends can run here?
uv run ocq backends

# 3. Which models suit this PC?
uv run ocq recommend --limit 10 --use-case coding

# 4. Can this model run here, and at which quantization?
uv run ocq feasibility Qwen/Qwen2.5-7B-Instruct

# 5. Show the plan without doing anything
uv run ocq plan Qwen/Qwen2.5-7B-Instruct --goal speed

# 6. Quantize, validate and benchmark
uv run ocq optimize Qwen/Qwen2.5-0.5B-Instruct --execute
```

| Command | Description |
|---------|-------------|
| `profile` | CPU, RAM, GPU, memory bandwidth, PCIe rate, free disk. |
| `backends` | Which quantization backends are installed and usable on this PC, and why not. |
| `recommend` | Models that suit this PC (llmfit catalog, or a built-in list of open models). |
| `feasibility` | Fit, run mode (`gpu`, `gpu_offload`, `cpu`, `unfit`) and predicted tokens/sec per quantization level. |
| `plan` | The quantization that would run, with warnings about disk, RAM and VRAM. |
| `optimize` | Full pipeline: analyze -> plan -> quantize -> validate -> benchmark. Dry run unless `--execute`. |

### Quantization control

`plan` and `optimize` take the same options. Anything left out is decided automatically.

| Option | Meaning |
|--------|---------|
| `--goal`, `-g` | `balanced` (default), `speed`, `quality`, `memory` |
| `--backend`, `-b` | `gguf`, `awq`, `gptq`, `bnb` |
| `--bits` | Bit width. Rejected with a clear message if the backend does not support it. |
| `--gguf-type` | Exact GGUF type: `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0` |
| `--group-size` | AWQ/GPTQ: how many weights share one scale (default 128). Lowered automatically if it does not divide the model's layers. |
| `--samples` | AWQ/GPTQ: number of calibration samples (wikitext-2) |
| `--seq-len` | AWQ/GPTQ: tokens per calibration sample (default 512) |
| `--context`, `-c` | Context size in tokens the model will run with (default 4096). It decides how much memory the key/value cache needs, and the GGUF benchmark runs with it. |
| `--output`, `-o` | Output folder (default `./ocq-output/<model>`) |

```sh
uv run ocq optimize Qwen/Qwen2.5-0.5B-Instruct --gguf-type Q5_K_M --execute
uv run ocq optimize Qwen/Qwen2.5-0.5B-Instruct -b awq --samples 128 --execute
uv run ocq optimize Qwen/Qwen2.5-0.5B-Instruct -b gptq --bits 8 --group-size 64 --execute
uv run ocq optimize Qwen/Qwen2.5-0.5B-Instruct -b bnb --bits 4 --execute
```

### What `optimize --execute` produces

```
<output>/artifact/    the quantized model (.gguf file, or a folder for awq/gptq/bnb)
<output>/workspace/   logs of the convert, quantize and benchmark steps (gguf)
<output>/error.log    full traceback, only when something failed
```

It then prints the size on disk next to the predicted size, the validation checks (artifact exists, is smaller than the original, loads, writes readable text), the measured tokens/sec and time to first token, and a sample of the generated text. The exit code is non-zero when validation fails.

GGUF artifacts are benchmarked with llama.cpp: speed is the average of 3 runs of 128 tokens from `llama-bench`. AWQ, GPTQ and bitsandbytes artifacts are benchmarked with transformers, which is slower than a serving runtime such as vLLM, so read those numbers as "it works" rather than as deployment speed.

---

## How the numbers are calculated

**Memory needed = weights + context.**

- **Weights** come from the name and shape of every tensor in the model, read from the safetensors file headers on HuggingFace (the weights themselves are not downloaded). For GGUF, each tensor gets the type llama-quantize would give it, including the mixed types inside `Q4_K_M` and friends and the fallback types for widths that are not a multiple of 256. Checked against `llama-quantize --dry-run` for all 12 types on two models: within 0.02%. For AWQ, GPTQ and bitsandbytes the packed size is within 1% of the real artifacts.
- **Context** is the key/value cache (`2 x layers x context x kv_heads x head_dim x 2 bytes`, from the model's `config.json`) plus llama.cpp's output buffer (`vocab x 512 x 4 bytes`). The cache matches llama.cpp's load log to the megabyte. A few tens of MB of scratch buffers are not counted.
- **Fit** compares that with the VRAM and RAM that are free at this moment, not the totals.

If a repo has no readable safetensors headers or `config.json`, sizes fall back to parameter count x average bits and the output says "estimated".

**Hardware.**

| Value | Where it comes from |
|-------|---------------------|
| RAM and VRAM, total and free | The OS and `nvidia-smi`, at the time of the command |
| RAM peak bandwidth | Windows: configured speed x channels of the installed modules. Linux: `dmidecode` when readable. Otherwise a table by CPU generation. |
| RAM live bandwidth | Measured copy test |
| GPU bandwidth, live | Measured copy test in VRAM (PyTorch with CUDA) |
| GPU bandwidth, spec | A table by GPU model. The driver does not expose it. |

**Speed.** There is no formula that gives exact tokens/sec from hardware specs, so OCQ does not pretend to have one:

- `<= N` is a limit: every token reads all the weights once, so speed cannot exceed bandwidth / model size. On the test machine a 7B model reached 98% of this limit on the GPU; small models and CPU runs stay well below it.
- `~ N` is the expected speed, fitted to GGUF benchmarks already run on this PC (`seconds per token = fixed + per GB x size`). It appears once two models of different sizes have been benchmarked on the same device, and every `optimize --execute` adds a data point. On the test machine the fit reproduces its own measurements within 6% on CPU, and within about 20% on GPU for models under 1 GB and 1% at 7B.

Time to first token is always measured, never predicted.

Remembered between runs, all under `~/.cache/ocq/`: `measurements.json` (benchmark results), `gpu_bandwidth.json` (VRAM probe), `models/` (tensor shapes per model revision), `llama.cpp/` (downloaded tools). Delete any of them to start fresh.

---

## Project Layout

```
src/ocq/
  cli.py                  Terminal commands
  hardware.py             Hardware detection: llmfit first, built-in probe as fallback
  hardware_resolver.py    Memory and interconnect bandwidth: live micro-probe, bus telemetry, UMA
  llmfit_bridge.py        llmfit CLI JSON wrapper
  model_analyzer.py       Model metadata and fit per quantization level
  model_size.py           Exact sizes: per-tensor GGUF types, packed sizes, key/value cache
  measurements.py         Speed: bandwidth limit, and the fit to this PC's own benchmark runs
  engine.py               Picks backend, bit width and calibration settings
  plan.py                 Builds the plan and its warnings
  pipeline.py             End-to-end orchestration
  validator.py            Post-quantization checks
  types.py                Shared data types
  quantizers/
    protocol.py           The steps every backend follows
    base.py               Shared code for the transformers based backends
    oneshot.py            Calibrated quantization with llm-compressor
    awq.py, gptq.py       The AWQ and GPTQ recipes
    bnb.py                bitsandbytes
    gguf.py               GGUF conversion, quantization and benchmark
    llama_cpp.py          Finds or downloads the llama.cpp tools
tests/tests/              Unit tests, plus test_quantize_real.py (real quantization, marked slow)
```

---

## Development & Testing

```sh
uv run pytest             # unit tests, a few seconds
uv run pytest -m slow     # downloads a 135M model and quantizes it with every backend
uv run ruff check .
```

The slow tests skip any backend that is not usable on the machine and say why.

---

## Environment Variables

- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`: HuggingFace token for gated models (e.g. Llama, Gemma).
- `OCQ_LLMFIT_BIN`: Path to the `llmfit` binary if it is not installed with the project.
- `OCQ_LLAMA_CPP_BIN`: Folder with your own llama.cpp build (`llama-quantize`, `llama-completion`) to use instead of the downloaded release.
- `OCQ_LLAMA_CPP`: llama.cpp source folder whose `convert_hf_to_gguf.py` should be used.
