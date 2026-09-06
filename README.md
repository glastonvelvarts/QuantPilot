# QuantPilot- An OCQ(One click optimizer)

Terminal-only tool: **model + hardware + goal -> quantization plan -> (future) quantized artifact**.

Uses [llmfit](https://github.com/AlexsJones/llmfit) as an external binary for hardware detection and catalog fit scores. Everything else is OCQ.

## Setup

```sh
uv sync --group dev
uv tool install llmfit   # external dependency, once
```

## Commands

```sh
uv run ocq profile
uv run ocq feasibility Qwen/Qwen2.5-Coder-7B-Instruct
uv run ocq plan meta-llama/Llama-3.1-8B-Instruct --goal speed
uv run ocq optimize Qwen/Qwen2.5-7B-Instruct --goal balanced --dry-run
```

| Command | Purpose |
|---------|---------|
| `profile` | Hardware (via `llmfit --json system`) |
| `feasibility` | Fit/speed for a model |
| `plan` | Optimization plan only |
| `optimize` | Full pipeline (`--dry-run` default) |

Goals: `balanced` | `speed` | `quality` | `memory`

## Models

- **In llmfit catalog** (~13k): full fit scores via `llmfit info`
- **Any HuggingFace repo**: OCQ fetches HF API metadata as fallback
- Refresh catalog: `llmfit update`

## Layout

```
src/ocq/
  cli.py              Terminal commands
  llmfit_bridge.py    llmfit system + info (2 calls only)
  model_analyzer.py   HuggingFace fallback for any repo
  engine.py           Pick AWQ/GPTQ/BNB + bit width
  plan.py             Optimization plan
  pipeline.py         Orchestration
  quantizers/         Pluggable backends (stubs -> real impl next)
tests/
```

## Dev

```sh
make sync   # uv sync --group dev
make test   # uv run pytest
make check  # ruff + pytest
```

Optional quant backends: `uv sync --group dev --extra awq|gptq|bnb`

## Environment

- `OCQ_LLMFIT_BIN` or `--llmfit-bin` — path to llmfit if not on PATH
- `HF_TOKEN` — optional, for gated HuggingFace models
