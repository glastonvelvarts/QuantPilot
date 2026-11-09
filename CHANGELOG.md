# Changelog

What changed in each version of QuantPilot. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/); the JSON and Python API has its own `schema_version` ([docs/api.md](docs/api.md)).

## [Unreleased]

### Changed

- README: how to open the app, for each way of installing QuantPilot (`ocq`, `uvx quantpilot`, `uv run ocq`).

## [1.5.0] - 2026-11-09

The first release on PyPI: `uv tool install quantpilot`.

### Recommendations

- `ocq recommend` and the app's **Check a model** choose the model, format, runtime and context cache for a kind of work: chat, coding, RAG, agentic, reasoning, long context, low latency, high throughput.
- Candidates from the models on the PC (Ollama, LM Studio, the Hugging Face cache, QuantPilot's own, your folders) and popular models on the Hugging Face Hub.
- Hard limits first (runtime, memory, context, architecture, quantization, disk), each with its reason; then the workload's priorities in order, never one blended score.
- Every pick says why it won, what the next options would give and cost, and its confidence (Low to Very High) with the evidence.

### Sizing

- Weights and context memory from `config.json` and the tensor headers, without downloading the weights, as llama.cpp, vLLM and transformers each allocate them: sliding window, chunked, MLA, mixture of experts (experts in RAM with `--n-cpu-moe`), Mamba, convolution and linear-attention state, several GPUs, `--kv-cache q8_0 / q4_0`.
- `ocq verify` compares it with llama.cpp's own log: checked on Windows with NVIDIA, Linux (WSL2) with NVIDIA and CPU only, macOS on Apple Silicon.
- Every fact about a model is labelled with its source and how sure it is (`ocq inspect`).

### Benchmarks

- `ocq bench` and the app's **Benchmark** page: writing and reading speed and the time to the first token (llama-bench), placed as QuantPilot plans; `--quality` measures perplexity on WikiText-2 (llama.cpp's method, in llama.cpp or transformers) and what the format keeps of the original model, measuring the original once from the Hugging Face cache.
- Every run is saved with the PC, the GPU, the runtime version and the placement; speed predictions, the time to the first token and the quality in recommendations learn from them.

### Quantizing

- `ocq optimize` and the app's **Quantize**: GGUF (llama.cpp, downloaded on first use), AWQ, GPTQ, bitsandbytes, NVFP4, from a Hugging Face model or your own fine-tune, checked and benchmarked after.
- What QuantPilot makes records the repo it came from, so it is compared with its original wherever it is moved.

### The app, scripts and installing

- `ocq` alone opens the full-screen app; Settings keeps a Hugging Face token for gated models in a file only you can read.
- `--json` on `recommend`, `bench`, `models`, `inspect`, `feasibility`, `profile`, `history` and `doctor`, and the same data from Python (`from ocq import api`), schema version 1.
- `ocq doctor` says what is ready on the PC and the command that fixes what is not.
- Installs with `uv tool install quantpilot`, `uvx quantpilot` or pip on Linux, macOS and Windows, Python 3.10 to 3.13; quantization tools as extras, added the way QuantPilot was installed.
- On Windows the CLI writes UTF-8, so its status marks (✓, ✗) show in consoles set to cp1252, the default of many terminals and of CI; where they cannot, plain `+` and `x`.

[Unreleased]: https://github.com/glastonvelvarts/QuantPilot/compare/v1.5.0...HEAD
[1.5.0]: https://github.com/glastonvelvarts/QuantPilot/releases/tag/v1.5.0
