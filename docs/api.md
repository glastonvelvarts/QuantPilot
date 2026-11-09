# QuantPilot's JSON and Python API

Everything QuantPilot works out is available as data, for scripts and agents:

```sh
ocq recommend --workload coding --json
ocq bench ./model-Q4_K_M.gguf --quality --json
```

```python
from ocq import api

hardware = api.hardware()  # read this PC once
pick = api.recommend(workload="coding", hardware=hardware)["best"]
print(pick["model"]["id"], pick["format"]["label"], pick["format"]["tokens_per_second"])
```

`ocq <command> --json` prints exactly what the matching `api` function returns. On stdout there is only the JSON; progress lines (a download, a benchmark step) go to stderr.

## Stability

Every result starts with:

| Key | |
|---|---|
| `schema_version` | `1`. Within one version, keys are only added: never renamed or removed, and a value keeps its type and unit. Anything else is a new version. |
| `quantpilot_version` | The QuantPilot that made it, e.g. `1.5.0` |
| `kind` | Which result it is: `profile`, `inspect`, `feasibility`, `models`, `recommend`, `bench`, `history`, `doctor` |

Units are in the names: `_gb` is GiB (1024³ bytes), `_gbs` is GB/s, `_seconds`, `tokens_per_second`. A value that is not known is `null`.

A failure: the command prints `{"schema_version": 1, "error": "<what went wrong, for people>"}` and exits with 1; the Python function raises `ocq.types.OcqError` (`api.OcqError`) with the same message.

## The functions

| Python | Command | Returns |
|---|---|---|
| `api.hardware(llmfit_bin=None)` | | This PC, to pass to the others as `hardware=` (not JSON) |
| `api.profile(hardware=None)` | `ocq profile --json` | This PC |
| `api.inspect(model)` | `ocq inspect <model> --json` | Every fact about a model |
| `api.feasibility(model, context=4096, kv_cache="f16", hardware=None)` | `ocq feasibility <model> --json` | Every format of a model on this PC |
| `api.models(folders=(), include_partial=False, context=4096, kv_cache="f16", hardware=None)` | `ocq models --json` | The models on this PC, and how each fits |
| `api.recommend(models=(), workload="chat", context=0, kv_caches=(), folders=(), hardware=None, limit=8)` | `ocq recommend --json` | The pick, alternatives, and what was ruled out |
| `api.bench(model, quality="", context=4096, kv_cache="f16", hardware=None)` | `ocq bench <model> --json` | Speed, and with `quality="fixed"` or `"full"` (`--quality`, `--full`) quality |
| `api.history(model="")` | `ocq history --json` | This PC's saved benchmarks |
| `api.doctor(online=True, hardware=None)` | `ocq doctor --json` | What is ready, what is not, and the fixes |

A model is a Hugging Face id (`Qwen/Qwen3-8B`), a local folder or `.gguf` file, or `ollama:name:tag`. Workloads: `chat`, `coding`, `rag`, `agentic`, `reasoning`, `long-context`, `low-latency`, `throughput`. Cache types: `f16`, `q8_0`, `q4_0`.

`bench` runs for real (minutes) and saves to the history like `ocq bench`; the others only read.

## Shapes

### hardware (in `profile`, `feasibility`, `recommend`)

`os`, `cpu`, `cpu_cores`, `ram_gb`, `ram_free_gb`, `ram_bandwidth_gbs` (specification), `ram_bandwidth_measured_gbs`, `backend` (`CUDA`, `Metal`, `CPU`, ...), `unified_memory`, and `gpus`: a list of `name`, `count`, `vram_gb`, `vram_free_gb`, `backend`, `bandwidth_gbs`, `compute_capability`.

`profile` adds `disk_free_gb` (where the command runs).

### fact (in `inspect`)

`facts` has `parameters`, `active_parameters`, `precision`, `stored_bytes`, `architecture`, `context_length`, `kv_cache` (bytes per token), `quantization`, each `{value, source, kind, note}`. `kind` says how sure: `exact` (read from one of the model's files), `cross-checked` (two sources agree), `estimate` (derived, or the sources disagree), `unknown`.

### format (in `feasibility`, `models`, `recommend`)

One way to run a model on this PC:

| Key | |
|---|---|
| `label` | `Q4_K_M`, `AWQ 4-bit`, `BNB 4-bit double`, `BF16 (as stored)`, ... |
| `kind` | `gguf`, `awq`, `gptq`, `bnb`, `nvfp4`, or `stored` (the files as they are) |
| `runtime` | `llama.cpp`, `vLLM`, `transformers`, `MLX` |
| `bits_per_weight`, `weights_gb`, `context_gb` | Its size, and its context memory |
| `memory_required_gb`, `memory_free_gb` | What it needs, against what is free |
| `fit` | `Perfect`, `Good`, `Marginal`, `Too Tight` |
| `runs_on` | `gpu`, `gpu_offload` (layers split), `gpu_experts_ram` (experts of some layers in RAM), `cpu`, `unfit` |
| `gpu_layers`, `experts_in_ram_layers` | The split, when it is one |
| `tokens_per_second`, `tokens_per_second_basis` | Writing speed: `upper bound` (what memory speed allows) or `from runs on this PC` |
| `prompt_tokens_per_second` | Reading a 512 token prompt, learnt from this PC's runs (`null` before any) |
| `note` | Anything worth knowing about running it so |

### recommend

`workload` (`key`, `label`, `priorities` in order), `context`, `hardware`, `best` (a recommendation, or `null` when nothing runs here), `alternatives` (recommendations, best first), `considered` (count), `ruled_out` (each a candidate with `failed`: `{check, text}`), `unread` and `too_big` (`{model, why}`).

A recommendation:

| Key | |
|---|---|
| `model` | `ref` (as named), `id`, `on_this_pc`, `tags` (`coder`, `reasoning`, ...) |
| `format` | A format, above |
| `context`, `kv_cache` | What it was sized at |
| `needs_quantizing`, `command` | Whether QuantPilot makes the file first, and the `ocq optimize` command that does |
| `quality` | `kept`: the share of the model's quality the format keeps (0 to 1); `measured`: from its perplexity on this PC (else llama.cpp's numbers, scaled by this model's measured formats) |
| `values` | What the ranking compared: `quality_parameters` (what the model is worth in parameters, its format counted), `tokens_per_second`, `memory_gb`, and per workload `context_room_tokens`, `throughput_tokens_per_second`, `first_token_seconds` |
| `reasons` | `{kind, text}`, kind `good`, `tradeoff`, `bad`: why it won, or why it lost to the best |
| `warnings` | Texts |
| `confidence` | `level` (`Low`, `Medium`, `High`, `Very High`) and `evidence`: `{text, known}` |
| `lost_on` | For an alternative: the priority it fell short on, `drawbacks`, or `tie` |

### bench

`model` (the repo it was made from, when QuantPilot made it), `label`, `placement` (`gpu`, `cpu`, `gpu+cpu (...)`, `gpu (experts of N layers in ram)`), `kv_cache`, and:

- `speed` (GGUF; `null` for a model only transformers runs): `tokens_per_second`, `tokens_per_second_spread`, `prompt_tokens_per_second` and `first_token_seconds` by prompt length (`"512"`, `"4096"`), `expected_tokens_per_second` and `expected_basis` (what QuantPilot predicted before), `runtime_version` (`b11429`), `runs_of_this_setup`.
- `quality` (with `quality=`): `dataset`, `slice` (`fixed`, `full`), `chunks`, `perplexity`, `perplexity_error`, `runtime`, `is_the_original`, `kept` (against the reference, or `null`), `reference` (`label`, `perplexity`, `same_device`), `note` (why there is nothing to compare with).

### history

`speed_runs` and `quality_results`, newest first, as saved in `~/.cache/ocq/measurements.json` and `quality.json`. Their keys are the files' own and may grow; `date`, `model`, `label`, `runtime` and, for speed `tps`, for quality `ppl` are always there.

### doctor

`ready` (false when something blocks real work) and `checks`: `{name, status, detail, fix}`, status `ok`, `info`, `warn`, `fail`; `fix` is the command to type, or `null`.

### models

`context`, `kv_cache`, and `models`: `label`, `ref` (what the other functions take), `source` (`Ollama`, `LM Studio`, `Hugging Face cache`, `QuantPilot`, ...), `path`, `format`, `quantization`, `size_gb`, `complete`, `note`, `fit` (a format, as it is on disk, or `null`).
