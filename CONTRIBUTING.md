# Contributing to QuantPilot

Thanks for helping. This page covers setting up, testing, the way QuantPilot is written, and making a release.

## Set up

```sh
git clone https://github.com/glastonvelvarts/QuantPilot && cd QuantPilot
uv sync --group dev --extra gguf      # add --extra awq / gptq / bnb / nvfp4 on an NVIDIA PC
uv run ocq doctor
```

## Test

```sh
make check                 # lint, format check and the fast tests: run before every pull request
uv run pytest -m slow      # real quantizations of SmolLM2-135M with each installed backend
make test-gpu              # on an NVIDIA PC: every backend for real, into one report in results/
```

- The fast tests never use the network, your models or your settings: `tests/tests/conftest.py` gives each test its own caches. Model sizes are tested against real files: `tests/tests/data/` holds the `config.json`, safetensors headers (no weights) and Hugging Face counts of real repos, and llama.cpp's own logs and reports. A new formula comes with the real data that checks it.
- GitHub runs the fast tests on Linux, macOS and Windows with Python 3.10 and 3.13 on every push and pull request, checks that the package installs with `uv tool install` and `uvx`, and runs the real GGUF test on all three every night.
- AWQ, GPTQ, NVFP4 and bitsandbytes need an NVIDIA GPU, so they run on a PC with one: `make test-gpu` writes a report that starts with what passed and what could not run there.

## How QuantPilot is written

- **Measured, not guessed.** A number comes from the model's files, from a measurement on the PC, or from a published measurement that is named in the code. When it is an estimate, it says so, and the confidence goes down.
- **Plain words.** Messages say what happened and what to do, without jargon; the app shows no command-line flags. A message from the core can end with a command for the CLI (`Run: ...`): the app leaves it out.
- **One engine.** The CLI, the app and `ocq.api` call the same functions; nothing is worked out twice.
- **The API is a promise.** A key of `ocq.api` (and `--json`) is never renamed or removed within a `schema_version`; `tests/tests/test_api.py` holds the keys. Add keys there and in `docs/api.md`.

```
src/ocq/
  cli.py              the commands
  api.py              the same, as data (--json)
  tui/                the full-screen app (Textual)
  recommend.py        candidates, hard limits, ranking, reasons, confidence
  model_analyzer.py   reading a model (Hub, folder, GGUF, Ollama) and sizing each format
  model_size.py       weight and context memory
  gguf_file.py        GGUF headers
  runtimes.py         how llama.cpp, vLLM and transformers use memory
  hardware.py         this PC (with llmfit, or QuantPilot's own probe)
  measurements.py     the speed history and speed predictions
  bench.py            the standard benchmark (llama-bench)
  quality.py          perplexity on WikiText-2, and what a format keeps
  verify.py           llama.cpp's own memory against the prediction
  pipeline.py         ocq optimize
  quantizers/         GGUF, AWQ, GPTQ, bitsandbytes, NVFP4
  discovery.py        the models already on the PC
  installation.py     how QuantPilot was installed, and adding a tool the same way
  doctor.py           ocq doctor
```

## The pictures in the README

`assets/` is made from the real app on a PC, with real models: run it again when the app changes.

```sh
uv run --with pillow python scripts/screenshots.py --gguf <a .gguf on this PC>
uv run --with pillow python scripts/screenshots.py --banner-only   # only the banner
uv run --with pillow python scripts/screenshots.py --own-model     # only + Add your model (own-model.gif)
```

The own-model pictures are made in a demo home folder holding only a copy of SmolLM2 as `my-finetune` (your own folders stay out of the picture, your history untouched). It needs Chrome (or `rsvg-convert`). The banner is 1280 x 640: also set it as the repository's social preview (Settings, General, Social preview).

## Making a release

Once, on pypi.org: add a trusted publisher for the project `quantpilot` (Your account, Publishing, Add a new pending publisher): owner `glastonvelvarts`, repository `QuantPilot`, workflow `release.yml`, environment `pypi`. On GitHub, create the environment `pypi` (Settings, Environments); a required reviewer there makes each release wait for a click.

Each release:

1. Set the version in `pyproject.toml` (`uv version 0.2.0`), and move the changelog's **Unreleased** notes under `## [0.2.0] - <date>`, with the link at the end.
2. Commit, merge to `main`, then tag it: `git tag v0.2.0 && git push origin v0.2.0`.
3. The `release` workflow checks that the tag, the version and the changelog agree, runs the tests, builds and test-installs the package, publishes it to PyPI and makes the GitHub release with the changelog's notes.

`uv tool install quantpilot` and `uvx quantpilot` then get the new version (`uv tool upgrade quantpilot` for an installed one).
