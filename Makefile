.PHONY: sync test test-slow test-gpu lint fmt check

sync:
	uv sync --group dev

test:
	uv run pytest -q

test-slow:
	uv run pytest -m slow -q

# Every format for real (the NVIDIA ones need an NVIDIA GPU), into one report in results/.
test-gpu:
	uv sync --group dev --all-extras
	uv run python scripts/gpu_report.py

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .

check: lint test
