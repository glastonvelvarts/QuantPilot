.PHONY: sync test lint fmt check

sync:
	uv sync --group dev

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

check: lint test
