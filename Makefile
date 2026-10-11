.PHONY: sync test test-slow lint fmt check

sync:
	uv sync --group dev

test:
	uv run pytest -q

test-slow:
	uv run pytest -m slow -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

check: lint test
