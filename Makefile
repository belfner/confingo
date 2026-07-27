.PHONY: clean build publish publish-test install dev test lint format typecheck

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:
	uv build

publish: clean build
	@set -a && . ./.env && set +a && uv publish

publish-test: clean build
	@set -a && . ./.env && set +a && uv publish \
		--publish-url https://test.pypi.org/legacy/ \
		--token "$$UV_PUBLISH_TOKEN_TESTPYPI"

install:
	uv pip install -e .

dev:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyrefly check
