# Makefile — convenience targets wrapping uv for container / non-uv environments
#
# If `uv` is available, commands run through `uv run`.
# Otherwise, they fall back to the active Python environment directly.

UV := $(shell command -v uv 2>/dev/null)

ifdef UV
  RUN := uv run
  SYNC := uv sync
else
  RUN :=
  SYNC := @echo "uv not found — skipping sync (using active Python environment)"
endif

.PHONY: help install install-dev lint format test test-unit test-functional pre-commit clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	$(SYNC)

install-dev: ## Install with dev + recipes extras
	$(SYNC) --group dev --group recipes

lint: ## Run ruff linter with auto-fix
	$(RUN) ruff check --fix .

format: ## Run ruff formatter
	$(RUN) ruff format .

check: lint format ## Run lint + format

test: test-unit ## Run all unit tests (alias)

test-unit: ## Run unit tests
	$(RUN) python -m pytest tests/unit_tests/ -x -q

test-functional: ## Run functional tests
	$(RUN) python -m pytest tests/functional_tests/ -x -q

test-k: ## Run tests matching a keyword (usage: make test-k K=test_llama)
	$(RUN) python -m pytest tests/ -k "$(K)" -x -q

pre-commit: ## Install pre-commit hooks
	$(RUN) pre-commit install

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .ruff_cache/
