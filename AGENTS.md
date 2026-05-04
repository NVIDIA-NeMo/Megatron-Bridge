# AGENTS.md — Megatron Bridge

> **Project:** PyTorch-native bridge between Hugging Face and Megatron-Core.
> Bidirectional checkpoint conversion, pretraining, SFT, and LoRA recipes
> with optimized NVIDIA GPU performance. Package: `megatron.bridge` (Python 3.12).

## Skills

The `skills/` directory contains structured guides for common tasks (adding
models, running experiments, debugging multi-node jobs, performance tuning,
etc.). Read the relevant `SKILL.md` before starting a task it covers.

## Boundaries

**NEVER:**
- Modify files inside `3rdparty/Megatron-LM/` — changes go through the upstream repo
- Run the full test suite — run only the specific tests relevant to your change
- Add required (non-optional) dependencies — use optional extras; submit dependency changes as a separate PR
- Commit secrets, tokens, `.env` files, or environment-specific paths / account names (e.g. `/home/yuya/…`, usernames, cluster hostnames)
- Use bare `print()` — use `logging.getLogger(__name__)` or `print_rank_0()`

**ASK FIRST:**
- Before adding any new dependency to `pyproject.toml`
- Before modifying CI workflows (`.github/workflows/`)
- Before changing public API signatures in `models/conversion/`

**ALWAYS:**
- Run `uv run pre-commit run --all-files` before committing
- Add NVIDIA copyright headers to new Python files (except under `tests/`)
- Sign off commits: `git commit -s -m "message"`
- Use `uv run python -m pytest` and `uv run python -m torch.distributed.run`, not bare `pytest` / `torchrun`
- Use the current year (2026) in generated content — do not default to 2025 or any past year

## Toolchain

| Action | Command | Config |
|--------|---------|--------|
| Install deps | `uv sync` | `pyproject.toml`, `uv.lock` |
| Install dev tools | `uv sync --group dev` | |
| Lint + format | `uv run pre-commit run --all-files` | `ruff.toml`, `.pre-commit-config.yaml` |
| Unit tests | `uv run python -m pytest tests/unit_tests/` | `pyproject.toml [tool.pytest]` |
| Distributed run | `uv run python -m torch.distributed.run --nproc_per_node=2 script.py` | |
| Type check | `uv run mypy --strict path/to/file.py` | |
| Regen lock file | `uv lock` | |
| Init submodule | `git submodule update --init` | `.gitmodules` |

## Code Style

Lint and format are enforced by pre-commit hooks (ruff). See `ruff.toml` for
the authoritative rules. For judgment calls not covered by tooling, see
`skills/code-style/SKILL.md`. Key points the linter cannot catch:

- Type hints required on all public API functions (`X | None`, not `Optional[X]`)
- Google-style docstrings on public classes and functions
- Use `*` separator for functions with multiple same-type parameters
- No arbitrary defaults for config values — be explicit

## Contributing

See `CONTRIBUTING.md` for the full contributor guide, including:

- Commit and PR title format (`[{areas}] {type}: {description}`)
- PR labeling taxonomy (type, area, state, risk labels)
- Testing conventions (unit preferred, functional max 2 GPUs, L0/L1/L2 tiers)
- Dependency management policy
- DCO sign-off requirements
- CI triggering (`/ok to test <commit-SHA>`)

## Architecture

### Bridge Pattern

Each supported model family lives under `src/megatron/bridge/models/<family>/` with:

| File | Role |
|------|------|
| `bridge.py` | HF ↔ Megatron conversion logic |
| `config_mapping.py` | Maps HF config → Megatron config |
| `param_mapping.py` | Maps parameter names between formats |
| `hf_pretrained/` | HF model definition provider |

Recipes live in `src/megatron/bridge/recipes/<family>/`.

`AutoBridge` (`models/conversion/auto_bridge.py`) auto-selects the correct
bridge from a HF model name or path.

### Megatron-Core Submodule

`3rdparty/Megatron-LM/` is a git submodule pinned to a specific commit,
installed as an editable package via `[tool.uv.sources]` in `pyproject.toml`.
Use `scripts/switch_mcore.sh` to switch versions.

## Tool Compatibility

This file is the single source of agent instructions. For Claude Code
compatibility, create a symlink: `ln -s AGENTS.md CLAUDE.md`
