# Label Taxonomy

From `CONTRIBUTING.md`. Apply exactly one **type** and one **area** label per issue.

## Type Labels (one per issue)

| Label | Use for |
|---|---|
| `bug` | Incorrect behavior, regressions, broken workflows |
| `feature` | New capabilities, enhancements, enablement |
| `support` | Questions, help requests, guidance gaps |
| `docs` | Documentation-only updates or debt |
| `ci` | CI, automation, test queue, workflow infra |

## Area Labels (one primary per item)

| Label | Scope |
|---|---|
| `area:model` | Model implementations and HF bridge logic |
| `area:recipe` | Training recipes and launch configs |
| `area:training` | Training loop, callbacks, runtime integration |
| `area:data` | Dataset builders, preprocessing, samplers |
| `area:ckpt` | Checkpoint conversion, loading, export, save |
| `area:peft` | PEFT methods (LoRA, adapters), adapter export |
| `area:perf` | Performance optimizations, kernel integration, throughput |
| `area:distill` | Knowledge distillation |
| `area:diffusion` | Diffusion model implementations and training |
| `area:prune` | Pruning and sparsity |
| `area:quant` | Quantization (PTQ, QAT, FP8 recipes) |
| `area:build` | Dependencies, packaging, images, env setup |
| `area:misc` | Cross-cutting utilities, logging, helpers |

## State Labels (at most one primary, see exceptions below)

| Label | Meaning |
|---|---|
| `needs-triage` | New, needs classification and ownership |
| `needs-review` | PR is ready for code review |
| `needs-author` | Author action required |
| `needs-follow-up` | Initial triage done, needs further follow-up |
| `blocked` | External dependency not cleared |
| `ready-to-merge` | Approved, CI-only gate |

**Allowed combinations:** `needs-author` + `needs-follow-up` can co-exist (e.g., waiting on author but oncall should keep tracking).

## Risk Labels (apply when relevant)

| Label | Meaning |
|---|---|
| `breaking-change` | Public API or behavior changes |
| `high-complexity` | Conflict-prone, needs extra test coverage |
| `needs-more-tests` | Requires additional test coverage |

## Orthogonal Labels (leave as-is)

- Release labels (e.g. `r0.3.0`)
- `community-request`
- Partner labels (`x-*`)

## File-to-Area Mapping (for PRs)

| Path prefix | Area |
|---|---|
| `src/megatron/bridge/models/` | `area:model` |
| `src/megatron/bridge/recipes/` | `area:recipe` |
| `src/megatron/bridge/training/` | `area:training` |
| `src/megatron/bridge/data/` | `area:data` |
| `src/megatron/bridge/models/conversion/`, `models/hf_pretrained/` | `area:ckpt` |
| `src/megatron/bridge/peft/` | `area:peft` |
| `scripts/performance/` | `area:perf` |
| `src/megatron/bridge/distill/` | `area:distill` |
| `src/megatron/bridge/diffusion/` | `area:diffusion` |
| `src/megatron/bridge/prune/` | `area:prune` |
| `src/megatron/bridge/quant/` | `area:quant` |
| `pyproject.toml`, `uv.lock`, `docker/`, `Dockerfile*` | `area:build` |
| Everything else | `area:misc` |

## Issue Template Auto-Labels

Issue templates auto-apply type + `needs-triage`:
- Bug report → `bug` + `needs-triage`
- Feature request → `feature` + `needs-triage`
- Support request → `support` + `needs-triage`
- Model support request → `feature` + `area:model` + `needs-triage`

For templated issues, read the **Affected area** dropdown to apply the `area:*` label.
