---
name: expert-parallel-overlap
description: Operational guide for enabling MoE expert-parallel communication overlap in Megatron-Bridge, including config knobs, code anchors, pitfalls, and verification.
---

# MoE Expert-Parallel Overlap Skill

For stable background and recommendation level, see:

- `docs/training/communication-overlap.md`

## Enablement

Minimal Bridge override with plain `alltoall`:

```python
cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
cfg.comm_overlap.delay_wgrad_compute = False

cfg.model.expert_model_parallel_size = 8
cfg.model.num_moe_experts = 64
cfg.model.moe_token_dispatcher_type = "alltoall"
cfg.model.moe_shared_expert_overlap = False
cfg.model.bf16 = True
cfg.model.fp16 = False
```

Minimal Bridge override with DeepEP or HybridEP:

```python
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend

cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
cfg.comm_overlap.delay_wgrad_compute = True
cfg.model.moe_shared_expert_overlap = False

apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend="deepep")
# or: apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend="hybridep")
```

Required constraints:

- `expert_model_parallel_size > 1`
- `num_moe_experts > 1`
- `moe_token_dispatcher_type in {"alltoall", "flex"}`
- base precision is BF16 or FP16
- PyTorch `>= 2.6.0`
- if `PP > 1`, set `virtual_pipeline_model_parallel_size`

## Code Anchors

Bridge overlap validation:

```463:520:src/megatron/bridge/training/comm_overlap.py
if self.user_comm_overlap_cfg.overlap_moe_expert_parallel_comm is True:
    assert model_cfg.expert_model_parallel_size > 1, ...
    assert model_cfg.num_moe_experts > 1, ...
    assert model_cfg.moe_token_dispatcher_type in ["alltoall", "flex"], ...
    assert model_cfg.bf16 or model_cfg.fp16, ...
    assert is_torch_min_version("2.6.0"), ...
...
assert (
    model_cfg.overlap_moe_expert_parallel_comm
    or self.user_comm_overlap_cfg.overlap_moe_expert_parallel_comm
), "overlap_moe_expert_parallel_comm is required for delay_wgrad_compute"
```

Flex-dispatcher activation:

```27:69:src/megatron/bridge/training/flex_dispatcher_backend.py
def apply_flex_dispatcher_backend(...):
    ...
    model_config.moe_token_dispatcher_type = "flex"
    model_config.moe_flex_dispatcher_backend = moe_flex_dispatcher_backend
    model_config.moe_shared_expert_overlap = False
```

Perf harness overlap enablement:

```148:155:scripts/performance/utils/overrides.py
if moe_a2a_overlap:
    recipe.comm_overlap.overlap_moe_expert_parallel_comm = True
    recipe.comm_overlap.delay_wgrad_compute = True
    recipe.model.moe_shared_expert_overlap = False
```

## Pitfalls

1. `moe_flex_dispatcher_backend` is metadata unless the recipe also calls `apply_flex_dispatcher_backend(...)`.
2. `delay_wgrad_compute` is stricter than plain overlap and requires overlap first.
3. CUDA graph plus delayed wgrad needs extra TE and graph-scope constraints.
4. MoE overlap and shared-expert overlap are mutually exclusive.
5. If `PP > 1`, virtual pipeline parallelism is required for MoE overlap.
6. TP/CP overlap tuning can conflict with DeepEP or HybridEP launch tuning.

## Verification

Run the existing unit coverage for Bridge MoE overlap validation and DeepEP or HybridEP helper logic:

```bash
uv run python -m pytest \
  tests/unit_tests/training/test_comm_overlap.py \
  tests/unit_tests/training/test_deepep.py -q
```

Success criteria:

- Pytest reports `41 passed`
- `test_comm_overlap.py` covers MoE overlap and delayed-wgrad validation
- `test_deepep.py` covers DeepEP or HybridEP helper activation and GPU gating
