---
name: expert-parallel-overlap
description: Validate and use MoE expert-parallel communication overlap in Megatron-Bridge, including `overlap_moe_expert_parallel_comm`, `delay_wgrad_compute`, and `flex` dispatcher backends such as DeepEP and HybridEP.
---

# MoE Expert-Parallel Overlap

## References

- `docs/training/communication-overlap.md`
- `skills/perf-techniques/expert-parallel-overlap/card.yaml`
- `scripts/experiments/ep_overlap_qwen3_moe.py`

## What It Is

MoE expert-parallel overlap hides token dispatch or combine communication behind
expert compute through
`comm_overlap.overlap_moe_expert_parallel_comm`.
`comm_overlap.delay_wgrad_compute` is a stricter additive option that delays
expert weight-gradient work to create more overlap opportunity.

This is separate from choosing a dispatcher backend. If you want DeepEP or
HybridEP, the dispatcher must actually switch to `flex`. Setting only
`moe_flex_dispatcher_backend` is not enough.

## Quick Decision

| Situation | Recommendation | Why |
|---|---|---|
| First EP-overlap rollout on a working MoE recipe | Start with overlap only, not delayed wgrad | Smaller change surface and best current evidence. |
| You want DeepEP or HybridEP | Call `apply_flex_dispatcher_backend(...)` | That is what actually flips the dispatcher to `flex`. |
| `EP <= 4` with `alltoall` on `<= 2` nodes | Do not expect a speedup | Repo measurements show flat or worse performance. |
| `PP > 1` and no VPP | Stop and add VPP first | Bridge asserts on this combination. |
| Full recompute or shared-expert overlap must stay enabled | Do not use EP overlap | Unsupported combination. |
| You need benchmark-backed throughput guidance | Use the wrapper experiment script and compare against baseline | Unit tests validate logic, not speedup. |

## Enablement

1. Start from a working MoE run with `expert_model_parallel_size > 1`.
2. Ensure the recipe has a `CommOverlapConfig`. If it leaves `comm_overlap=None`,
   initialize it in code before applying overrides.
3. Set `overlap_moe_expert_parallel_comm=True`.
4. Leave `delay_wgrad_compute=False` for the first rollout unless you have a
   reason to benchmark it separately.
5. If using DeepEP or HybridEP, call `apply_flex_dispatcher_backend(...)` so the
   dispatcher actually becomes `flex`.

## Compatibility And Constraints

Bridge validation currently requires:

- `expert_model_parallel_size > 1`
- `num_moe_experts > 1`
- `moe_token_dispatcher_type in {"alltoall", "flex"}`
- model precision is BF16 or FP16
- PyTorch `>= 2.6.0`
- if `pipeline_model_parallel_size > 1`, set `virtual_pipeline_model_parallel_size`
- `recompute_granularity != "full"`
- `recompute_method is None`
- `recompute_num_layers is None`
- `moe_shared_expert_overlap is False`
- `mtp_num_layers is None or 1`

Additional delayed-wgrad constraints:

- `delay_wgrad_compute` requires EP overlap to be enabled
- if `overlap_grad_reduce=True` or `gradient_accumulation_fusion=True`, TE must be `>= 2.7.0`
- if CUDA graphs capture the affected wgrad path, TE must be `>= 2.12.0`
- CUDA graphs plus delayed wgrad also requires `gradient_accumulation_fusion=True`
- CUDA graphs plus delayed wgrad does not support attention bias today

## Minimal Working Config

### Plain `alltoall`

```python
from megatron.bridge.training.comm_overlap import CommOverlapConfig

cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
cfg.comm_overlap.delay_wgrad_compute = False

cfg.model.expert_model_parallel_size = 8
cfg.model.num_moe_experts = 64
cfg.model.moe_token_dispatcher_type = "alltoall"
cfg.model.moe_shared_expert_overlap = False
cfg.model.bf16 = True
cfg.model.fp16 = False
```

### DeepEP or HybridEP

```python
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend

cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
cfg.comm_overlap.delay_wgrad_compute = False
cfg.model.moe_shared_expert_overlap = False

apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend="deepep")
# or: apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend="hybridep")
```

## Minimal Runnable Command

Use the measured wrapper script when the recipe leaves `comm_overlap=None`:

```bash
uv run python scripts/experiments/ep_overlap_qwen3_moe.py \
  --model qwen3_30b \
  --variant overlap \
  model.tensor_model_parallel_size=2 \
  model.pipeline_model_parallel_size=2 \
  model.virtual_pipeline_model_parallel_size=4 \
  model.expert_model_parallel_size=4 \
  model.sequence_parallel=True \
  model.moe_token_dispatcher_type=alltoall \
  model.moe_shared_expert_overlap=False \
  train.train_iters=20 \
  train.global_batch_size=8 \
  train.micro_batch_size=1
```

Run the same command with `--variant baseline` and `--variant overlap-wgrad` to
get an apples-to-apples comparison.

## Verification

### Local logic coverage

```bash
uv run python -m pytest \
  tests/unit_tests/training/test_comm_overlap.py \
  tests/unit_tests/training/test_deepep.py -q
```

Success criteria:

- both test files pass
- `test_comm_overlap.py` covers EP overlap and delayed-wgrad validation logic
- `test_deepep.py` covers `flex` backend activation and GPU gating

### End-to-end activation checks

In benchmark logs, confirm:

- `overlap_moe_expert_parallel_comm: True`
- `delay_wgrad_compute: True` only for the delayed-wgrad variant
- overlap and baseline use the same non-EP settings

### Current measured evidence

- Qwen3-30B-A3B, EP=4, 2 nodes, `alltoall`: overlap is numerically safe at
  GBS=8 but not faster, and is about 13% slower at GBS=64.
- Qwen3-Next-80B-A3B, EP=8, 8 nodes, `alltoall`: overlap variants are stable,
  but `delay_wgrad_compute` is about 4.8% slower than overlap-only.

## Code Anchors

Bridge validation logic:

```465:522:src/megatron/bridge/training/comm_overlap.py
# MOE expert parallel comm overlap
if self.user_comm_overlap_cfg.overlap_moe_expert_parallel_comm is True:
    assert model_cfg.expert_model_parallel_size > 1, ...
    assert model_cfg.num_moe_experts > 1, ...
    assert model_cfg.moe_token_dispatcher_type in ["alltoall", "flex"], ...
    assert model_cfg.bf16 or model_cfg.fp16, ...
    assert is_torch_min_version("2.6.0"), ...
    ...
if self.user_comm_overlap_cfg.delay_wgrad_compute is True:
    ...
    assert (
        model_cfg.overlap_moe_expert_parallel_comm
        or self.user_comm_overlap_cfg.overlap_moe_expert_parallel_comm
    ), "overlap_moe_expert_parallel_comm is required for delay_wgrad_compute"
```

Delayed-wgrad CUDA-graph checks:

```524:556:src/megatron/bridge/training/comm_overlap.py
# CUDA graph scope-specific validations for delayed wgrad.
cuda_graph_scope = getattr(model_cfg, "cuda_graph_scope", []) or []
...
if wgrad_in_graph_scope:
    assert is_te_min_version("2.12.0"), ...
    assert model_cfg.gradient_accumulation_fusion, ...
    if attn_scope_enabled:
        assert not model_cfg.add_bias_linear and not model_cfg.add_qkv_bias, ...
```

Flex-dispatcher activation:

```27:72:src/megatron/bridge/training/flex_dispatcher_backend.py
def apply_flex_dispatcher_backend(
    model_config: TransformerConfig,
    moe_flex_dispatcher_backend: str | None = None,
) -> None:
    ...
    model_config.moe_token_dispatcher_type = "flex"
    model_config.moe_flex_dispatcher_backend = moe_flex_dispatcher_backend
    model_config.moe_shared_expert_overlap = False
```

Benchmark wrapper config injection:

```128:139:scripts/experiments/ep_overlap_qwen3_moe.py
cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
cfg.comm_overlap.delay_wgrad_compute = False

if args.variant in ("overlap", "overlap-wgrad"):
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute = args.variant == "overlap-wgrad"

cfg = process_config_with_overrides(cfg, cli_overrides=cli_overrides or None)
```

## Failure Diagnosis

- `AttributeError` or `NoneType` around `comm_overlap.*`:
  the recipe left `comm_overlap=None`. Initialize `CommOverlapConfig` in code or
  use `scripts/experiments/ep_overlap_qwen3_moe.py`.
- Assertion about `virtual_pipeline_model_parallel_size`:
  EP overlap with `PP > 1` needs VPP.
- Assertion about recompute or `moe_shared_expert_overlap`:
  disable full recompute, recompute method, recompute layers, and shared-expert
  overlap before enabling EP overlap.
- Overlap not actually using DeepEP or HybridEP:
  call `apply_flex_dispatcher_backend(...)`; setting only
  `moe_flex_dispatcher_backend` is just metadata.
- Overlap is slower than baseline:
  check whether the run is small-EP, `alltoall`, or has too few micro-batches.
  That is consistent with current measurements, not necessarily a bug.

## Known Limitations

- Repo measurements do not yet show a throughput win for `alltoall` EP overlap.
- `delay_wgrad_compute` has been slower in both measured workloads.
- The highest-confidence future win case is still unmeasured here: larger EP,
  more nodes, more micro-batches, and `flex` backends such as DeepEP or HybridEP.
- Public recipes are conservative and may expose the feature without enabling it
  by default.
