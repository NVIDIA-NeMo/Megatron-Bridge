# CUDA Graphs

## Overview

CUDA graphs capture a sequence of GPU operations once and replay them with
minimal host overhead, eliminating repeated kernel-launch and driver costs on
every training step. Megatron Bridge supports two capture implementations and
fine-grained scope selection to balance performance gain against memory cost.

## When to Use

CUDA graphs are most effective when:

- **Tensor shapes are static** across training steps (fixed sequence length,
  fixed micro-batch size). Variable-length sequences break graph replay
  assumptions.
- **Host overhead is significant** relative to GPU compute — smaller models
  or high step rates benefit most.
- **Memory budget allows it** — graph capture allocates static buffers,
  typically adding a few GB. Models with `PP > 1` can consume over 10 GB
  of additional memory.

## Implementations

| `cuda_graph_impl` | Mechanism | Scope support |
|---|---|---|
| `"local"` | MCore `CudaGraphManager` / `FullCudaGraphWrapper` | `full_iteration` (whole fwd+bwd) |
| `"transformer_engine"` | TE `make_graphed_callables()` per layer | `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, `mamba` |
| `"none"` (default) | Disabled | — |


### Local full-iteration graphs

Captures the entire forward-backward pass as one graph. Provides the highest
host-overhead reduction but requires disabling NaN checks and has the largest
memory footprint.

### Transformer Engine scoped graphs

Captures individual layer components (attention, MLP, MoE router, etc.)
through TE. More flexible, works with MoE models where only dense modules
can be graphed, and supports selective scope combinations.

## Configuration

```python
cfg.model.cuda_graph_impl = "transformer_engine"        # or "local"
cfg.model.cuda_graph_scope = ["attn", "moe_router"]     # scope list
cfg.model.cuda_graph_warmup_steps = 3                   # warmup before capture
cfg.rng.te_rng_tracker = True                           # required
```

### Key constraints

- `cfg.rng.te_rng_tracker` must be `True` when `cuda_graph_impl != "none"`.
- `full_iteration` scope requires `cuda_graph_impl = "local"` and
  `rerun_state_machine.check_for_nan_in_loss = False`.
- MoE models with token-dropless routing have limited graph support
  (dense modules only).
- `cuda_graph_impl = "none"` automatically clears `cuda_graph_scope`.

## MoE Considerations

MoE models often cannot graph the full expert dispatch path due to dynamic
token routing. Common practice:

- Graph `moe_router` and `moe_preprocess` (static portions).
- Add `attn` scope for the dense attention blocks.
- Leave expert dispatch in eager mode.

Do not combine `moe` scope with `moe_router` scope — they are mutually
exclusive.

## Memory Impact

CUDA graphs allocate static buffers that persist for the duration of training.
Expect a few GB of additional memory. With `PP > 1`, memory overhead can
exceed 10 GB due to pipeline-stage buffering. Plan activation memory
accordingly.

## Related

- Operational details: [skills/perf-techniques/cuda-graphs/SKILL.md](../../skills/perf-techniques/cuda-graphs/SKILL.md)
- Performance guide: [docs/performance-guide.md](../performance-guide.md)
- Communication overlap interaction: [docs/training/communication-overlap.md](communication-overlap.md)
