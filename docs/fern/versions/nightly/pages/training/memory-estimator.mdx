# Theoretical Memory Estimator

Megatron Bridge includes a formula-based training memory estimator for GPT-like
model providers. It is intended for early configuration planning and for the
training-time theoretical memory report printed alongside CUDA memory metrics.

The estimator is available as a Python API:

```python
from megatron.bridge.training.utils.theoretical_memory_utils import (
    estimate_training_memory,
    format_training_memory_estimate,
)

estimate = estimate_training_memory(cfg, num_microbatches=num_microbatches)
print(format_training_memory_estimate(estimate, unit="GB"))
```

`estimate_training_memory` returns a structured result with:

- dense and embedding parameter memory on the most-loaded GPU shard
- routed MoE expert parameter memory when `num_moe_experts` is set
- activation memory when requested
- total global parameter count covered by the estimator
- assumptions attached to the estimate

The training loop continues to use the same report hook:

```text
Theoretical memory footprints: weight and optimizer=..., activation=..., total=...
```

## What It Covers

The model-state estimate covers weights, gradients, FP32 master weights, and
Adam optimizer states. It uses the same byte model as the legacy utility:

- `18` bytes per parameter when the distributed optimizer is disabled
- `6 + 12 / shard_size` bytes per parameter when the distributed optimizer is enabled

For dense parameters, `shard_size` is `data_parallel_size * context_parallel_size`.
For routed MoE experts, expert parameters are divided by
`expert_model_parallel_size * expert_tensor_parallel_size`, and optimizer state
uses the expert data-parallel shard size, including context parallel ranks.

The activation estimate follows the existing Megatron-LM theoretical formula and
adds Bridge-aware accounting for MoE active expert width and context parallel
partitioning. It is most meaningful when sequence parallelism and selective
activation recomputation are enabled, which is also the condition used by the
training-time report.

## Assumptions

The estimate reports the most-loaded GPU shard. Embeddings are assigned to the
first and last pipeline stages; untied input/output embeddings both land on the
same GPU only when `pipeline_model_parallel_size == 1`.

The estimator does not model runtime allocator fragmentation, CUDA kernel
workspace, NCCL buffers, CUDA graph static buffers, token-routing imbalance,
dispatcher workspace, CPU offloading, or full activation recomputation. Use
profiling and CUDA memory reports to validate final launch configurations.

## MoE Notes

MoE layer count follows `moe_layer_freq`, including list patterns. Routed expert
parameters are counted separately from dense attention, dense MLP, shared expert,
normalization, and embedding parameters. This keeps expert parallelism effects
visible in the returned component breakdown.

For memory tuning decisions, use this estimator as a planning signal together
with the runtime guidance in [CPU Offloading](cpu-offloading.md),
[Activation Recomputation](activation-recomputation.md), and
[Megatron FSDP](megatron-fsdp.md).
