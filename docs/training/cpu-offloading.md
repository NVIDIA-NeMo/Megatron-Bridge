# CPU Offloading

CPU offloading reduces per-GPU memory by moving data to host (CPU) memory
during training, trading throughput for the ability to train models or
configurations that would otherwise not fit in GPU memory.

For operational setup, code anchors, and verification commands, see
[skills/perf-cpu-offloading/SKILL.md](../skills/perf-cpu-offloading/SKILL.md).

## What It Is

Megatron Bridge supports two independent CPU offloading mechanisms:

| Mechanism | What gets offloaded | Implementation |
|---|---|---|
| **Activation offloading** | Activations (and optionally weights) per transformer layer | MCore `cpu_offloading_context` in transformer block |
| **Optimizer offloading** | Optimizer states (Adam momentum + variance) | MCore `HybridDeviceOptimizer` with configurable GPU/CPU split |

Activation offloading moves layer activations to CPU during forward and
reloads them during backward. Optimizer offloading keeps a configurable
fraction of Adam optimizer states on CPU and runs the optimizer step there.

These are independent features addressing different memory pools. They can
be used separately but not always together due to constraint conflicts.

## What Problem It Solves

Large models, especially MoE architectures, can exhaust GPU memory even with
standard parallelism techniques (TP, PP, EP). The two offloading mechanisms
target different bottlenecks:

- **Activation offloading** helps when activation memory dominates — common
  with long sequences, large batch sizes, or when recomputation is disabled.
- **Optimizer offloading** helps when optimizer state memory dominates — Adam
  keeps two state tensors (momentum + variance) per parameter, doubling the
  parameter memory footprint. For a 30B MoE model this can be 15+ GB per GPU.

## Impacted Training Dimensions

| Dimension | Effect | Confidence | Rationale |
|-----------|--------|------------|-----------|
| Speed | 1.9x–4.2x slower step time (scales linearly with offload fraction) | high | CPU Adam compute and D2H/H2D transfers add latency. Measured on Qwen3-30B-A3B TP2 PP2 EP4. D2H/H2D overlap reduces 100% penalty from 4.2x to 3.9x. |
| Memory | 3.8 GB saved per 25% of optimizer offload fraction (up to 15.3 GB / 32% at 100%) | high | Measured on Qwen3-30B-A3B (47.2 GB baseline). Activation offload saves proportional to layers offloaded. |
| Scale | enables otherwise-OOM configurations | medium | Can free memory for larger batch sizes or additional parallelism. |
| Convergence | no change (loss delta < 0.001 across all fractions) | high | All optimizer offload fractions (25–100%) produce identical loss across 20 iterations. |
| Stability | no issues observed | high | No errors, hangs, or NCCL issues across 120 total iterations tested (6 configurations). |

D2H (device-to-host) and H2D (host-to-device) refer to data transfers between
GPU and CPU memory. Each optimizer step copies gradients to CPU (D2H), runs
Adam on CPU, then copies updated parameters back (H2D). The
`overlap_cpu_optimizer_d2h_h2d` flag overlaps these transfers with compute.
On Qwen3-30B-A3B MoE this provided only ~7% speedup because CPU-side Adam
compute — not the transfers — was the dominant bottleneck. Other models with
different parameter counts or optimizer configurations may see different
transfer-to-compute ratios.

## When to Use It

- GPU memory is tight and throughput regression is acceptable
- The model requires PP > 1 to fit — use **optimizer offloading** (activation
  offloading requires PP=1)
- You want a tunable memory-speed tradeoff via `optimizer_offload_fraction`
- Activation memory is the bottleneck and the model fits with PP=1 and no
  recompute — use **activation offloading**

## When Not to Use It

- Throughput is the primary concern — offloading always adds overhead
- The model already fits comfortably in GPU memory
- CUDA graphs are enabled — activation offloading is incompatible
- The model is large (30B+ MoE) and requires PP > 1 — activation offloading
  is blocked by the PP=1 constraint
- Alternative memory techniques (FSDP, activation recomputation) provide
  sufficient savings without the throughput penalty

## Feature Interactions

| Feature | Interaction | Details |
|---------|-------------|---------|
| Pipeline parallelism (PP > 1) | **Blocks** activation offloading | Hard MCore constraint. Use optimizer offloading instead. |
| Activation recomputation | **Blocks** activation offloading | Hard MCore constraint. Cannot combine. |
| CUDA graphs | **Blocks** activation offloading | Hard MCore constraint. Optimizer offloading is unaffected. |
| Fine-grained activation offloading | **Mutual exclusion** with layer-level activation offloading | Use one or the other. Fine-grained works with PP > 1. |
| Distributed optimizer | **Required** for optimizer offloading | `use_distributed_optimizer=True` (default in most recipes). |
| Megatron FSDP | Alternative | Shards parameters across DP ranks. Different tradeoff profile. |
| Expert parallelism | Compatible | Both offloading mechanisms work with EP. |

## Bridge Configuration

CPU offloading is configured through two independent config namespaces:

- **Optimizer offloading**: `optimizer.optimizer_cpu_offload`,
  `optimizer.optimizer_offload_fraction`, and
  `optimizer.overlap_cpu_optimizer_d2h_h2d`
- **Activation offloading**: `model.cpu_offloading`,
  `model.cpu_offloading_num_layers`, and related `model.cpu_offloading_*` fields

For config examples, parameter tables, and runnable commands, see
[skills/perf-cpu-offloading/SKILL.md](../skills/perf-cpu-offloading/SKILL.md).

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Currently there is no support for Pipeline parallelism with CPU offloading` | Activation offload with PP > 1 | Set PP=1 or switch to optimizer offloading |
| `CPU offloading does not work when activation recomputation is enabled` | Activation offload with recompute enabled | Set `recompute_granularity=null` |
| `CUDA graphs not supported with CPU offloading` | Activation offload with CUDA graphs | Set `cuda_graph_impl="none"` |
| `fine_grained_activation_offloading cannot be enabled with cpu_offloading` | Both offloading types enabled | Use one or the other |
| OOM with activation offloading on large model | Model too large for PP=1 | Switch to optimizer offloading (works with PP > 1) |
| >4x throughput regression | 100% optimizer offload, CPU Adam bottleneck | Reduce fraction or enable `overlap_cpu_optimizer_d2h_h2d` |

## Related Docs

- [docs/training/activation-recomputation.md](activation-recomputation.md)
- [docs/training/megatron-fsdp.md](megatron-fsdp.md)
- [docs/training/optimizer-scheduler.md](optimizer-scheduler.md)
- [docs/training/cuda-graphs.md](cuda-graphs.md)
- [skills/perf-cpu-offloading/SKILL.md](../skills/perf-cpu-offloading/SKILL.md)
