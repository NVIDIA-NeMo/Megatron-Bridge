# CPU Offloading

CPU offloading reduces per-GPU memory by moving data to host (CPU) memory
during training, trading throughput for the ability to train models or
configurations that would otherwise not fit in GPU memory.

For operational setup, code anchors, and verification commands, see
[skills/perf-techniques/cpu-offloading/SKILL.md](../../skills/perf-techniques/cpu-offloading/SKILL.md).

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
| Speed | `--` | high | Throughput penalty scales linearly with offload fraction. CPU Adam compute and D2H/H2D transfers add latency. Measured 1.9x–4.2x slower on Qwen3-30B-A3B. |
| Memory | `++` | high | Optimizer offload saves 3.8 GB per 25% of fraction on Qwen3-30B-A3B (up to 15.3 GB / 32% at 100%). Activation offload saves activation memory proportional to layers offloaded. |
| Scale | `+` | medium | Enables training configurations that would otherwise OOM. Can free memory for larger batch sizes or additional parallelism. |
| Convergence | `0` | high | All optimizer offload fractions (25–100%) produce identical loss within 0.001 across 20 iterations, validation, and test. |
| Stability | `0` | high | No errors, hangs, or NCCL issues across 120 total iterations tested (6 configurations). |

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

### Optimizer CPU offloading

```python
cfg.optimizer.optimizer_cpu_offload = True
cfg.optimizer.optimizer_offload_fraction = 0.5     # 0.0 to 1.0
cfg.optimizer.overlap_cpu_optimizer_d2h_h2d = True # overlap transfers with compute
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer_cpu_offload` | `False` | Master switch |
| `optimizer_offload_fraction` | `0.0` | Fraction of optimizer states on CPU (0.0–1.0) |
| `overlap_cpu_optimizer_d2h_h2d` | `False` | Overlap GPU↔CPU transfers with compute |
| `use_torch_optimizer_for_cpu_offload` | `False` | Use `torch.optim` instead of fused optimizer for CPU portion |

### Activation CPU offloading

```python
cfg.model.cpu_offloading = True
cfg.model.cpu_offloading_num_layers = 16
cfg.model.cpu_offloading_activations = True
cfg.model.cpu_offloading_weights = False
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cpu_offloading` | `False` | Master switch |
| `cpu_offloading_num_layers` | `0` | Number of transformer layers to offload (0 to num_layers-1) |
| `cpu_offloading_activations` | `True` | Offload activations |
| `cpu_offloading_weights` | `False` | Offload weights |
| `cpu_offloading_double_buffering` | `False` | Double-buffer across layers while reloading |

## Minimal Runnable Example

Optimizer offload with 50% fraction on Qwen3-30B-A3B pretrain:

```bash
uv run python scripts/training/run_recipe.py \
  --recipe qwen3_30b_a3b_pretrain_config \
  optimizer.optimizer_cpu_offload=True \
  optimizer.optimizer_offload_fraction=0.5 \
  train.train_iters=20 \
  train.global_batch_size=8 \
  train.micro_batch_size=1
```

## Expected Metric Changes

| Metric | Direction | Magnitude | Conditions | Evidence |
|--------|-----------|-----------|------------|----------|
| Steady-state memory (allocated) | down | 3.8 GB per 25% of optimizer offload fraction | Qwen3-30B-A3B, TP2 PP2 EP4, 2 nodes H100 | measured |
| Step time | up | 1.9x at 25%, 2.5x at 50%, 3.2x at 75%, 4.2x at 100% offload | Same config | measured |
| Step time with D2H/H2D overlap | up | 3.9x at 100% offload (vs 4.2x without overlap) | Same config + `overlap_cpu_optimizer_d2h_h2d=True` | measured |
| Loss | neutral | max delta < 0.001 across all fractions | Same config, 20 iterations | measured |

Memory savings and throughput penalty both scale linearly with
`optimizer_offload_fraction`. The D2H/H2D overlap provides ~7% speedup at
100% because CPU-side Adam compute — not the data transfers — is the
dominant bottleneck.

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

- [Activation Recomputation](activation-recomputation.md)
- [Megatron FSDP](megatron-fsdp.md)
- [Optimizer & Scheduler](optimizer-scheduler.md)
- [CUDA Graphs](cuda-graphs.md)
- [skills/perf-techniques/cpu-offloading/SKILL.md](../../skills/perf-techniques/cpu-offloading/SKILL.md)
