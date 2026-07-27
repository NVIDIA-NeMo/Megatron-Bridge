---
name: nemo-mbridge-perf-cpu-offloading
description: Validate and use CPU offloading in Megatron Bridge, including layer-level and fine-grained activation offloading plus fractional optimizer state offloading with HybridDeviceOptimizer. Use for GPU-memory reduction, CPU-offload OOM or crash investigation, cpu_offloading, optimizer_cpu_offload, optimizer_offload_fraction, HybridDeviceOptimizer, or moving optimizer state to CPU.
license: Apache-2.0
---

# CPU Offloading

## References

- Stable docs: @docs/training/cpu-offloading.md
- Structured metadata: @skills/nemo-mbridge-perf-cpu-offloading/card.yaml

## What It Is

Three mutually composable or exclusive mechanisms, depending on the row below,
move data from GPU to CPU memory:

| Mechanism | Config namespace | What gets offloaded | PP restriction |
|---|---|---|---|
| Activation offloading | `model.cpu_offloading*` | Activations (and optionally weights) per transformer layer | PP must be 1 |
| Fine-grained activation offloading | `model.fine_grained_activation_offloading` | Saved activations at selected submodule boundaries | Supports PP > 1 |
| Optimizer offloading | `optimizer.optimizer_cpu_offload` | Adam optimizer states (momentum + variance) via `HybridDeviceOptimizer` | None |

## Quick Decision

| Situation | Recommendation |
|---|---|
| Large MoE model (30B+), needs PP > 1 | Optimizer offloading — activation offloading is blocked by PP=1 |
| Small/medium model, PP=1 fits, activation memory dominates | Activation offloading |
| Need only a few GiB while preserving a larger MBS/topology | Fine-grained offloading of the largest saved activations; start with a small fraction |
| Want tunable memory-speed tradeoff | Optimizer offloading with fractional `optimizer_offload_fraction` |
| Throughput is top priority | Don't enable — offloading always adds overhead |
| CUDA graphs are needed | Only optimizer offloading — activation offloading is incompatible |
| Memory pressure is moderate | Optimizer offload at 25–50% fraction for best efficiency |

## Enablement

### Optimizer CPU offloading (recommended for large models)

```python
cfg.optimizer.optimizer_cpu_offload = True
cfg.optimizer.optimizer_offload_fraction = 1.0
cfg.optimizer.overlap_cpu_optimizer_d2h_h2d = True
```

CLI overrides:

```bash
optimizer.optimizer_cpu_offload=True \
optimizer.optimizer_offload_fraction=0.5 \
optimizer.overlap_cpu_optimizer_d2h_h2d=True
```

### Activation CPU offloading (small/medium models only)

```python
cfg.model.cpu_offloading = True
cfg.model.cpu_offloading_num_layers = 16
cfg.model.cpu_offloading_activations = True
cfg.model.cpu_offloading_weights = False

cfg.model.pipeline_model_parallel_size = 1
cfg.model.recompute_granularity = None
cfg.model.cuda_graph_impl = "none"
```

### Fine-grained activation offloading

```python
cfg.model.fine_grained_activation_offloading = True
cfg.model.offload_modules = ["expert_fc1"]
cfg.model.min_offloaded_tensor_size = 1_048_576
cfg.model.activation_offload_fraction = 0.2
```

The first training iteration is discovery warmup and offloads every eligible
group. After warmup, the fraction is applied separately to each cached
microbatch/chunk after size, PP-rank, and last-group filters. The implementation
keeps the first fraction of groups in forward execution order offloaded. Plan
enough host memory and exclude the warmup iteration from performance results.

`offload_modules` names activate boundaries wired into the corresponding MCore
module forwards. A custom `forward` replacement that bypasses those
`off_interface(...)` scopes must recreate the group-start/group-offload
boundaries and preserve non-offload markers on parameter-derived views.
Saved-tensor hooks may see a transpose/view as a plain tensor rather than an
`nn.Parameter`; mark the exact operand passed to the custom autograd operation.
Setting the config alone cannot offload activations that the replacement never
registers.

### Bandwidth gate before allocation

Before using fine-grained offload to unlock a larger MBS, estimate the minimum
transfer volume:

1. Measure the HBM shortfall at the same optimizer-state phase.
2. Divide by bytes released per eligible group to get the minimum groups that
   must remain offloaded after warmup.
3. Multiply group bytes by groups, microbatches per optimizer step, and two for
   D2H plus H2D.
4. Divide by the sustained host-link bandwidth and compare that ideal lower
   bound with the step-time reduction required by the performance gate.

Reject the candidate before allocation if even the ideal transfer lower bound
exceeds the available step-time margin. Async streams may hide some copies,
but they cannot create host-link bandwidth, and forward D2H plus backward H2D
do not become free merely because they use separate streams.

For a 16-H100 Qwen3.5-35B-A3B TP1/EP16 example, measured MBS1 and prior MBS2
memory phases implied roughly 5--6 GiB of steady relief. A static MBS2
`expert_fc1` input was about 282 MiB, requiring roughly 20 of 40 layer groups.
Across 32 microbatches that is about 180 GiB D2H plus 180 GiB H2D per GPU per
step. At an ideal 60 GB/s PCIe rate, the approximately 6-second transfer lower
bound already exceeded the 1.84-second improvement budget to reach the
287.305-TFLOP/s/GPU gate. The queued job was canceled before allocation:
offload could make the shape fit, but could not meet that throughput target.

## Config Parameter Reference

### Optimizer offloading

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer_cpu_offload` | `False` | Master switch |
| `optimizer_offload_fraction` | `0.0` | Fraction of optimizer states on CPU (0.0–1.0) |
| `overlap_cpu_optimizer_d2h_h2d` | `False` | Overlap GPU↔CPU transfers with compute |
| `use_torch_optimizer_for_cpu_offload` | `False` | Use `torch.optim` instead of fused optimizer for CPU portion |

### Activation offloading

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cpu_offloading` | `False` | Master switch |
| `cpu_offloading_num_layers` | `0` | Number of transformer layers to offload (0 to num_layers-1) |
| `cpu_offloading_activations` | `True` | Offload activations |
| `cpu_offloading_weights` | `False` | Offload weights |
| `cpu_offloading_double_buffering` | `False` | Double-buffer across layers while reloading |

### Fine-grained activation offloading

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fine_grained_activation_offloading` | `False` | Enable submodule-level saved-activation offload |
| `offload_modules` | `None` | Boundaries such as `core_attn`, `attn_proj`, `expert_fc1`, or `moe_act` |
| `min_offloaded_tensor_size` | `1_048_576` | Skip smaller saved tensors, in elements |
| `activation_offload_fraction` | `1.0` | Fraction of eligible groups per cached microbatch/chunk after warmup |
| `delta_offload_bytes_across_pp_ranks` | `0` | Keep progressively more bytes on higher PP ranks |

## Compatibility And Constraints

### Activation offloading

- `pipeline_model_parallel_size` must be 1
- `recompute_granularity` must be `None`
- Cannot combine with `fine_grained_activation_offloading`
- Cannot combine with CUDA graphs
- `cpu_offloading_num_layers` must be in `[0, num_layers-1)`

### Optimizer offloading

- Requires `use_distributed_optimizer = True` (default in most recipes)
- No PP, recompute, or CUDA graph restrictions
- `optimizer_offload_fraction` must be in `[0.0, 1.0]`

### Fine-grained activation offloading

- Cannot combine with layer-level `cpu_offloading`
- `offload_modules` must name boundaries implemented by the active module forward
- The first discovery iteration offloads all eligible groups before
  `activation_offload_fraction` is applied
- Expert `expert_fc1`/`moe_act` boundaries work with PP=1 or PP>1 and can be
  used without CUDA graphs
- CUDA-graph combinations have additional implementation, scope, and version
  constraints; see the MCore fine-grained activation-offloading guide

### Practical: large MoE models

Activation offloading is blocked for Qwen3-30B-A3B and similar large MoE
models. The PP=1 constraint means each GPU holds all 48 layers; model
weights + optimizer states alone (~70 GB) exceed H100 80 GB capacity.

## Minimal Runnable Command

```bash
uv run python scripts/training/run_recipe.py \
  --recipe qwen3_30b_a3b_pretrain_config \
  optimizer.optimizer_cpu_offload=True \
  optimizer.optimizer_offload_fraction=0.5 \
  train.train_iters=20 \
  train.global_batch_size=8 \
  train.micro_batch_size=1
```

## Verification

### Unit tests

```bash
uv run python -m pytest \
  tests/unit_tests/models/test_gpt_full_te_layer_autocast_spec.py -k "cpu_offload" \
  tests/unit_tests/peft/test_utils.py -k "cpu_offload" -q
```

### Success criteria

- Config validation passes for the selected offloading mode
- Training completes without OOM or NCCL errors
- Loss matches the non-offloaded baseline (max delta < 0.001)
- Memory usage drops proportionally to offload fraction

## Code Anchors

### MCore activation offload constraints

```1296:1310:3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
        if self.cpu_offloading and (
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(...)

        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                "Currently there is no support for Pipeline parallelism with CPU offloading"
            )

        if self.cpu_offloading and self.recompute_granularity is not None:
            raise ValueError(
                "CPU offloading does not work when activation recomputation is enabled"
            )
```

### MCore CUDA graph incompatibility

```1943:1944:3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
            if self.cpu_offloading:
                raise ValueError("CUDA graphs not supported with CPU offloading.")
```

### MCore fine-grained offloading mutual exclusion

```1427:1430:3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
        if self.fine_grained_activation_offloading:
            assert (
                not self.cpu_offloading
            ), "fine_grained_activation_offloading cannot be enabled with cpu_offloading."
```

### MCore warmup and fraction semantics

```566:632:3rdparty/Megatron-LM/megatron/core/pipeline_parallel/fine_grained_activation_offload.py
        for chunk in self._cached_chunks_forward:
            chunk.is_warmup = False
        # ...
        for chunk in self._cached_chunks_backward:
            eligible_offload_groups = [
                group for group in chunk.offload_groups
                if group.offload and group.total_offload_bytes > 0
            ]
            disabled_groups_count = int(
                len(eligible_offload_groups) * (1 - self._activation_offload_fraction)
            )
            for group in reversed(eligible_offload_groups):
                # Disable later forward groups first.
```

```1085:1093:3rdparty/Megatron-LM/megatron/core/pipeline_parallel/fine_grained_activation_offload.py
        if self.is_warmup:
            return True
        if not group.offload:
            return False
```

### MCore HybridDeviceOptimizer instantiation

```480:518:3rdparty/Megatron-LM/megatron/core/optimizer/__init__.py
        if config.optimizer_cpu_offload:
            # ... setup cpu/gpu optimizer classes ...
            optimizer = HybridDeviceOptimizer(
                param_groups,
                offload_fraction=config.optimizer_offload_fraction,
                cpu_optimizer_cls=cpu_optimizer_cls,
                gpu_optimizer_cls=gpu_optimizer_cls,
                overlap_cpu_optimizer_d2h_h2d=config.overlap_cpu_optimizer_d2h_h2d,
                pin_cpu_grads=config.pin_cpu_grads,
                pin_cpu_params=config.pin_cpu_params,
            )
```

### Bridge CUDA graph guard

```232:234:src/megatron/bridge/models/gpt_full_te_layer_autocast_spec.py
        assert not config.cpu_offloading and config.recompute_granularity is None, "Cudagraphs not supported"
```

### Bridge activation offloading in PEFT

```621:631:src/megatron/bridge/peft/utils.py
        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = self.linear_in(x)
        x = self.activation(x)
        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = self.linear_out(x)
```

## Failure Diagnosis

| Symptom | Likely Cause | How To Confirm | Fix |
|---|---|---|---|
| `Currently there is no support for Pipeline parallelism with CPU offloading` | Activation offload + PP > 1 | Check `pipeline_model_parallel_size` | Set PP=1 or use optimizer offloading |
| `CPU offloading does not work when activation recomputation is enabled` | Activation offload + recompute | Check `recompute_granularity` | Set `recompute_granularity=null` |
| `fine_grained_activation_offloading cannot be enabled with cpu_offloading` | Both offloading modes enabled | Check both flags | Use one or the other |
| `CUDA graphs not supported with CPU offloading` | CUDA graphs + activation offload | Check `cuda_graph_impl` | Set `cuda_graph_impl="none"` |
| OOM with activation offloading | Model too large for PP=1 | Check allocated memory vs 80 GB | Use optimizer offloading with PP > 1 |
| Extreme slowdown (>4x) | 100% optimizer offload, CPU Adam bottleneck | Compare iter time at different fractions | Reduce fraction or enable `overlap_cpu_optimizer_d2h_h2d` |
| OOM at partial optimizer offload | Insufficient offload for this config | Check memory at different fractions | Increase fraction or add PP |
| Fine-grained offload has no memory effect under a custom module | Replacement `forward` bypassed MCore's offload boundaries | Confirm `off_interface` group start/commit calls execute | Restore or recreate the exact boundary scopes |
| Fine-grained custom kernel copies weights to CPU | A Parameter transpose/view lost the base parameter's non-offload marker | Inspect the warmup summary/profile and saved operands | Mark the exact parameter-derived tensor operand as non-offloadable |
| First fine-grained-offload step is much slower than steady state | Discovery warmup offloads all eligible groups | Check the post-warmup offload summary | Exclude warmup; reduce module set or tensor eligibility if warmup itself is infeasible |
| Larger MBS may fit only at a large offload fraction | Required saved bytes imply repeated per-microbatch host traffic | Compute the ideal D2H+H2D bandwidth lower bound | Reject before allocation when the lower bound exceeds the target step-time margin |

## Known Limitations

- Activation offloading requires PP=1, making it impractical for large models
  (30B+ MoE) that need pipeline parallelism.
- Optimizer offloading throughput penalty scales linearly (~1.9x at 25%,
  ~4.2x at 100% for Qwen3-30B-A3B).
- D2H/H2D overlap provides only ~7% speedup because CPU Adam compute is
  the dominant bottleneck.
- `fine_grained_activation_offloading` is a separate module-level approach
  that works with PP > 1 but cannot be combined with layer-level
  `cpu_offloading`.
- Its fraction is adaptive only after discovery warmup; the first iteration
  offloads all eligible groups and may need substantially more host bandwidth
  and pinned memory than steady state.
- A configuration that fits through offload can still be incapable of meeting
  a throughput gate. Size the full per-step bidirectional transfer volume, not
  only the instantaneous HBM savings.
