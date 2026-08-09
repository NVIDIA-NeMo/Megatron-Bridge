---
name: nemo-mbridge-perf-memory-tuning
description: Techniques for reducing peak GPU memory in Megatron Bridge, including expandable segments, optimizer-state precision, PEFT plus SP input re-gather, parallelism resizing, activation recompute, CPU-offloading constraints, and common OOM fixes. Use for GPU OOM, peak-memory reduction, LoRA or PEFT sequence-parallel activation memory, memory fragmentation, expandable_segments, sequence_parallel_input_regather, PYTORCH_CUDA_ALLOC_CONF, or memory-regression investigation.
license: Apache-2.0
---

# Memory Tuning

Stable docs: @docs/parallelisms.md
Card: @skills/nemo-mbridge-perf-memory-tuning/card.yaml

## What It Is

GPU OOM failures during training often stem from memory **fragmentation** rather
than raw capacity.  PyTorch's default CUDA allocator can leave unusable gaps
between allocations.  The single most effective fix is:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This tells PyTorch to use expandable (non-fixed-size) memory segments, which
dramatically reduces fragmentation and often eliminates borderline OOM without
any model or parallelism changes.

Beyond fragmentation, actual peak memory is determined by:

- **Parameter + optimizer state memory** — controlled by TP, PP, DP sharding
  (distributed optimizer, FSDP)
- **Activation memory** — controlled by activation recompute, sequence length,
  micro-batch size, and PEFT-specific retention of gathered inputs
- **Temporary / workspace memory** — CUDA kernels, NCCL buffers, CUDA graphs

For configuration planning, use the Bridge theoretical estimator before launching
large jobs:

```python
from megatron.bridge.training.utils.theoretical_memory_utils import estimate_training_memory

estimate = estimate_training_memory(cfg, num_microbatches=num_microbatches)
```

The estimator reports the most-loaded GPU shard and separates dense/embedding,
routed MoE expert, and activation components. It does not include allocator
fragmentation, CUDA/NCCL workspace, CUDA graph buffers, token imbalance, or
dispatcher workspace, so validate final configs with runtime memory metrics.

## Quick Decision

When a training run OOMs or is close to the memory limit:

1. **Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` first.** This often
   fixes fragmentation-induced OOM without a measurable performance cost in
   conventional schedules. Most Slurm launch templates already include it.
   For fine-grained multi-stream schedules, benchmark the allocator as part of
   the exact workload instead of assuming that allocator behavior is neutral.
2. **For LoRA with sequence parallelism, enable input re-gather**
   (`LoRA(sequence_parallel_input_regather=True)`). This avoids retaining the
   full gathered LoRA-A input in every eligible layer; it has no effect when SP
   is disabled.
3. **If step 1 passes but step 2 OOMs, inspect optimizer-state precision.**
   Adam states may be materialized after the first iteration. On BF16 training,
   a precision-aware optimizer with BF16 gradients and moments can recover
   enough capacity to keep PP low; validate numerical behavior for the target
   workload.
4. **Add selective activation recompute** (`recompute_modules=[core_attn]`) if
   not already enabled. See @skills/nemo-mbridge-perf-activation-recompute/SKILL.md.
5. **Avoid increasing TP** as a memory fix — doubling TP dramatically increases
   NVLink all-reduce volume and often kills throughput (-28% on Llama3 70B).
6. **Avoid increasing PP at the cost of DP** — halving DP doubles gradient
   accumulation steps and hurts throughput (~6%).
7. Consider `mlp` recompute if still OOM. Saves ~3 GB but costs ~16% GPU
   utilization on large dense models (Llama3 70B).
8. CPU offloading is **blocked when PP > 1**.

## Enablement

### Expandable segments (recommended first step)

Set in the job's environment before launching:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

In Slurm scripts this is typically placed alongside other env vars:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

No model config changes are needed. Conventional training schedules commonly
show no measurable throughput cost, but allocator neutrality is not universal;
see the fine-grained overlap note below.

### Allocator diagnosis in fine-grained overlap schedules

Fine-grained MoE schedules can release storage on one CUDA stream while other
streams still own related work. In this case allocator runtime calls and rank
skew can become part of the distributed critical path:

1. Compare an exact matched run with `expandable_segments:True` and
   `backend:native`; do not change the dispatcher, overlap, batch shape, or
   topology at the same time.
2. Inspect `cudaEventSynchronize`, `cuMemCreate`, `cuMemMap`,
   `cuMemSetAccess`, `cuMemUnmap`, and `cuMemRelease` in an Nsight Systems
   profile. Large `cuMem*` totals show allocator activity, but do not by
   themselves prove that expandable segments are the root cause.
3. Compare live utilization and memory across every rank. A mixture of idle and
   fully busy GPUs near the physical memory limit indicates rank skew or
   fragmentation feeding collective rendezvous.
4. Require at least two optimizer steps. A finite first step can hide delayed
   optimizer allocation, fragmentation, or a later cross-rank stall.

On Qwen3.5-35B-A3B with exact 2-node/16-H100 HybridEP overlap, the expandable
allocator profile contained roughly 30,000 calls in each major `cuMem*`
category. Switching only to the native allocator did not fix the slowdown:
iteration 1 took 151.9 seconds and iteration 2 failed to complete after more
than six minutes, while ranks occupied about 79.2--80.9 GiB and alternated
between 0% and 100% utilization. The matched expandable run was also slow at
about 82.7 seconds per step, but materially better. Treat allocator VMM traffic
as a correlated symptom until an isolated allocator A/B establishes causality.

Do not disable fine-grained storage release wholesale as the next reaction. In
the same Qwen3.5 shape, retaining every MoE-combine input raised rank-0
first-step peak allocation from about 63.15 to 68.93 GiB. The first iteration
was finite, but all 16 ranks then OOMed while requesting another 1.89 GiB as
optimizer state was materialized. Early reclamation was required for capacity;
any scheduler fix must preserve that memory benefit while making ownership and
cross-stream retirement explicit.

A dependency-aware control did preserve it: after combine recorded its
completion event, the compute/owner stream waited on that event and resized the
storage there instead of registering the input on the communication stream.
First-step peak allocation returned to 63.15 GiB and iteration-2 peak remained
72.03 GiB, while the first steady step improved from 82.70 to 67.98 seconds.
This is diagnostic evidence for cross-stream retirement cost, not a general
allocator recipe; the overlap schedule was still far slower than no overlap.

The matched profile confirmed that this was not timing noise. Capture span
contracted from 85.453 to 73.024 seconds and idle gaps from 24.187 to 19.916
seconds. `cudaEventSynchronize` fell from 8,952 calls / 12.513 seconds to
4,524 calls / 4.346 seconds. Major VMM call counts fell by about 25%:
`cuMemUnmap` from 29,502 to 22,049, `cuMemCreate` from 30,912 to 23,156, and
`cuMemMap` from 29,606 to 22,358. HybridEP dispatch also contracted by 22.4%.
This validates a cross-stream allocator/event interaction. It does not validate
the overlap schedule: dispatch and NCCL were still orders of magnitude above
the no-overlap trace.

Do not assume `backend:cudaMallocAsync` is an interchangeable escape hatch for
this case. On the same exact 2-node Qwen3.5 shape, with owner-stream retirement,
one connection, and plain EP overlap, switching to `cudaMallocAsync` exhausted
all 79.11 GiB devices in the first fine-grained backward. A representative
rank requested another 1.89 GiB with only 9.56 MiB CUDA-free, and no optimizer
step completed. The run also emitted repeated AccumulateGrad stream-mismatch
warnings. This is a capacity/lifetime failure, not a performance result:
allocator backends must pass the same multi-step memory gate before timing.

Selective architecture-specific recompute can distinguish lifetime pressure
from allocator choice. On the same owner-release/connections=1 Qwen3.5 overlap
shape, `recompute_modules=["gdn"]` lowered iteration-2 rank-0 peak allocated
memory from 72.028 to 59.516 GiB and reduced the steps 2-3 mean from 44.4646 to
25.3615 seconds. All three steps were finite. The 42.96% contraction proves
that the retained GDN-heavy activation lifetime was a major amplifier, but the
run still trailed the no-overlap winner by 13.52%; memory recovery and
throughput acceptance are separate gates.

### Validate that MoE paged stash is actually active

`moe_paged_stash=true` in a config dump is not proof that expert activations
were paged. MCore's saved-tensor hook only captures tensors carrying the
`grouped_tensor_scale_inv` marker normally supplied by Transformer Engine's
fused grouped operations. A custom grouped-GEMM or activation path must provide
an equivalent marker only on eligible token-major tensors; never tag expert
weights, offsets, or unrelated saved tensors.

Require both of these runtime signals before treating a run as a paged-stash
measurement:

1. after the capture iteration, the log contains one or more
   `allocate_stash_buffers cuda:` entries with dtype and hidden size; and
2. the later iteration reports no stash overflow or unintended host spill.

If the log instead says `Paged stash: max_tokens_dict is None, skipping stash
buffer allocation`, the hook captured no eligible tensors. Treat every timing
from that run as an unpaged matched control. On the exact 2-node/16-H100
Qwen3.5 owner-release overlap shape, such an invalid run completed with steady
steps of 44.3824, 44.3754, and 45.1325 seconds, matching the unpaged control;
it provides no evidence for or against paged-stash performance.

A corrected exact-2-node run did pass that activation gate: after its finite
158.8747-second capture iteration it allocated BF16 buffers with shapes
`[1478272, 2048]`, `[1478272, 1024]`, and `[1478272, 512]`, plus an FP32
`[1478272, 1]` buffer. It then OOMed on all 16 ranks before iteration 2 while
requesting another 1.89 GiB for optimizer state; PyTorch already held about
70.11 GiB and only about 1.6--1.7 GiB remained CUDA-free. The 1,478,272 rows
show that PP1 buffer sizing covered eligible activations across the in-flight
layer schedule, not one roughly 32K-token expert group. Passing the activation
gate is therefore necessary but not a capacity or throughput pass. Combine
paged stash with an independently validated lifetime reduction, lower the
number of eligible saved tensors, or change parallelism before timing.

Selective GDN recompute supplied enough headroom for that exact combination,
but did not make it a throughput win. The combined run completed all three
steps with finite numerics and allocated the same four stash buffers. Its
steps 2 and 3 took 30.1803 and 29.7948 seconds, averaging 29.98755 seconds and
about 196.4 model TFLOP/s/GPU. Rank-0 iteration-2 peak allocated/reserved
memory was 65.891/72.343 GiB. This was 18.24% slower than recompute alone and
34.23% slower than the accepted no-overlap path. Paged pack/unpack work can
turn recovered capacity back into a scheduler cost; benchmark combined memory
features rather than assuming their benefits compose.

### Parallelism resizing

If the model genuinely does not fit (not fragmentation), adjust parallelism:

| Strategy | Memory effect | Throughput cost | Notes |
|---|---|---|---|
| Increase PP (keeping DP) | Fewer layers per stage | Moderate (~6% if DP halved) | Only if GPU count allows |
| Increase TP | Fewer params per GPU | Severe (-28% on 70B) | Last resort |
| Distributed optimizer | Shards optimizer state across DP ranks | ~1-2% | Recommended for large models |
| FSDP | Shards params + grads + optimizer | Varies | See @skills/nemo-mbridge-perf-megatron-fsdp/SKILL.md |

### Delayed optimizer-state OOM

A finite first iteration does not prove that a layout fits. Optimizer state can
be allocated lazily during or after the first optimizer step, making iteration
2 the first point where full steady-state memory is visible. Distinguish this
from activation memory before adding recompute or PP:

1. Record peak allocated memory after both iterations 1 and 2.
2. Keep the batch shape and model topology fixed.
3. For BF16 training, test the precision-aware optimizer with BF16 gradients
   and Adam moments:

```python
import torch

cfg.optimizer.use_precision_aware_optimizer = True
cfg.optimizer.main_grads_dtype = torch.bfloat16
cfg.optimizer.exp_avg_dtype = torch.bfloat16
cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
```

The optimizer can retain FP32 main parameters and parameter remainders. Treat
reduced-precision states as a numerical choice, not only a capacity switch:
require multiple finite steps with zero skipped/NaN iterations.

### Activation recompute

See @skills/nemo-mbridge-perf-activation-recompute/SKILL.md for full details.

### PEFT + sequence-parallel input re-gather

For `LoRA` training with sequence parallelism, eligible column-parallel
`linear_qkv` and `linear_fc1` adapters consume a gathered LayerNorm output.
Because LoRA-A is trainable, the default path retains that full gathered input
until backward for the LoRA-A weight gradient.

Enable input re-gather when constructing the PEFT config:

```python
from megatron.bridge.peft.lora import LoRA

cfg.peft = LoRA(
    # Keep the recipe's existing LoRA settings here.
    sequence_parallel_input_regather=True,
)
```

With this option, forward still materializes the full input temporarily for the
LoRA-A GEMM, but MCore autograd retains only its sequence-local shard. Backward
asynchronously gathers the full input again, overlaps the collective with
dgrad when possible, computes the LoRA-A weight gradient, and then reuses the
temporary communication buffer.

This is a memory-for-communication tradeoff, not conventional activation
checkpointing: no LayerNorm, attention, MLP, or LoRA GEMM is rerun. Some
throughput degradation is expected, and the benefit grows with the amount of
eligible LoRA-A activation retained. The option has no effect when sequence
parallelism is disabled.

### CPU offloading

```python
cfg.model.cpu_offloading = True
```

**Incompatible with PP > 1.** Only usable when `pipeline_model_parallel_size = 1`.

## A Note on VPP

Virtual pipeline parallelism (VPP) is primarily a **throughput** optimization
that reduces pipeline bubble overhead by interleaving smaller model chunks. Its
effect on peak memory is minimal — changing VPP does not meaningfully change
the total activation, parameter, or optimizer memory on a GPU.

In earlier experiments we incorrectly attributed an OOM fix to VPP tuning
(VPP 5→10). The actual fix was `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
which eliminated memory fragmentation. The VPP=10 run actually used slightly
**more** peak memory (60.2 GB vs 58.8 GB) but did not OOM because expandable
segments prevented fragmentation.

VPP should be tuned for pipeline bubble reduction (see @docs/parallelisms.md),
not as a memory fix.

## Compatibility and Constraints

- `expandable_segments:True` is incompatible with `--use-nccl-ub` (NCCL
  user-buffer registration). See Megatron-FSDP docs.
- When using CUDA graphs with `expandable_segments:True`, set
  `NCCL_GRAPH_REGISTER=0` (required on pre-Blackwell GPUs, enforced by MCore
  `CudaGraphManager`).
- CPU offloading requires `pipeline_model_parallel_size = 1`.
- Distributed optimizer requires `use_distributed_optimizer = True` in the
  optimizer config.
- `sequence_parallel_input_regather` applies only to eligible non-expert
  column-parallel LoRA-A projections. Row-parallel adapters, expert adapters,
  TP=1, CUDA graphs, CPU activation offload, and overlapping full-layer or
  selective MLP activation recompute fall back to the existing path.

## Measured Results

Llama3 70B SFT on 32x H100 80GB, FP8 (Current Scaling):
- Baseline: TP=4, PP=4, VPP=5, DP=2, MBS=1, GBS=32, seq_len=4096
- Golden GPU utilization: 709.93 TFLOP/s/GPU
- Regression threshold: 5%

### Strategy comparison: parallelism changes for memory reduction

| Experiment | TP | PP | VPP | DP | TFLOP/s/GPU | vs Golden | Peak Mem (GB) | Result |
|---|---|---|---|---|---|---|---|---|
| Baseline | 4 | 4 | 5 | 2 | ~704 | -0.8% | 58.8 | OOM (fragmentation) |
| More PP | 4 | 8 | 5 | 1 | 668.0 | -5.9% | 53.2 | Borderline perf |
| More TP | 8 | 4 | 5 | 1 | 508.7 | -28.4% | 50.2 | Severe regression |
| Baseline + expandable_segments | 4 | 4 | 5 | 2 | ~704 | -0.8% | ~59 | **Passed** |

Key takeaways:

- **`expandable_segments:True` is the winner for this Llama3 workload.** The
  baseline OOM was caused by memory fragmentation, not insufficient capacity.
  Setting this env var eliminated the OOM with no measured throughput cost and
  no parallelism changes.
- **PP=8 works for memory but loses DP** (2→1), meaning 32 gradient accumulation
  steps per batch, which hurts throughput by ~6%.
- **TP=8 is catastrophic** (-28%) because doubling TP increases all-reduce
  communication volume proportionally across NVLink, and DP=1 means no
  micro-batch overlap.

### CPU offloading: blocked

| Experiment | offload_layers | Result |
|---|---|---|
| Exp 4 | 2 | Incompatible (PP > 1) |
| Exp 5 | 4 | Incompatible (PP > 1) |
| Exp 6 | 6 | Incompatible (PP > 1) |

`ValueError: Currently there is no support for Pipeline parallelism with CPU
offloading.` This approach is blocked for any model using PP > 1.

### Activation recompute: expensive alternative

Selective activation recompute with `mlp` saved ~3 GB peak memory but cost
~16% GPU utilization on this workload. See
@skills/nemo-mbridge-perf-activation-recompute/SKILL.md for full results.

### LoRA + SP input re-gather

Real-checkpoint H100 training with SQuAD showed lower peak memory in all tested
configurations, with workload-dependent throughput cost:

| Model/config | Baseline peak | Input re-gather peak | Memory saved | Throughput change |
|---|---:|---:|---:|---:|
| Qwen3-8B, TP2, seq 8192 | 47.545 GB | 42.814 GB | 4.731 GB (10.0%) | -6.74% |
| Qwen3-30B-A3B, TP4/EP4 | 29.890 GB | 28.321 GB | 1.569 GB (5.2%) | -2.89% |
| GPT-OSS-120B, TP2/EP8 | 52.185 GB | 51.371 GB | 0.814 GB (1.6%) | -0.34% |

All runs had finite losses with zero skipped or NaN iterations. Two-rank BF16
and FP32 checks matched the baseline for outputs, input gradients, LoRA-A and
LoRA-B gradients, and two-microbatch fused `main_grad` accumulation.

## Code Anchors

### LoRA sequence-parallel input re-gather

```text
src/megatron/bridge/peft/lora.py
    LoRA.sequence_parallel_input_regather

src/megatron/bridge/peft/utils.py
    ParallelLinearAdapter._sequence_parallel_input_regather_eligibility()
    ParallelLinearAdapter.forward()
```

### CPU offloading PP incompatibility (MCore)

```1303:1306:3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                "Currently there is no support for Pipeline parallelism with CPU offloading"
            )
```

### VPP config and layer divisibility validation (MCore)

```1581:1592:3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
            if pipeline_parallel_size and self.virtual_pipeline_model_parallel_size is not None:
                num_layers_per_middle_pipeline_rank = num_layers // pipeline_parallel_size
                if (
                    not num_layers_per_middle_pipeline_rank
                    % self.virtual_pipeline_model_parallel_size
                    == 0
                ):
                    raise ValueError(
                        f"number of layers on each middle pipeline rank:"
                        f"{num_layers_per_middle_pipeline_rank} must be divisible by virtual"
                        f"pipeline parallel degree {self.virtual_pipeline_model_parallel_size}"
                    )
```

### Parallelism docs on interleaved pipeline schedule

```116:124:docs/parallelisms.md
To minimize the pipeline bubble, the computation on each GPU can be divided into multiple subsets of layers (referred to as model chunks), rather than a single contiguous block. Enable this by setting `virtual_pipeline_model_parallel_size`:

model_config = GPTModelProvider(
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2,  # 2 model chunks per pipeline stage
    # ... other model parameters
)
```

## Failure Diagnosis

| Symptom | Cause | Confirm | Fix |
|---|---|---|---|
| OOM on a single rank despite headroom on others | Memory fragmentation | check if `expandable_segments:True` is set | set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Step 1 passes, step 2 OOMs on all ranks | delayed Adam-state materialization | compare peak memory after iterations 1 and 2; inspect optimizer dtypes | use a validated precision-aware optimizer configuration, distributed optimizer/FSDP, or more model parallelism |
| OOM with `expandable_segments` already set | Genuine capacity limit | check `nvidia-smi` for param/optimizer memory | increase PP, use distributed optimizer, or add recompute |
| Fine-grained overlap shows many `cuMem*` calls and severe rank skew | Allocator activity may amplify a cross-stream lifetime or scheduler problem | run an exact expandable-vs-native allocator A/B and compare all-rank memory/utilization | keep the better allocator; profile the schedule's storage release and event dependencies rather than assuming VMM calls are causal |
| Native allocator makes overlap stall longer and pushes ranks near physical capacity | Fragmentation or delayed reuse in the native allocator | compare iteration 2, per-rank memory, and idle/busy rank split against expandable segments | reject the native allocator for that shape and keep investigating the schedule |
| Disabling fine-grained storage release makes iteration 1 pass but iteration 2 OOM | Retained schedule inputs consume optimizer-state headroom | compare first-step peak allocation and the optimizer-state allocation request | restore early release; change release ordering/dependencies instead of retaining inputs until backward |
| Owner-stream retirement preserves memory but overlap is still slow | Cross-stream retirement was one contributor, not the whole scheduler critical path | compare matched iteration-2 time, peak allocation, dispatch/NCCL unions, and event/allocator APIs | profile the new schedule before changing another dependency; do not adopt it as a default from short timing alone |
| Owner-stream retirement cuts event-sync and VMM activity but no-overlap is still much faster | The release path amplified allocator/rendezvous overhead, but another fine-grained schedule dependency remains | compare three traces: no overlap, normal release, and owner-stream release | keep the allocator and early-release settings that preserve capacity; investigate remaining event/rank-skew inflation |
| `cudaMallocAsync` OOMs in the first fine-grained backward although expandable segments runs | The stream-ordered pool cannot reuse enough memory under the combined activation lifetime | compare the exact request/free/device-limit diagnostics and require a completed optimizer step | reject that allocator for the shape; reduce activation lifetime or parallelism before another timing run |
| `moe_paged_stash=true` but the log skips buffer allocation because `max_tokens_dict is None` | No eligible saved tensor carried the paged-stash marker | require `allocate_stash_buffers cuda:` after capture; inspect markers on token-major expert activations | fix eligibility marking and rerun; classify existing timings as an unpaged control |
| Paged stash allocates buffers after capture but optimizer state immediately OOMs | Full-schedule stash buffers plus capture/optimizer lifetime exceed device capacity | record every allocated buffer shape and the optimizer request/free diagnostics | reduce independently measured activation lifetime or eligible tensor scope before timing; do not claim throughput |
| Mixed FP8 first step is much slower and memory rises near capacity | FP8 compilation, scaling state, and non-expert precision boundaries can add cold-start workspace even when experts remain BF16 | sample all-rank utilization and memory through iteration 1, then compare a post-compile steady window with the exact BF16 control | require both finite steady throughput and optimizer-state headroom; reject the precision mode if it only fits transiently or regresses end to end |
| Estimated memory exceeds GPU capacity before launch | model state or activations genuinely too large | run `estimate_training_memory` and inspect the largest component | adjust PP/TP/CP/EP, distributed optimizer, or recompute before launching |
| LoRA + SP retains unexpectedly high activation memory | full gathered LoRA-A inputs are retained until backward | check whether `cfg.peft.sequence_parallel_input_regather` is enabled and the target is eligible | set `LoRA(sequence_parallel_input_regather=True)`; verify fallback constraints |
| `ValueError: PP + CPU offloading` | using cpu_offloading with PP > 1 | check PP config | disable CPU offloading or set PP=1 |
| `RuntimeError` with `--use-nccl-ub` + expandable segments | NCCL UB incompatible with expandable allocator | check env vars | remove `expandable_segments:True` or disable `--use-nccl-ub` |

## Known Limitations

- CPU offloading is blocked when PP > 1
- Parallelism resizing (TP/PP) often has significant throughput costs
- The theoretical estimator is formula-based and does not replace runtime
  profiling or CUDA memory reports
- Expandable segments are not universally throughput-neutral in fine-grained
  multi-stream schedules; allocator runtime traffic must be interpreted with a
  matched allocator A/B
- LoRA input re-gather does not cover row-parallel or expert adapters and may
  have negligible benefit when few eligible LoRA-A activations dominate memory

## Verification

Quick check that `expandable_segments:True` is active in the current process:

```python
import os
assert "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
```

For Slurm jobs, verify the env var is exported before the training command
in the launch script. If a recipe config dump still prints its default, inspect
the environment of a live training PID (for example,
`tr '\0' '\n' < /proc/<pid>/environ`) because launcher values can take
precedence while the serialized recipe continues to display its configured
default.

For LoRA + SP input re-gather, run the focused configuration tests and the real
two-rank MCore backward-parity test:

```bash
uv run python -m pytest \
  tests/unit_tests/peft/test_utils.py -k "sequence_parallel_input_regather" \
  tests/unit_tests/peft/test_lora.py -k "sequence_parallel_input_regather"

uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
  tests/unit_tests/peft/test_lora_sp_input_regather_distributed.py
```
