---
name: memory-tuning
description: Techniques for reducing peak GPU memory in Megatron Bridge — VPP tuning, parallelism resizing, CPU offloading constraints, and future memory reduction strategies.
---

# Memory Tuning

Stable docs: `docs/parallelisms.md`
Card: `card.yaml` (co-located)

## What It Is

Virtual pipeline parallelism (VPP) divides each pipeline stage's layers into
multiple smaller chunks that are processed in an interleaved schedule. Increasing
VPP reduces the number of transformer layers in-flight per micro-batch in each
chunk, which directly reduces peak activation memory:

```
layers_per_chunk = num_layers / (PP * VPP)
```

Doubling VPP halves the layers per chunk and roughly halves the peak activation
footprint from those layers — often enough to fix borderline OOM.

## Quick Decision

When a training run OOMs or is close to the memory limit with PP already in use:

1. **Try VPP increase first.** It preserves TP, PP, and DP, so throughput impact
   is minimal (~1-2%).
2. **Avoid increasing TP** as a memory fix — doubling TP dramatically increases
   NVLink all-reduce volume and often kills throughput (-28% on Llama3 70B).
3. **Avoid increasing PP at the cost of DP** — halving DP doubles gradient
   accumulation steps and hurts throughput (~6%).
4. Consider `mlp` recompute only if VPP increase is not possible (layer count
   not divisible). See `skills/perf-techniques/activation-recompute/SKILL.md`.
5. CPU offloading is **blocked when PP > 1**.

## Enablement

### Calculating VPP

```
num_layers = 80  (Llama3 70B)
PP = 4

VPP=5:  layers_per_chunk = 80 / (4 * 5)  = 4 layers/chunk
VPP=10: layers_per_chunk = 80 / (4 * 10) = 2 layers/chunk  ← winner
VPP=20: layers_per_chunk = 80 / (4 * 20) = 1 layer/chunk
```

Constraint: `num_layers % (PP * VPP) == 0`

### Config

```python
cfg.model.pipeline_model_parallel_size = 4
cfg.model.virtual_pipeline_model_parallel_size = 10  # was 5, doubled to fix OOM
```

### Performance harness CLI

```bash
python scripts/performance/run_performance_workload.py \
  --pipeline_model_parallel_size 4 \
  --virtual_pipeline_model_parallel_size 10 \
  ...
```

## Compatibility and Constraints

- Requires `pipeline_model_parallel_size > 1`
- `num_layers` must be divisible by `PP * VPP` on each pipeline stage
- `account_for_embedding_in_pipeline_split` and
  `account_for_loss_in_pipeline_split` change the layer count per stage and
  must still be divisible by VPP
- Very high VPP with few micro-batches per global batch can worsen the pipeline
  bubble ratio (more interleaved chunks but less compute to hide the bubble)
- VPP increases P2P send/recv volume proportionally (each chunk needs
  activation/gradient transfers between stages)

## Measured Results

Llama3 70B SFT on 32x H100 80GB, FP8 (Current Scaling):
- Baseline: TP=4, PP=4, VPP=5, DP=2, MBS=1, GBS=32, seq_len=4096
- Golden GPU utilization: 709.93 TFLOP/s/GPU
- Regression threshold: 5%

### Strategy comparison: memory reduction approaches

| Experiment | TP | PP | VPP | DP | TFLOP/s/GPU | vs Golden | Peak Mem (GB) | Result |
|---|---|---|---|---|---|---|---|---|
| Baseline | 4 | 4 | 5 | 2 | ~704 | -0.8% | 58.8 | OOM |
| More PP | 4 | 8 | 5 | 1 | 668.0 | -5.9% | 53.2 | Borderline perf |
| More TP | 8 | 4 | 5 | 1 | 508.7 | -28.4% | 50.2 | Severe regression |
| **More VPP** | **4** | **4** | **10** | **2** | **698.9** | **-1.6%** | **60.2** | **Passed** |

Key takeaways:

- **VPP=10 is the winner.** Doubling virtual pipeline chunks (5→10) reduced
  layers-in-flight per chunk from 4 to 2, while keeping TP/PP/DP unchanged.
  Only -1.6% GPU utilization — well within the 5% threshold.
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
`skills/perf-techniques/activation-recompute/SKILL.md` for full results.

## Code Anchors

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

### Llama3 70B SFT H100 workload config (VPP=5 baseline)

```551:560:scripts/performance/configs/llama/llama3_workload_base_configs.py
_LLAMA3_70B_SFT_CONFIG_H100 = replace(
    BASE_LLAMA3_70B_CONFIG,
    num_gpus=32,
    peft="none",
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    micro_batch_size=1,
    global_batch_size=32,
)
```

### CPU offloading PP incompatibility (MCore)

```1303:1306:3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                "Currently there is no support for Pipeline parallelism with CPU offloading"
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
| `ValueError: must be divisible by VPP` | num_layers / PP not divisible by VPP | check `num_layers % (PP * VPP)` | choose VPP that divides evenly |
| Throughput regression > budget | too many virtual chunks or too few micro-batches | profile P2P comm time | reduce VPP or increase GBS |
| Still OOM after VPP increase | memory pressure from non-activation sources | check `nvidia-smi` for param/optimizer memory | consider FSDP or distributed optimizer |
| `ValueError: PP + CPU offloading` | using cpu_offloading with PP > 1 | check PP config | use VPP or recompute instead |

## Known Limitations

- Not all layer counts allow arbitrary VPP values (divisibility constraint)
- Memory reduction is indirect — fewer layers in-flight, not a fixed GB value
- Optimal VPP depends on model depth, PP, GBS, and hardware
- Very high VPP with low GBS can worsen pipeline bubble ratio

## Verification

Quick check that VPP=10 initializes and trains:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python -m torch.distributed.run --nproc_per_node=8 \
  scripts/training/run_recipe.py \
  --recipe llama3_70b_pretrain_config \
  model.pipeline_model_parallel_size=4 \
  model.tensor_model_parallel_size=2 \
  model.virtual_pipeline_model_parallel_size=10 \
  train.train_iters=3 train.global_batch_size=8 train.micro_batch_size=1 \
  scheduler.lr_warmup_iters=0 \
  validation.eval_iters=0 validation.eval_interval=0 \
  checkpoint.save_interval=0 \
  logger.log_interval=1
```

Success criteria:

- exit code 0
- finite loss at iteration 3
- log shows PP=4 VPP=10 layout
