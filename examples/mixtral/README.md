# Mixtral with Megatron-Bridge

> **⚠️ MIGRATED TO RECIPE PATTERN**
> The Mixtral training examples have been migrated to the modern recipe pattern.
> **Please use:** [`examples/recipes/mixtral/`](../recipes/mixtral/)

---

## 🚀 New Recipe-Based Examples (Recommended)

All training and finetuning should now use the recipe pattern:

```bash
# Pretraining with YAML + CLI overrides
torchrun --nproc_per_node=8 examples/recipes/mixtral/pretrain_mixtral_8x7b.py \
    --config-file conf/my_config.yaml \
    model.expert_model_parallel_size=8

# Finetuning with pretrained checkpoint
torchrun --nproc_per_node=8 examples/recipes/mixtral/finetune_mixtral_8x7b.py \
    --pretrained-checkpoint /path/to/ckpt \
    --data-path /path/to/data.jsonl
```

**Benefits of recipe pattern:**
- ✅ YAML configuration support
- ✅ Hydra-style CLI overrides
- ✅ Type-safe configuration
- ✅ 74% less code (413 vs 1,605 lines)
- ✅ Consistent with all other models

**Documentation:** See [`examples/recipes/mixtral/README.md`](../recipes/mixtral/README.md)

---

## Utilities (This Directory)

This directory contains utility scripts for Mixtral models:

### Checkpoint Conversion

Convert HuggingFace checkpoints to Megatron format for efficient loading:

```bash
torchrun --nproc_per_node=2 examples/mixtral/convert_checkpoint.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --output_path=/path/to/megatron/checkpoint \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4
```

### Text Generation (Inference)

**Single-GPU inference:**
```bash
python examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --prompt="Explain quantum computing:" \
    --max_tokens=200
```

**Multi-GPU with model parallelism:**
```bash
torchrun --nproc_per_node=8 examples/mixtral/generate_text.py \
    --load=/path/to/megatron/checkpoint \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4 \
    --prompt="Explain quantum computing:" \
    --max_tokens=200
```

---

## Migration Guide

If you were using the old training scripts:

### Old Approach (Deprecated)
```bash
# ❌ OLD: Manual argparse, no YAML support
torchrun --nproc_per_node=8 examples/mixtral/train_mixtral.py \
    --tensor-model-parallel-size=2 \
    --expert-model-parallel-size=4 \
    --train-iters=100000 \
    --global-batch-size=32 \
    --micro-batch-size=2 \
    # ... 60+ more arguments
```

### New Approach (Recipe Pattern)
```bash
# ✅ NEW: YAML configs + CLI overrides
torchrun --nproc_per_node=8 examples/recipes/mixtral/pretrain_mixtral_8x7b.py \
    --config-file conf/my_config.yaml \
    model.tensor_model_parallel_size=2 \
    model.expert_model_parallel_size=4 \
    train.train_iters=100000
```

**See full migration guide:** [`examples/recipes/mixtral/README.md`](../recipes/mixtral/README.md#migration-from-legacy-scripts)

---

## MoE-Specific Parallelism

Mixtral uses three types of parallelism:

| Type | Flag | Description |
|------|------|-------------|
| **Tensor Parallel (TP)** | `tensor_model_parallel_size` | Splits tensors across GPUs |
| **Pipeline Parallel (PP)** | `pipeline_model_parallel_size` | Splits layers across GPUs |
| **Expert Parallel (EP)** | `expert_model_parallel_size` | Splits MoE experts across GPUs |

**Recommended for 8x A100 80GB:**
- TP=2, PP=1, EP=4 (uses all 8 GPUs)
- Enable `sequence_parallel=true` when TP > 1

---

## Files in This Directory

| File | Purpose | Status |
|------|---------|--------|
| `convert_checkpoint.py` | HF ↔ Megatron checkpoint conversion | ✅ Active |
| `generate_text.py` | Text generation / inference | ✅ Active |
| `sample_instructions.jsonl` | Sample instruction data | ✅ Active |
| ~~`train_mixtral.py`~~ | ~~Pretraining~~ | ❌ Removed - Use recipe |
| ~~`finetune_mixtral.py`~~ | ~~Finetuning~~ | ❌ Removed - Use recipe |

---

## Additional Resources

- **Recipe Examples:** [`examples/recipes/mixtral/`](../recipes/mixtral/)
- **Recipe Module:** [`src/megatron/bridge/recipes/mixtral/`](../../src/megatron/bridge/recipes/mixtral/)
- **YAML Configs:** [`examples/recipes/mixtral/conf/`](../recipes/mixtral/conf/)
