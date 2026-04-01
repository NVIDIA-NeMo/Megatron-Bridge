# Qwen3-Next Examples

This directory contains example scripts for Qwen3-Next 80B-A3B, a hybrid MoE language model combining Gated Delta Net (GDN) linear attention with standard softmax attention.

## Architecture

Qwen3-Next 80B-A3B uses a hybrid architecture with MambaModel:
- **48 logical layers** expanded to **96 physical layers** (each logical layer = attention + MoE FFN)
- **75% GDN layers** (linear attention) + **25% standard attention** (`full_attention_interval=4`)
- **512 routed experts** with top-10 routing + shared expert with gating
- **Multi-Token Prediction (MTP)** with 1 depth (standard attention + MoE)
- Pattern: `GEGEGE*E` repeated 12 times, where `G`=GDN, `*`=standard attention, `E`=MoE FFN

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

See the [conversion.sh](conversion.sh) script for checkpoint conversion examples.

The conversion uses the Qwen3-Next bridge which converts HuggingFace checkpoints to MambaModel format with physical layer indexing (HF layer N -> physical layers 2N and 2N+1).

### Import HF -> Megatron

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --megatron-path ${WORKSPACE}/models/Qwen3-Next-80B-A3B-Instruct \
    --trust-remote-code
```

### Export Megatron -> HF

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --megatron-path ${WORKSPACE}/models/Qwen3-Next-80B-A3B-Instruct/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Qwen3-Next-80B-A3B-Instruct-hf-export
```

### Round-trip Validation

Multi-GPU round-trip validation between formats:

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id Qwen/Qwen3-Next-80B-A3B-Instruct \
    --megatron-load-path ${WORKSPACE}/models/Qwen3-Next-80B-A3B-Instruct/iter_0000000 \
    --tp 1 --pp 4 \
    --trust-remote-code
```

## Training Recipes

- See: [bridge.recipes.qwen.qwen3_next](../../../src/megatron/bridge/recipes/qwen/qwen3_next.py)
- Available recipes:
  - `qwen3_next_80b_a3b_pretrain_config`: Pretraining configuration
  - `qwen3_next_80b_a3b_sft_config`: Full SFT configuration
  - `qwen3_next_80b_a3b_peft_config`: PEFT configuration (not yet supported)

Before training, ensure the following are configured:
1. **Container Image**: Set `CONTAINER_IMAGE` in the SLURM scripts to your container path
2. **Container Mounts**: (optional) Set `CONTAINER_MOUNTS` for data and workspace directories
3. **Environment Variables**:
   - `HF_TOKEN`: to download models from HF Hub
   - `HF_HOME`: (optional) to avoid re-downloading models and datasets

All training scripts use SLURM for containerized multi-node training.

### Pretrain

See the [slurm_pretrain.sh](slurm_pretrain.sh) script. Recommended parallelism: TP=1, PP=4, EP=8.

### Supervised Fine-Tuning (SFT)

See the [slurm_sft.sh](slurm_sft.sh) script. Recommended parallelism: TP=1, PP=4, EP=8.

Note: Packed sequence is NOT supported for Qwen3-Next. The SFT recipe uses `packed_sequence=False`.

### Configuration Overrides

See [conf/qwen3_next_80b_a3b_finetune_override_example.yaml](conf/qwen3_next_80b_a3b_finetune_override_example.yaml) for an example of overriding recipe defaults via YAML config.
