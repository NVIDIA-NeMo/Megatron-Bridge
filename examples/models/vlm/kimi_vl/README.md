# Kimi K2.5 VL Examples

This directory contains example scripts for the Kimi K2.5 Vision-Language model (~1T parameters, 384 MoE experts).

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

### Import HF → Megatron

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model moonshotai/Kimi-K2.5 \
  --megatron-path ${WORKSPACE}/models/Kimi-K2.5 \
  --trust-remote-code
```

### Export Megatron → HF

```bash
python examples/conversion/convert_checkpoints.py export \
  --hf-model moonshotai/Kimi-K2.5 \
  --megatron-path ${WORKSPACE}/models/Kimi-K2.5/iter_0000000 \
  --hf-path ${WORKSPACE}/models/Kimi-K2.5-hf-export \
  --trust-remote-code
```

See the [conversion.sh](conversion.sh) script for more examples including multi-GPU round-trip validation.

## Inference

Kimi K2.5 VL uses a model-specific generation script that handles PP layout, pre-expanding image placeholders for pipeline parallelism, and Kimi processor patching.

### Single-Node Inference (≤ 8 GPUs)

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/models/vlm/kimi_vl/hf_to_megatron_generate_vlm.py \
  --hf_model_path moonshotai/Kimi-K2.5 \
  --trust_remote_code \
  --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
  --prompt "Describe this image." \
  --tp 2 --ep 4
```

See the [inference.sh](inference.sh) script for additional inference configurations.

### Multi-Node Inference (Full Model)

The full Kimi K2.5 VL model requires 12 nodes (96 GPUs) with TP=2, EP=48.
See the [slurm_inference.sh](slurm_inference.sh) script for multi-node Slurm-based inference.

## Finetune Recipes

- Available recipes:
  - `kimi_k25_vl_sft_config`: Full model SFT with Muon optimizer

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Pretrain

Pretraining is not verified for this model.

### Supervised Fine-Tuning (SFT)

See the [slurm_sft.sh](slurm_sft.sh) script for full parameter fine-tuning.

Recommended parallelism: TP=4, PP=4, EP=32 → 128 GPUs (16 nodes).

## Evaluation

Coming soon.
