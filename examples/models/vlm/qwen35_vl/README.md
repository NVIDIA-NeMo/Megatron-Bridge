# Qwen3.5-VL Examples

This directory contains example scripts for Qwen3.5-VL vision-language models.

For model introduction and architecture details, see the [Qwen3.5-VL documentation](../../../../docs/models/vlm/qwen35-vl.md).

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
To import the HF VL model to your desired Megatron path:
```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3.5-35B-A3B \
  --megatron-path ${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
  --hf-model Qwen/Qwen3.5-35B-A3B \
  --megatron-path ${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B/iter_0000000 \
  --hf-path ${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B-hf-export
```

See the [conversion.sh](conversion.sh) script for more examples including multi-GPU round-trip validation.

## Inference

### Run Inference on Converted Checkpoint

```bash
python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path Qwen/Qwen3.5-35B-A3B \
  --megatron_model_path ${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B/iter_0000000 \
  --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
  --prompt "Describe this image." \
  --max_new_tokens 100 \
  --tp 2 --pp 2 --ep 4
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward.
- You can also use image URLs: `--image_path="https://example.com/image.jpg"`
- For MoE models, set `--ep` to the desired expert parallelism degree.

See the [inference.sh](inference.sh) script for commands to:
- Run inference with Hugging Face checkpoints
- Run inference with imported Megatron checkpoints
- Run inference with exported Hugging Face checkpoints

For multi-node distributed inference—required for the largest 397B model—see the [slurm_inference.sh](slurm_inference.sh) script.

## Finetune Recipes

- Available recipes:
  - `qwen35_vl_800m_sft_config` / `qwen35_vl_800m_peft_config`: 0.8B dense model
  - `qwen35_vl_2b_sft_config` / `qwen35_vl_2b_peft_config`: 2B dense model
  - `qwen35_vl_4b_sft_config` / `qwen35_vl_4b_peft_config`: 4B dense model
  - `qwen35_vl_9b_sft_config` / `qwen35_vl_9b_peft_config`: 9B dense model
  - `qwen35_vl_27b_sft_config` / `qwen35_vl_27b_peft_config`: 27B dense model
  - `qwen35_vl_35b_a3b_sft_config` / `qwen35_vl_35b_a3b_peft_config`: 35B-A3B MoE model
  - `qwen35_vl_122b_a10b_sft_config` / `qwen35_vl_122b_a10b_peft_config`: 122B-A10B MoE model
  - `qwen35_vl_397b_a17b_sft_config` / `qwen35_vl_397b_a17b_peft_config`: 397B-A17B MoE model

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Pretrain

Pretraining is not verified for this model.

### Supervised Fine-Tuning (SFT)

See the [slurm_sft.sh](slurm_sft.sh) script for full parameter fine-tuning with configurable model sizes.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [slurm_peft.sh](slurm_peft.sh) script for LoRA fine-tuning with configurable model sizes.

### Expected Training Dynamics
We provide a [Weights & Biases report](https://api.wandb.ai/links/nvidia-nemo-fw-public/rt6uzrvf) for the expected loss curves and grad norms.

## Evaluation

Coming soon.
