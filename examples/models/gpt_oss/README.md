# GPT-OSS Examples

This directory contains example scripts for GPT-OSS 20B language models.

For model introduction and architecture details, see the GPT-OSS documentation.

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

- **Import**: Use `openai/gpt-oss-20b` as the source Hugging Face model.
- **Export**: Use `unsloth/gpt-oss-20b-BF16` as the reference HF model for export because the exported Megatron checkpoint is unquantized (bf16), which matches that repo's format.

### Import HF → Megatron

To import the HF model to your desired Megatron path:

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model openai/gpt-oss-20b \
    --megatron-path ${WORKSPACE}/models/gpt-oss-20b \
    --trust-remote-code
```

### Export Megatron → HF

The export uses `unsloth/gpt-oss-20b-BF16` as the reference so the saved HF checkpoint matches that unquantized format:

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model unsloth/gpt-oss-20b-BF16 \
    --megatron-path ${WORKSPACE}/models/gpt-oss-20b/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gpt-oss-20b-hf-export
```

### Round-trip Validation

Multi-GPU round-trip validation between formats:

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id unsloth/gpt-oss-20b-BF16 \
    --megatron-load-path ${WORKSPACE}/models/gpt-oss-20b/iter_0000000 \
    --tp 2 --pp 2 \
    --trust-remote-code
```

## Training Recipes

- See: [bridge.recipes.gpt_oss](../../../src/megatron/bridge/recipes/gpt_oss/gpt_oss.py)
- Available recipes:
  - `gpt_oss_20b_pretrain_config`: Pretraining configuration for 20B
  - `gpt_oss_20b_finetune_config`: Finetuning configuration for 20B with PEFT support
  - `gpt_oss_120b_pretrain_config`: Pretraining configuration for 120B
  - `gpt_oss_120b_finetune_config`: Finetuning configuration for 120B with PEFT support

Before training, ensure the following are configured:
1. **Container Image**: Set `CONTAINER_IMAGE` in the SLURM scripts to your container path
2. **Container Mounts**: (optional) Set `CONTAINER_MOUNTS` for data and workspace directories
3. **Environment Variables**:
   - `HF_TOKEN`: to download models from HF Hub (if required)
   - `HF_HOME`: (optional) to avoid re-downloading models and datasets
   - `WANDB_API_KEY`: (optional) to enable WandB logging

All training scripts use SLURM for containerized multi-node training.

### Pretrain

Pretrain uses the **DCLM** dataset by default when `DCLM_DATA_DIR` and `DCLM_CACHE` are set (see [slurm_pretrain.sh](slurm_pretrain.sh)).

To use your own preprocessed DCLM data, set the dataset config as follows (e.g. in the recipe or via overrides):

```python
cfg.dataset.blend = [
    [f"/path/to/dclm/preprocessed/dclm_{i:02d}_text_document" for i in range(1, 11)],
    None,
]
cfg.dataset.split = "9999,8,2"
cfg.dataset.path_to_cache = "/path/to/cache"
```

Preprocess your data using the [DCLM data preprocessing tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/tutorials/data/dclm).

W&B report coming soon.

### Supervised Fine-Tuning (SFT)

See the [slurm_sft.sh](slurm_sft.sh) script for full parameter fine-tuning. The recipe uses sequence packing by default.

W&B report coming soon.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [slurm_peft.sh](slurm_peft.sh) script for LoRA fine-tuning. The recipe uses sequence packing by default.

W&B report coming soon.

## Inference

See [inference.sh](inference.sh) for text generation with:
- Hugging Face checkpoint (`unsloth/gpt-oss-20b-BF16`)
- Imported Megatron checkpoint (after [conversion.sh](conversion.sh) import)
- Exported HF checkpoint (after conversion export)
- **SFT (finetuned) checkpoint**: set `SFT_CHECKPOINT` to your [slurm_sft.sh](slurm_sft.sh) result dir and run:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path unsloth/gpt-oss-20b-BF16 \
    --megatron_model_path ${WORKSPACE}/results/gpt_oss_20b_finetune_tp2_pp2_ep4_spTrue_cp1 \
    --prompt "Hello, how are you?" \
    --max_new_tokens 64 \
    --tp 2 --pp 2 --ep 2 --etp 1 \
    --trust-remote-code
```

TP×PP×EP must equal `--nproc_per_node`. Adjust parallelism to match your SFT run.

## Evaluation

Coming soon.