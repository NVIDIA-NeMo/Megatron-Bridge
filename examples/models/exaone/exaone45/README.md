# EXAONE 4.5 VL Examples

Scripts for [EXAONE-4.5-33B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B), a 33B-parameter vision-language model from LG AI Research.

| Property | Value |
|---|---|
| HF model ID | `LGAI-EXAONE/EXAONE-4.5-33B` |
| Architecture | Vision-language model (`Exaone4_5_ForConditionalGeneration`) |
| Default dtype | BF16 |
| Default inference parallelism | `TP=4, PP=1, EP=1` |
| Sequence length in recipes | 8192 |

**Requirements:** use `--trust_remote_code` when loading the Hugging Face checkpoint.

## Workspace Configuration

The conversion script uses a `WORKSPACE` environment variable as the base directory for converted checkpoints. It defaults to `/workspace`.

```bash
export WORKSPACE=/your/custom/path
```

Converted checkpoints are written under `${WORKSPACE}/models/`.

## Inference (Megatron)

[inference.sh](inference.sh) loads the Hugging Face checkpoint, converts it to Megatron in memory, and runs vision-language generation on 4 local GPUs.

```bash
examples/models/exaone/exaone45/inference.sh
```

The script uses the sample NVIDIA H100 table image and the prompt `Describe this image.`:

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 \
  examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path LGAI-EXAONE/EXAONE-4.5-33B \
  --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
  --prompt "Describe this image." \
  --max_new_tokens 64 \
  --tp 4 \
  --pp 1 \
  --ep 1 \
  --trust_remote_code
```

You can replace `--image_path` with a local image path or another image URL.

## Checkpoint Conversion

[conversion.sh](conversion.sh) imports the Hugging Face checkpoint to Megatron format:

```bash
examples/models/exaone/exaone45/conversion.sh
```

By default, the converted checkpoint is written to:

```text
${WORKSPACE}/models/EXAONE-4.5-33B
```

Equivalent direct command:

```bash
uv run python examples/conversion/convert_checkpoints.py import \
  --hf-model LGAI-EXAONE/EXAONE-4.5-33B \
  --megatron-path "${WORKSPACE}/models/EXAONE-4.5-33B"
```

## Finetune Recipes

Available recipes:

- `exaone45_vl_sft_config`: Full supervised fine-tuning for EXAONE 4.5 VL.
- `exaone45_vl_peft_config`: Parameter-efficient fine-tuning with LoRA or DoRA.

Recipe defaults:

| Recipe | TP | PP | Notes |
|---|---:|---:|---|
| `exaone45_vl_sft_config` | 4 | 1 | Full SFT, sequence parallel enabled |
| `exaone45_vl_peft_config` | 1 | 1 | PEFT, sequence parallel disabled |

The recipes freeze the vision encoder by default (`freeze_vision_model=True`) and keep the language model and vision projection trainable.

Before training, set the usual runtime environment variables as needed:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face access token, if required |
| `HF_HOME` | Shared Hugging Face cache directory |
| `WANDB_API_KEY` | Optional W&B logging key |

## Validation

Focused EXAONE conversion coverage lives in `tests/functional_tests/test_groups/models/exaone/test_exaone_conversion.py`.
