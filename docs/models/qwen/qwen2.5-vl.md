# Qwen2.5-VL

Qwen2.5-VL is a series of vision-language models developed by Alibaba Cloud that enable multimodal understanding across text, images, and videos. The models support various vision-language tasks including image understanding, visual question answering, and multimodal reasoning.

NeMo Megatron Bridge supports finetuning Qwen2.5-VL models (3B, 7B, 32B, and 72B variants) on single-image and multi-image datasets.
The finetuned model can be converted back to the 🤗 Hugging Face format for downstream evaluation.

```{tip}
We use the following environment variables throughout this page
- `HF_MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct` (it can also be set to `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`)
- `MEGATRON_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct` (feel free to set your own path)

Unless explicitly stated, any megatron model path in the commands below should NOT contain the iteration number
`iter_xxxxxx`. For more details on checkpointing, please see
[here](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html#checkpoint-contents)
```

## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF model to your desired `$MEGATRON_MODEL_PATH`, run the following command.
```bash
uv run python examples/conversion/convert_checkpoints.py import \
--hf-model $HF_MODEL_PATH \
--megatron-path $MEGATRON_MODEL_PATH
```

### Export Megatron → HF
You can export a trained model with the following command.
```bash
uv run python examples/conversion/convert_checkpoints.py export \
--hf-model $HF_MODEL_PATH \
--megatron-path <trained megatron model path> \
--hf-path <output hf model path>
```

### Run In-Framework Inference on Converted Checkpoint
You can run a quick sanity check on the converted checkpoint with the following command.
```bash
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path $HF_MODEL_PATH \
--megatron_model_path $MEGATRON_MODEL_PATH \
--image_path <example image path> \
--prompt "Describe this image." \
--max_new_tokens 100
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward. If
  specified, the script will just load the megatron model
- `--max_new_tokens` controls the number of tokens to generate.
- You can also use image URLs: `--image_path="https://example.com/image.jpg"`


## Finetuning Recipes
Before training, ensure the following environment variables are set.
1. `SAVE_DIR`: to specify a checkpoint and log saving directory, used in the commands below.
2. `HF_TOKEN`: to download models from HF Hub (if required).
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets every time.
4. `WANDB_API_KEY`: (optional) to enable WandB logging.

### Full Finetuning

Example usage for full parameter finetuning:

```bash
uv run python -m torch.distributed.run --nproc-per-node=8 scripts/training/run_recipe.py \
--recipe qwen25_vl_3b_sft_config \
--mode sft \
--step-func vlm_step \
--from $MEGATRON_MODEL_PATH \
--dataset cord-v2 \
train.global_batch_size=<batch size> \
train.train_iters=<number of iterations> \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name>
```

Note:
- The `--recipe` parameter selects the model size configuration. Available options:
  - `qwen25_vl_3b_sft_config` - for 3B model
  - `qwen25_vl_7b_sft_config` - for 7B model
  - `qwen25_vl_32b_sft_config` - for 32B model
  - `qwen25_vl_72b_sft_config` - for 72B model
- Config fields can be overridden directly. For example, set `train.global_batch_size=<batch size>` in the command.
- The dataset format should be JSONL with conversation format (see dataset section below).
- After training, you can run inference with `hf_to_megatron_generate_vlm.py` by supplying the trained megatron checkpoint.
  You can also export the trained checkpoint to Hugging Face format.

### Parameter-Efficient Finetuning (PEFT)
Parameter-efficient finetuning (PEFT) using LoRA or DoRA is selected with `--mode lora` or `--mode dora`:

```bash
uv run python -m torch.distributed.run --nproc-per-node=8 scripts/training/run_recipe.py \
--recipe qwen25_vl_3b_peft_config \
--mode lora \
--step-func vlm_step \
--from $MEGATRON_MODEL_PATH \
--dataset cord-v2 \
train.global_batch_size=<batch size> \
checkpoint.save=$SAVE_DIR/<experiment name>
```

Use an SFT recipe with `--mode sft` for full finetuning.

You can also combine PEFT with freeze options to control which components are trainable:
- `model.freeze_language_model`: Set to `True` to freeze the language model
- `model.freeze_vision_model`: Set to `True` to freeze the vision encoder
- `model.freeze_vision_projection`: Set to `True` to freeze the vision projection layer

Example with LoRA and freeze options:
```bash
uv run python -m torch.distributed.run --nproc-per-node=8 scripts/training/run_recipe.py \
--recipe qwen25_vl_3b_peft_config \
--mode lora \
--step-func vlm_step \
--from $MEGATRON_MODEL_PATH \
--dataset cord-v2 \
model.freeze_language_model=True \
model.freeze_vision_model=False \
model.freeze_vision_projection=False \
checkpoint.save=$SAVE_DIR/<experiment name>
```


## Example Datasets

Megatron Bridge supports various vision-language dataset examples which can be used to finetune Qwen 2.5 VL:
| Dataset | Dataset preset | Description |
|---------|------------|-------------|
| [cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | `cord_v2` | OCR receipts: Single-image-text dataset for receipt understanding, outputs xml-like annotated text. |
| [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) | `medpix` | Medical VQA: Single-image question-answer dataset covering clinical medical images and free-form answers. |
| [The Cauldron (Raven subset)](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | `raven` | Visual reasoning: Multi-image, vision reasoning dataset for analogical reasoning in different visual layouts. |

For a built-in dataset, set only `dataset.source.dataset_name`. Raven and RDR are train-only, so also set `dataset.do_validation=false dataset.do_test=false`; MedPix requires `dataset.do_test=false`. To replace the recipe's preset with a custom native-conversation source from the CLI, set `dataset.source.dataset_name=null` and `dataset.source.path_or_dataset=<id>`. Custom non-native schemas additionally require a registered `dataset.source.schema_adapter`.


## Hugging Face Model Cards
- Qwen2.5-VL-3B: `https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct`
- Qwen2.5-VL-7B: `https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct`
- Qwen2.5-VL-32B: `https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct`
- Qwen2.5-VL-72B: `https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct`
