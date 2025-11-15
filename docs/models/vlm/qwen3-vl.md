# Qwen3-VL

Qwen3-VL is the latest generation of vision-language models from Alibaba Cloud, supporting multimodal understanding across text, images, and videos. Qwen3-VL includes both dense models and Mixture-of-Experts (MoE) variants for improved efficiency.

NeMo Megatron Bridge supports finetuning Qwen3-VL models (8B dense and 30B MoE variants).

```{tip}
We use the following environment variables throughout this page
- `HF_MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct` (or `Qwen/Qwen3-VL-30B-A3B-Instruct` for MoE)
- `MEGATRON_MODEL_PATH=/models/Qwen3-VL-8B-Instruct` (feel free to set your own path)
Unless explicitly stated, any megatron model path in the commands below should NOT contain the iteration number 
`iter_xxxxxx`. For more details on checkpointing, please see 
[here](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html#checkpoint-contents) 
```

## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF model to your desired `$MEGATRON_MODEL_PATH`, run the following command.
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model $HF_MODEL_PATH \
--megatron-path $MEGATRON_MODEL_PATH
```

### Export Megatron → HF
You can export a trained model with the following command.
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model $HF_MODEL_PATH \
--megatron-path <trained megatron model path> \
--hf-path <output hf model path>
```

### Run In-Framework Inference on Converted Checkpoint
You can run a quick sanity check on the converted checkpoint with the following command.
```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path $HF_MODEL_PATH \
--megatron_model_path $MEGATRON_MODEL_PATH \
--image_path <example image path> \
--prompt "Describe this image." \
--max_new_tokens 100
```

## Finetuning Recipes
Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: to specify a checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Full Finetuning

Example usage for full parameter finetuning:

```bash
torchrun --nproc-per-node=8 examples/recipes/qwen_vl/finetune_qwen_vl.py \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--recipe qwen3_vl_8b_finetune_config \
--dataset-type hf \
dataset.maker_name=make_cord_v2_dataset \
train.global_batch_size=<batch size> \
train.train_iters=<number of iterations> \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name>
```

For MoE models with expert parallelism:
```bash
torchrun --nproc-per-node=8 examples/recipes/qwen_vl/finetune_qwen_vl.py \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--recipe qwen3_vl_3b_active_30b_moe_finetune_config \
--dataset-type hf \
dataset.maker_name=make_cord_v2_dataset \
train.global_batch_size=<batch size> \
train.train_iters=<number of iterations> \
checkpoint.save=$SAVE_DIR/<experiment name>
```

Note:
- The `--recipe` parameter selects the model configuration:
  - `qwen3_vl_8b_finetune_config` - for 8B dense model
  - `qwen3_vl_3b_active_30b_moe_finetune_config` - for 30B MoE model
- For dataset formats and additional information, refer to the [Qwen2.5-VL documentation]
- See the full script with examples at [`examples/recipes/qwen_vl/finetune_qwen_vl.py`](../../../examples/recipes/qwen_vl/finetune_qwen_vl.py)

## Hugging Face Model Cards
- Qwen3-VL-8B: `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`
- Qwen3-VL-30B-A3B (MoE): `https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct`
