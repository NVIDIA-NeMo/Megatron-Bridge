# Qwen 3 VL - Vision Language Model

This directory contains example scripts for Qwen 3 vision-language models.

For model introduction and architecture details, see the [Qwen 3 - VL documentation](../../../../docs/models/vlm/qwen3-vl.md).

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
  --hf-model Qwen/Qwen3-VL-8B-Instruct \
  --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct
```

### Export Megatron → HF
```bash
python examples/conversion/convert_checkpoints.py export \
  --hf-model Qwen/Qwen3-VL-8B-Instruct \
  --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct/iter_0000000 \
  --hf-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct-hf-export
```

## Inference

### Run Inference on Converted Checkpoint

```bash
python -m torch.distributed.run --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path Qwen/Qwen3-VL-8B-Instruct \
  --megatron_model_path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct/iter_0000000 \
  --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
  --prompt "Describe this image." \
  --max_new_tokens 100 
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward.
- You can also use image URLs: `--image_path="https://example.com/image.jpg"`

See the [inference.sh](inference.sh) script for commands to:
- Run inference with Hugging Face checkpoints
- Run inference with imported Megatron checkpoints
- Run inference with exported Hugging Face checkpoints

**Expected output:**
```
...
Generation step 46
Generation step 47
Generation step 48
Generation step 49
======== GENERATED TEXT OUTPUT ========
Image: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png
Prompt: Describe this image.
Generated: <|im_start|>user
<|vision_start|><|image_pad|><|image_pad|>
...
<|image_pad|><|vision_end|>Describe this image.<|im_end|>
<|im_start|>assistant
This image displays a **technical specifications table** comparing two variants of NVIDIA's H100 GPU: the **H100 SXM** and the **H100 NVL**.

The table is organized into rows, each detailing a specific performance or hardware characteristic, with columns showing the corresponding value for each GPU variant.

Here is a breakdown of the key specifications:

**Performance (FLOPS & TOPS):**
*   **FP64 (Double Precision):** The
=======================================
```

## Finetune Recipes

- Available recipes:
  - `qwen3_vl_8b_finetune_config`: Finetuning for 8B VL model with PEFT support
  - `qwen3_vl_30b_a3b_finetune_config`: Finetuning for 30B-A3B VL model with PEFT support
  - `qwen3_vl_235b_a22b_finetune_config`: Finetuning for 235B-A22B VL model with PEFT support
    
Before training, ensure the following environment variables are set:
1. `HF_TOKEN`: to download models from HF Hub (if required)
2. `HF_HOME`: (optional) to avoid re-downloading models and datasets
3. `WANDB_API_KEY`: (optional) to enable WandB logging

### Pretrain

- Available recipes:
  - `qwen3_vl_8b_pretrain_config`: Pretraining for 8B VL model with PEFT support
  - `qwen3_vl_30b_a3b_pretrain_config`: Pretraining for 30B-A3B VL model with PEFT support
  - `qwen3_vl_235b_a22b_pretrain_config`: Pretraining for 235B-A22B VL model with PEFT support

### Supervised Fine-Tuning (SFT)

See the [sft_unpacked.sh](sft_seq_unpacked.sh) script for full parameter fine-tuning with configurable model parallelisms, with unpacked sequences..
See the [sft_packed.sh](sft_seq_packed.sh) script for for full parameter fine-tuning with sequence-packing.

W&B report coming soon.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [peft_unpacked.sh](peft_seq_unpacked.sh) script for LoRA fine-tuning with configurable tensor and pipeline parallelism, with unpacked sequences.
See the [peft_packed.sh](peft_seq_packed.sh) script for LoRA fine-tuning with sequence-packing.

W&B report coming soon.

**Note:** LoRA/DoRA significantly reduces memory requirements, allowing for larger batch sizes and fewer GPUs.

## Finetuning with Energon Dataset

Follow the instructions [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/multimodal#pretraining) to prepare `LLaVA-Pretrain` dataset in Energon format. Change the file `.nv-meta/dataset.yaml` to the following:

```yaml
__module__: megatron.bridge.recipes.qwen_vl.data.energon.task_encoder
__class__: ChatMLWebdataset
field_map:
  imgs: jpg
  conversation: json
```

Then, update the dataset path (`dataset.path=/path/to/energon/dataset`) in [energon_test.sh](energon_test.sh) and run teh script.

## Evaluation

Coming soon.
