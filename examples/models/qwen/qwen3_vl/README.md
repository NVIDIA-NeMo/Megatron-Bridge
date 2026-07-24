# Qwen 3 VL - Vision Language Model

This directory contains example scripts for Qwen 3 vision-language models.

For model introduction and architecture details, see the [Qwen 3 - VL documentation](../../../../docs/models/qwen/qwen3-vl.md).

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
./scripts/conversion/convert.sh import \
  --hf-model Qwen/Qwen3-VL-8B-Instruct \
  --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct
```

### Export Megatron → HF
```bash
./scripts/conversion/convert.sh export \
  --hf-model Qwen/Qwen3-VL-8B-Instruct \
  --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct/iter_0000000 \
  --hf-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct-hf-export
```

## Inference

### Run Inference on Converted Checkpoint

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_vlm.py \
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

| Model | Full SFT | PEFT |
| --- | --- | --- |
| 8B | `qwen3_vl_8b_sft_config` (2 GPUs) | `qwen3_vl_8b_peft_config` (1 GPU) |
| 30B-A3B | `qwen3_vl_30b_a3b_sft_config` (8 GPUs) | `qwen3_vl_30b_a3b_peft_config` (4 GPUs) |
| 235B-A22B | `qwen3_vl_235b_a22b_sft_config` (32 GPUs) | `qwen3_vl_235b_a22b_peft_config` (16 GPUs) |

Before training, ensure the following environment variables are set:
1. `HF_TOKEN`: to download models from HF Hub (if required)
2. `HF_HOME`: (optional) to avoid re-downloading models and datasets
3. `WANDB_API_KEY`: (optional) to enable WandB logging

### Pretrain

The shipped pretraining recipes use synthetic VLM data for bring-up: `qwen3_vl_8b_pretrain_mock_config`, `qwen3_vl_30b_a3b_pretrain_mock_config`, and `qwen3_vl_235b_a22b_pretrain_mock_config`.

### Supervised Fine-Tuning (SFT)

See the [sft_unpacked.sh](sft_unpacked.sh) script for full parameter fine-tuning with configurable model parallelisms, with unpacked sequences.
See the [sft.sh](sft.sh) script for full parameter fine-tuning with sequence-packing.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [peft_unpacked.sh](peft_unpacked.sh) script for LoRA fine-tuning with configurable tensor and pipeline parallelism, with unpacked sequences.
See the [peft.sh](peft.sh) script for LoRA fine-tuning with sequence-packing.

**Note:** LoRA/DoRA significantly reduces memory requirements, allowing for larger batch sizes and fewer GPUs.

For hosted or local Hugging Face data and a complete one-GPU Qwen3-VL run, start with the
[Hugging Face multimodal tutorial](../../../../tutorials/data/hf-multimodal/README.md). For sharded WebDataset data,
use the [multimodal Energon tutorial](../../../../tutorials/data/energon/README.md).

## Controlling Energon visual-token computation budget

These fields belong to `QwenVLEnergonTaskEncoderConfig`; they do not configure the Direct-HF source.

Three independent CLI-overridable controls bound a sample's GPU cost. They compose:
- **`dataset.task_encoder.min_pixels` / `dataset.task_encoder.max_pixels`** — image/frame resolution lower and upper bounds (defaults `200704` / `1003520`).
- **`dataset.task_encoder.max_num_images` / `dataset.task_encoder.max_num_frames`** — image/frame count limits (defaults `10` / `60`). Too many images cause a sample to be dropped; excess frames are truncated.
- **`dataset.task_encoder.max_visual_tokens`** — total visual-token limit across all images and frames, computed post-rescaling as `prod(T,H,W) // merge_size²` (default `16384`; set to `None` to disable). This catches cases the other two limits miss. Exceeding samples are dropped.

## Finetuning with Energon Dataset

The [multimodal Energon tutorial](../../../../tutorials/data/energon/README.md) documents the tar-member contract, version-compatible indexing, canonical `ChatMLWebdataset` YAML, and a one-GPU launch. [peft_energon.sh](peft_energon.sh) provides a larger packing/CP experiment matrix after the baseline works.

### Expected Training Dynamics
We provide a [Weights & Biases report](https://api.wandb.ai/links/nvidia-nemo-fw-public/lczz4ixx) for the expected loss curves and grad norms.

## Dataset with Multiple Images

Below is an example for finetuning on a dataset containing multiple images in a sample, using a subset of [TIGER-Lab/Mantis-Instruct](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct) dataset.

1. Download the `llava_665k_multi` subset of TIGER-Lab/Mantis-Instruct dataset from Hugging Face and unzip the images folder (NOTE: 44GB of disk space required):

    ```
    pip install -U "huggingface_hub[cli]"
    huggingface-cli download TIGER-Lab/Mantis-Instruct \
        --include "llava_665k_multi/*" \
        --repo-type dataset \
        --local-dir /path/to/Mantis-Instruct-LLaVA    
    ```

2. Convert and index the downloaded subsets. The helper writes deterministic train/validation shards, calls Energon's preparation API, and writes `.nv-meta/dataset.yaml`:

    ```bash
    uv run python examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \
        --source-dir /path/to/Mantis-Instruct-LLaVA \
        --output-dir /path/to/Mantis-Instruct-LLaVA/wds \
        --max-samples-per-tar 10000 \
        --validation-fraction 0.01
    ```

3. Set `dataset.path=/path/to/Mantis-Instruct-LLaVA/wds` in the canonical Energon launch. No interactive class selection or manual YAML replacement is required.


## Evaluation

Coming soon.
