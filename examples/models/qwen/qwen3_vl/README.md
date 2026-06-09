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
uv run python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-VL-8B-Instruct \
  --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct
```

### Export Megatron → HF
```bash
uv run python examples/conversion/convert_checkpoints.py export \
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

See the [sft_unpacked.sh](sft_unpacked.sh) script for full parameter fine-tuning with configurable model parallelisms, with unpacked sequences.
See the [sft.sh](sft.sh) script for full parameter fine-tuning with sequence-packing.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [peft_unpacked.sh](peft_unpacked.sh) script for LoRA fine-tuning with configurable tensor and pipeline parallelism, with unpacked sequences.
See the [peft.sh](peft.sh) script for LoRA fine-tuning with sequence-packing.

**Note:** LoRA/DoRA significantly reduces memory requirements, allowing for larger batch sizes and fewer GPUs.

## Controlling visual tokens computation budget
Three independent CLI-overridable controls bound a sample's GPU cost. They compose:
- **`dataset.min_pixels` / `dataset.max_pixels`** — image/frame resolutions lower and upper bound (defaults `200704` / `1003520`). 
- **`dataset.max_num_images` / `dataset.max_num_frames`** - limit count of images/frames (defaults `10` / `60`). Too many images → sample is dropped. Too many frames → frame list truncated.
- **`dataset.max_visual_tokens`** — limit total visual tokens across all images and frames in a sample, computed post-rescaling as `prod(T,H,W) // merge_size²` (default `16384`; set to `None` to disable). Catches cases the other two miss (few images at high resolution, or many at low resolution). Exceeding samples are dropped.

## Finetuning with Energon Dataset

Follow the instructions [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/multimodal#pretraining) to prepare `LLaVA-Pretrain` dataset in Energon format. Change the file `.nv-meta/dataset.yaml` to the following:

```yaml
__module__: megatron.bridge.models.qwen_vl.data.energon
__class__: ChatMLWebdataset
field_map:
  imgs: jpg
  conversation: json
```

Then, update the dataset path (`dataset.path=/path/to/energon/dataset`) in [peft_energon.sh](peft_energon.sh) and run the script.

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

2. Run the following script to convert the data to webdataset format:

    ```
    python examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \
        --source-dir/path/to/Mantis-Instruct-LLaVA \
        --output-dir /path/to/Mantis-Instruct-LLaVA/wds \
        --max-samples-per-tar 10000
    ```

3. Run the following command to convert to megatron-energon format:

    ```
    cd /path/to/Mantis-Instruct-LLaVA/wds
    energon prepare ./
    ```

    select the following values for the presented options:

    ```
    > Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
    > Do you want to create a dataset.yaml interactively? [Y/n]: Y
    > Please enter a number to choose a class: 9 (VQASample)
    > Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y
    > Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
    > Please enter a webdataset field name for 'context' (<class 'str'>): json[0][value]
    > Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): json[1][value]
    > Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):
    ```

4. Change the file `.nv-meta/dataset.yaml` to the following:

    ```yaml
    __module__: megatron.bridge.models.qwen_vl.data.energon
    __class__: ChatMLWebdataset
    field_map:
      imgs: jpgs
      conversation: json
    ```

Follow previous instruction to run the finetuning with the prepared dataset.


## Evaluation

Coming soon.
