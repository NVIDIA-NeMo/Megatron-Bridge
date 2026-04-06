# Preparing OpenVid-1M Dataset for Wan Fine-tuning

This guide walks through downloading a subset of the [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) dataset, preprocessing it, and converting it into the Energon WebDataset format for Wan training.

## Prerequisites

- `wget`, `unzip` available on your system
- A Hugging Face account with access token (`HF_TOKEN`)
- Python environment with Megatron-Bridge installed

## Step 1: Download a Small Portion of OpenVid-1M

```bash
PROCESSED_DATA_PATH=<path/to/dataset_dir>
cd ${PROCESSED_DATA_PATH}

export HF_TOKEN=<HF_API_KEY>
wget https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD/OpenVidHD_part_1.zip
wget https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv
```

## Step 2: Extract a Subset of Videos

Extract 200 videos as an example (adjust the number as needed):

```bash
unzip -Z1 OpenVidHD_part_1.zip | head -n 200 | xargs -I {} unzip OpenVidHD_part_1.zip "{}" -d OpenVidHD_part_1
```

## Step 3: Create Per-Video Caption JSON Files

The preprocessing script reads the CSV caption file and creates a sidecar `.json` file next to each `.mp4`:

```bash
cd ${MBRIDGE_PATH}
python examples/diffusion/recipes/wan/prepare_dataset/openvid1M_preprocess.py \
  --csv_path ${PROCESSED_DATA_PATH}/OpenVidHD.csv \
  --video_dir ${PROCESSED_DATA_PATH}/OpenVidHD_part_1
```

Each video will have a corresponding `.json` file with `video` and `caption` fields.

## Step 4: Convert to Energon WebDataset Format

Run the dataset preparation script using `torchrun` with 8 GPUs:

```bash
DATASET_SRC=${PROCESSED_DATA_PATH}/OpenVidHD_part_1
DATASET_PATH=${PROCESSED_DATA_PATH}/prepared_dataset_wds

torchrun --nproc_per_node=8 \
  examples/diffusion/recipes/wan/prepare_dataset/prepare_dataset_wan.py \
  --video_folder "${DATASET_SRC}" \
  --output_dir "${DATASET_PATH}" \
  --output_format energon \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --mode video \
  --height 480 \
  --width 832 \
  --resize_mode bilinear \
  --center-crop
```

The output directory `${DATASET_PATH}` can then be passed directly as `dataset.path` when launching Wan training.

## Step 5: Launch Wan Training

Set the required environment variables and run the training script:

```bash
DATASET_PATH=${PROCESSED_DATA_PATH}/prepared_dataset_wds
CHECKPOINT_DIR=<path/to/save/checkpoints>
EXP_NAME=<experiment_name>

NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe wan_1_3b_pretrain_config \
    --step_func wan_step \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.context_parallel_size=4 \
    model.sequence_parallel=false \
    model.qkv_format=thd \
    model.bias_activation_fusion=false \
    dataset.path=${DATASET_PATH} \
    dataset.packing_buffer_size=200 \
    dataset.num_workers=10 \
    checkpoint.save=${CHECKPOINT_DIR} \
    checkpoint.load=${CHECKPOINT_DIR} \
    checkpoint.load_optim=false \
    checkpoint.save_interval=200 \
    optimizer.lr=5e-5 \
    optimizer.min_lr=5e-5 \
    train.eval_iters=0 \
    scheduler.lr_decay_style=constant \
    scheduler.lr_warmup_iters=1000 \
    model.seq_length=40000 \
    dataset.seq_length=40000 \
    train.global_batch_size=8 \
    train.micro_batch_size=1 \
    dataset.global_batch_size=8 \
    dataset.micro_batch_size=1 \
    logger.log_interval=1
```
