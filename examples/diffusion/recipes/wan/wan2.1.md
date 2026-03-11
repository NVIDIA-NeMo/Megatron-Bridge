## 🚀 Megatron WAN

### 📋 Overview
An open-source implementation of [WAN 2.1](https://github.com/Wan-Video/Wan2.1) (large-scale text-to-video/image generative models) built on top of [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for scalable and efficient training. It supports advanced parallelism strategies (data, tensor, sequence, and context parallelism) and optimized kernels (e.g., Transformer Engine fused attention).

---

### 📦 Dataset Preparation
This recipe uses NVIDIA's [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) as an efficient multi-modal data loader. Datasets should be in the WebDataset-compatible format (typically sharded `.tar` archives). Energon supports large-scale distributed loading, sharding, and sampling for video-text and image-text pairs. Set `dataset.path` to your WebDataset directory or shard pattern. See Megatron-Energon docs for format details, subflavors, and advanced options.

If you do not have a dataset yet or only need to validate performance/plumbing, see the "Quick Start with Mock Dataset" section below.

---

#### 🗂️ Dataset Preparation Example
Starting with a directory containing raw .mp4 videos and their corresponding .json metadata files containing captions, you can turn the data into WAN-ready WebDataset shards using our helper script. We then use Energon to process those shards and create its metadata. After this, you can set training script's `dataset.path` argument to the output processed data folder and start training.

```bash
# 1) Define your input (raw videos) and output (WebDataset shards) folders. For example:
DATASET_SRC=/opt/raw_videos            # contains .mp4 and  per-video .jsonl captions
DATASET_PATH=/opt/wan_webdataset      # output WebDataset shards

# 2) (Optional) If your WAN models require auth on first download
export HF_TOKEN=<your_huggingface_token>

# 3) Create WAN shards with latents + text embeddings
# Wan's VAE encoder and T5 encoder is used to extract videos' latents and caption embeddings offline before training, using the following core arugments:
#    --output_format: select output format of "automodel" or "energon"
#    --height/--width: control resize target (832x480 is supported for both 1.3B and 14B model)
#    --center-crop: run center crop to exact target size after resize
#    --mode: to process video or frames of video
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 8 \
  examples/diffusion/recipes/wan/prepare_dataset_wan.py \
  --video_folder "${DATASET_SRC}" \
  --output_dir "${DATASET_PATH}" \
  --output_format energon \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --mode video \
  --height 480 \
  --width 832 \
  --resize_mode bilinear \
  --center-crop

# 4) Use Energon to process shards and create its metadata/spec
energon prepare "${DATASET_PATH}"
# In the interactive prompts:
# - Enter a train/val/test split, e.g., "8,1,1"
# - When asked for the sample type, choose: "Crude sample (plain dict for cooking)"
```

What gets produced:
- Each shard contains:
  - pth: contain WAN video latents
  - pickle: contain text embeddings
  - json: contain useful side-info (text caption, sizes, processing choices, etc.)
- Energon writes a `.nv-meta` directory with dataset info and a `dataset.yaml` you can version/control.

You’re ready to launch training. In the training config, we will point the WAN config (or CLI overrides) to the processed data output direcotry as `dataset.path=${DATASET_PATH}`.

---

### 🐳 Build Container

Please follow the instructions in the container section of the main README:

- DFM container guide: https://github.com/NVIDIA-NeMo/DFM#-built-your-own-container

---

### 🏋️ Pretraining

This recipe leverages sequence packing to maximize throughput. When a batch containing videos with different shapes or resolution, naive batching and padding method require significant numner of padded tokens, due to the inherit size of videos. Sequence packing stacks multiple samples (with dirrent resolutions) into a single sequence instead of padding; hence no computation is wasted on padded tokens. When using sequence packing:
- Set `train.micro_batch_size=1` and `dataset.micro_batch_size=1`
- Ensure `model.qkv_format=thd` (required with context parallelism and recommended with sequence packing)

Multiple parallelism techniques including tensor, sequence, and context parallelism are supported and configurable per your hardware.

Wan training is driven by `examples/diffusion/recipes/wan/pretrain_wan.py`, which supports both a YAML config file and CLI overrides.

The script exposes a `--training-mode` with `pretrain` and `finetune` presets for flow-matching hyperparameters as a starting point for experiments. This presets specify that pretraining uses noisier, biased sampling (e.g., logit-normal, higher logit_std, lower flow_shift) for stability and broad learning, while finetuning uses uniform, lower-noise settings (e.g., uniform sampling, lower logit_std, higher flow_shift) to refine details and improve quality.

**Notes**: If you use `logger.wandb_project` and `logger.wandb_exp_name`, export `WANDB_API_KEY`.

#### Pretraining script example

We provide example scripts for running 1.3B and 14B model sizes on mock dataset (see `wan_1_3B.yaml` and `wan_14B.yaml` under `examples/megatron/recipes/wan/conf`). From these starting points, users can set their own configuration by copy one of the example override configs and update it with your settings (e.g., with actual processed data path, and specific configurations based on available hardware, etc.). Users can learn more about arugments detail at [Megatron-Bridge docs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/megatron-lm-to-megatron-bridge.md).


```bash
cp examples/megatron/recipes/wan/conf/wan_1_3B.yaml examples/megatron/recipes/wan/conf/my_wan.yaml
# Edit my_wan.yaml to set:
# - dataset.path: Path to your WebDataset directory
# - train.global_batch_size/micro_batch_size: Keep micro_batch_size=1
# - model.tensor_model_parallel_size / model.context_parallel_size: Based on GPUs
# - checkpoint.save and checkpoint.load: Checkpoint directory
```

Then run:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/diffusion/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  --config-file examples/megatron/recipes/wan/conf/my_wan.yaml
```

You can also override any config values from the command line. For example:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/diffusion/recipes/wan/pretrain_wan.py \
  --config-file examples/megatron/recipes/wan/conf/my_wan.yaml \
  --training-mode pretrain \
  dataset.path=/opt/wan_webdataset \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  model.tensor_model_parallel_size=2 \
  model.context_parallel_size=4 \
  checkpoint.save=/opt/pretrained_checkpoints \
  checkpoint.load=/opt/pretrained_checkpoints
```

#### 🧪 Quick Start with Mock Dataset
If you want to run without a real dataset (for debugging or performance measurement), pass `--mock`:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/diffusion/recipes/wan/pretrain_wan.py \
  --config-file examples/megatron/recipes/wan/conf/wan_1_3B.yaml \
  --training-mode pretrain \
  --mock
```

You may adjust mock shapes (`F_latents`, `H_latents`, `W_latents`) and packing behavior (`number_packed_samples`) in `WanMockDataModuleConfig` (see `src/megatron/bridge/diffusion/recipes/wan/wan.py`) to simulate different data scenarios.

---

### 🎬 Inference

After training, users can run inferencing with `examples/diffusion/recipes/wan/inference_wan.py`. Set `--checkpoint_step` to use specific checkpoint for inference. Set `--sizes` and `--frame_nums` to specify video shape (frames, height, width). Set `--sample_steps` (default to 50) for number of noise diffusion steps.

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 1 \
  examples/diffusion/recipes/wan/inference_wan.py  \
  --task t2v-1.3B \
  --frame_nums 81 \
  --sizes 480*832 \
  --checkpoint_dir /opt/pretrained_checkpoints \
  --checkpoint_step 10000 \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --sample_steps 50
```

**Note**: Current inference path is single-GPU. Parallel inference is not yet supported.


---

### 🔄 Checkpoint Converting (optional)

If you plan to fine-tune Wan using a pre-trained model, you must first convert the HuggingFace checkpoint (e.g., `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) into the Megatron format. The provided script supports bidirectional conversion, allowing you to move between HuggingFace and Megatron formats as needed.

Follow these steps to convert your checkpoints:
```
  # Download the HF checkpoint locally
  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir /root/.cache/huggingface/wan2.1 \
  --local-dir-use-symlinks False

  # Import a HuggingFace model to Megatron format
  python examples/diffusion/recipes/wan/conversion/convert_checkpoints.py import \
  --hf-model /root/.cache/huggingface/wan2.1 \
  --megatron-path /workspace/checkpoints/megatron_checkpoints/wan_1_3b

  # Export a Megatron checkpoint to HuggingFace format
  python examples/diffusion/recipes/wan/conversion/convert_checkpoints.py export \
  --hf-model /root/.cache/huggingface/wan2.1 \
  --megatron-path /workspace/checkpoints/megatron_checkpoints/wan_1_3b/iter_0000000 \
  --hf-path /workspace/checkpoints/hf_checkpoints/wan_1_3b_hf

```

**Note**: The exported checkpoint from Megatron to HuggingFace (`/workspace/checkpoints/hf_checkpoints/wan_1_3b_hf`) contains only the DiT transformer weights. To run inference, you still require the other pipeline components (VAE, text encoders, etc.).
To assemble a functional inference directory:
- Duplicate the original HF checkpoint directory.
- Replace the `./transformer` folder in that directory with your newly exported `/transformer` folder.

---

### ⚡ Parallelism Support

The table below shows current parallelism support for different model sizes:

| Model | Data Parallel | Tensor Parallel | Sequence Parallel | Context Parallel | FSDP |
|---|---|---|---|---|---|
| 1.3B | ✅ | ✅ | ✅ | ✅ |Coming Soon|
| 14B  | ✅ | ✅ | ✅ | ✅ |Coming Soon|


### References
Wan Team. (2025). Wan: Open and advanced large-scale video generative models (Wan 2.1). GitHub. https://github.com/Wan-Video/Wan2.1/