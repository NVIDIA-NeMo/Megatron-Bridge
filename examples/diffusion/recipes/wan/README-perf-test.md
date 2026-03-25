## WAN Model Setup and Usage (for Perf Test)

This guide provides concise steps to set up the environment and run WAN pretraining and inference. It pins repo commits and shows explicit commands for the 1.3B and 14B configurations.

## Container Launch

```bash
CONT="nvcr.io/nvidia/nemo:25.09.00"
MOUNT="/lustre/fsw/:/lustre/fsw/"

srun -t 02:00:00 \
  --account <your_slurm_account> \
  -N 1 \
  -J <your_job_name> \
  -p batch \
  --exclusive \
  --container-image="${CONT}" \
  --container-mounts="${MOUNT}" \
  --pty bash
```

## Setup Inside the Container

Setup DFM, Megatron-Bridge, Megatron-LM with specific commits, and other dependencies.

```bash
cd /opt/

# DFM (pinned)
git clone --no-checkout https://github.com/NVIDIA-NeMo/DFM.git
git -C DFM checkout 174bb7b34de002ebbbcae1ba8e2b12363c7dee01
export DFM_PATH=/opt/DFM

# Megatron-Bridge (pinned)
rm -rf /opt/Megatron-Bridge
git clone --no-checkout https://github.com/huvunvidia/Megatron-Bridge.git
git -C Megatron-Bridge checkout 713ab548e4bfee307eb94a7bb3f57c17dbb31b50

# Megatron-LM (pinned)
rm -rf /opt/Megatron-LM
git clone --no-checkout https://github.com/NVIDIA/Megatron-LM.git
git -C Megatron-LM checkout ce8185cbbe04f38beb74360e878450f2e8525885

# Python path
export PYTHONPATH="${DFM_PATH}/.:/opt/Megatron-Bridge/.:/opt/Megatron-LM"

# Python deps
python3 -m pip install --upgrade diffusers==0.35.1
pip install easydict imageio imageio-ffmpeg
```

## Pretraining
Set data path and checkpoint directory:

```bash
DATASET_PATH="/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/datasets/shared_datasets/processed_arrietty_scene_automodel"
EXP_NAME=wan_debug_perf
CHECKPOINT_DIR="/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/results/wan_finetune/${EXP_NAME}"

export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_api_key>
cd ${DFM_PATH}
```


### 1.3B configuration

```bash
NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=8 examples/diffusion/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=4 \
  model.crossattn_emb_size=1536 \
  model.hidden_size=1536 \
  model.ffn_hidden_size=8960 \
  model.num_attention_heads=12 \
  model.num_layers=30 \
  model.qkv_format=thd \
  dataset.path="${DATASET_PATH}" \
  checkpoint.save="${CHECKPOINT_DIR}" \
  checkpoint.load="${CHECKPOINT_DIR}" \
  checkpoint.load_optim=false \
  checkpoint.save_interval=200 \
  optimizer.lr=5e-6 \
  optimizer.min_lr=5e-6 \
  train.eval_iters=0 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=0 \
  model.seq_length=2048 \
  dataset.seq_length=2048 \
  train.global_batch_size=2 \
  train.micro_batch_size=1 \
  dataset.global_batch_size=2 \
  dataset.micro_batch_size=1 \
  logger.log_interval=1 \
  logger.wandb_project="wan" \
  logger.wandb_exp_name="${EXP_NAME}" \
  logger.wandb_save_dir="${CHECKPOINT_DIR}"
```

### 14B configuration

```bash
NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=8 examples/diffusion/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  model.tensor_model_parallel_size=2 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=4 \
  model.recompute_granularity=full \
  model.recompute_method=uniform \
  model.recompute_num_layers=1 \
  model.crossattn_emb_size=5120 \
  model.hidden_size=5120 \
  model.ffn_hidden_size=13824 \
  model.num_attention_heads=40 \
  model.num_layers=40 \
  model.qkv_format=thd \
  dataset.path="${DATASET_PATH}" \
  checkpoint.save="${CHECKPOINT_DIR}" \
  checkpoint.load="${CHECKPOINT_DIR}" \
  checkpoint.load_optim=false \
  checkpoint.save_interval=200 \
  optimizer.lr=5e-6 \
  optimizer.min_lr=5e-6 \
  train.eval_iters=0 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=0 \
  model.seq_length=2048 \
  dataset.seq_length=2048 \
  train.global_batch_size=2 \
  train.micro_batch_size=1 \
  dataset.global_batch_size=2 \
  dataset.micro_batch_size=1 \
  logger.log_interval=1 \
  logger.wandb_project="wan" \
  logger.wandb_exp_name="${EXP_NAME}" \
  logger.wandb_save_dir="${CHECKPOINT_DIR}"
```

### Using mock data (optional, for debugging)

- Using `--mock` argument.
- Adjust `video_size` (F_latents, H_latents, W_latents) and `number_packed_samples` of `WanMockDataModuleConfig` in `wan.py`. Total `seq_len = F * H * W * number_packed_samples`.

## Inference

```bash
cd ${DFM_PATH}
export HF_TOKEN=<your_huggingface_token>

T5_DIR="/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/wan_checkpoints/t5"
VAE_DIR="/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/wan_checkpoints/vae"
CKPT_DIR="/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/datasets/shared_checkpoints/megatron_checkpoint_1.3B"

NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=1 examples/diffusion/recipes/wan/inference_wan.py  \
  --task t2v-1.3B \
  --sizes 480*832 \
  --checkpoint_dir "${CKPT_DIR}" \
  --checkpoint_step 0 \
  --t5_checkpoint_dir "${T5_DIR}" \
  --vae_checkpoint_dir "${VAE_DIR}" \
  --frame_nums 81 \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --tensor_parallel_size 1 \
  --context_parallel_size 1 \
  --pipeline_parallel_size 1 \
  --sequence_parallel False \
  --base_seed 42 \
  --sample_steps 50
```

## Notes

- Replace placeholders (tokens, account, dataset/checkpoint paths) with your own.
- Keep the specified commit hashes for compatibility.
- `NVTE_FUSED_ATTN=1` enables fused attention where supported.
