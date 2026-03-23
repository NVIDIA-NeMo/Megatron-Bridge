# Nemotron 3 Examples

This directory contains example scripts for Nemotron 3 language models.

For model introduction and architecture details, see the Nemotron 3 documentation.

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

### Import HF → Megatron

To import the HF model to your desired Megatron path:

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --megatron-path ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --trust-remote-code
```

### Export Megatron → HF

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --megatron-path ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/iter_0000000 \
    --hf-path ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-hf-export
```

### Round-trip Validation

Multi-GPU round-trip validation between formats:

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --megatron-load-path ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/iter_0000000 \
    --tp 2 --pp 2 \
    --trust-remote-code
```

## Training Recipes

- See: [bridge.recipes.nemotronh](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py)
- Available recipes:
  - `nemotron_3_nano_pretrain_config`: Pretraining configuration
  - `nemotron_3_nano_finetune_config`: Finetuning configuration with PEFT support

Before training, ensure the following are configured:
1. **Container Image**: Set `CONTAINER_IMAGE` in the SLURM scripts to your container path
2. **Container Mounts**: (optional) Set `CONTAINER_MOUNTS` for data and workspace directories
3. **Environment Variables**:
   - `HF_TOKEN`: to download models from HF Hub (if required)
   - `HF_HOME`: (optional) to avoid re-downloading models and datasets
   - `WANDB_API_KEY`: (optional) to enable WandB logging

All training scripts use SLURM for containerized multi-node training.

### Pretrain

See the [slurm_pretrain.sh](slurm_pretrain.sh) script for pretraining with configurable model parallelisms.

W&B report coming soon.

### Supervised Fine-Tuning (SFT)

See the [slurm_sft.sh](slurm_sft.sh) script for full parameter fine-tuning.

W&B report coming soon.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [slurm_peft.sh](slurm_peft.sh) script for LoRA fine-tuning.

W&B report coming soon.

## Evaluation

Within the NeMo Framework container, evaluation uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via the NeMo Evaluator. All benchmarks supported by lm-eval can be run against a deployed Megatron-Bridge model.

For full documentation, see: https://docs.nvidia.com/nemo/evaluator/latest/deployment/nemo-fw/mbridge.html

### nemo-run Script

The script `evaluation_with_nemo_run.py` from the Evaluator repo submits a 2-task Slurm job: one task deploys the model via Ray Serve, the other runs lm-eval against the endpoint. Both run concurrently; the deploy stops automatically when evaluation finishes.

#### Parallelism for Nemotron-3-Nano-30B-A3B

This is a MoE model with 32 experts. Use **TP=2, EP=8** — do not use TP=8 without EP. TransformerEngine's grouped GEMM kernel requires Expert Parallelism to distribute experts across GPUs; high TP alone causes deadlock.

#### SSHTunnel

```bash
NEMORUN_HOME=${WORKSPACE}/.nemorun /opt/venv/bin/python /opt/Evaluator/scripts/evaluation_with_nemo_run.py \
    --megatron_checkpoint ${WORKSPACE}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/iter_0000000 \
    --serving_backend ray \
    --slurm \
    --ssh_user <your_username> \
    --ssh_host <login_node_hostname> \
    --job_dir ${WORKSPACE}/nemo-experiments \
    --account <slurm_account> \
    --partition <partition> \
    --nodes 1 --devices 8 \
    --tensor_parallelism_size 2 --expert_model_parallel_size 8 \
    --eval_task gsm8k \
    --batch_size 8 --parallel_requests 8 \
    --server_port <port> \
    --evaluation_result_dir ${WORKSPACE}/results/eval \
    --container_image <container_image> \
    --custom_mounts "/lustre:/lustre,${WORKSPACE}/Megatron-Bridge:/opt/Megatron-Bridge" \
    --custom_env_vars "PYTHONPATH=/opt/megatron-lm:/opt/Megatron-Bridge/src,HF_HOME=<hf_cache_dir>"
```

#### Benchmark Results

| Model | Benchmark | Metric | Score |
|---|---|---|---|
| Nemotron-3-Nano-30B-A3B-Base-BF16 | GSM8K | flexible-extract | 83.78% |
| Nemotron-3-Nano-30B-A3B-Base-BF16 | GSM8K | strict-match | 83.47% |

