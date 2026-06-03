# Nemotron 3 Ultra Examples

This directory contains examples for Nemotron 3 Ultra conversion, inference,
DCLM pretraining, packed OpenMathInstruct-2 full SFT, and packed
OpenMathInstruct-2 LoRA PEFT.

Nemotron 3 Ultra is a 550B total / A55B active hybrid Mamba-Transformer MoE
model. See the
[Nemotron 3 Ultra Base model guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Ultra-Base/README.html)
for model details.

## Workspace Configuration

The scripts use `WORKSPACE` as the base directory for checkpoints, packed data,
and results. Defaults:

```bash
export WORKSPACE=/workspace
export MODEL_HOME=${WORKSPACE}/models/nvidia
export HF_MODEL_PATH=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
export MEGATRON_MODEL_PATH=${MODEL_HOME}/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-megatron
export PRETRAINED_CHECKPOINT=${MEGATRON_MODEL_PATH}
```

Use shared filesystems for multi-node jobs:

```bash
export HF_HOME=${WORKSPACE}/cache/hf
export NEMO_HOME=${WORKSPACE}/cache/nemo
export UV_CACHE_DIR=${WORKSPACE}/cache/uv
```

The BF16 Hugging Face cache and imported Megatron checkpoint are each about
1.1 TB. Reserve at least 2.5 TB for model storage before starting checkpoint
conversion, plus additional space for training outputs and logs.

## Hardware Starting Points

The checked-in Slurm scripts default to 8xH100 nodes unless noted below.
When running on 4xGB200 nodes, update the `#SBATCH --nodes`,
`#SBATCH --ntasks-per-node`, `#SBATCH --gpus-per-node`, and parallelism
environment variables to the GB200 values in this table.

| Workflow | 8xH100 nodes | 4xGB200 nodes |
| --- | --- | --- |
| Checkpoint import | 1 node with [conversion.sh](conversion.sh), CPU import path | 6 nodes with [slurm_conversion.sh](slurm_conversion.sh), `TP=1 PP=6 EP=4` |
| Base inference | 4 nodes, `TP=1 PP=4 EP=8`, `KV_CACHE_BUFFER_SIZE_GB=4` | 3 nodes, `TP=1 PP=3 EP=4` |
| DCLM pretraining | 48 nodes, `TP=2 PP=12 EP=32` script default; validate on your DCLM mount before long runs | 24 nodes, `TP=2 PP=3 EP=32`, selective recompute on `moe+layernorm+core_attn+moe_act+mlp+shared_experts` |
| OpenMath SFT | 48 nodes, `TP=2 PP=12 EP=32`, selective recompute on `moe+layernorm+core_attn+moe_act` | 48 nodes, `TP=2 PP=3 EP=32`, selective recompute on `moe+layernorm+core_attn+moe_act` |
| OpenMath PEFT | 4 nodes, `TP=2 PP=4 EP=8`, selective recompute on `moe+layernorm+core_attn+moe_act+mlp+shared_experts` | 4 nodes, `TP=2 PP=1 EP=16`, selective recompute on `moe+layernorm+core_attn+moe_act` |

These are bring-up and convergence starting points, not universal optima.
Keep `TP` within a node-local NVLink domain and scale with `PP`, `EP`, and
data parallelism when moving between hardware. For MoE sizing, the minimum GPU
count is `PP * max(TP * CP, EP * ETP)`, then additional GPUs increase dense
DP and expert DP.

## Day-0 Code

Start from the NeMo 26.04.01 container:

```bash
docker run --rm -it --gpus all --shm-size=24g \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.04.01 \
    bash
```

Inside the container, clone the Day-0 Megatron Bridge branch and initialize the
bundled Megatron-Core submodule:

```bash
export WORKSPACE=/workspace
export MEGATRON_BRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
export MEGATRON_BRIDGE_BRANCH=nemotron_3_ultra

cd "$WORKSPACE"
git clone --branch "$MEGATRON_BRIDGE_BRANCH" "$MEGATRON_BRIDGE_REPO"
cd Megatron-Bridge

git submodule update --init --recursive --depth 1 3rdparty/Megatron-LM

uv lock
uv sync --extra ssm
```

Run dependency sync during setup. The Slurm job scripts use `uv run --no-sync`
so they do not download dependencies or mutate the Python environment during
allocated jobs.

Verify that `megatron.core` and `megatron.bridge` resolve to this checkout:

```bash
uv run python - <<'PY'
import megatron.bridge
import megatron.core
from megatron.bridge.recipes.nemotronh import nemotron_3_ultra_pretrain_config

print("core:", list(megatron.core.__path__))
print("bridge:", list(megatron.bridge.__path__))
print("recipe:", nemotron_3_ultra_pretrain_config.__name__)
PY
```

Expected paths should include:

```text
/workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core
/workspace/Megatron-Bridge/src/megatron/bridge
```

## Checkpoint Conversion

Use [conversion.sh](conversion.sh) for CPU checkpoint import when the node has
enough host RAM to materialize Nemotron 3 Ultra, for example an 8xH100 node.
This is the preferred path when available because it avoids distributed GPU
memory pressure during import.

```bash
bash conversion.sh
```

Set these variables for your environment:

- `WORKSPACE`
- `HF_HOME`
- `UV_CACHE_DIR`
- `HF_MODEL_PATH`
- `MEGATRON_MODEL_PATH`

Use [slurm_conversion.sh](slurm_conversion.sh) for distributed GPU checkpoint
import when host RAM is not large enough, for example a 4xGB200 setup with less
than 1 TB of host RAM. The checked-in distributed example defaults to 6
8-GPU nodes with `TP=1 PP=6 EP=8`; for 4xGB200, use 6 nodes and set
`#SBATCH --ntasks-per-node=4`, `#SBATCH --gpus-per-node=4`, and
`TP=1 PP=6 EP=4`.

```bash
sbatch slurm_conversion.sh
```

Set these variables for your environment:

- `CONTAINER_IMAGE`
- `CONTAINER_MOUNTS`
- `WORKDIR`
- `WORKSPACE`
- `HF_MODEL_PATH`
- `MEGATRON_MODEL_PATH`

## Inference

Use [slurm_inference.sh](slurm_inference.sh) for 4-node text generation with
`TP=1 PP=4 EP=8`. The script defaults `KV_CACHE_BUFFER_SIZE_GB=4` to keep
the inference KV/context buffer within H100 memory for the default prompt
lengths. On 4xGB200 nodes, use 3 nodes with `TP=1 PP=3 EP=4`.

```bash
sbatch slurm_inference.sh
```

Set `MEGATRON_MODEL_PATH` to generate from an imported Megatron checkpoint.
Leave it unset to load from the Hugging Face checkpoint path.

## DCLM Pretraining

Use [slurm_pretrain.sh](slurm_pretrain.sh) for DCLM pretraining with
`TP=2 PP=12 EP=32` on 8xH100 nodes. On 4xGB200 nodes, use 24 nodes with
`TP=2 PP=3 EP=32`.

```bash
sbatch slurm_pretrain.sh
```

Set `DCLM_DATA_DIR` to a preprocessed DCLM directory containing
`*_text_document.bin` / `*_text_document.idx` files. The script defaults to
matching `dclm_01_*_text_document.bin`.

## OpenMath Packed Data

Pre-pack OpenMath data before training:

```bash
sbatch pack_data_job.sh
```

Use the same `SEQ_LENGTH`, `HF_MODEL_PATH`, and `NEMO_HOME` for packing and
training. `NEMO_HOME` must point at a shared filesystem visible on all nodes.

## Training

PEFT:

```bash
sbatch slurm_peft.sh
```

Full SFT:

```bash
sbatch slurm_sft.sh
```

The scripts default to OpenMath convergence settings: `TRAIN_ITERS=1000`,
`GLOBAL_BATCH_SIZE=128`, `SEQ_LENGTH=4096`, and `LR_WARMUP_ITERS=250`. W&B
logging is disabled by default. SFT and PEFT save at the final training
iteration by default; the SFT script removes older intermediate `iter_*`
checkpoints after a successful run to avoid retaining multiple full-model
checkpoints.

Current OpenMath starting points are:

- PEFT: 4 nodes, `TP=2 PP=4 EP=8`, selective recompute on
  `moe+layernorm+core_attn+moe_act+mlp+shared_experts`.
- Full SFT: 48 nodes, `TP=2 PP=12 EP=32`, selective recompute on
  `moe+layernorm+core_attn+moe_act`. This is the current H100 starting point
  for 4096-token packed OpenMath SFT.

For 4xGB200 nodes:

- PEFT: 4 nodes, `TP=2 PP=1 EP=16`, selective recompute on
  `moe+layernorm+core_attn+moe_act`.
- Full SFT: 48 nodes, `TP=2 PP=3 EP=32`, selective recompute on
  `moe+layernorm+core_attn+moe_act`.

Advanced VPP, pipeline-layout, and recompute sweeps are intentionally left out
of these starter scripts; add those overrides only for targeted performance
experiments.

## W&B

W&B logging is disabled by default:

```bash
WANDB_ENTITY=nvidia-nemo-fw-public
WANDB_PROJECT=megatron-bridge-nemotron-ultra
WANDB_MODE=disabled
```

To enable online W&B logging, set `WANDB_MODE=online` and make `WANDB_API_KEY`
visible in the submit environment.

Run names include model, OpenMath, mode, TP/PP/EP, recompute, and Slurm job ID.
