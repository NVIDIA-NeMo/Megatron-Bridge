# Nemotron 3.5 Lightning

This directory is the day-0 release entrypoint for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on Megatron Bridge 0.5.1.
Use it with the `nvcr.io/nvidia/nemo:26.06.01` runtime and a Megatron Bridge
0.5.1 checkout that contains Nemotron 3.5 Lightning support (i.e. the `nemotron-3.5-lightning-mb-0.5.1` branch).

The support based on Megatron Bridge 0.6.0 (NeMo 26.08 container) is available on the main branch.

## Contents

| File | Purpose |
|---|---|
| `conversion.sh` | Import a Hugging Face checkpoint or export a Megatron checkpoint |
| `inference.sh` | Generate from the HF checkpoint or an imported Megatron checkpoint |
| `pretrain.sh` | Pretraining recipe with H100 and GB200 profiles |
| `sft.sh` | Packed OpenMathInstruct-2 full SFT with H100 and GB200 profiles |
| `lora.sh` | Rank-32 LoRA fine-tuning with H100 and GB200 profiles |
| `adapter.sh` | Export a LoRA adapter or merge a Megatron LoRA checkpoint |

All commands in this README use Docker directly and are independent of
external launch tooling.

## Fresh-checkout bootstrap

Clone the release branch and initialize the Megatron-LM submodule before
starting the container:

```bash
git clone --branch nemotron-3.5-lightning-mb-0.5.1 \
  https://github.com/NVIDIA-NeMo/Megatron-Bridge.git megatron-bridge
cd megatron-bridge
git submodule update --init 3rdparty/Megatron-LM

test -f 3rdparty/Megatron-LM/megatron/core/__init__.py
git rev-parse HEAD
git submodule status 3rdparty/Megatron-LM
```

Run the remaining host commands from this repository root. Keep the workspace
outside the checkout so generated files do not dirty the source tree:

```bash
export REPO_ROOT="$PWD"
export LIGHTNING_STATE="${LIGHTNING_STATE:-$PWD/../nemotron-3.5-lightning-workspace}"
mkdir -p "$LIGHTNING_STATE"/{cache,models,results}
```

Plan storage before downloading or training. Approximate observed sizes are:

| Artifact | Space |
|---|---:|
| Pinned HF source cache | 70 GiB |
| Imported Megatron checkpoint | 70 GiB |
| Each full-model HF export or merged checkpoint | 70 GiB |
| Full-SFT checkpoint | 70 GiB |
| LoRA checkpoint and exported adapter | 5 GiB |
| 100-step pretraining checkpoint with optimizer state | 450 GiB |

Allow at least 160 GiB for import plus export, or about 1 TiB to retain every
documented artifact and both dataset caches at once.

## Override the Megatron Bridge folder

The NeMo 26.06.01 image is `nvcr.io/nvidia/nemo:26.06.01`. 
(`nvcr.io/nvidia/nemo@sha256:912033288c982a8c4af05df46a1d670c34350f1427c758f5da9c485bdec57264`).
The image does not contain this release-specific model support. Override the
Megatron Bridge folder with the following steps:

```bash
docker pull nvcr.io/nvidia/nemo:26.06.01

docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO_ROOT:/opt/Megatron-Bridge:ro" \
  -v "$LIGHTNING_STATE:/workspace" \
  -e HF_HOME=/workspace/cache/huggingface \
  -e HF_HUB_CACHE=/workspace/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets \
  -e NEMO_HOME=/workspace/cache/nemo \
  -e NEMO_DATASETS_CACHE=/workspace/cache/nemo/datasets \
  -w /opt/Megatron-Bridge \
  nvcr.io/nvidia/nemo:26.06.01 bash
```

In the container, activate the bundled environment and export the paths used
by every subsequent command:

```bash
source /opt/venv/bin/activate
cd /opt/Megatron-Bridge

export PYTHONPATH="$PWD/src:$PWD/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export WORKSPACE=/workspace
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_CACHE=/workspace/cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets
export NEMO_HOME=/workspace/cache/nemo
export NEMO_DATASETS_CACHE=/workspace/cache/nemo/datasets
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$NEMO_DATASETS_CACHE" /workspace/{models,results}
```

The scripts use `uv run --active --no-sync` with the bundled environment. No
package installation or dependency change is required.

## Checkpoint conversion

Import the pinned local HF snapshot with TP1/EP8. Eight local ranks are
required:

```bash
export HF_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
export BASE_MEGATRON_ROOT=/workspace/models/nemotron-3.5-lightning-megatron
export BASE_MEGATRON_CHECKPOINT="$BASE_MEGATRON_ROOT/iter_0000000"

NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$BASE_MEGATRON_ROOT" TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh import

printf '%s\n' "$HF_MODEL_REVISION" > "$BASE_MEGATRON_ROOT/source_hf_revision.txt"
test "$(<"$BASE_MEGATRON_ROOT/latest_checkpointed_iteration.txt")" = 0
test -f "$BASE_MEGATRON_CHECKPOINT/run_config.yaml"
```

Export that checkpoint back to native HF format:

```bash
export BASE_HF_EXPORT=/workspace/models/nemotron-3.5-lightning-hf-export

NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$BASE_MEGATRON_CHECKPOINT" HF_EXPORT_PATH="$BASE_HF_EXPORT" \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh export

test -f "$BASE_HF_EXPORT/model.safetensors.index.json"
```

## Inference

Generate from the imported Megatron checkpoint:

```bash
NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$BASE_MEGATRON_CHECKPOINT" \
PROMPT="The capital of France is" TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```

Generate by importing the pinned HF snapshot in memory:

```bash
NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" MEGATRON_PATH= \
PROMPT="The capital of France is" TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```

Both commands should complete without an exception and produce a continuation
beginning with `The capital of France is Paris.`.

## Select a training profile

Choose one hardware profile in the container. The profile determines the
recipe, local process count, and full-SFT tensor parallelism; EP is 8 for all
documented workflows.

```bash
# Choose exactly one value: h100 or gb200.
export HARDWARE=gb200

case "$HARDWARE" in
  h100)
    export PRETRAIN_NPROC=16 SFT_NPROC=16 SFT_TP=2
    ;;
  gb200)
    export PRETRAIN_NPROC=8 SFT_NPROC=8 SFT_TP=1
    ;;
  *)
    echo "HARDWARE must be h100 or gb200" >&2
    return 2
    ;;
esac
export LORA_NPROC=8
```

The GB200 SFT and LoRA profiles select HybridEP exactly as implemented by the
scripts. The commands below do not apply an undocumented dispatcher fallback.

## Pretraining

The default dataset is mock data. The 100-step default saves optimizer state
and can create a checkpoint of approximately 450 GiB.

```bash
export PRETRAIN_OUTPUT=/workspace/results/nemotron-3.5-lightning-pretrain-$HARDWARE

NPROC_PER_NODE="$PRETRAIN_NPROC" HARDWARE="$HARDWARE" \
WORKSPACE=/workspace OUTPUT_DIR="$PRETRAIN_OUTPUT" \
  ./examples/models/nemotron/nemotron_3_5_lightning/pretrain.sh

test "$(<"$PRETRAIN_OUTPUT/latest_checkpointed_iteration.txt")" = 100
test -f "$PRETRAIN_OUTPUT/iter_0000100/run_config.yaml"
```

Pass normal Hydra-style recipe overrides after `pretrain.sh` to select real
indexed data. For example, set `dataset.blend` to the mounted dataset prefix.

## Full SFT

The SFT workflow uses packed OpenMathInstruct-2 prepared
above. Its input is the imported `iter_0000000` checkpoint.

```bash
export SFT_OUTPUT=/workspace/results/nemotron-3.5-lightning-sft-$HARDWARE
export SFT_CHECKPOINT="$SFT_OUTPUT/iter_0000100"

NPROC_PER_NODE="$SFT_NPROC" HARDWARE="$HARDWARE" \
PRETRAINED_CHECKPOINT="$BASE_MEGATRON_CHECKPOINT" OUTPUT_DIR="$SFT_OUTPUT" \
  ./examples/models/nemotron/nemotron_3_5_lightning/sft.sh \
  "dataset.hf_kwargs={revision:${OPENMATH_REVISION}}"

test "$(<"$SFT_OUTPUT/latest_checkpointed_iteration.txt")" = 100
test -f "$SFT_CHECKPOINT/run_config.yaml"
```

Export the exact iteration-100 full-SFT checkpoint with the TP value selected
for the profile:

```bash
export SFT_HF_EXPORT=/workspace/models/nemotron-3.5-lightning-sft-$HARDWARE-hf

NPROC_PER_NODE="$SFT_NPROC" HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$SFT_CHECKPOINT" HF_EXPORT_PATH="$SFT_HF_EXPORT" \
TP="$SFT_TP" EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh export

test -f "$SFT_HF_EXPORT/model.safetensors.index.json"
```

Verify both forms of the trained artifact by generating from each:

```bash
NPROC_PER_NODE="$SFT_NPROC" HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$SFT_CHECKPOINT" TP="$SFT_TP" EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh

NPROC_PER_NODE=8 HF_MODEL="$SFT_HF_EXPORT" MEGATRON_PATH= TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```

## LoRA

The LoRA recipe uses rank 32, alpha 32, zero dropout, and targets attention,
MLP, Mamba, expert, shared-expert, and MTP linear layers. Its packed SQuAD
dataset is explicitly pinned by the final override below.

```bash
export LORA_OUTPUT=/workspace/results/nemotron-3.5-lightning-lora-$HARDWARE
export LORA_CHECKPOINT="$LORA_OUTPUT/iter_0000100"

NPROC_PER_NODE="$LORA_NPROC" HARDWARE="$HARDWARE" \
PRETRAINED_CHECKPOINT="$BASE_MEGATRON_CHECKPOINT" OUTPUT_DIR="$LORA_OUTPUT" \
  ./examples/models/nemotron/nemotron_3_5_lightning/lora.sh \
  "dataset.hf_kwargs={revision:${SQUAD_REVISION}}"

test "$(<"$LORA_OUTPUT/latest_checkpointed_iteration.txt")" = 100
test -f "$LORA_CHECKPOINT/run_config.yaml"
```

Export either hardware profile's Megatron adapter to the standard HF PEFT
layout:

```bash
export HF_ADAPTER_PATH=/workspace/models/nemotron-3.5-lightning-lora-$HARDWARE-adapter

NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
LORA_CHECKPOINT="$LORA_CHECKPOINT" HF_ADAPTER_PATH="$HF_ADAPTER_PATH" \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/adapter.sh export
```

Merge the same Megatron LoRA checkpoint with its imported Megatron base into a
standalone HF checkpoint:

```bash
export HF_MERGED_PATH=/workspace/models/nemotron-3.5-lightning-lora-$HARDWARE-merged

NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
PRETRAINED_CHECKPOINT="$BASE_MEGATRON_CHECKPOINT" \
LORA_CHECKPOINT="$LORA_CHECKPOINT" HF_MERGED_PATH="$HF_MERGED_PATH" \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/adapter.sh merge

test -f "$HF_MERGED_PATH/model.safetensors.index.json"
```

Verify that the merged standalone artifact loads and generates:

```bash
NPROC_PER_NODE=8 HF_MODEL="$HF_MERGED_PATH" MEGATRON_PATH= TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```