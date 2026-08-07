# Nemotron 3.5 Lightning on Megatron Bridge 0.5.1

This directory is the release-specific customer entrypoint for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on Megatron Bridge 0.5.1.
Use it with the `nvcr.io/nvidia/nemo:26.06.01` runtime and a Megatron Bridge
0.5.1 checkout that contains Nemotron 3.5 Lightning support.

The unmodified 26.06.01 image does not contain this model support or the newer
generic `scripts/training/train.sh` and `scripts/conversion/convert.sh`
wrappers. It does contain the Python entrypoints used here. These model-local
wrappers are intentionally scoped to the 0.5.1 release line; Megatron Bridge
0.6 and newer use the generic launchers instead.

## Contents

| File | Purpose |
|---|---|
| `conversion.sh` | Import a Hugging Face checkpoint or export a Megatron checkpoint |
| `inference.sh` | Generate from the HF checkpoint or an imported Megatron checkpoint |
| `pretrain.sh` | Pretraining recipe with H100 and GB200 profiles |
| `sft.sh` | Packed OpenMathInstruct-2 full SFT with H100 and GB200 profiles |
| `lora.sh` | Rank-32 LoRA fine-tuning with H100 and GB200 profiles |
| `adapter.sh` | Export a LoRA adapter or merge a Megatron LoRA checkpoint |

All scripts must be launched from the repository root. They use the mounted
checkout for both Bridge and Megatron-Core:

```bash
export PYTHONPATH="$PWD/src:$PWD/3rdparty/Megatron-LM:${PYTHONPATH:-}"
```

When executed by `srun`, each task starts one Python rank. Outside Slurm, the
scripts use `uv run python -m torch.distributed.run` and `NPROC_PER_NODE`
(default: 8).

## Container setup

The checkout must override the older source bundled in 26.06.01:

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD:/opt/Megatron-Bridge" \
  -v /workspace:/workspace \
  -w /opt/Megatron-Bridge \
  nvcr.io/nvidia/nemo:26.06.01 \
  bash -lc 'source /opt/venv/bin/activate; ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh import'
```

For Slurm/Pyxis, mount the checkout and workspace, activate `/opt/venv`, and
launch one task per GPU. Two representative allocations are:

| Hardware | Nodes | GPUs per node | Total ranks | Intended workflows |
|---|---:|---:|---:|---|
| H100 80 GB | 2 | 8 | 16 | Pretraining and full SFT |
| H100 80 GB | 1 | 8 | 8 | Conversion, inference, and LoRA |
| GB200 | 2 | 4 | 8 | All workflows |

Scheduler partitions, accounts, and topology flags are site-specific and are
therefore not hard-coded in these examples.

## Checkpoint conversion

Import the pinned public checkpoint with TP1/EP8:

```bash
WORKSPACE=/workspace TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh import
```

Export a Megatron checkpoint back to native Hugging Face format:

```bash
MEGATRON_PATH=/workspace/models/nemotron-3.5-lightning-megatron/iter_0000000 \
HF_EXPORT_PATH=/workspace/models/nemotron-3.5-lightning-hf-export \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh export
```

The source checkpoint used for the 0.5.1 validation was revision
`b3caaabed0263651a17dc1f2d4ce97e794f76c44`.

## Inference

Generate from the imported Megatron checkpoint:

```bash
MEGATRON_PATH=/workspace/models/nemotron-3.5-lightning-megatron/iter_0000000 \
PROMPT="The capital of France is" TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```

Leave `MEGATRON_PATH` unset to import the HF weights in memory before
generation. Set `HF_MODEL` to an exported SFT or merged-LoRA directory to test
that artifact.

## Pretraining

The default pretraining dataset is mock data. Set the recipe's
`dataset.blend` override after the script arguments for real indexed data.

```bash
# 16 H100 ranks: TP1, CP2, EP8
HARDWARE=h100 WORKSPACE=/workspace \
  ./examples/models/nemotron/nemotron_3_5_lightning/pretrain.sh

# 8 GB200 ranks: TP1, CP1, EP8
HARDWARE=gb200 WORKSPACE=/workspace \
  ./examples/models/nemotron/nemotron_3_5_lightning/pretrain.sh
```

The full BF16 optimizer-state recipe does not fit on one eight-GPU H100 node,
even at shorter sequence lengths. Use the documented 16-H100 topology for
pretraining.

## Full SFT

The SFT examples use a pinned OpenMathInstruct-2 revision and cache packed
sequences as Parquet. The first run needs dataset access to prepare the cache;
later offline runs require that complete cache to be mounted.
`PRETRAINED_CHECKPOINT` must point to the imported `iter_0000000` directory.

```bash
# 16 H100 ranks: TP2, EP8
HARDWARE=h100 \
PRETRAINED_CHECKPOINT=/workspace/models/nemotron-3.5-lightning-megatron/iter_0000000 \
OUTPUT_DIR=/workspace/results/nemotron-3.5-lightning-sft-h100 \
  ./examples/models/nemotron/nemotron_3_5_lightning/sft.sh

# 8 GB200 ranks: TP1, EP8, HybridEP
HARDWARE=gb200 \
PRETRAINED_CHECKPOINT=/workspace/models/nemotron-3.5-lightning-megatron/iter_0000000 \
OUTPUT_DIR=/workspace/results/nemotron-3.5-lightning-sft-gb200 \
  ./examples/models/nemotron/nemotron_3_5_lightning/sft.sh
```

Export the resulting full-model checkpoint with `conversion.sh export`, using
the same TP/EP values as the training profile.

## LoRA

The LoRA recipe uses rank 32, alpha 32, zero dropout, and targets attention,
MLP, Mamba, expert, shared-expert, and MTP linear layers. Its default dataset
is packed SQuAD.

```bash
# 8 H100 ranks: TP1, EP8, DeepEP
HARDWARE=h100 \
PRETRAINED_CHECKPOINT=/workspace/models/nemotron-3.5-lightning-megatron/iter_0000000 \
OUTPUT_DIR=/workspace/results/nemotron-3.5-lightning-lora-h100 \
  ./examples/models/nemotron/nemotron_3_5_lightning/lora.sh

# 8 GB200 ranks: TP1, EP8, HybridEP
HARDWARE=gb200 \
PRETRAINED_CHECKPOINT=/workspace/models/nemotron-3.5-lightning-megatron/iter_0000000 \
OUTPUT_DIR=/workspace/results/nemotron-3.5-lightning-lora-gb200 \
  ./examples/models/nemotron/nemotron_3_5_lightning/lora.sh
```

Export the Megatron adapter to the standard Hugging Face PEFT layout:

```bash
LORA_CHECKPOINT=/workspace/results/nemotron-3.5-lightning-lora-h100/iter_0000100 \
HF_ADAPTER_PATH=/workspace/models/nemotron-3.5-lightning-lora-adapter \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/adapter.sh export
```

Merge the Megatron LoRA checkpoint with its Megatron base checkpoint into a
standalone Hugging Face checkpoint. This uses the repository's common
`examples/peft/merge_lora.py` path; it does not consume the separately exported
Hugging Face PEFT package.

```bash
PRETRAINED_CHECKPOINT=/workspace/models/nemotron-3.5-lightning-megatron/iter_0000000 \
LORA_CHECKPOINT=/workspace/results/nemotron-3.5-lightning-lora-h100/iter_0000100 \
HF_MERGED_PATH=/workspace/models/nemotron-3.5-lightning-lora-merged \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/adapter.sh merge
```

The common merger reconstructs the training model, loads the base and LoRA
distributed checkpoints, and asks the Bridge conversion path to fuse adapter
weights while exporting every supported model tensor, including MTP. This is
preferable to maintaining a Lightning-only copy of adapter merge logic.

When operating offline, set `HF_MODEL` to a mounted local snapshot for import,
export, inference, and merge. A Hub ID alone cannot resolve
weight shards unless the snapshot is stored in the standard Hugging Face cache
layout.
