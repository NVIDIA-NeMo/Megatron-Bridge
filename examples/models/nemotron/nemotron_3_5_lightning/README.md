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

The exact 26.06.01 image bundles PEFT 0.19.1 with Transformers 5.8.1. For this
model, PEFT reconstructs Transformers `WeightConverter` objects with
`distributed_operation` and `quantization_operation`, but that Transformers
constructor does not accept those arguments. Consequently, direct loading of
the exported PEFT package in this exact image is not a supported verification
path. The export remains a standard PEFT artifact for use with a compatible
PEFT/Transformers environment; no model-local monkeypatch is installed.

## Manual 26.06.01 verification

These are two-step correctness runs, not performance benchmarks. They used
the pinned public BF16 checkpoint and the unmodified 26.06.01 runtime with this
branch mounted over the bundled Bridge source.

| Workflow | Hardware and configuration | Iteration 1: LM / MTP-1 / MTP-2 | Iteration 2: LM / MTP-1 / MTP-2 | Result |
|---|---|---|---|---|
| Pretrain | 2 nodes, 16 H100 80 GB; 8K, TP1/CP2/EP8 | 12.13518 / 12.19819 / 12.17519 | 12.13271 / 12.19786 / 12.17710 | Passed; reloadable checkpoint, skipped 0, NaN 0 |
| Full SFT | 2 nodes, 16 H100 80 GB; packed OpenMath 4K, TP2/EP8 | 0.4134090 / 0.6536090 / 0.7314808 | 0.4127115 / 0.6384317 / 0.7196919 | Passed; reloadable checkpoint, skipped 0, NaN 0 |
| LoRA | 1 node, 8 H100 80 GB; packed SQuAD, TP1/EP8, DeepEP | 0.2351837 / 2.486913 / 2.843087 | 0.1531711 / 2.473096 / 2.837737 | Passed; reloadable adapter checkpoint, skipped 0, NaN 0 |
| Full SFT | 2 nodes, 8 GB200; packed OpenMath 4K, TP1/EP8, `alltoall` fallback | 0.4482034 / 0.6759753 / 0.7569838 | 0.3785085 / 0.5998533 / 0.6831521 | Passed; reloadable checkpoint, skipped 0, NaN 0 |
| LoRA | 2 nodes, 8 GB200; packed SQuAD, TP1/EP8, `alltoall` fallback | 0.2281463 / 2.485532 / 2.841834 | 0.1542726 / 2.476500 / 2.838086 | Passed; reloadable adapter checkpoint, skipped 0, NaN 0 |

The GB200 recipe still selects HybridEP. The GB200 validation rows use plain
`alltoall` only as a correctness fallback: HybridEP was installed and selected,
but the 26.06.01 build first rejected its default 128-token chunk because the
4,080-token per-rank capacity is not divisible by 128. With a 16-token chunk,
two separate same-NVLink-block runs completed iteration 1 and then timed out in
HybridEP's all-gather during iteration 2. DeepEP is the H100 backend used above,
not a supported substitute for HybridEP on GB200 NVL72. The fallback results
therefore validate the model, data, checkpoint, and non-HybridEP recipe path;
they do not establish HybridEP stability in this container.

Import/export on eight H100s round-tripped all 6,513 tensors and
32,913,266,240 parameters exactly (`max_abs_diff=0.0`). The imported Megatron
checkpoint and the exported HF checkpoint both generated `The capital of
France is Paris.`. Exporting the full-SFT checkpoint also produced a complete
6,513-tensor HF artifact and the same continuation.

The H100 and GB200 LoRA checkpoints each exported to a standard PEFT package
containing 12,532 adapter tensors, including 524 MTP tensors (262 A/B pairs).
The public-revision GB200 LoRA checkpoint was also merged through the common
Megatron-checkpoint path. The standalone artifact preserved all 6,513 base
tensors by name, shape, and native dtype; 6,266 LoRA-targeted tensors changed,
including all 262 MTP tensors with adapter pairs, while the other eight MTP
tensors remained unchanged. Distributed inference from that artifact generated
`The capital of France is Paris.  `.
Full-SFT export and inference generated `The capital of France is Paris.`.

When operating offline, set `HF_MODEL` to a mounted local snapshot for import,
export, inference, and merge. A Hub ID alone cannot resolve
weight shards unless the snapshot is stored in the standard Hugging Face cache
layout.
