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
| `adapter.sh` | Export, verify, or merge a LoRA adapter |

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

The SFT examples use the pinned OpenMathInstruct-2 revision and offline packed
sequences. `PRETRAINED_CHECKPOINT` must point to the imported `iter_0000000`
directory.

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

Verify that PEFT loads the adapter, that it changes the base-model logits, and
that its top tokens match Megatron's in-memory merged export:

```bash
LORA_CHECKPOINT=/workspace/results/nemotron-3.5-lightning-lora-h100/iter_0000100 \
HF_ADAPTER_PATH=/workspace/models/nemotron-3.5-lightning-lora-adapter \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/adapter.sh verify
```

Finally, merge the HF adapter into a standalone HF checkpoint. This operation
materializes the full base model on one device; use a GPU with enough memory
(for example, GB200) or select a suitable `--device-map` when invoking
`merge_adapter.py` directly.

```bash
HF_ADAPTER_PATH=/workspace/models/nemotron-3.5-lightning-lora-adapter \
HF_MERGED_PATH=/workspace/models/nemotron-3.5-lightning-lora-merged \
  ./examples/models/nemotron/nemotron_3_5_lightning/adapter.sh merge
```

The merge utility checks that the adapter changes the base logits, that merged
and unmerged PEFT logits agree within the declared tolerance, and that their
top-five tokens are identical before saving.

## Manual 26.06.01 verification

The following are correctness runs, not performance benchmarks. Exact SFT,
LoRA, and adapter-export results will be recorded here after running these
customer scripts on the 0.5.1 branch.

The already-completed base-checkpoint validation used eight H100s for
TP1/PP1/EP8 import and export. All 6,513 tensors and 32,913,266,240 parameters
round-tripped exactly (`max_abs_diff=0.0`), and both the imported Megatron
checkpoint and exported HF checkpoint generated `The capital of France is
Paris.`. The canonical 16-H100 pretraining recipe completed two finite-loss
steps with no skipped or NaN iterations.

