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

The merge utility checks that the adapter changes the base logits and that the
BF16-fused model preserves PEFT's top-1 token and greedy continuation while
meeting cosine-similarity and relative-L2 thresholds. Transformers does not
instantiate Lightning's training-only MTP modules during inference, so the
utility separately applies every MTP LoRA pair and carries the remaining MTP
tensors unchanged into the standalone checkpoint. The result therefore keeps
all 6,513 base-model tensors and can be used for either inference or subsequent
training.

## Manual 26.06.01 verification

These are two-step correctness runs, not performance benchmarks. They used
the pinned public BF16 checkpoint and the unmodified 26.06.01 runtime with this
branch mounted over the bundled Bridge source.

| Workflow | Hardware and configuration | Iteration 1: LM / MTP-1 / MTP-2 | Iteration 2: LM / MTP-1 / MTP-2 | Result |
|---|---|---|---|---|
| Pretrain | 2 nodes, 16 H100 80 GB; 8K, TP1/CP2/EP8 | 12.13518 / 12.19819 / 12.17519 | 12.13271 / 12.19786 / 12.17710 | Passed; reloadable checkpoint, skipped 0, NaN 0 |
| Full SFT | 2 nodes, 16 H100 80 GB; packed OpenMath 4K, TP2/EP8 | 0.4134090 / 0.6536090 / 0.7314808 | 0.4127115 / 0.6384317 / 0.7196919 | Passed; reloadable checkpoint, skipped 0, NaN 0 |
| LoRA | 1 node, 8 H100 80 GB; packed SQuAD, TP1/EP8, DeepEP | 0.2351837 / 2.486913 / 2.843087 | 0.1531711 / 2.473096 / 2.837737 | Passed; reloadable adapter checkpoint, skipped 0, NaN 0 |

Import/export on eight H100s round-tripped all 6,513 tensors and
32,913,266,240 parameters exactly (`max_abs_diff=0.0`). The imported Megatron
checkpoint and the exported HF checkpoint both generated `The capital of
France is Paris.`. Exporting the full-SFT checkpoint also produced a complete
6,513-tensor HF artifact and the same continuation.

The H100 LoRA checkpoint exported to a standard PEFT package containing 12,532
adapter tensors, including 524 MTP tensors (262 A/B pairs). Adapter loading and
standalone merge were exercised in the same container. For the BF16 merge,
PEFT and fused models had cosine similarity
0.998045444 and relative L2 error 0.06499264, retained the same top-1 token,
and generated the same four-token continuation, ` Paris.  `. The final merged
artifact contains all 6,513 tensors, including all 270 MTP tensors. A
key-by-key safetensors-header audit found the same names, shapes, and native
dtypes as the base checkpoint, and distributed inference from the artifact
generated `The capital of France is Paris.`.

When operating offline, set `HF_MODEL` to a mounted local snapshot for import,
export, inference, verification, and merge. A Hub ID alone cannot resolve
weight shards unless the snapshot is stored in the standard Hugging Face cache
layout.
