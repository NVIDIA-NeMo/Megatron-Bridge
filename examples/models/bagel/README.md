# BAGEL pretraining

This example runs BAGEL pretraining through Megatron Bridge with a deterministic
WebDataset/Energon input pipeline. It is currently a stacked integration: model
construction requires the BAGEL MIMO implementation from
[Megatron-LM PR #3635](https://github.com/NVIDIA/Megatron-LM/pull/3635), and data
transforms/tokenization require a local checkout of the
[official BAGEL repository](https://github.com/ByteDance-Seed/Bagel). The
validated stack uses Megatron-Energon 7.4.1.

## Data flow

The example keeps the official BAGEL data semantics while changing storage and
iteration infrastructure:

```text
official raw data -> deterministic WDS -> Energon raw samples
                  -> official-equivalent source planning and packing
                  -> Megatron external loader
```

Use `data/convert_bagel_dataset_to_wds.py` to create the three WDS datasets and
`data/prepare_bagel_energon.py` to generate their Energon metadata. The
conversion preserves original image bytes and source positions. The cooker,
order planner, and packer live in `src/megatron/bridge/models/bagel/data/`.

## Initialization and training

First export the official random initialization, then convert that exact state
to an iteration-0 Bridge checkpoint. Run each command below through a
distributed launcher with one process per GPU:

```bash
python examples/models/bagel/export_official_init.py \
  --bagel-repo <bagel-repo> \
  --model-path <bagel-model-files> \
  --seed 42 \
  --output <native-init.safetensors>

python examples/models/bagel/export_mcore_init.py \
  --recipe pretrain-32gpu \
  --bagel-repo <bagel-repo> \
  --model-path <bagel-model-files> \
  --dataset-root <prepared-wds-root> \
  --tokenizer-model <bagel-tokenizer> \
  --native-model-checkpoint <native-init.safetensors> \
  --seed 42 \
  --output <bridge-init-directory>
```

Launch training with the same one-process-per-GPU topology:

```bash
python examples/models/bagel/pretrain_bagel.py \
  --recipe pretrain-32gpu \
  --bagel-repo <bagel-repo> \
  --model-path <bagel-model-files> \
  --dataset-root <prepared-wds-root> \
  --tokenizer-model <bagel-tokenizer> \
  --mcore-checkpoint <bridge-init-directory> \
  --seed 42 \
  --train-iters <iterations>
```

The 32-GPU recipe is pure data parallelism with Megatron FSDP; tensor,
pipeline, and context parallel sizes are all one. The validated data topology
uses one logical data worker per rank. Other model-parallel topologies and data
worker counts are not yet supported by the training recipe.

## Alignment evidence

`pretrain_official_alignment.py` runs the official implementation from the
same native initialization and records CE, MSE, gradient norm, learning rate,
token throughput, MFU, and memory. `compare_loss_curves.py` compares its JSONL
trace with the Bridge trace and can emit JSON, CSV, and an SVG loss graph.

The alignment run uses the official sample data and is a numerical integration
test, not a model-quality benchmark. Record the exact BAGEL, Megatron-LM, and
Bridge SHAs together with seed, world size, worker topology, and checkpoint
hashes when publishing results.
