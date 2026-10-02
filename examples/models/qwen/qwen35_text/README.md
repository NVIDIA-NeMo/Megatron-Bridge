# Qwen3.5 35B-A3B text-only 128K MXFP8 SFT

`qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config` trains the
text decoder and one pretrained MTP layer on 16 GB200 GPUs within one NVLink
domain: TP1, PP1, CP8, EP16, ETP1, global batch 32 and microbatch 1.
It enables MXFP8 compute and parameter gather, CuTeDSL grouped MLP, single
expert-weight storage, FP32 gradient reduction, learned expert routing,
HybridEP, and selective `gdn_norm_out`/`moe` recomputation.

## 1. Prepare the text checkpoint

Download `Qwen/Qwen3.5-35B-A3B` at revision
`59d61f3ce65a6d9863b86d2e96597125219dc754` into a local HF snapshot. Extract its
text weights, then import them into Megatron format:

```bash
uv run python scripts/conversion/extract_qwen35_text_checkpoint.py \
  --source "$HF_SNAPSHOT" --output "$TEXT_HF"

./scripts/conversion/convert.sh import \
  --executor local --device cpu \
  --hf-model "$TEXT_HF" --megatron-path "$TEXT_MEGATRON" \
  --torch-dtype bfloat16
```

The extractor retains decoder, output-head and MTP weights and removes the
vision tower. CPU conversion needs enough host memory for the full text model;
use the conversion launcher's distributed GPU backend when necessary.

## 2. Prepare the data

The default builder selects `togethercomputer/CoderForge-Preview`, configuration
`trajectories`, split `SWE_Rebench`, pinned to revision
`060fca96cf723b2ebab3181e9e59fafd273df3cb`. It normalizes messages and tools,
derives a seeded 5% validation split, applies the checkpoint's chat template,
and packs conversations into 131072-token budgets. Conversations are aligned
to 16 tokens for CP8; their attention boundaries remain separate. Assistant
responses are supervised; system/user/tool content and padding are masked.
A 128K pack can contain several shorter conversations.

Set `NEMO_HOME` to shared storage for materialized and packed data. Do not pass
`--dataset coderforge` when retaining the recipe's pinned source and validation
settings: a dataset preset replaces them. Prebuilt packs must use the same
checkpoint tokenizer and CP8 alignment. Their exact paths can be supplied as
shown in the validation overrides below.

## 3. Launch training

The validated runtime is NeMo 26.10.rc3 with MCore revision
`b5581a9e874fc5be2b91e74cdd3ccdc64a1b5204` from the pending
[MTP padding-mask fix #7611](https://github.com/NVIDIA/Megatron-LM/pull/7611).
That fix is required for packed MTP correctness and is **not included** in this
recipe PR or the current MCore main pin. Older TE/cuDNN stacks may lack the
grouped-tensor API or padded THD attention support for 256-wide heads.

For four GPUs per node, request four nodes within one NVLink domain using your
cluster's topology settings. Export `WANDB_API_KEY` before launching:

```bash
./scripts/training/train.sh \
  --nodes 4 --gpus-per-node 4 \
  --account "$ACCOUNT" --partition "$PARTITION" \
  --container-image "$CONTAINER_IMAGE" \
  --mount "$WORKSPACE" --env HF_HOME --env NEMO_HOME --env WANDB_API_KEY \
  --mount "$PWD/src:/opt/Megatron-Bridge/src" \
  --mount "$PWD/scripts:/opt/Megatron-Bridge/scripts" \
  --mount "$MCORE_SOURCE:/opt/Megatron-Bridge/3rdparty/Megatron-LM" \
  --recipe qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config \
  --pretrained_checkpoint "$TEXT_MEGATRON" \
  --save_dir "$WORKSPACE/results/qwen35-text-mxfp8" \
  "logger.wandb_project=$WANDB_PROJECT" \
  "logger.wandb_entity=$WANDB_ENTITY"
```

Run from this Bridge checkout and set `MCORE_SOURCE` to a separate MCore
checkout containing the revision above. All mounted paths must be accessible
on the allocated nodes. The recipe sets the CuTeDSL and HybridEP environment flags before imports
through the public runner. It runs 500 updates, with warmup 200 and LR decay
horizon 300000. A 100-update check remains within warmup and demonstrates
short-run stability, not completed-schedule convergence.

The Python recipe accepts `seq_length`, `hf_path` and `hf_revision`. For an
extracted local HF config use `hf_path=TEXT_HF, hf_revision=None`. Config and
tokenizer share a source. Context lengths must be positive multiples of 16 and
fit the model's position limit. Resizing CP/TP/EP requires revalidating packing
alignment, batch divisibility and dispatcher layout.

## Validation scope

The reference launcher and migrated recipe each completed 100 updates on
16 GB200 GPUs. Every update had finite LM, MTP and router losses, zero skipped
or NaN iterations, and evaluations at steps 50 and 100. All four loss histories
matched within `1e-6 + 0.01 * abs(reference)` at every update.

| Measurement | Reference launcher | Bridge recipe |
| --- | ---: | ---: |
| Final LM loss | 0.1486210 | 0.1485086 |
| Final MTP loss | 0.1949851 | 0.1948212 |
| Last 10 updates, seconds/update | 28.9344 | 28.5919 |
| Last 10 updates, model TFLOP/s/GPU | 332.21 | 336.19 |

Runs: [reference](https://wandb.ai/nvidia/qwen35-text-sft-mcore/runs/qczs0k0o)
and [Bridge](https://wandb.ai/nvidia/qwen35-text-sft-mcore/runs/9hdu7ypa).
These used Bridge base `00ca266c4abc5f8435821823a1dbe05d0c3d6033` plus this
recipe's precursor, with the execution settings now made recipe defaults.
The published change has separate construction and configuration checks on
its newer main base; the 100-step evidence predates that rebase.

The comparison reused the reference's prebuilt data: a seeded 99:1 split of
77169 trajectories, first 20% of the training split (15279 trajectories), and
all 772 validation trajectories. `first_fit_shuffle` packing produced 6491
training packs and 333 validation packs. Seed 1234 and the full 300000-step LR
horizon were preserved. This does not qualify the default builder's 5% split.
For the same bounded workload, append these overrides to the launch command:

```bash
train.train_iters=100 \
validation.eval_global_batch_size=32 validation.eval_micro_batch_size=1 \
dataset.hf_dataset=null dataset.hf_validation_proportion=null \
"dataset.dataset_root=$MATERIALIZED_DATA" \
"dataset.offline_packing_specs.packed_train_data_path=$PACKED_DATA/training.parquet" \
"dataset.offline_packing_specs.packed_val_data_path=$PACKED_DATA/validation.parquet" \
"dataset.offline_packing_specs.packed_metadata_path=$PACKED_DATA/metadata.jsonl" \
"tokenizer.tokenizer_model=$TEXT_HF" '++tokenizer.hf_tokenizer_kwargs.revision=null' \
"checkpoint.hf_source_path=$TEXT_HF" \
checkpoint.load=null checkpoint.load_optim=false checkpoint.load_rng=false \
checkpoint.save=null checkpoint.save_interval=null checkpoint.async_save=false \
logger.log_interval=1 logger.log_throughput=true
```

The recipe initializes from pretrained weights with FP32 master-parameter
loading, and disables optimizer/RNG loading. For a full training resume, set
`checkpoint.load_main_params_from_ckpt=false`, `checkpoint.load_optim=true`,
`checkpoint.load_rng=true` and an explicit `checkpoint.load`.

Checkpoint saving was disabled for both comparison runs; checkpoint resume and
post-SFT export were not tested. BF16 qualification is tracked separately and
this change exposes only the verified MXFP8 variant.
