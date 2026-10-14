# Qwen4-Exp / Qwen3.8-Flash-Next image training

**Early recipe: performance is unoptimized.** This label applies to throughput
and resource efficiency, not to the validity of the measured training loss.

[`finetune_nemotron_image_v3.py`](finetune_nemotron_image_v3.py) publishes the
full-checkpoint VL training recipe used with **Qwen/Qwen3.8-Flash-Next** and
**Nemotron-Image-Training-v3**. It builds a `ConfigContainer`, loads the complete
HF checkpoint through an AutoBridge pre-wrap hook, and calls Bridge `pretrain`
with an image-aware forward step. Language, vision, and projector parameters
are all trainable. It is an executable recipe, not a registered
`scripts/training/run_recipe.py` preset.

The measured run completed **5,349 steps in 3h58m on 64 GB200 GPUs**. First/last
10-step mean loss was **0.31155 / 0.01778**; no recorded update was skipped or
nonfinite. The [loss curve and run results](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6354#issuecomment-6160448312)
remain valid training measurements. This is repeated-subset training without
held-out evaluation, not a numerical parity or performance claim.

## Environment and checkpoint

The measured run used NeMo **26.10.rc4**, Bridge commit
`228077d7b63b13d8b3b7f276c78f53c8110c57ac`, and a Transformers **5.16.1** overlay
for Qwen4-Exp model/processor support. Use an environment with that support;
the stock container's Transformers installation may need the same isolated
overlay. These scripts do not install or change dependencies.

Download the full checkpoint and processor assets before training. The run used
HF revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Set these portable paths
in your shell, and run the commands below from the Bridge checkout:

```bash
export CHECKPOINT=/path/to/Qwen3.8-Flash-Next
export DATA_SOURCE=/path/to/nemotron-image-v3-downloads
export DATASET=/path/to/prepared-ecd-subset
export OUTPUT=/path/to/new-training-output
```

## Prepare the image subset

The companion script reuses the repository's
[`prepare_nemotron_image_v3.py`](../../../../tutorials/data/energon/prepare_nemotron_image_v3.py)
converter for media resolution, conversation normalization, shard writing, and
Energon indexing. It downloads pinned ECD text and image payloads and selects
the first **8,192 complete, supervised conversations** that fit sequence2048
in source order. It skips overlength conversations instead of truncating them.
The result is 32 training shards, with no validation split. The ECD images come
from `ChartFoundation/ECD-10k-Images`.

```bash
CUDA_VISIBLE_DEVICES='' uv run python \
  examples/models/qwen/qwen4_exp/prepare_nemotron_image_v3_subset.py \
  --checkpoint "$CHECKPOINT" --source-dir "$DATA_SOURCE" \
  --output-dir "$DATASET"
```

Preparation uses four CPU workers and does not load model weights. Use a fresh
output directory. `--max-samples` can select a smaller subset for data checks.
The manifest binds processor assets, selection records, and shard hashes;
training rejects different processor assets or changed selected token/label/mask
prefixes. Existing subsets prepared by the verification driver are compatible.

Run a CPU-only check of the real data encoder and finalized configuration:

```bash
WORLD_SIZE=64 RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES='' uv run python \
  examples/models/qwen/qwen4_exp/finetune_nemotron_image_v3.py \
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" \
  --output "$OUTPUT/config-check" --check-config --check-batches 4
```

## Train

The defaults reproduce the measured configuration: **64 GPUs, TP8 with SP,
EP64, expert TP1, PP1, CP1**, microbatch1, global batch8, sequence2048, seed1234,
BF16 parameters, FP32 gradient buffers, distributed Adam with constant LR1e-6,
weight decay0, and clip1.0. This early recipe deliberately rejects other world
sizes, EP sizes, sequence lengths, and global batch sizes.

Within an existing 16-node allocation with four visible GPUs per node, run the
following once per node. Set `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` for
your launcher. Stage inputs and use suitable node-local caches according to
your cluster's storage policy.

```bash
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python -m torch.distributed.run \
  --nnodes=16 --nproc_per_node=4 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  examples/models/qwen/qwen4_exp/finetune_nemotron_image_v3.py \
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" --output "$OUTPUT/train" \
  --max-runtime-minutes 240 --exit-reserve-seconds 180
```

The runtime budget starts when the Python driver starts, including model load;
it is not the scheduler's allocation start time. To stop before an allocation's
actual deadline, supply the same `--stop-at-epoch UNIX_SECONDS` on every rank;
this overrides the relative runtime budget. The callback reserves cleanup time
and stops after complete optimizer steps. `--steps` is an optional upper bound
(default1,000,000). Output directories for training must be fresh.

## Training behavior and limits

- One still image per conversation, min/max pixels65,536/262,144. The processor
  and vision patch/merge geometry come from the HF checkpoint.
- Unpacked sequences; EP configuration finalization pads to2,048. The original
  complete sample and its labels/mask are checked against preparation records.
  Image/user/padding labels are masked; assistant-token loss is normalized once
  by the global supervised-token count, with SUM gradient collectives.
- Native allgather MoE dispatcher and grouped GEMM, no fused permutation/shared
  expert overlap, gradient-accumulation fusion, communication overlap,
  activation recomputation, or CUDA graphs. No auxiliary MoE loss or dropout.
- Each rank writes loss, preclip gradient norm, and sampled owned FP32 master
  updates for language/vision/projector parameters. These checks add overhead;
  they do not establish every parameter's gradient or update correctness.
- No checkpoint saving/resume or validation is enabled in this recipe.
  Gradient spikes in the measured run remain unexplained (maximum230.02);
  clipping was active. This recipe does not establish HF/native backward parity
  or correctness across TP/EP/PP/CP settings.

The publication preserves the measured driver's training and preprocessing
operations while adding portable CLI defaults and source discovery. The
four-hour loss curve comes from the original run, not a second full training
run of this packaged script. **Performance optimization remains future work.**
