# Qwen3.8-Flash-Next image SFT

**Early recipe: performance is unoptimized.** This applies to throughput and
resource efficiency. The measured training loss remains valid.

The library recipe
[`qwen38_flash_next_sft_64gpu_gb200_bf16_config`](../../../../src/megatron/bridge/recipes/qwen_vl/qwen38_flash_next.py)
uses the standard training runner, Qwen VL forward step, Energon data loader,
and HF checkpoint initialization. Language, vision, and projector parameters
are all trainable. It replaces the standalone verification training script.

## Prepare data

Use NeMo **26.10.rc4** with Transformers **5.16.1** and tokenizers **0.23.1**
for Qwen4-Exp processor/model support. These commands do not install dependencies.
Download **Qwen/Qwen3.8-Flash-Next**, revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, including its processor assets.
Use the same revision for preparation and weight loading. The recipe pins its
model config, processor, and tokenizer to this revision.

The preparation script reuses the existing Nemotron converter and selects the
first **8,192 complete, supervised conversations** from the pinned ECD portion
of **nvidia/Nemotron-Image-Training-v3**. Images come from
**ChartFoundation/ECD-10k-Images**. Overlength conversations are skipped whole,
without truncation; the output has 32 shards and no validation split.

Run from the repository root, with portable paths supplied by the caller:

```bash
CUDA_VISIBLE_DEVICES='' uv run python \
  scripts/data/prepare_nemotron_image_v3_subset.py \
  --checkpoint /path/to/Qwen3.8-Flash-Next \
  --source-dir /path/to/downloads --output-dir /path/to/prepared-ecd
```

Preparation uses four CPU workers and writes a manifest with processor,
selection, and shard hashes. Existing subsets from the measured run are usable.
Keep the pinned processor and preprocessing limits when reusing the subset;
the standard runner does not replay the old driver's per-sample hash checks.

## Use the recipe

Within an existing 16-node allocation with four GPUs per node, invoke the
standard runner once per node with the launcher-provided rank and rendezvous:

```bash
uv run python -m torch.distributed.run \
  --nnodes=16 --nproc_per_node=4 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  scripts/training/run_recipe.py \
  --recipe qwen38_flash_next_sft_64gpu_gb200_bf16_config \
  --pretrained_checkpoint /path/to/Qwen3.8-Flash-Next \
  --max_steps 100 \
  dataset.path=/path/to/prepared-ecd \
  logger.save_config_filepath=/path/to/output/config.yaml \
  logger.tensorboard_dir=/path/to/output/tensorboard
```

Create the output directory first. The
[verification card](../../../model_verification_cards/qwen3.8-flash-next/card.yaml)
uses the public `scripts/training/train.sh` launcher with this same recipe.
Supply scheduler configuration through the launcher's normal external setup.
Do not select a different `--dataset` preset: it would replace the prepared
Energon dataset configuration. `--dryrun` resolves the launch configuration
without allocating the model; it is not a runtime or correctness check.

Defaults preserve the measured training settings: **64 GB200 GPUs, TP8/SP,
EP64, expert TP1, PP1, CP1**, sequence2048, microbatch1, global batch8, BF16
parameters, FP32 gradient reduction, distributed Adam with constant LR1e-6,
weight decay0, clip1, and seed1234. The image limits are65,536–262,144 pixels.
Sequences are unpacked and runtime finalization pads to2048 for EP. Auxiliary
MoE loss and dropout are disabled. Allgather dispatch, grouped GEMM, and
unfused permutation are used, with no overlap, recomputation, or CUDA graphs.
No evaluation or checkpoint saving is enabled.

The standard SFT path loads the full checkpoint and refreshes optimizer master
parameters before the first step. Each step logs loss, pre-clipping gradient
norm, skipped/NaN counts, iteration time, and estimated model TFLOP/s/GPU.
TensorBoard logging is enabled when its output path is supplied. Save the
post-setup config and complete keyed step records. Record initial/final loss
and average step time and TFLOP/s over the same final ten steps in the card;
derive token-slot throughput as `2048 * 8 / (step_seconds * 64)`.
TFLOP/s uses Bridge's partial analytical architecture estimate, not hardware
counters or full-model MFU. It includes GDN, dense attention, MoE, logits, and
vision work, but omits Qwen4-specific gated-residual, PLE, and QSA-indexer work.
Those omissions are material; compare only measurements using the same formula.

## Recorded loss

The original full-checkpoint run completed **5,349 steps in3h58m on64 GB200s**
from Bridge commit `228077d7b63b13d8b3b7f276c78f53c8110c57ac`.
Initial/final loss was **0.31087 / 0.01342**; first/last ten-step mean loss was
**0.31155 / 0.01778**, with zero recorded skipped or nonfinite updates.
The [loss curve](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6354#issuecomment-6160448312)
remains valid evidence from that run. It did not capture trustworthy TFLOP/s.

That run used the previous verification driver. The new standard-runner command
requires a fresh run before its metrics can be marked verified. The old
per-rank sampled-gradient/update callback and deadline controller are not part
of the library recipe. Repeated-subset training does not establish held-out
quality or HF/backward/parallelism parity; gradient spikes (maximum230.02 before
clipping) remain unexplained. **Performance optimization remains future work.**
