# Qwen3.8-Flash-Next image SFT

**Selected configuration: TP4 with HybridEP and fusions.** A 100-step native
training run completed with finite loss. Same-input numerical differences from
the earlier configuration remain unresolved; this is functional training
evidence, not a parity result. Historical loss curves remain valid.

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
Energon dataset configuration. `--dry-run` resolves the launch configuration
without allocating the model; it is not a runtime or correctness check.

Defaults preserve the measured training settings: **64 GB200 GPUs, TP4/SP,
EP64, expert TP1, PP1, CP1**, sequence2048, microbatch1, global batch16, BF16
parameters, FP32 gradient reduction, distributed Adam with constant LR1e-6,
weight decay0, clip1, and seed1234. The image limits are65,536–262,144 pixels.
Sequences are unpacked and explicitly padded to2048. Auxiliary MoE loss and
dropout are disabled. HybridEP flex dispatch and grouped GEMM are enabled with
permutation fusion, permutation fusion into HybridEP, router fusion,
bias-activation fusion, and gradient-accumulation fusion. Shared-expert overlap,
gradient-reduction/parameter-gather overlap, recomputation, and CUDA graphs are disabled.
No evaluation or checkpoint saving is enabled. HybridEP requires the 64-rank
expert group to fit a supported NVLink domain; confirm placement and backend
availability before using this configuration.

The standard SFT path loads the full checkpoint and refreshes optimizer master
parameters before the first step. Each step logs loss, pre-clipping gradient
norm, skipped/NaN counts, iteration time, and estimated model TFLOP/s/GPU.
TensorBoard logging is enabled when its output path is supplied. Save the
runtime-synchronized configuration produced at setup entry and complete keyed
step records. Record initial/final loss and average step time and TFLOP/s over the same final ten steps in the card;
derive token-slot throughput as `2048 * 16 / (step_seconds * 64)`.
TFLOP/s uses Bridge's analytical matrix-work estimate, not hardware counters
or MFU. It includes GDN, dense attention, MoE, logits, vision, gated-residual
and PLE projections. At the recipe's sequence length2048, the dense QSA path
bypasses the indexer.
Norms, activations, embedding lookups, PLE depthwise convolution, communication,
and duplicated work across parallel ranks are outside the matrix estimate.
Longer or forced-sparse QSA configurations warn that their estimate is approximate.

## Recorded loss

The original full-checkpoint run completed **5,349 steps in3h58m on64 GB200s**
from Bridge commit `228077d7b63b13d8b3b7f276c78f53c8110c57ac`.
Initial/final loss was **0.31087 / 0.01342**; first/last ten-step mean loss was
**0.31155 / 0.01778**, with zero recorded skipped or nonfinite updates.
The [loss curve](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6354#issuecomment-6160448312)
remains valid evidence from that run. It did not capture trustworthy TFLOP/s.

A historical baseline run of the public `train.sh` command completed **100 optimizer
steps** from Bridge commit `99188699f92108885b7f8823f87029369d2eb447`.
Loss was **0.3097828 / 0.2084261**, with zero skipped or NaN iterations and a
clean exit. Over the final ten steps, mean time was **1,403.24 ms**, mean
analytical throughput was **7.75 TFLOP/s/GPU**, and derived token-slot throughput
was **182.43 tokens/s/GPU**. These are early, unoptimized TP8/GBS8 allgather measurements.
The 5,349-step run used the previous verification driver; its per-rank sampled
gradient/update callback and deadline controller are not part of the library
recipe. Repeated-subset training does not establish held-out quality or
HF/backward/parallelism parity; historical gradient spikes (maximum230.02
before clipping) remain unexplained.

The selected configuration completed **100 native Energon-loader steps** from
commit `62f79249661338ea921f00d927d704bde10833cb` with TP4/GBS16, HybridEP, and the
five fusions listed above. Its explicit run overrides are now recipe defaults;
the public command above expresses the same workload. Loss was
**0.350771964 / 0.167497709**, with zero skipped or nonfinite updates and a clean
exit. Final-ten mean time was **1,525.19 ms**, mean analytical throughput
**14.26 TFLOP/s/GPU**, and padded token-slot throughput **335.70 tokens/s/GPU**.
This native-loader run did not use canonical batch replay or profiling. It has
no input-matched native-loader baseline, so these numbers do not establish a
native-loader speedup over the historical GBS8 run.

A separate 40-step comparison used identical canonical global batches at GBS16.
Against its TP8 baseline, the selected TP4 combination had maximum absolute
loss difference **0.008722812** and preclip gradient-norm difference **28.2429**.
Differences began at step one and remain unexplained. Finite training and faster
measured execution do not establish numerical equivalence. Recomputation stays
disabled; this recipe does not claim qualified shared-expert recomputation.
