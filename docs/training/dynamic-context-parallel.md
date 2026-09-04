# Dynamic Context Parallelism

Dynamic context parallelism (DCP) chooses a context-parallel group for each
variable-length sequence instead of running every sequence on the configured
maximum CP size. It is useful for long-context SFT and RL batches whose sequence
lengths vary enough that static CP leaves many ranks underutilized.

This guide uses CoderForge GPT SFT with 128K runtime in-batch packing as a
concrete example. DCP replaces `GPTSFTDataset`'s collate-time packer after the
logical global batch has been selected; it does not redefine that batch.

## Ownership Boundaries

DCP has four separate stages:

1. The framework dataloader selects the logical global batch. Token-count,
   unmasked-token-count, or RL-specific batch policies stay here.
2. The MCore scheduler assigns the selected sequences to runtime CP groups.
3. MCore transports caller-selected, sequence-aligned tensors to the assigned
   ranks without interpreting their field names or values.
4. The framework materializes each rank-local THD batch, including tokens,
   labels, loss masks, position IDs, padding, and packed-sequence metadata.

This split keeps model- and objective-specific fields out of MCore. Bridge's
GPT path materializes the standard SFT fields. An RL framework can reuse the
same MCore placement and transport APIs while supplying its own materializer
for fields such as advantages, returns, or old log-probabilities. Multimodal
callers similarly remain responsible for their model-specific metadata.

The configured global batch size still counts source samples, not the
rank-local THD rows produced by DCP. The number of execution microbatches may
change after scheduling, but optimizer and learning-rate sample accounting
continue to use the original global batch.

## Current Bridge Scope

The initial Bridge integration supports:

- text-only GPT SFT with `enable_in_batch_packing=True`; offline-packed input is
  intentionally rejected
- `dataloader_type="single"`, `"cyclic"`, or `"batch"`; the configured
  micro-batch size remains a logical data-selection quantity
- pipeline parallel size 1 with no virtual pipeline parallelism
- eager training and loss evaluation
- runtime CP groups selected from power-of-two factors of the configured
  `DP * CP` pool, plus the full pool size

The rank-local non-loss evaluation collector is not supported. Bridge's GPT
forward path also does not automatically carry arbitrary RL or multimodal
fields; those frameworks should provide their own materialization and forward
contracts.

## Megatron Core Prerequisite

This integration requires the companion MCore Dynamic CP V2 stack, including
runtime CP-group-aware `PackedSeqParams`, attention group restoration, and the
framework-facing `gather_global_sequence_lengths()` and
`reroute_tensor_fields_to_dcp_ranks()` APIs. The validation below used MCore
commit `70ef3ebbf65f97da207a8ee069c105844e60f78e`. Until equivalent changes are
included in Bridge's published MCore pin, check out that companion commit in
`3rdparty/Megatron-LM` before running this guide.

## 128K CoderForge Example

The following command starts Qwen3-30B-A3B SFT on 16 H100 GPUs. Replace the
checkpoint and output roots with paths available to your environment.

```bash
./scripts/training/train.sh \
  --nodes 2 --gpus-per-node 8 \
  --recipe qwen3_30b_a3b_sft_16gpu_h100_bf16_config \
  --mode sft --dataset coderforge \
  --pretrained_checkpoint work/model-verification/qwen3-30b-a3b/imported-megatron/iter_0000000 \
  --max_steps 12 --seq_length 131072 \
  --global_batch_size 32 --micro_batch_size 2 \
  -tp 1 -pp 1 -cp 16 -ep 16 -etp 1 \
  'dataset.hf_dataset.split="SWE_Rebench[:2048]"' \
  'dataset.hf_dataset.load_kwargs={revision:"060fca96cf723b2ebab3181e9e59fafd273df3cb",data_files:{SWE_Rebench:"trajectories/SWE_Rebench-*"},verification_mode:no_checks}' \
  '++tokenizer.hf_tokenizer_kwargs.revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"' \
  dataset.hf_output_root=work/data/coderforge/qwen3-30b-a3b-128k-dcp \
  dataset.hf_rewrite=true dataset.seed=1234 rng.seed=5678 \
  dataset.do_validation=false dataset.hf_validation_proportion=null \
  dataset.dataloader_type=cyclic dataset.enable_in_batch_packing=true \
  model.dynamic_context_parallel=true \
  model.sequence_packing_scheduler=default_dynamic_cp \
  model.max_seqlen_per_dp_cp_rank=8192 \
  model.min_dynamic_context_parallel_size=8 \
  model.calculate_per_token_loss=true \
  model.cross_entropy_loss_fusion=false \
  model.recompute_granularity=full \
  model.recompute_method=uniform model.recompute_num_layers=1 \
  ddp.average_in_collective=false ddp.nccl_ub=false \
  dist.use_decentralized_pg=false \
  scheduler.lr_decay_iters=12 \
  validation.eval_iters=0 validation.eval_interval=0 \
  checkpoint.load=null checkpoint.save=null \
  logger.log_interval=1 logger.log_throughput=true
```

Bridge derives the per-sequence THD alignment from the topology, so CP16 uses a
multiple of 32. `max_seqlen_per_dp_cp_rank=8192` is the scheduler's per-rank
sequence-length budget; lower values favor larger runtime CP groups, while
higher values give short sequences fewer ranks. Tune it against memory
headroom and step time rather than treating it as a model context limit. The
example bounds the runtime to CP8 or CP16 so a smaller TE P2P topology cannot
be introduced for the first time after steady-state memory is resident.

After the first run materializes the GPTSFT JSONL, set
`dataset.hf_rewrite=false` for comparisons. To measure the static-CP baseline,
keep the same source rows, order, seed, logical batch, and topology, then set:

```text
model.dynamic_context_parallel=false
model.sequence_packing_scheduler=null
```

With DCP disabled, the same `enable_in_batch_packing=True` request goes through
the original `GPTSFTDataset._collate_in_batch` path. Keep logical MBS greater
than one for this matched legacy baseline.

Check the log for the number of source sequences, scheduled execution
microbatches, and runtime CP group histogram. Compare the last ten completed
steps after warmup; a one-step smoke test establishes functionality but is not
a performance result.

## Diagnostic 128K Comparison

A bring-up comparison used the same materialized source snapshot, seeds,
logical GBS/MBS 32/2, and TP1/PP1/CP16/EP16/ETP1 topology for both modes. It did
not capture and replay exact logical batches, so the independent runs are not
convergence-parity evidence. Each mode completed 12 optimizer steps on 16 H100
GPUs; the table averages steps 3 through 12.

| Mode | Step time (ms) | Configured tokens/s/GPU | Reported TFLOP/s/GPU | Peak allocated (GiB) | Peak reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static CP16 | 55,885.210 | 4,690.758 | 406.270 | 57.615 | 70.578 |
| Dynamic CP8/16 | 75,769.620 | 3,459.751 | 116.780 | 54.466 | 74.113 |
| DCP change | +35.581% | -26.243% | not comparable | -5.466% | +5.009% |

Both completed runs had finite loss with zero skipped and zero NaN iterations.
DCP preserved the 32-sample logical global batch, scheduled 13--16 execution
microbatches per step, and used runtime CP8 and CP16 groups. The original
collate-time packer already produced 16 efficient THD microbatches, so the
small reduction in execution-microbatch count did not offset runtime-group and
scheduling overhead. DCP was therefore slower for this length distribution.

An earlier run with `min_dynamic_context_parallel_size=1` completed ten steps,
then failed when CP4 first appeared: Transformer Engine's unbatched P2P path
tried to create a new pairwise NCCL communicator after steady-state memory was
resident. Restricting the run to CP8/16 completed all 12 steps. The diagnostic
launcher also omitted the recipe's allocator environment defaults, so treat
the memory values and failure threshold as bring-up evidence rather than a
canonical performance result.

Use step time or the identically computed configured-token rate for this
comparison. The reported TFLOP/s values use different FLOP numerators: the
static CP path falls back to a configured 128K BSHD estimate, while DCP supplies
the actual per-sequence attention lengths. Configured-token throughput still
includes padded and masked capacity; measure supervised-token throughput
separately when comparing dataset utility.

## Tuning and Failure Checks

- Use identical materialized source rows, order, and logical batch settings for
  DCP on/off comparisons.
- Keep the largest physical sequence divisible by twice the largest runtime CP
  size. For a configured CP of 16, pad internal THD segments to a multiple of
  32.
- If most sequences select CP16, DCP has little room to improve utilization.
  Confirm the length distribution and memory budget before changing the
  scheduler threshold.
- If short sequences use smaller groups but step time does not improve, inspect
  scheduler imbalance, MoE communication, and host-side launch gaps separately.
- Leave headroom for runtime-CP communicator initialization. A CP size that
  first appears in a later batch can trigger lazy NCCL P2P communicator
  allocation after model and activation memory are already resident. If that
  fails near the memory limit, raise `min_dynamic_context_parallel_size`, lower
  the per-rank sequence budget to avoid the smaller group, or warm the actual
  Transformer Engine P2P peer pattern before training. A collective barrier on
  each process group does not initialize these pairwise communicators.
- Treat useful-token throughput and model TFLOP/s as different measurements.
  Packed padding and masked labels can make configured token capacity larger
  than actual supervised tokens.

## Related Documentation

- [Packed Sequences](packed-sequences.md)
- [Hierarchical Context Parallel](hierarchical-context-parallel.md)
- [MoE Optimization](moe-optimization.md)
