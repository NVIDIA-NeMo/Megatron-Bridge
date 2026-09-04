# Dynamic Context Parallelism

Dynamic context parallelism (DCP) chooses a context-parallel group for each
variable-length sequence instead of running every sequence on the configured
maximum CP size. It is useful for long-context SFT and RL batches whose sequence
lengths vary enough that static CP leaves many ranks underutilized.

This guide uses offline-packed CoderForge SFT at 128K as a concrete example.
DCP changes where the already-selected samples execute; it does not redefine
the logical global batch.

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

The configured global batch size still counts logical packed rows. The number
of execution microbatches may change after DCP scheduling, but optimizer and
learning-rate sample accounting continue to use the original global batch.

## Current Bridge Scope

The initial Bridge integration supports:

- text-only GPT SFT with offline-packed THD data
- `dataloader_type="batch"` and `micro_batch_size=1`
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
  --max_steps 20 --seq_length 131072 \
  --global_batch_size 32 --micro_batch_size 1 \
  -tp 1 -pp 1 -cp 16 -ep 16 -etp 1 \
  'dataset.hf_dataset.split="SWE_Rebench[:2048]"' \
  'dataset.hf_dataset.load_kwargs={revision:"060fca96cf723b2ebab3181e9e59fafd273df3cb",data_files:{SWE_Rebench:"trajectories/SWE_Rebench-*"},verification_mode:no_checks}' \
  '++tokenizer.hf_tokenizer_kwargs.revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"' \
  dataset.hf_output_root=work/data/coderforge/qwen3-30b-a3b-128k-dcp \
  dataset.hf_rewrite=true dataset.seed=1234 rng.seed=5678 \
  dataset.do_validation=false dataset.hf_validation_proportion=null \
  dataset.enable_offline_packing=true \
  'dataset.offline_packing_specs={packed_sequence_size:131072,pad_seq_to_mult:32,num_tokenizer_workers:8}' \
  model.dynamic_context_parallel=true \
  model.sequence_packing_scheduler=default_dynamic_cp \
  model.max_seqlen_per_dp_cp_rank=8192 \
  model.min_dynamic_context_parallel_size=1 \
  model.calculate_per_token_loss=true \
  model.cross_entropy_loss_fusion=false \
  model.recompute_granularity=full \
  model.recompute_method=uniform model.recompute_num_layers=1 \
  ddp.average_in_collective=false ddp.nccl_ub=false \
  dist.use_decentralized_pg=false \
  scheduler.lr_decay_iters=20 \
  validation.eval_iters=0 validation.eval_interval=0 \
  checkpoint.load=null checkpoint.save=null \
  logger.log_interval=1 logger.log_throughput=true \
  '~env_vars.NCCL_GRAPH_REGISTER=0' \
  '~env_vars.NCCL_NVLS_ENABLE=0' \
  '~env_vars.PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' \
  '~env_vars.TORCH_NCCL_AVOID_RECORD_STREAMS=1' \
  '~env_vars.TORCH_NCCL_HIGH_PRIORITY=1'
```

`pad_seq_to_mult=32` supplies the `2 * CP` alignment required by THD context
parallelism. `max_seqlen_per_dp_cp_rank=8192` is the scheduler's per-rank
sequence-length budget; lower values favor larger runtime CP groups, while
higher values give short sequences fewer ranks. Tune it against memory headroom
and step time rather than treating it as a model context limit.

After the first run prepares the packed dataset, set `dataset.hf_rewrite=false`
for comparisons. To measure the static-CP baseline, keep the same data, seed,
batch, and topology, then set:

```text
model.dynamic_context_parallel=false
model.sequence_packing_scheduler=null
```

Check the log for the number of source sequences, scheduled execution
microbatches, and runtime CP group histogram. Compare the last ten completed
steps after warmup; a one-step smoke test establishes functionality but is not
a performance result.

## Measured 128K Comparison

The command above was measured with the same packed artifacts, seeds, and
TP1/PP1/CP16/EP16/ETP1 topology for both modes. Each run completed 12 optimizer
steps on 16 H100 GPUs; the table averages steps 3 through 12.

| Mode | Step time (ms) | Configured tokens/s/GPU | Reported TFLOP/s/GPU | Peak reserved memory (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Static CP16 | 134,556.920 | 1,948.202 | 336.750 | 67.650 |
| Dynamic CP | 127,308.300 | 2,059.127 | 168.790 | 63.955 |
| DCP change | -5.387% | +5.694% | not comparable | -5.462% |

Both runs completed with zero skipped and zero NaN iterations. The packed
dataset was 93.38% efficient and averaged 2.387 sequences per packed row. DCP
preserved the 32-row logical global batch, materialized roughly 70--85 internal
sequences per step, and selected runtime CP4, CP8, and CP16 groups; most groups
were CP8 or CP16, which limits the available gain for this particular length
distribution.

Use step time or the identically computed configured-token rate for this
comparison. The reported TFLOP/s values intentionally use different FLOP
numerators: the static path estimates fixed 128K pack slots, while DCP uses the
actual per-sequence attention lengths and drops zero-logical tail padding.
Consequently, the lower DCP TFLOP/s number does not mean that DCP executed more
slowly. Configured-token throughput still includes padded and masked capacity;
measure supervised-token throughput separately when comparing dataset utility.

## Tuning and Failure Checks

- Use identical packed artifacts for DCP on/off comparisons. Repacking can
  change sample membership and invalidate the comparison.
- Keep the largest physical sequence divisible by twice the largest runtime CP
  size. For a configured CP of 16, pad internal THD segments to a multiple of
  32.
- If most sequences select CP16, DCP has little room to improve utilization.
  Confirm the length distribution and memory budget before changing the
  scheduler threshold.
- If short sequences use smaller groups but step time does not improve, inspect
  scheduler imbalance, MoE communication, and host-side launch gaps separately.
- Treat useful-token throughput and model TFLOP/s as different measurements.
  Packed padding and masked labels can make configured token capacity larger
  than actual supervised tokens.

## Related Documentation

- [Packed Sequences](packed-sequences.md)
- [Hierarchical Context Parallel](hierarchical-context-parallel.md)
- [MoE Optimization](moe-optimization.md)
