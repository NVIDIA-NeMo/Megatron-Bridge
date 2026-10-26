# Dynamic Context Parallel

Dynamic context parallelism (dynamic CP) runs every packed microbatch on a
context-parallel group sized for its longest sequence, instead of splitting
every microbatch across the full `context_parallel_size` group. It builds on
[global-batch online packing](packed-sequences.md#global-batch-online-packing):
the packing scheduler already sees the whole global batch across data-parallel
ranks, and with dynamic CP it also picks a context-parallel group for each
packed bin from the pool of `data_parallel_size x context_parallel_size` ranks.

Dynamic CP runs on the Megatron-Core `main` pin this repository ships (dynamic
CP v2); no submodule switch is needed. `ConfigContainer.validate` reports what
an older pin is missing. On `main` it supports full-attention and MoE layers.
Gated DeltaNet (and KDA) layers and multi-token prediction are not supported
under it yet, and validation rejects them with an explanation.

## When to Use It

Static context parallelism is sized for the longest sequence a run must fit.
On a variable-length stream most sequences are much shorter, and splitting them
across the whole group adds context-parallel communication and smaller kernels
without needing the memory. Dynamic CP pays off when both of these hold:

- Static CP is required by memory for the long tail. A large vocabulary can
  make the logits buffer the limit, for example.
- The stream has a meaningful share of sequences short enough to run on one GPU
  or a small group.

When nearly every sequence needs the full group, dynamic CP matches static CP up
to a small scheduling overhead.

## How It Works

`model.max_seqlen_per_dp_cp_rank` is the token capacity of one rank in one
microbatch. With `default_dynamic_cp`, a sequence of at most that length runs on
one GPU, a sequence of up to twice that length on a two-GPU group, and so on up
to the whole pool. Megatron-Core creates the runtime groups up front, one set
per valid group size: the power-of-two divisors of the pool, plus the whole
pool. The scheduler only selects among those sizes, and every rank runs the
same number of microbatches in a step.

The scheduler is packing-aware. It forms bins longest first, and it may place a
short sequence on a larger group than the sequence needs when that lowers the
workload of the busiest rank. Fewer sequences therefore run on one GPU than the
length histogram suggests.

Bridge wires this through the global-batch packing path:

- `model.dynamic_context_parallel=True` makes validation select the
  `default_dynamic_cp` scheduler instead of `dp_balanced`.
- The packed batch fetch receives the dynamic-CP flag and returns a
  `PackedSeqParams` that carries the microbatch's context-parallel group, which
  attention and rotary embeddings use in place of the static group.
- The per-sequence alignment multiple covers every group the pool can form:
  `2 x dp x cp`, and also `dp x cp x tp` under sequence parallelism.

## Configuration

```python
cfg.dataset.enable_global_batch_packing = True
cfg.dataset.dataloader_type = "single"
cfg.model.context_parallel_size = 4  # static CP domain inside each data-parallel replica
cfg.model.dynamic_context_parallel = True
cfg.model.min_dynamic_context_parallel_size = 1
cfg.model.max_seqlen_per_dp_cp_rank = cfg.model.seq_length // 4
cfg.model.calculate_per_token_loss = True
cfg.ddp.average_in_collective = False
cfg.train.micro_batch_size = 1
```

Validation applies every global-batch packing rule and adds these:

- The `dp x cp` pool is a power of two with at least two ranks.
- `min_dynamic_context_parallel_size` is a power of two no larger than the pool.
- `dist.use_decentralized_pg=False`, because Megatron-Core resolves dynamic
  groups from its global parallel state.
- The pool times `max_seqlen_per_dp_cp_rank` covers `seq_length`.
- The model has no Gated DeltaNet or KDA layers, and `model.mtp_num_layers` is
  unset or zero.

## Example on Mock Data

`examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py
--tiny-model` trains a small Qwen3.5-shaped attention and MoE model on a
generated variable-length SFT file, with no download and no checkpoint. Add
`--static-cp` for the static-CP baseline on the same packing path:

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 \
  examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py \
  --tiny-model model.context_parallel_size=4 train.global_batch_size=128 \
  train.train_iters=40 logger.log_interval=1
```

Measured on four GB200 GPUs in one node (NeMo 26.10 rc5 container, Megatron-Core
`main` pin). Both modes process identical global batches and reach the same
loss; the table compares step time after dropping the first 12 steps as warm-up.
The CP4 / 128 row comes from two runs in opposite order.

| Topology | Global batch | Static CP median step | Dynamic CP median step | Static / dynamic total time |
|---|---|---|---|---|
| DP2 x CP2 | 32 | 702 ms | 638 ms | 0.89 |
| CP4 | 32 | 771 ms | 645 ms | 1.10 |
| CP4 | 128 | 2.82 s | 2.10 to 2.15 s | 1.31 to 1.46 |

Dynamic CP wins where static CP is expensive. With CP2 replicas on one NVLink
domain, context-parallel communication is cheap and the tiny model is dominated
by per-step overhead, so dynamic CP is slightly slower. With one CP4 group
across all four GPUs and a larger global batch, short samples run on single GPUs
with no context-parallel exchange, and the gain grows. The example's README has
the full setup and the caveats on step-to-step variance.

## Reference Recipe

`qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config` fine-tunes
Qwen3.5-35B-A3B on CoderForge-Preview SWE-Rebench trajectories with TP1, PP1,
CP4, and EP8 on eight GB200 GPUs and a 32K-token cap. Its 3:1 Gated DeltaNet
layout and its multi-token-prediction layer are not supported under dynamic CP
on Megatron-Core `main`:

- Megatron-Core sizes Gated DeltaNet's context-parallel exchange from the
  static CP group, so a bin placed on a smaller runtime group fails in the
  forward pass.
- With microbatches on different context-parallel groups, the multi-token
  prediction token roll hangs.

`ConfigContainer.validate` therefore rejects the recipe with an explanation.
Use the mock-data example above until Megatron-Core supports runtime groups for
both. The static-CP path (`dp_balanced`) is unaffected.

## Models with Wide Attention Heads

Transformer Engine does not train with cuDNN fused attention at head dimension
256 on Blackwell, so Qwen3.5 attention runs on FlashAttention. FlashAttention
rejects THD bins that contain padding between sequences. Such models need two
changes:

- `fold_alignment_padding=True` on the dataset config, which reports each
  sequence's alignment padding as part of that sequence. The padding stays
  loss-masked and follows the sequence's last token, so causal attention and
  recurrent layers compute the same outputs for real tokens.
- A Megatron-Core that derives `pad_between_seqs` from the actual `cu_seqlens`,
  proposed in [Megatron-LM #7417](https://github.com/NVIDIA/Megatron-LM/pull/7417).

Leave `fold_alignment_padding` off for head dimensions of 128 or less, where
cuDNN fused attention handles the padding itself.

## Practical Caveats

- The first step that uses a new group size pays communicator and kernel setup,
  and each newly seen packed shape pays a one-time attention kernel setup cost,
  so expect slow early steps and occasional slow steps later. Compare total time
  over the same steps rather than single steps.
- The scheduler may place short sequences on larger groups to keep every rank
  busy, so fewer sequences run on one GPU than the length histogram suggests.
- A finite dataset with `dataloader_type="single"` ends the run at the end of
  its epoch. Use `cyclic` when `train_iters` times the global batch size can
  exceed the dataset size, as the mock-data example does.
- Transformer Engine's FusedAdam draws parameter handles from a pool of roughly
  20,000. Many experts on few expert-parallel ranks can exhaust it with the
  precision-aware optimizer. Use more expert-parallel ranks, as the recipe does
  with EP8, or offload part of the optimizer to the CPU.
- The fused cross-entropy materializes an fp32 buffer of tokens per rank times
  vocabulary size. At a 248K vocabulary this buffer sets
  `max_seqlen_per_dp_cp_rank` and makes context parallelism necessary at long
  caps.

## Related Docs

- [Packed Sequences](packed-sequences.md)
- [Hierarchical Context Parallel](hierarchical-context-parallel.md)
- [Activation Recomputation](activation-recomputation.md)
