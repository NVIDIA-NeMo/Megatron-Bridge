# Dynamic Context Parallel

Dynamic context parallelism (dynamic CP) runs every packed microbatch on a
context-parallel group sized for its longest sequence, instead of splitting
every microbatch across the full `context_parallel_size` group. It builds on
[global-batch online packing](packed-sequences.md#global-batch-online-packing):
the packing scheduler already sees the whole global batch across data-parallel
ranks, and with dynamic CP it also picks a context-parallel group for each
packed bin from the pool of `data_parallel_size x context_parallel_size` ranks.

Dynamic CP needs the Megatron-Core `dev` submodule pin:

```bash
./scripts/switch_mcore.sh dev && uv sync
```

`ConfigContainer.validate` reports what an older pin is missing.

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
to the whole pool. The scheduler forms bins longest first and gives each bin a
power-of-two group, and every rank runs the same number of microbatches in a
step.

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

## Reference Recipe

`qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config` fine-tunes
Qwen3.5-35B-A3B on CoderForge-Preview SWE-Rebench trajectories with TP1, PP1,
CP4, and EP8 on eight GB200 GPUs and a 32K-token cap. Launch it with the
recipe runner and a converted pretrained checkpoint:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
  --recipe qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config --mode sft \
  checkpoint.pretrained_checkpoint=/path/to/megatron/qwen3.5-35b-a3b \
  train.train_iters=20 logger.log_interval=1
```

Add these overrides for the static-CP baseline on the same packing path:

```bash
model.dynamic_context_parallel=false model.sequence_packing_scheduler=dp_balanced
```

### Measured on the recipe's data

Eight GB200 GPUs in one NVL72 rack, NeMo 26.06.01 container, Megatron-Core dev
pin with [Megatron-LM #7417](https://github.com/NVIDIA/Megatron-LM/pull/7417),
random initialization, the first 4,000 SWE-Rebench trajectories, and 12 steps
of which the first four are dropped. Throughput is Bridge's model TFLOP/s per
GPU.

| Configuration | Median step | Median TFLOP/s per GPU | Peak allocated per GPU |
|---|---|---|---|
| Recipe: dynamic CP, full recompute, 16 loader workers | 88 s | 90 | 108 GB |
| Static CP4 (`dp_balanced`), otherwise identical | 83 s | 93 | 108 GB |
| Recipe with 2 loader workers | 254 s | 31 | 108 GB |
| Recipe with selective recompute of `moe_act` and `layernorm` | 75 s over four steps, then NCCL out of memory | 106 | 163 GB |

- About nine in ten of these trajectories are longer than the 32K cap, so
  nearly every packed sequence needs the full CP4 group and dynamic CP has no
  short sequences to place on smaller groups. Its median step is 5% slower
  than static CP4 here, within this run's step-to-step noise. Dynamic CP pays
  off on streams with a meaningful share of short sequences.
- GPT-SFT tokenizes each trajectory lazily, which takes about 9 seconds per
  sample here, and the packing scheduler pulls a whole step's samples at once.
  With two loader workers the GPUs wait on data; the recipe uses 16.
- Selective recompute runs about 13% faster per step, but it reserves over
  170 GB per GPU, and the next communicator dynamic CP creates for a new group
  size fails to allocate. Full recompute leaves that headroom.

### Example on mixed-length data

[`examples/agentic_sft`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/agentic_sft)
trains the full 40-layer Qwen3.5-35B-A3B on one four-GPU GB200 node with static
or dynamic CP, on generated data whose sample lengths follow a mixed
distribution (half of the samples fit one rank's 16K tokens, 30% need CP2, 20%
need CP4). On identical data, 200 steps, dynamic CP was 9.6% faster than static
CP4 in throughput (95% bootstrap interval 6.0% to 13.2%). Its README describes
the setup and the comparison tool.

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
  so expect a few slow early steps.
- The scheduler may place short sequences on larger groups to keep every rank
  busy, so fewer sequences run on one GPU than the length histogram suggests.
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
