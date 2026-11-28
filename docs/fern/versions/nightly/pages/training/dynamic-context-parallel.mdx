# Dynamic Context Parallel

Dynamic context parallelism (dynamic CP) runs every packed microbatch on a
context-parallel group sized for its longest sequence, instead of splitting
every microbatch across the full `context_parallel_size` group. It builds on
[global-batch online packing](packed-sequences.md#global-batch-online-packing):
the packing scheduler already sees the whole global batch across data-parallel
ranks, and with dynamic CP it also picks a context-parallel group for each
packed bin from the pool of `data_parallel_size x context_parallel_size` ranks.

Dynamic CP needs a Megatron-Core with dynamic context parallelism. Models with
Gated DeltaNet layers, such as Qwen3.5, also need the runtime CP group
propagated through those layers
([Megatron-LM #7048](https://github.com/NVIDIA/Megatron-LM/pull/7048)).
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
- The pool times `max_seqlen_per_dp_cp_rank` covers `seq_length`, and the
  offline packed row length when the dataset is offline packed.
- For models with Gated DeltaNet layers, the pool size divides the Gated
  DeltaNet key and value head counts per tensor-parallel rank. Megatron-Core
  splits those heads across a bin's group, and the largest group is the whole
  pool: Qwen3.5-35B-A3B has 16 key heads, so its pool is at most 16 ranks.

Evaluation uses the same scheduler and does not cycle the validation split:
a run needs `(train_iters // eval_interval + 1) x eval_iters x
eval_global_batch_size` validation samples (rows, with offline packing), and
data loading fails up front when the split holds fewer.

### Fixed-Width Bins

A packed bin holds whatever its sequences add up to, so nearly every
microbatch gives a rank a token count it has not run before. Kernels on the
training path are specialized per sequence length: Megatron-Core's fused Gated
DeltaNet kernels are autotuned for each new length, and CuTe DSL kernels are
compiled for it. Each new length then costs seconds on one rank while the ranks
it communicates with wait, and with many ranks most steps contain one.

```python
cfg.model.pad_packed_seq_alignment = "max"
```

pads every bin to its group's full capacity with a masked dummy sequence, so a
rank always runs exactly `max_seqlen_per_dp_cp_rank` tokens. The padding is
computed but not trained on; what remains is a one-time setup per CP group
size. Use it when the model's kernels are length-specialized, as for the
Qwen3.5 recipe below; an integer value pads to a multiple instead.

### Offline Packed Rows

Dynamic CP also runs on offline packed data. Set `enable_offline_packing` with
`offline_packing_specs` next to `enable_global_batch_packing`: each packed row
then yields one sample per sequence it holds, and the scheduler re-bins those
sequences and picks a CP group for every bin, exactly as for unpacked samples.
`train.global_batch_size` counts packed rows, so offline packing only decides
which sequences share a step. Each sequence is re-padded to the dynamic-CP
alignment, `2 x dp x cp`, whatever `pad_seq_to_mult` the rows were prepared
with. See [Offline Packed Rows](packed-sequences.md#offline-packed-rows).

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

Eight GB200 GPUs (two four-GPU nodes in one NVL72 rack), NeMo 26.10 container,
Megatron-Core with [Megatron-LM #7048](https://github.com/NVIDIA/Megatron-LM/pull/7048),
random initialization, the first 4,000 SWE-Rebench trajectories
(`dataset.hf_dataset.split='SWE_Rebench[:4000]'`), and 12 steps of which the
first four are dropped. Throughput is Bridge's model TFLOP/s per GPU; memory is
the peak allocated on rank 0 over the first two steps.

| Configuration | Median step | Median TFLOP/s per GPU | Peak allocated per GPU |
|---|---|---|---|
| Recipe: dynamic CP, full recompute, 16 loader workers | 79 s | 100 | 100 GB |
| Static CP4 (`dp_balanced`), otherwise identical | 75 s | 106 | 99 GB |
| Recipe with 2 loader workers | 261 s | 30 | 99 GB |
| Recipe with selective recompute of `moe_act` and `layernorm` | 58 s | 132 | 150 GB |

- About nine in ten of these trajectories are longer than the 32K cap, so
  nearly every packed sequence needs the full CP4 group and dynamic CP has no
  short sequences to place on smaller groups. Its median step is 5% slower
  than static CP4 here, within the step-to-step noise of 12-step runs on
  different node pairs. Dynamic CP pays off on streams with a meaningful share
  of short sequences; see the mock-data measurements of the 16-GPU recipe
  below.
- GPT-SFT tokenizes each trajectory lazily, which takes seconds per sample
  here, and the packing scheduler pulls a whole step's samples at once. With
  two loader workers the GPUs wait on data; the recipe uses 16.
- Selective recompute runs about 26% faster per step and completed all 12
  steps, but it peaks at 150 GB allocated per GPU. On an earlier container the
  same setting failed with NCCL out of memory when dynamic CP created a
  communicator for a new group size, so the recipe keeps full recompute's
  headroom.

## Recipe at a 128K Context on 16 GPUs

`qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_dynamic_cp_config`
fine-tunes the full Qwen3.5-35B-A3B plus one MTP layer on CoderForge-Preview
SWE-Rebench trajectories at a 131,072-token context on 16 GPUs. It is the
static recipe `qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config`
(TP1, base CP8, EP16, offline packed rows, 32 rows per step, MXFP8, selective
recompute, NCCL EP) with global-batch packing and dynamic CP switched on, and
three settings changed for it:

- `model.pad_packed_seq_alignment="max"`, for fixed-width bins as above.
- `validation.eval_iters=2` instead of 8, so that the evaluations of a run fit
  the 5% validation split without cycling.
- `model.gdn_pre_gated_delta_rule_fusion=false`. A bin can run on the whole
  16-rank pool, where each rank holds one Gated DeltaNet key head and two value
  heads; the fused pre-delta-rule kernels need the channels a rank holds to be
  a multiple of 8 and fail there. The unfused path is a few percent slower.

```bash
srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 bash -c '
  CUDA_VISIBLE_DEVICES=$SLURM_LOCALID uv run python scripts/training/run_recipe.py \
  --recipe qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_dynamic_cp_config --mode sft \
  checkpoint.pretrained_checkpoint=/path/to/megatron/qwen3.5-35b-a3b-text'
```

Add `model.dynamic_context_parallel=false` for static CP8 on the same packing
path. Prepare the data before training as `examples/agentic_sft` describes;
otherwise rank 0 tokenizes the whole split while the other ranks wait.

Measured on four GB300 nodes of one NVL72 rack with the NeMo 26.10.01
container, one GPU visible per rank. CoderForge: the pretrained text
checkpoint, the first 8,000 SWE-Rebench trajectories, 100 steps of 32 rows.
Mock data: the same recipe on `MockGPTSFTDatasetConfig` samples of mixed
lengths (40% of them fit one GPU), from random weights. Throughput is runtime
tokens per second over the steps from 11 on:

| Data | Static CP8 | Dynamic CP | Dynamic vs static |
|---|---:|---:|---:|
| CoderForge rows, 32 per step | 152,732 | 145,985 | −4.4% |
| Mock samples, 64 per step, packed online | 119,375 | 147,069 | +23.2% |
| Mock samples packed offline into rows, 32 per step | 164,872 | 140,111 | −15.0% |

- The unchanged static recipe reaches 146,651 tokens/s on the same CoderForge
  rows, so static CP8 through global-batch packing costs nothing, and dynamic
  CP is level with the static recipe (−0.5%, 95% interval −4.8% to +3.1%).
- Both modes train the same model. On identical CoderForge rows from the same
  checkpoint, the loss is 0.19822 with static CP8 and 0.19820 with dynamic CP
  at step 1 and 0.13327 and 0.13341 at step 100; the validation loss is 0.13190
  and 0.13189 at step 100.
- With fixed-width bins the cost of a step follows its number of microbatches.
  On CoderForge, where almost no trajectory fits one GPU, the scheduler forms
  17.7 microbatches per step with dynamic CP against 16 with static CP8. On the
  mock samples packed online it forms 8.1 against 9.5, and dynamic CP is 23.2%
  faster (95% interval +17.1% to +29.1%). Rows packed offline are full, so
  static CP8 runs each as one bin and dynamic CP has nothing to gain.
- The first microbatch on each group size takes about a minute. Peak device
  memory is 153 GiB per GPU with dynamic CP against 140 GiB with static CP8.
- The recipe carries the GB200 name of the static recipe it extends. All of
  these measurements are from GB300 GPUs; it has not been run on GB200
  hardware.
- Without fixed-width bins, the same mock run takes 40–66 s per step at 32
  samples per step, against 4–13 s with them.

`examples/agentic_sft` has the full tables, the commands, and the launcher that
runs the recipe on mock data.

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
- Variable-length bins also give each rank token counts it has not seen
  before, and length-specialized kernels are tuned or compiled on first use.
  Pad the bins to a fixed width when the model has such kernels; see
  [Fixed-Width Bins](#fixed-width-bins).
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
