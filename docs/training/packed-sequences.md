# Packed Sequences

Packed sequences are a fine-tuning technique that reduces padding waste by
concatenating multiple examples into one pack while preserving sequence
boundaries for attention. In Megatron Bridge, this is primarily a supervised
fine-tuning and PEFT optimization rather than a general pretraining feature.

This page is the stable overview for what packed sequences are, when to use
them, and which constraints are durable. For operational setup, code anchors,
and verification commands, see [skills/nemo-mbridge-perf-sequence-packing/SKILL.md](../skills/nemo-mbridge-perf-sequence-packing/SKILL.md).

## What It Is

Fine-tuning datasets often contain examples with highly variable lengths. When
those examples are batched conventionally, many tokens in each batch are just
padding. Packed sequences reduce that waste by building longer packs from
multiple examples and carrying boundary metadata into the attention path.

In Bridge today, there are four distinct packing paths plus long-context
enablement through context parallelism:

| Path | Use case | Key config |
|---|---|---|
| Offline packed SFT | Text-only finetuning | `enable_offline_packing=True` plus `offline_packing_specs` |
| Runtime in-batch packing | GPT-SFT JSONL, Direct Hugging Face, and supported VLM finetuning | `enable_in_batch_packing=True` |
| Energon online packing | Qwen-VL data using the model-owned Energon collator | `packing_buffer_size=<candidate samples per worker>` |
| Global-batch online packing | Variable-length SFT / pretraining across DP x CP, dynamic context parallel | `enable_global_batch_packing=True` (+ `model.dynamic_context_parallel`) |
| Long-context (CP) | Pretrain / finetune at 16K-128K+ | `context_parallel_size > 1` |

These are related but they are not the same knob. Offline packed SFT and
runtime in-batch packing solve padding waste; long-context training
primarily addresses activation memory and communication tradeoffs at larger
sequence lengths.

The shared implementation lives under `megatron.bridge.data.packing`: offline
GPT SFT materialization, packed Parquet runtime datasets, bin-packing
algorithms, and collate-time THD packing each have separate modules. Energon
online packing uses the task encoder's native `select_samples_to_pack` and
`pack_selected_samples` API while reusing the same canonical THD batch builder. Ordinary
non-packed padding remains in `megatron.bridge.data.collators`. Use
`scripts/training/prepare_gpt_sft_packed_data.py` when packed GPT SFT artifacts
should be prepared before launching training.

For `GPTSFTDatasetConfig`, in-batch packing works with both local mmap JSONL
schemas: prompt/completion (`GPTSFTDataset`) and chat
(`GPTSFTChatDataset`). Tokenization remains lazy: workers mmap the JSONL and
read, parse, and tokenize only the rows selected for the current logical
microbatch. Collation then concatenates those rows into one physical THD batch
row; it does not materialize an offline dataset or load the full source into
RAM. Use a microbatch-yielding `single` or `cyclic` dataloader; GPT-SFT
in-batch packing does not support the global-batch `batch` dataloader.

Local GPT-SFT data can also use `per_split_data_source_manifest_path` with MLM-style
alternating ratios and JSONL paths. Offline packing resolves the weighted raw
row stream first and writes one packed Parquet cache for the blended split.
The input JSONL files remain memory-mapped; blending does not concatenate them
into another raw file or load them all into host memory. A one-path split keeps
the established single-file behavior.

## When to Use It

Packed sequences are a good fit when all of the following are true:

- you are doing SFT, PEFT, or supported VLM finetuning using one of the three
  packing paths above
- your examples have variable lengths and padding waste is significant
- you can tolerate the micro-batch constraints of packed training

Packed sequences are usually not the right answer when:

- you are doing standard Megatron-style pretraining, which already concatenates
  documents during sampling
- you want long-context training in general, where context parallelism is often
  the main technique
- your model family or recipe explicitly opts out of packed-sequence support

## Choosing the Offline Pack Length

For text-only LLM SFT and PEFT, use 8192 as the first offline-pack target when
the model context limit, memory, and recipe support allow it. Compare candidate
lengths at the same token slots per optimizer step:

```text
token_slots_per_step = packed_sequence_size * global_batch_size
```

Thus 2K/GBS32, 4K/GBS16, and 8K/GBS8 each retain 65,536 token slots per step.
A longer pack can contain more source examples in the physical MBS1 row and
reduce gradient accumulation and launch overhead. It also consumes more
activation memory and can encounter fixed-width kernel constraints, so measure
the candidates rather than treating 8K as unconditional.

Offline packing requires MBS1. The selected GBS must be divisible by and no
smaller than data parallel size; an 8K/GBS8 run therefore requires DP no
larger than 8. Keep model, dataset, and packed sequence lengths equal, write
changed packing configurations to a fresh output root, and verify the resolved
post-setup configuration. Changing pack length can alter truncation and pack
membership even when token slots stay constant, so rerun loss and stability
checks before replacing existing verification evidence.

Derive the internal sequence alignment from the resolved topology for both SFT
and PEFT:

```text
cp_multiple = 2 * CP if CP > 1 else 1
sp_multiple = CP * TP if sequence parallelism is enabled and TP > 1 else 1
pad_seq_to_mult = lcm(cp_multiple, sp_multiple)
```

For example, TP1/CP1 with SP disabled uses 1, while TP4/CP1 with SP enabled
uses 4. The difference comes from execution topology, not from whether the
trainable set is full SFT or PEFT. Pin the derived value explicitly and rebuild
the packed output after changing topology because the alignment changes pack
membership.

Keep this internal alignment separate from fixed final pack width.
`pad_to_max_length=true` is needed when a dispatcher or kernel requires a
static width, such as a HybridEP combine kernel with a fixed token chunk, or
when using CUDA graphs. CUDA graphs additionally require
`pad_cu_seqlens=true` and packing metadata. Ordinary eager offline packing
does not universally require fixed-width padding.

## Choosing Runtime In-Batch Packing

Use GPT-SFT in-batch packing when retaining the original JSONL is preferable
to generating packed Parquet artifacts. Enable it directly on the dataset
config and use a logical micro-batch larger than one:

```text
dataset.enable_in_batch_packing=true
dataset.dataloader_type=single
train.micro_batch_size=4
```

The collator preserves each sample's prompt/completion or chat loss mask and
emits current MCore packed metadata (`cu_seqlens_q`, `cu_seqlens_kv`, and the
corresponding padded boundaries when CP/SP alignment is required). The model
sees one physical THD row. `enable_in_batch_packing` and
`enable_offline_packing` are mutually exclusive. The `batch` dataloader is not
supported for GPT-SFT in-batch packing; use `single` or `cyclic`.

## Stable Constraints

The durable constraints for packed sequences in Bridge are:

- offline packed SFT requires configured `micro_batch_size == 1`
- GPT-SFT/Direct-HF/VLM in-batch packing requires configured `micro_batch_size > 1`;
  collation flattens those input rows into one physical THD batch row
- GPT-SFT in-batch packing requires `dataloader_type="single"` or `"cyclic"`;
  the global-batch `"batch"` dataloader is not supported
- Energon online packing currently supports the eager Qwen-VL collator path,
  requires physical `micro_batch_size == 1`, the generic `vlm_step`, per-token loss, and
  `ddp.average_in_collective=False`
- standard eager `alltoall` expert parallelism has functional coverage for
  Qwen3.6-35B-A3B at TP1/PP1/EP8 with EP communication overlap disabled; this
  is not a performance claim; other EP dispatchers are accepted with fixed-width
  native packs but do not yet have equivalent runtime evidence; THD boundaries
  produce a padding mask that excludes fixed-width gaps from MoE auxiliary-loss,
  z-loss, and expert-bias statistics
- current MCore may still dispatch those padded positions; expert-capacity/token-
  dropping configurations do not yet have native-packing runtime coverage
- `packing_buffer_size` counts candidate samples independently in every Energon
  worker; it is not a byte cache or a packed-sequence length
- Energon native packing and collator-owned `enable_in_batch_packing=True` are
  mutually exclusive
- Energon native packing does not currently support CUDA graphs, Qwen3-VL
  DistTrain, or pipeline parallelism; MTP is supported via MCore's packed
  sequence boundary-aware token rolling; requested MoE expert-parallel
  communication overlap is disabled with a warning so training uses the
  non-overlapped path
- when context parallelism is used, sequence length must satisfy the standard
  CP divisibility constraints
- GPT-SFT and Direct-HF sequence length must also satisfy the LCM of the training and
  evaluation CP constraints and `CP * TP` when sequence parallelism is enabled
- for fine-tuning with CP enabled, per-token loss behavior and reduction
  settings matter
- combined recipes using offline, in-batch, or Energon native packing
  automatically enable safe uneven-input padding for eager HybridEP; unpacked
  BSHD recipes preserve their configured setting
- the THD safety path pads only to the group-wide aligned maximum before
  dispatch and trims the padding after combine
- CUDA-graph-friendly packed metadata requires additional padding constraints

Model-family support is not universal. Some families and recipe paths explicitly
opt out of packed sequences or related packing modes.

HybridEP CUDA-graph configs preserve their explicit uneven-input setting because
the safety path performs a host scalar synchronization that is not capture-safe.
They must provide equal per-rank dispatch shapes. Disable CUDA graphs when packed
THD token counts can differ so the recipe can enable safe padding. Direct model-
provider callers must explicitly enable the setting when they supply uneven THD
inputs because no combined recipe is available to infer their layout.

## Global-Batch Online Packing and Dynamic Context Parallel

The fourth packing mode forms bins **once per training step, after a
data-parallel all-gather of sample lengths**, using Megatron-Core's
sequence-packing scheduler. Its candidate pool is the whole global batch across
DP x CP ranks, the only mode whose pool crosses rank boundaries, which is what
makes dynamic context parallelism (DCP) possible: every packed bin runs on a
context-parallel group sized for its longest sequence.

| Mode | Switch | Bins formed | Candidate pool |
|---|---|---|---|
| Offline | `enable_offline_packing` + `offline_packing_specs` | ahead of time, to disk | whole dataset |
| In-batch | `enable_in_batch_packing` | collate time | one microbatch, one rank |
| Energon | `EnergonDatasetConfig.packing_buffer_size` | dataloader buffer | a local buffer, one rank |
| Global-batch | `enable_global_batch_packing` (+ `model.dynamic_context_parallel`) | per step, after a DP all-gather | global batch, across DP x CP |

### How it works

Each step Megatron-Core pulls this step's samples on every data-parallel rank,
all-gathers their lengths, and forms bins of at most
`model.max_seqlen_per_dp_cp_rank x cp` tokens. Each bin becomes one THD
microbatch (concatenated tokens, `cu_seqlens` boundaries, per-sequence position
ids, varlen attention), so the number of microbatches changes from step to step
and Bridge feeds the scheduled count to the pipeline schedule. Two schedulers
exist: `dp_balanced` (static CP: every bin is split across the full
`context_parallel_size` group; next-fit bins in stream order) and
`default_dynamic_cp` (a sequence of at most `max_seqlen_per_dp_cp_rank` tokens
runs on one GPU, up to twice that on a two-GPU group, and so on up to the whole
`dp x cp` pool; longest-first, packing-aware group formation).

Bridge wires the scheduler in through the same points the other modes use:
`dataset.enable_global_batch_packing` drives `model.sequence_packing_scheduler`
(`default_dynamic_cp` when `model.dynamic_context_parallel` is set, else
`dp_balanced`), it joins `uses_thd` so MoE dispatch gets the same safe-padding
treatment, it shares the runtime-THD constraints below with Energon packing, the
CP alignment multiple comes from the same `collate_padding_multiple` derivation
(`2 x dp x cp` under dynamic CP), and it marks the model for variable-length
pipeline shapes. `train_step` and `evaluate` wrap the raw iterator once per step
with Megatron-Core's `wrap_data_iterator` and run the scheduled microbatch count;
`get_batch` delegates to Megatron-Core's packed batch fetch, which returns a
finished `PackedSeqParams` with the per-microbatch CP group. The scheduler's
exact per-step token statistics feed the FLOPs accumulators, and MoE/MTP loss
scaling in the logs divides by the microbatches that actually ran. Helpers:
`megatron.bridge.data.packing.global_batch` (sample contract) and
`megatron.bridge.training.global_batch_packing` (loop glue).

### Datasets that yield unpacked samples

The scheduler consumes **unpacked** per-sample dicts delivered through an
identity collate (one sample per `micro_batch_size = 1` microbatch):
`tokens`/`labels`/`position_ids` (`int64 [L]`), `loss_mask` (`float32 [L]`),
`original_seq_len` and `padded_seq_len` (`int32 [1]`). Every `padded_seq_len`
is a multiple of the CP alignment (`2 x cp`, or `2 x dp x cp` under dynamic CP,
times TP with sequence parallelism). Datasets declare the capability through
`yields_unpacked_samples`; two providers implement it:

- `GPTSFTDatasetConfig(enable_global_batch_packing=True)`: real SFT data (for
  example `default_coderforge_config`) with a per-sample collate that shifts
  labels, builds the loss mask, and pads to the alignment multiple; use a
  `single` or `cyclic` dataloader.
- `GPTDatasetConfig` / `MockGPTDatasetConfig` with `enable_global_batch_packing=True`:
  Megatron's `VarlenDataset` (JSONL / Parquet / Hugging Face sources via `blend`)
  or `MockVarlenDataset` (`varlen_mock_dataset_config_json` shapes a lognormal or
  per-line length distribution), which emit exactly this schema.

`fold_alignment_padding=True` on either config reports the alignment padding as
part of the sequence (still loss-masked) so bins carry no padding between
sequences; see the wide-head note below.

### Configuration

```python
from megatron.bridge.recipes.utils.dataset_utils import default_coderforge_config

cfg.dataset = default_coderforge_config(seq_length=32768)
cfg.dataset.enable_global_batch_packing = True
cfg.dataset.dataloader_type = "single"
cfg.model.context_parallel_size = 4                          # static CP domain
cfg.model.dynamic_context_parallel = True                    # False for dp_balanced
cfg.model.max_seqlen_per_dp_cp_rank = cfg.model.seq_length // 4
cfg.model.calculate_per_token_loss = True
cfg.ddp.average_in_collective = False
cfg.train.micro_batch_size = 1
```

`ConfigContainer.validate` enforces, in addition to the shared runtime-THD
constraints (micro batch 1, per-token loss with `average_in_collective=False`,
no CUDA graphs, no pipeline parallelism): a `single`/`cyclic` dataloader, an
explicit `model.max_seqlen_per_dp_cp_rank` with `pool x max_seqlen_per_dp_cp_rank
>= seq_length` (`pool` is `cp` for static CP and `dp x cp` for dynamic CP), a
power-of-two `dp x cp` pool and `min_dynamic_context_parallel_size` under
dynamic CP, no decentralized process groups with dynamic CP, no virtual pipeline
parallelism, no Mamba hybrid models, no Megatron FSDP, mutual exclusivity with
the other packing modes, and a dataset with `yields_unpacked_samples`. It also
probes the pinned Megatron-Core for the requested scheduler and the dynamic-CP
entry points and reports `./scripts/switch_mcore.sh dev` when they are missing;
the scheduler lives in Megatron-Core `dev`.

`qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config` is the reference
recipe (Qwen3.5-35B-A3B, CoderForge, 32k cap, TP1 PP1 CP4 EP8) and
`examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py` launches
it; run it with `model.dynamic_context_parallel=false
model.sequence_packing_scheduler=dp_balanced` for the static-CP baseline with the
same bins.

### Notes for wide-head MoE models (Qwen3.5 and similar)

- Transformer Engine disables cuDNN fused attention for training at head
  dimensions of 256 and above, and FlashAttention rejects THD bins that contain
  padding between sequences. Megatron-Core currently flags every scheduler bin as
  padded; until it derives `pad_between_seqs` from the actual `cu_seqlens`
  ([Megatron-LM #7417](https://github.com/NVIDIA/Megatron-LM/pull/7417)), such
  models need that change plus `fold_alignment_padding=True`. Head dimensions up
  to 128 run on cuDNN without either.
- Transformer Engine's FusedAdam wraps every parameter slice in a handle from a
  pool of roughly 20k; 256 experts on few expert-parallel ranks exceed it with
  the precision-aware optimizer. Use more expert-parallel ranks (EP8 in the
  recipe) or partial optimizer CPU offload (`optimizer.optimizer_cpu_offload`
  with `use_precision_aware_optimizer=True`); a fully offloaded optimizer can
  exceed a node's host memory, and `torchrun` pins `OMP_NUM_THREADS=1`, so set it
  to `cores / local ranks` when offloading.
- The fused cross-entropy materializes an fp32 `[tokens_per_rank, vocab]` buffer;
  at a 248k vocabulary this is what makes CP (or TP) mandatory at long caps and
  what sizes `max_seqlen_per_dp_cp_rank`.
- The first iteration that introduces a new dynamic CP group size pays
  communicator and kernel setup; expect a few slow early steps.
- Dynamic CP pays when static CP is memory-mandatory and the stream has a
  short-sequence mass; on streams where every sequence needs the full group it is
  a small scheduling overhead. The scheduler may promote short sequences to larger
  groups to keep every rank busy, so fewer sequences run at CP1 than the length
  histogram alone suggests.

## Relationship to Long-Sequence Training

Packed sequences and long-sequence training are often mentioned together because
both affect sequence layout and memory behavior, but they solve different
problems:

- packed sequences mainly reduce padding waste in fine-tuning datasets
- long-sequence training mainly addresses activation memory and communication
  tradeoffs at larger sequence lengths

For long-sequence training guidance, see:

- `docs/performance-guide.md`
- `docs/training/hierarchical-context-parallel.md`

## Practical Caveats

The most stable caveats to remember are:

1. Packed-sequence support is recipe- and model-family-specific.
2. Fine-tuning sequence packing should not be assumed to work with every other
   training feature.
3. Setting a distinct evaluation CP only reserves compatible data shapes;
   activating it requires decentralized process groups and caller-managed eval
   groups. The eval-CP example demonstrates topology plumbing, not a complete
   real-data recipe; validation sharding and batch math must use the eval DP.
4. Packed sequences improve efficiency primarily by reducing padding waste, not
   by replacing long-context parallelism or memory-planning techniques.
5. An Energon checkpoint restores the loader's buffered samples and selected
   pack groups, but exact resumption still requires the same dataset, processor,
   topology, sequence length, and packing-buffer configuration.
6. `progress.txt` `Tokens` and the `time/tokens` runtime metric currently report
   configured token capacity (`consumed_train_samples * model.seq_length`), not
   exact packed-token utilization. Finite partial Energon packs can therefore
   make those values larger than the physical token slots executed, and neither
   metric excludes alignment padding to represent useful source tokens.

## Related Docs

- [docs/training/multi-token-prediction.md](multi-token-prediction.md)
- [docs/performance-guide.md](../performance-guide.md)
- [docs/training/hierarchical-context-parallel.md](hierarchical-context-parallel.md)
- [tutorials/data/energon/README.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/energon/README.md)
- [skills/nemo-mbridge-perf-sequence-packing/SKILL.md](../skills/nemo-mbridge-perf-sequence-packing/SKILL.md)
- [skills/nemo-mbridge-perf-sequence-packing/card.yaml](../skills/nemo-mbridge-perf-sequence-packing/card.yaml)
