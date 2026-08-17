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

In Bridge today, there are three distinct packing paths plus long-context
enablement through context parallelism:

| Path | Use case | Key config |
|---|---|---|
| Offline packed SFT | Text-only finetuning | `enable_offline_packing=True` plus `offline_packing_specs` |
| Runtime in-batch packing | GPT-SFT JSONL, Direct Hugging Face, and supported VLM finetuning | `enable_in_batch_packing=True` |
| Energon online packing | Qwen-VL data using the model-owned Energon collator | `packing_buffer_size=<candidate samples per worker>` |
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
- Energon native packing does not currently support MTP, CUDA graphs, Qwen3-VL
  DistTrain, or pipeline parallelism; requested MoE expert-parallel communication
  overlap is disabled with a warning so training uses the non-overlapped path
- when context parallelism is used, sequence length must satisfy the standard
  CP divisibility constraints
- GPT-SFT and Direct-HF sequence length must also satisfy the LCM of the training and
  evaluation CP constraints and `CP * TP` when sequence parallelism is enabled
- for fine-tuning with CP enabled, per-token loss behavior and reduction
  settings matter
- Megatron Bridge automatically enables safe uneven-input padding for eager
  HybridEP configs; this pads only to the group-wide aligned maximum before
  dispatch and trims the padding after combine
- CUDA-graph-friendly packed metadata requires additional padding constraints

Model-family support is not universal. Some families and recipe paths explicitly
opt out of packed sequences or related packing modes.

HybridEP CUDA-graph configs preserve their explicit uneven-input setting because
the safety path performs a host scalar synchronization that is not capture-safe.
They must provide equal per-rank dispatch shapes. Disable CUDA graphs when packed
runtime token counts can differ so Bridge can enable safe padding.

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
