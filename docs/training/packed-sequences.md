# Packed Sequences

Packed sequences are a fine-tuning technique that reduces padding waste by
concatenating multiple examples into one pack while preserving sequence
boundaries for attention. In Megatron Bridge, this is primarily a supervised
fine-tuning and PEFT optimization rather than a general pretraining feature.

This page is the stable overview for what packed sequences are, when to use
them, and which constraints are durable. For operational setup, code anchors,
and verification commands, see `skills/perf-techniques/sequence-packing.md`.

## What It Is

Fine-tuning datasets often contain examples with highly variable lengths. When
those examples are batched conventionally, many tokens in each batch are just
padding. Packed sequences reduce that waste by building longer packs from
multiple examples and carrying boundary metadata into the attention path.

In Bridge today, it is important to distinguish two related but different
things:

- offline packed sequence training for text-only fine-tuning
- long-sequence training, which is primarily handled through context
  parallelism and other memory-management techniques

Those features are related, but they are not the same knob.

## When to Use It

Packed sequences are a good fit when all of the following are true:

- you are doing text-only SFT or PEFT
- your examples have variable lengths and padding waste is significant
- you can tolerate the micro-batch constraints of packed training

Packed sequences are usually not the right answer when:

- you are doing standard Megatron-style pretraining, which already concatenates
  documents during sampling
- you want long-context training in general, where context parallelism is often
  the main technique
- your model family or recipe explicitly opts out of packed-sequence support

## Stable Constraints

The durable constraints for packed sequences in Bridge are:

- packed SFT requires `micro_batch_size == 1`
- when context parallelism is used, sequence length must satisfy the standard
  CP divisibility constraints
- for fine-tuning with CP enabled, per-token loss behavior and reduction
  settings matter
- CUDA-graph-friendly packed metadata requires additional padding constraints

Model-family support is not universal. Some families and recipe paths explicitly
opt out of packed sequences or related packing modes.

## Relationship to Long-Sequence Training

Packed sequences and long-sequence training are often mentioned together because
both affect sequence layout and memory behavior, but they solve different
problems:

- packed sequences mainly reduce padding waste in fine-tuning datasets
- long-sequence training mainly addresses activation memory and communication
  tradeoffs at larger sequence lengths

For long-sequence training guidance, see:

- `docs/performance-guide.md`
- `docs/training/hybrid-context-parallel.md`

## Practical Caveats

The most stable caveats to remember are:

1. Packed-sequence support is recipe- and model-family-specific.
2. Fine-tuning sequence packing should not be assumed to work with every other
   training feature.
3. Packed sequences improve efficiency primarily by reducing padding waste, not
   by replacing long-context parallelism or memory-planning techniques.

## Related Docs

- `docs/training/multi-token-prediction.md`
- `docs/performance-guide.md`
- `docs/training/hybrid-context-parallel.md`
- `skills/perf-techniques/sequence-packing.md`
- `skills/perf-techniques/sequence-packing/card.yaml`
