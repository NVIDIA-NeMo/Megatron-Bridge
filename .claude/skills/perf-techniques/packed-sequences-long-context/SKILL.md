---
name: packed-sequences-long-context
description: Sequence packing and long-context training in Megatron Bridge. Use when the user asks about packed sequences, sequence packing, long context training, PackedSequenceSpecs, pack_sequences_in_batch, or CP with packing.
---

# Packed Sequences & Long-Context Training

For what packed sequences are, the three packing paths, and when to use them, see:

- `docs/training/packed-sequences.md`
- `card.yaml` (co-located)

## Enablement

### Offline packed SFT

```python
cfg.train.micro_batch_size = 1
cfg.dataset.dataset_kwargs.pad_to_max_length = True
cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192  # match seq_length
```

### VLM in-batch packing

```python
cfg.dataset.pack_sequences_in_batch = True
cfg.train.micro_batch_size = 4  # must be > 1
```

### CP + packing (finetuning)

```python
cfg.model.context_parallel_size = 4
cfg.model.calculate_per_token_loss = True
cfg.ddp.average_in_collective = False
cfg.dataset.packed_sequence_specs.pad_seq_to_mult = 2 * 4  # 2 * CP

# If sequence_parallel is also enabled, pad_seq_to_mult must include TP:
# cfg.dataset.packed_sequence_specs.pad_seq_to_mult = 2 * CP * TP
```

## Code Anchors

- Packed sequence dataset: `src/megatron/bridge/data/datasets/packed_sequence.py`
- SFT dataset: `src/megatron/bridge/data/datasets/sft.py`
- Packed seq utils: `src/megatron/bridge/training/utils/packed_seq_utils.py`
- GPT step (packing logic): `src/megatron/bridge/training/gpt_step.py`
- VLM step (packing logic): `src/megatron/bridge/training/vlm_step.py`
- Finetune utils: `src/megatron/bridge/recipes/utils/finetune_utils.py`
- Functional test: `tests/functional_tests/training/test_seqpacking_cp_example.py`

## Pitfalls

1. **MBS constraint**: Offline packed SFT requires `micro_batch_size == 1`.
   VLM in-batch packing requires `micro_batch_size > 1`. Mixing these up
   produces silent data corruption.

2. **CP divisibility**: `seq_length` must be divisible by `2 * context_parallel_size`.
   When sequence parallelism (SP) is also enabled, the divisor becomes
   `2 * CP * TP`. Violations cause assertion errors during initialization.

3. **Per-token loss with CP**: Finetuning with `CP > 1` requires
   `calculate_per_token_loss=True` and `average_in_collective=False`.
   Without these, loss scaling is wrong across CP ranks.

4. **MTP incompatibility**: Sequence packing for finetuning is documented as
   unsupported with multi-token prediction.

5. **Model-family opt-outs**: Several model families explicitly disable packing:
   Qwen3-Next SFT, GLM-4.5 SFT/PEFT, Qwen3.5-VL. Check model-specific
   recipes before assuming packing is available.

## Verification

For offline packed SFT, verify that `cu_seqlens` and `seq_offsets` are
present in the batch dict during the forward pass. For CP + packing, look for
the `pad_seq_to_mult` validation message during config setup.
