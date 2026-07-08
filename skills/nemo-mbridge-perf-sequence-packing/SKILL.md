---
name: nemo-mbridge-perf-sequence-packing
description: Validate and use packed sequences and long-context training in Megatron-Bridge, distinguishing offline packed SFT from Direct-HF/VLM in-batch packing and applying the right CP constraints. Use when enabling or debugging PackedSequenceSpecs, enable_in_batch_packing, long-context SFT, or CP with packing.
license: Apache-2.0
---

# Sequence Packing Skill

For stable background and recommendation level, see:

- @docs/training/packed-sequences.md
- @skills/nemo-mbridge-perf-sequence-packing/card.yaml

## Enablement

Offline packed SFT for LLM finetuning:

```python
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

cfg.train.micro_batch_size = 1
cfg.dataset.seq_length = 4096
cfg.model.seq_length = 4096
cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
cfg.dataset.enable_offline_packing = True
cfg.dataset.offline_packing_specs = PackedSequenceSpecs(
    packed_sequence_size=4096,
    pad_seq_to_mult=1,
)
```

If CP is enabled:

```python
import math

cfg.model.context_parallel_size = 2
cfg.model.calculate_per_token_loss = True
cfg.ddp.average_in_collective = False

# Offline packing is not finalized by ConfigContainer. Align offline samples
# explicitly to every train/eval CP and sequence-parallel constraint.
train_cp = cfg.model.context_parallel_size
eval_cp = cfg.dist.eval_context_parallel_size or train_cp
tp = cfg.model.tensor_model_parallel_size
sp = cfg.model.sequence_parallel
cp_sizes = {train_cp, eval_cp}
cfg.dataset.offline_packing_specs.pad_seq_to_mult = math.lcm(
    *(2 * cp if cp > 1 else 1 for cp in cp_sizes),
    *(cp * tp if sp and tp > 1 else 1 for cp in cp_sizes),
)
```

Setting `eval_context_parallel_size` only reserves a compatible data shape; it
does not activate a different evaluation topology by itself. A distinct eval CP
requires `use_decentralized_pg=True`, `use_gloo_process_groups=False`, separate
train/eval `ProcessGroupCollection` and P2P objects, and evaluation inside
`eval_cp_context`. Follow
`examples/training_features/decentralized_pg/pretrain_qwen3_eval_cp.py` for the
topology plumbing. It is not a drop-in real-data template: a real validation
dataset must be sharded with `eval_pgs.dp`, and evaluation batch/microbatch math
must use the evaluation DP degree rather than static training DP.

If CUDA graphs are enabled for this packed path:

```python
cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
cfg.dataset.dataset_kwargs["pad_to_max_length"] = True
```

**Note:** The builder resolves `PackedSequenceSpecs.packed_metadata_path` or
generates a default metadata file beside its packed output. Packed Parquet needs
that runtime JSON only when `pad_cu_seqlens=True`; otherwise its THD boundaries
come from the Parquet rows. Legacy packed `.npy` datasets continue to load the
builder-resolved metadata whenever offline packing is enabled.

In-batch packing for Direct Hugging Face and supported VLM finetuning:

```python
cfg.dataset.enable_in_batch_packing = True
cfg.dataset.defer_in_batch_packing_to_step = False
cfg.train.micro_batch_size = 2
```

Long-context baseline:

```python
cfg.model.seq_length = 16384
cfg.dataset.seq_length = 16384
cfg.model.context_parallel_size = 2
```

## Code Anchors

LLM packed SFT config surface:

```116:131:src/megatron/bridge/recipes/utils/finetune_utils.py
dataset_kwargs = {}
offline_packing_specs = None
if packed_sequence:
    dataset_kwargs["pad_to_max_length"] = True
    offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

return _text_hf_dataset_config(
    source=HFDatasetSourceConfig(dataset_name="squad"),
    preprocessing=PromptCompletionSFTPreprocessingConfig(separator=" "),
    seq_length=seq_length,
    enable_offline_packing=packed_sequence,
    offline_packing_specs=offline_packing_specs,
    dataset_kwargs=dataset_kwargs,
    val_proportion=0.1,
    num_workers=1,
)
```

Bridge validation:

```1109:1159:src/megatron/bridge/training/config.py
enable_in_batch_packing = getattr(self.dataset, "enable_in_batch_packing", False)
enable_offline_packing = getattr(self.dataset, "enable_offline_packing", False)
offline_packing_specs = getattr(self.dataset, "offline_packing_specs", None)

if enable_offline_packing and enable_in_batch_packing:
    raise ValueError("enable_offline_packing and enable_in_batch_packing are mutually exclusive.")
if enable_offline_packing and offline_packing_specs is None:
    raise ValueError("offline_packing_specs must be set when enable_offline_packing=True.")
...
cp_size = getattr(self.model, "context_parallel_size", 1)
eval_cp_size = self.dist.eval_context_parallel_size
cp_sizes = {cp_size, eval_cp_size} if eval_cp_size is not None else {cp_size}
tp_size = getattr(self.model, "tensor_model_parallel_size", 1)
has_sp = getattr(self.model, "sequence_parallel", False)
cp_multiples = [2 * size if size > 1 else 1 for size in cp_sizes]
sp_multiples = [size * tp_size if has_sp and tp_size > 1 else 1 for size in cp_sizes]
collate_padding_multiple = math.lcm(*cp_multiples, *sp_multiples)

if enable_in_batch_packing:
    self.model._enable_in_batch_packing = True
    if hasattr(self.dataset, "in_batch_packing_pad_to_multiple_of"):
        self.dataset.in_batch_packing_pad_to_multiple_of = collate_padding_multiple
elif isinstance(self.dataset, DirectHFSFTDatasetConfig):
    self.dataset.pad_to_multiple_of = math.lcm(
        self.dataset.pad_to_multiple_of,
        collate_padding_multiple,
    )
```

```1311:1353:src/megatron/bridge/training/config.py
if self.model.context_parallel_size > 1:
    assert self.model.seq_length % (self.model.context_parallel_size * 2) == 0, ...
    if isinstance(self.dataset, (GPTSFTDatasetConfig, DirectHFSFTDatasetConfig)):
        assert self.model.calculate_per_token_loss, ...
        assert not self.ddp.average_in_collective, ...
...
if enable_offline_packing and self.train.micro_batch_size > 1:
    raise ValueError(...)
...
if enable_in_batch_packing and self.train.micro_batch_size == 1:
    raise ValueError(...)
```

Direct-HF text in-batch runtime:

```102:136:src/megatron/bridge/data/collators/sft.py
if enable_in_batch_packing:
    sequence_rows = []
    ...
    batch = build_mcore_thd_sequence_batch_from_rows(
        sequence_rows,
        sequence_length=max_length,
        pad_token_id=...,
        ignore_index=ignore_index,
        pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
    )
    batch["tokens"] = batch["input_ids"]
    ...
    return batch
```

The builder forwards the config flags through `DirectSFTDataset`. Direct-HF text
uses immediate collate-time packing. Deferral is only valid for model-specific
providers and training steps that explicitly implement it; generic `gpt_step`
does not defer packing, and generic `vlm_step` rejects deferred packing.

Generic collate-time runtime used by model-specific and VLM collators:

```397:449:src/megatron/bridge/data/sequence_batching.py
def prepare_padded_or_packed_sequence_batch(
    batch,
    *,
    sequence_length,
    ...
    enable_in_batch_packing=False,
    in_batch_packing_pad_to_multiple_of=1,
    ...
):
    ...
    if enable_in_batch_packing:
        pack_right_padded_sequence_batch_to_mcore_thd(
            batch,
            sequence_length=sequence_length,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            ...
        )
        return
```

Packed THD runtime constraint:

```94:108:src/megatron/bridge/training/gpt_step.py
if batch.get("cu_seqlens_q") is not None:
    cu_seqlens = batch.get("cu_seqlens_q_padded")
    if cu_seqlens is None:
        cu_seqlens = batch["cu_seqlens_q"]
    if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
        raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
    return cu_seqlens.squeeze()

cu_seqlens = batch["cu_seqlens"]
if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
    raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
```

This THD “micro-batch size 1” is the physical batch dimension after Direct-HF
collation has flattened multiple input rows. Direct-HF in-batch packing still
requires configured `train.micro_batch_size > 1` before collation.

## Pitfalls

1. Offline packed SFT and Direct-HF/VLM in-batch packing are different features with opposite micro-batch rules.
2. Direct-HF `seq_length` must be divisible by the full train/eval CP and SP LCM shown above, not only `2 * context_parallel_size`. For train CP 2, eval CP 4, TP 2, and SP enabled, the required multiple is 8.
3. For finetuning with CP, `calculate_per_token_loss=True` and `ddp.average_in_collective=False` are required.
4. `pad_cu_seqlens=True` also requires `pad_to_max_length=True`.
5. Packing support is model-family-specific. `Qwen3-Next`, `GLM-4.5`, and `Qwen3.5-VL` contain explicit opt-outs in different paths.
6. MTP finetuning is documented as incompatible with packed sequences.
7. Synthetic padding rows, including negative indices remapped through `samples_mapping`, must retain an all-zero loss mask.

## Verification

Use the checked-in unit coverage:

```bash
uv run python -m pytest tests/unit_tests/training/utils/test_packed_seq_utils.py -v && \
uv run python -m pytest tests/unit_tests/training/test_config.py -k "packed_sequence or offline_packing or enable_in_batch_packing or direct_hf_ or offline_and_in_batch_packing_are_mutually_exclusive or context_parallel_seq_length_divisibility or context_parallel_finetuning_validations" -v && \
uv run python -m pytest tests/unit_tests/data/vlm_datasets/test_batching.py -v && \
uv run python -m pytest tests/unit_tests/data/collators/test_sft.py -k "packs_sequences_for_gpt_step or pads_packed_sequences_to_multiple" -v && \
uv run python -m pytest tests/unit_tests/data/builders/test_direct_hf_sft.py -k "forwards_runtime_packing_to_collate" -v && \
uv run python -m pytest tests/unit_tests/data/builders/test_gpt_sft_config.py -k "pack_metadata or metadata_forwarding" -v && \
uv run python -m pytest tests/unit_tests/data/datasets/test_direct_sft.py -k "forwards_supported_packing_options" -v && \
uv run python -m pytest tests/unit_tests/training/test_gpt_step.py -k "packed_cp_partition or partition_current_packed_batch or partition_packed_batch" -v && \
uv run python -m pytest tests/unit_tests/training/test_vlm_step.py -k "deferred_in_batch_packing or packed_metadata" -v && \
uv run python -m pytest tests/unit_tests/data/datasets/test_packed_parquet.py -k "negative_index_zeroes_loss_mask" -v && \
uv run python -m pytest tests/unit_tests/data/datasets/test_gpt_sft.py -k "mapped_padding_rows_do_not_contribute_to_loss" -v
```

Success criteria:

- all selected tests pass
- offline and in-batch configuration validation remains mutually exclusive
- packed metadata reaches the training step in MCore THD form
- mapped padding rows do not contribute to loss
