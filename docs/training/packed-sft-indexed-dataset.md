# Packed SFT Indexed Dataset Tutorial

Megatron Bridge stores offline-packed text SFT and PEFT data in Megatron
Core's binary indexed format by default. One logical prefix produces a pair of
files:

```text
/data/packed/training_4096.sft.bin
/data/packed/training_4096.sft.idx
```

Pass `/data/packed/training_4096.sft` to Bridge, without the final `.bin` or
`.idx`. This gives pretraining and prepared SFT the same file-container model,
while the packed SFT payload keeps the additional loss-mask and sequence-boundary
information required for supervised training.

This format applies to text-only offline packing. Direct Hugging Face and VLM
in-batch packing use a different runtime path.

## Configure Offline Packing

Use `GPTSFTDatasetConfig` with `PackedSequenceSpecs`. The builder materializes
the indexed pair on rank zero the first time it is needed, then reuses it on
later launches.

```python
from megatron.bridge.data.builders import (
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)
from megatron.bridge.data.packing import PackedSequenceSpecs

dataset = GPTSFTDatasetConfig(
    seq_length=4096,
    hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
    hf_validation_proportion=0.1,
    do_test=False,
    preprocessing=PromptCompletionSFTPreprocessingConfig(
        separator=" ",
        loss_mode="completion",
    ),
    enable_offline_packing=True,
    offline_packing_specs=PackedSequenceSpecs(
        packed_sequence_size=4096,
        pad_seq_to_mult=1,
    ),
)
```

Builder-managed outputs live under a fingerprinted directory below the
materialized dataset root:

```text
<dataset-root>/packed/<tokenizer-and-fingerprint>/training_4096.sft.bin
<dataset-root>/packed/<tokenizer-and-fingerprint>/training_4096.sft.idx
<dataset-root>/packed/<tokenizer-and-fingerprint>/validation_4096.sft.bin
<dataset-root>/packed/<tokenizer-and-fingerprint>/validation_4096.sft.idx
```

The fingerprint protects against accidentally reusing data prepared with
different tokenization, preprocessing, sequence length, or dataset options.
Set `hf_rewrite=True` only when the builder should regenerate its managed
JSONL and packed artifacts.

## Prepare Data Before a Training Job

For a recipe that already enables offline packing, prepare the data on a CPU
node before reserving training GPUs:

```bash
uv run python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --num-tokenizer-workers 8
```

To choose the artifact locations explicitly, provide a prefix rather than a
file suffix:

```bash
uv run python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --train-input-path /data/sft/training.jsonl \
    --val-input-path /data/sft/validation.jsonl \
    --packed-train-data-path /data/packed/training_4096.sft \
    --packed-val-data-path /data/packed/validation_4096.sft \
    --packed-metadata-path /data/packed/4096_metadata.json \
    --num-tokenizer-workers 8
```

Reference prebuilt pairs from a dataset config after both files exist:

```python
offline_packing_specs=PackedSequenceSpecs(
    packed_sequence_size=4096,
    packed_train_data_path="/data/packed/training_4096.sft",
    packed_val_data_path="/data/packed/validation_4096.sft",
    packed_metadata_path="/data/packed/4096_metadata.json",
)
```

A prefix, either pair filename, a glob, or a directory of canonical
`*.sft.bin/*.sft.idx` pairs is accepted. Directory and glob inputs are read in
sorted shard order. Every pair must be complete.

For `msc://` prefixes, MCore streams `.bin` ranges and caches each `.idx`
locally. Bridge enables MCore's Multi-Storage Client integration when it sees
the prefix; this only toggles MCore's feature flag. Before launch, install and
configure the storage provider and named profile according to the
[Multi-Storage Client configuration guide](https://nvidia.github.io/multi-storage-client/). Bridge does not create
profiles or credentials. It uses
`$NEMO_DATASETS_CACHE/packed_sft_index_cache` (or the corresponding default
NeMo cache) unless an explicit cache is configured. In multi-node jobs the
cache path must be visible to every rank:

```python
offline_packing_specs=PackedSequenceSpecs(
    packed_sequence_size=4096,
    packed_train_data_path="msc://profile/sft/training_4096.sft",
    packed_val_data_path="msc://profile/sft/validation_4096.sft",
    object_storage_cache_path="/shared/cache/packed-sft-indices",
    # MCore defaults to 256 MiB range-read chunks; tune only after measuring.
    object_storage_bin_chunk_nbytes=256 * 1024 * 1024,
)
```

The canonical builder downloads remote indices on rank zero before its
distributed barrier and automatically disables mmap for remote `.bin` files.
Remote artifacts are read-only: prepare the pair locally, then upload both
files before constructing the recipe. For example:

```python
import multistorageclient as msc

msc.upload_file("msc://profile/sft/training_4096.sft.bin", "/data/packed/training_4096.sft.bin")
msc.upload_file("msc://profile/sft/training_4096.sft.idx", "/data/packed/training_4096.sft.idx")
```

Do not upload `.idx` until `.bin` has completed. Direct packing or rewriting to
an object-storage prefix is rejected because a two-object replacement cannot
provide the local pair's reader/writer atomicity.

## Launch Training

The normal recipe launcher consumes the indexed pair through the configured
`GPTSFTDatasetBuilder`; no storage-specific training flag is required:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 \
    scripts/training/run_recipe.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --mode sft \
    checkpoint.pretrained_checkpoint=/checkpoints/llama32_1b
```

Offline packed SFT requires `micro_batch_size=1`. Keep `model.seq_length`,
`dataset.seq_length`, and `packed_sequence_size` equal. If context parallelism
is enabled, set `pad_seq_to_mult` to satisfy the training topology; the usual
minimum is `2 * context_parallel_size`, combined with sequence-parallel
alignment when applicable. Rebuild the packed pair whenever preprocessing,
tokenizer, sequence length, or topology-derived alignment changes.

## Validate Against Parquet

Parquet remains available as an explicit compatibility format. Prepare the
same input and configuration twice with the same seed: use a `.sft` output
prefix for indexed data and an output ending in `.parquet` for the comparison
copy. Then validate every row and measure sequential decode throughput:

```bash
uv run python scripts/training/compare_packed_sft_formats.py \
    --parquet /data/packed/training_4096.idx.parquet \
    --indexed /data/packed/training_4096.sft
```

Use `--max-rows N` for a quick sample. The command fails on any difference in
`input_ids`, the target-aligned `loss_mask`, or `seq_start_id`, and reports row
rate, token rate, elapsed time, and bytes on disk for each format. Read-rate
results are microbenchmarks; validate end-to-end dataloader and training
throughput on the target filesystem before making capacity decisions.

## On-Disk Schema

Each IndexedDataset item is one complete packed row. Its int32 payload contains
a versioned header, the sequence start offsets, and one word per token. The
lower 31 bits store the token ID and the high bit stores the binary loss mask.
The writer validates mask and token ranges. The reader validates the magic
value, version, row dimensions, and strictly increasing boundaries before
constructing a training sample.

Writers validate every row under the versioned schema, build a temporary pair,
take an exclusive filesystem lock, and publish `.idx` last as the completion
point. Local readers take the shared side of that lock while opening both files.
If either publication step raises, the writer restores the previous pair. The
internal `.sft.lock` sidecar intentionally remains for later readers and
writers.

This is a packed-SFT schema, not an ordinary pretraining token stream. Both use
the MCore `.bin/.idx` container; local data uses mmap and remote data uses range
reads. Their payloads are not interchangeable.

## Compatibility Notes

- Existing packed `.parquet` and deprecated `.npy` paths remain readable when
  explicitly configured.
- The default for newly prepared data is `.sft.bin/.sft.idx`.
- Multi-Token Prediction is not supported with offline packed SFT.
- VLM packing is unchanged and does not use this indexed schema.
