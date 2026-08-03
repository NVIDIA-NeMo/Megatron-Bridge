# Packed SFT with MCore Indexed Datasets

This tutorial covers the complete text-only offline-packed SFT and PEFT
workflow: why the indexed format exists, how to prepare raw data, how to attach
the resulting artifacts to a recipe, how to start training, and how to migrate
from packed Parquet.

Newly prepared packed SFT data uses Megatron Core's binary indexed container by
default. One logical prefix produces two files:

```text
/data/packed/training_4096.sft.bin
/data/packed/training_4096.sft.idx
```

Pass `/data/packed/training_4096.sft` to Bridge, without the final `.bin` or
`.idx`. Packed Parquet remains readable when it is selected explicitly.

## Background

Fine-tuning examples are often much shorter than the model context length.
Padding every example to that length wastes computation. Offline packing
tokenizes the examples ahead of training and combines several examples into one
row up to the selected packing capacity while preserving:

- the token IDs
- the target-aligned loss mask, so prompt and padding tokens can be excluded
- the start offset of every source sequence, so attention does not cross
  example boundaries

Rows are padded to a fixed width only when the dataset or runtime configuration
requires it, for example for a separately supported static-shape kernel path.

The resulting data flow is:

```text
local JSONL or Hugging Face source
  -> schema normalization
  -> tokenization and loss-mask construction
  -> offline bin packing
  -> .sft.bin + .sft.idx + packing metadata
  -> GPTSFTDatasetBuilder
  -> packed THD training batch
```

Previously, prepared SFT rows were normally stored in Parquet while pretraining
used MCore `.bin/.idx`. The indexed SFT format removes that storage-path split:
both workflows now use MCore's indexed container, mmap-backed local reads, and
the same object-storage reader infrastructure.

The payloads are intentionally different and are not interchangeable:

| Property | Pretraining indexed data | Packed SFT indexed data |
| --- | --- | --- |
| Typical prefix | `corpus_text_document` | `training_4096.sft` |
| One item contains | A tokenized document or sentence | One complete packed SFT row |
| Supervision metadata | Derived by the pretraining sampler | Encoded loss mask and sequence boundaries |
| Bridge config | `GPTDatasetConfig` | `GPTSFTDatasetConfig` plus `PackedSequenceSpecs` |

This tutorial applies to text-only offline packing. Direct Hugging Face and VLM
in-batch packing are separate runtime paths and do not use this schema.

## Choose a Workflow

| Starting point | Recommended path |
| --- | --- |
| Trying the feature | Run the local JSONL end-to-end example below |
| A recipe's built-in Hugging Face dataset | Run the preparation script with only `--recipe`, then launch the same recipe |
| Custom local JSONL | Set `dataset_root`, preprocessing, and offline packing in `GPTSFTDatasetConfig` |
| Prebuilt local shards | Set explicit packed prefixes in `PackedSequenceSpecs` |
| Prebuilt object-storage shards | Upload complete local pairs and use `msc://` prefixes |
| Existing packed Parquet | Keep it explicit during migration, regenerate indexed data from the same normalized input, and run the parity tool |

## Prerequisites

Run commands from the Megatron Bridge repository root in an environment created
with `uv sync` or in the project container. The example uses the gated
`meta-llama/Llama-3.2-1B` tokenizer and model configuration, so authenticate
with Hugging Face before preparation if they are not already cached.

SFT and PEFT also require base weights. `checkpoint.pretrained_checkpoint` may
point to a native Megatron checkpoint or a local Hugging Face full-model
directory. A remote Hugging Face model ID is not a checkpoint path. For a
repeatable production setup, import it first:

```bash
./scripts/conversion/convert.sh import \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path /data/checkpoints/llama32_1b
```

Use `checkpoint.pretrained_checkpoint` to initialize a new fine-tuning run. Use
`checkpoint.load` instead when resuming optimizer, scheduler, RNG, and dataloader
state from a complete native training checkpoint. The shipped SFT recipes have
a default `checkpoint.load` directory, so explicitly set it to `null`/`None`
when the run must start from `pretrained_checkpoint` rather than resume an old
run found in the working directory.

## End-to-End Local Quickstart

The tiny dataset in this section is only a plumbing smoke test. It is too small
for meaningful model quality evaluation.

### 1. Create and inspect raw JSONL

Generate prompt-completion train, validation, and test files:

```bash
uv run python tutorials/data/text-only-sft/prepare_example_data.py \
    --output-dir /tmp/bridge-text-only-sft
```

The packing path expects conventional split names:

```text
/tmp/bridge-text-only-sft/
  training.jsonl
  validation.jsonl
  test.jsonl
```

Each line is one JSON object. The example uses `input` and `output`:

```json
{"input": "What is SFT?", "output": "Supervised fine-tuning."}
```

Validate the files before tokenization:

```bash
uv run python - <<'PY'
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
for path in sorted(Path("/tmp/bridge-text-only-sft").glob("*.jsonl")):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    logger.info("%s: %d rows", path.name, len(rows))
PY
```

For a production prompt-completion dataset, set the actual column names with
`PromptCompletionSFTPreprocessingConfig`. For conversation data, store a
`messages` list and select `ChatSFTPreprocessingConfig`; do not flatten a
multi-turn conversation implicitly.

### 2. Prepare `.sft.bin/.sft.idx`

Prepare on a CPU node before reserving training GPUs. The named recipe supplies
the tokenizer, prompt-completion schema, sequence length, padding settings, and
random seed:

```bash
mkdir -p /tmp/bridge-packed-sft

uv run python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --train-input-path /tmp/bridge-text-only-sft/training.jsonl \
    --val-input-path /tmp/bridge-text-only-sft/validation.jsonl \
    --packed-train-data-path /tmp/bridge-packed-sft/training_4096.sft \
    --packed-val-data-path /tmp/bridge-packed-sft/validation_4096.sft \
    --packed-metadata-path /tmp/bridge-packed-sft/4096_metadata.json \
    --num-tokenizer-workers 1
```

Increase `--num-tokenizer-workers` after the single-worker path succeeds. The
explicit inputs must match the preprocessing schema of the selected recipe. If
the columns, chat template, or loss policy differ, use the custom config path
in [Integrate with a Python Recipe](#integrate-with-a-python-recipe) so
preparation and training share one config definition.

The command creates:

```text
/tmp/bridge-packed-sft/training_4096.sft.bin
/tmp/bridge-packed-sft/training_4096.sft.idx
/tmp/bridge-packed-sft/validation_4096.sft.bin
/tmp/bridge-packed-sft/validation_4096.sft.idx
/tmp/bridge-packed-sft/4096_metadata.json
```

`.idx` is the metadata and offset table for MCore's indexed reader; it is not a
standalone dataset. The additional JSON file records packing statistics and is
required when `pad_cu_seqlens=True`.

### 3. Inspect the prepared rows

Check that both members of every pair exist and decode the first row:

```bash
ls -lh /tmp/bridge-packed-sft

uv run python - <<'PY'
import logging

from megatron.bridge.data.packing.indexed import PackedSFTIndexedDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
dataset = PackedSFTIndexedDataset("/tmp/bridge-packed-sft/training_4096.sft")
sample = dataset[0]
logger.info(
    "rows=%d first_row_tokens=%d first_row_sequences=%d",
    len(dataset),
    len(sample["input_ids"]),
    len(sample["seq_start_id"]),
)
PY
```

This reader validates the SFT magic value, schema version, row dimensions, and
increasing sequence boundaries. Token range and binary loss-mask validation
happen when the row is encoded by the writer.

### 4. Attach the exact pair to the recipe

The recipe already enables offline packing. The following overrides replace its
built-in SQuAD source with the local JSONL source and point it at the pair just
prepared:

```text
dataset.hf_dataset=null
dataset.hf_validation_proportion=null
dataset.dataset_root=/tmp/bridge-text-only-sft
dataset.offline_packing_specs.packed_train_data_path=/tmp/bridge-packed-sft/training_4096.sft
dataset.offline_packing_specs.packed_val_data_path=/tmp/bridge-packed-sft/validation_4096.sft
dataset.offline_packing_specs.packed_metadata_path=/tmp/bridge-packed-sft/4096_metadata.json
dataset.do_test=false
```

Keeping these overrides together is important: the source schema used to
prepare data and the packed prefixes consumed at training time must describe
the same dataset. The public `local-jsonl` preset starts with unpacked local
data; selecting it replaces the recipe's packed dataset object. The quickstart
therefore keeps the recipe dataset and changes its source and artifact paths.

### 5. Start a one-step training smoke

Launch with one GPU and a real local base checkpoint:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 \
    scripts/training/run_recipe.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --mode sft \
    --pretrained_checkpoint /data/checkpoints/llama32_1b \
    --save_dir /tmp/bridge-packed-sft-checkpoints \
    --max_steps 1 \
    --global_batch_size 1 \
    --micro_batch_size 1 \
    dataset.hf_dataset=null \
    dataset.hf_validation_proportion=null \
    dataset.dataset_root=/tmp/bridge-text-only-sft \
    dataset.offline_packing_specs.packed_train_data_path=/tmp/bridge-packed-sft/training_4096.sft \
    dataset.offline_packing_specs.packed_val_data_path=/tmp/bridge-packed-sft/validation_4096.sft \
    dataset.offline_packing_specs.packed_metadata_path=/tmp/bridge-packed-sft/4096_metadata.json \
    dataset.do_test=false \
    checkpoint.load=null
```

At startup, inspect the printed final configuration and confirm:

- `dataset.enable_offline_packing` is `true`
- the resolved train and validation prefixes are the expected `.sft` prefixes
- `model.seq_length`, `dataset.seq_length`, and `packed_sequence_size` are all
  `4096`
- `train.micro_batch_size` is `1`
- `checkpoint.pretrained_checkpoint` points to the intended base weights
- `checkpoint.load` is `null` for a new fine-tuning run

Because the pair already exists, rank zero logs that packed preparation is
being skipped. Training then reads the local `.bin` with mmap and saves to the
configured checkpoint directory.

For LoRA or DoRA, select the corresponding PEFT recipe and mode, then apply the
same dataset overrides. An indexed pair can be shared between full SFT and PEFT
only when tokenizer, preprocessing, sequence length, seed, padding, and packing
settings are identical.

## Use a Recipe-Managed Hugging Face Source

For a recipe that already enables offline packing, the shortest path is to let
the builder materialize and cache its configured source. For example, the
Llama 3.2 1B SFT recipe uses SQuAD:

```bash
uv run python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --num-tokenizer-workers 8
```

Then launch the same recipe without replacing its dataset:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 \
    scripts/training/run_recipe.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --mode sft \
    --pretrained_checkpoint /data/checkpoints/llama32_1b \
    --save_dir /data/checkpoints/llama32_1b_sft \
    checkpoint.load=null
```

Builder-managed outputs live under a fingerprinted directory below the
materialized dataset root:

```text
<dataset-root>/packed/<tokenizer-and-fingerprint>/training_4096.sft.bin
<dataset-root>/packed/<tokenizer-and-fingerprint>/training_4096.sft.idx
<dataset-root>/packed/<tokenizer-and-fingerprint>/validation_4096.sft.bin
<dataset-root>/packed/<tokenizer-and-fingerprint>/validation_4096.sft.idx
<dataset-root>/packed/<tokenizer-and-fingerprint>/4096_metadata.jsonl
```

The fingerprint separates different tokenizers, preprocessing settings,
sequence lengths, and dataset options. It does not hash the bytes of local
JSONL files. When local source content changes, use a versioned dataset root or
new explicit output prefixes. For a builder-managed Hugging Face source, set
`hf_rewrite=True` only when normalized JSONL and packed outputs should be
regenerated. It cannot be combined safely with explicit packed paths.

For reproducibility, record the source revision or content hash, tokenizer
revision, preprocessing and loss policy, seed, sequence length,
`pad_seq_to_mult`, and Bridge commit with each packed dataset.

## Integrate with a Python Recipe

Replace the dataset object when integrating custom local data into an existing
SFT or PEFT recipe. This example lets the builder choose fingerprinted output
paths and prepare them on the first call:

```python
from megatron.bridge.data.builders import (
    GPTSFTDatasetBuilder,
    GPTSFTDatasetConfig,
    PromptCompletionSFTPreprocessingConfig,
)
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.recipes.llama import llama32_1b_sft_config
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

cfg = llama32_1b_sft_config()
cfg.dataset = GPTSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    dataset_root="/data/sft-jsonl",
    preprocessing=PromptCompletionSFTPreprocessingConfig(
        prompt_column="input",
        completion_column="output",
        separator=" ",
        loss_mode="completion",
    ),
    dataset_kwargs={"pad_to_max_length": True},
    enable_offline_packing=True,
    offline_packing_specs=PackedSequenceSpecs(
        packed_sequence_size=cfg.model.seq_length,
        pad_seq_to_mult=1,
        num_tokenizer_workers=8,
    ),
    do_validation=True,
    do_test=False,
)
cfg.train.micro_batch_size = 1

tokenizer = build_tokenizer(cfg.tokenizer)
GPTSFTDatasetBuilder(config=cfg.dataset, tokenizer=tokenizer).prepare_data()
```

Run the final two lines once in a single CPU process before submitting the GPU
job. The normal training builder calls the same preparation method on rank zero
and skips work when the complete pair is already present.

For a Hugging Face source, replace `dataset_root` with a declarative source and
choose where normalized JSONL should be cached:

```python
from megatron.bridge.data.builders import HFDatasetSourceConfig

cfg.dataset.dataset_root = None
cfg.dataset.hf_dataset = HFDatasetSourceConfig(dataset_name="squad")
cfg.dataset.hf_validation_proportion = 0.1
cfg.dataset.hf_output_root = "/data/materialized-squad"
```

For conversation data, replace the prompt-completion preprocessing object with
`ChatSFTPreprocessingConfig(loss_mode="assistant")`. The selected tokenizer
must provide the intended chat template. Preparation and training must use the
same preprocessing object.

To train from that config, set the checkpoint fields and call the standard SFT
entry point from a script launched with `torch.distributed.run`:

```python
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

cfg.checkpoint.pretrained_checkpoint = "/data/checkpoints/llama32_1b"
cfg.checkpoint.load = None
cfg.checkpoint.save = "/data/checkpoints/llama32_1b_sft"
finetune(config=cfg, forward_step_func=forward_step)
```

If the complete example is saved as `train_packed_sft.py`, launch it with:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 train_packed_sft.py
```

Do not construct a different preprocessing config for the training job. Packing
captures tokenization, loss policy, truncation, padding, sequence alignment, and
pack membership; those settings are part of the artifact's identity.

## Use Explicit Local Prefixes and Shards

After both files exist, reference prebuilt pairs directly:

```python
cfg.dataset.offline_packing_specs = PackedSequenceSpecs(
    packed_sequence_size=4096,
    packed_train_data_path="/data/packed/training_4096.sft",
    packed_val_data_path="/data/packed/validation_4096.sft",
    packed_metadata_path="/data/packed/4096_metadata.json",
    pad_seq_to_mult=1,
)
```

The following path specifications are accepted:

- a prefix such as `/data/packed/training_4096.sft`
- either pair filename, such as `training_4096.sft.bin`
- a glob over canonical pairs
- a split-specific directory containing canonical `*.sft.bin/*.sft.idx` pairs

Directory and glob inputs are consumed in sorted shard order. Every resolved
prefix must have both files. Directory resolution loads every pair in that
directory, so never mix training, validation, and test pairs in one directory
when the directory itself is used as the path specification. Prefer
split-specific globs such as `training_*.sft` and `validation_*.sft`. Generate
shards into distinct prefixes; do not concatenate `.bin` or `.idx` files with
shell tools.

The preparation command writes one training pair and, optionally, one
validation pair per invocation. It does not split a large input automatically,
and the current packer materializes the tokenized input and packing output in
host memory. For a dataset that does not fit comfortably on one preparation
node:

1. split normalized JSONL deterministically before tokenization
2. run the preparation command once per raw shard with a stable, unique `.sft`
   prefix such as `training_00000.sft`
3. consume a split-specific sorted glob in the training config
4. record raw-row and packed-row counts so missing or duplicated shards are
   detected before training

For example, one invocation can prepare the first shard:

```bash
uv run python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --train-input-path /data/sft-shards/training_00000.jsonl \
    --packed-train-data-path /data/packed/training_00000.sft \
    --packed-metadata-path /data/packed/4096_metadata.json \
    --num-tokenizer-workers 8
```

Repeat sequentially for later shards. The preparation function appends packing
statistics to the metadata JSON; do not run multiple writers concurrently
against the same metadata path. When `pad_cu_seqlens=True`, the metadata passed
to training must contain statistics for every consumed shard.

Configure train and validation shards with separate globs:

```python
cfg.dataset.offline_packing_specs = PackedSequenceSpecs(
    packed_sequence_size=4096,
    packed_train_data_path="/data/packed/training_*.sft",
    packed_val_data_path="/data/packed/validation_*.sft",
    packed_metadata_path="/data/packed/4096_metadata.json",
)
```

Local writers validate every row, build a temporary pair, take an exclusive
filesystem lock, publish `.bin`, and publish `.idx` last as the completion
point. Readers take the shared side of that lock while opening the pair. The
`.sft.lock` sidecar intentionally remains for later readers and writers.

## Read Prebuilt Data from Object Storage

For `msc://` prefixes, MCore streams `.bin` ranges and caches each `.idx`
locally. Bridge enables MCore's Multi-Storage Client integration when it sees
the prefix, but it does not create storage profiles or credentials. Configure
the provider and named profile according to the
[Multi-Storage Client configuration guide](https://nvidia.github.io/multi-storage-client/).

Use a local index-cache directory visible to every rank in a multi-node job:

```python
cfg.dataset.offline_packing_specs = PackedSequenceSpecs(
    packed_sequence_size=4096,
    packed_train_data_path="msc://profile/sft/training_4096.sft",
    packed_val_data_path="msc://profile/sft/validation_4096.sft",
    packed_metadata_path="msc://profile/sft/4096_metadata.json",
    object_storage_cache_path="/shared/cache/packed-sft-indices",
    # MCore defaults to 256 MiB range-read chunks. Tune only after measuring.
    object_storage_bin_chunk_nbytes=256 * 1024 * 1024,
)
```

When no cache is set, Bridge uses
`$NEMO_DATASETS_CACHE/packed_sft_index_cache`, or the corresponding default
NeMo cache. Rank zero downloads the remote indices before the distributed
barrier, and remote `.bin` files automatically use range reads instead of mmap.

Remote artifacts are read-only. Prepare a local pair, upload `.bin` completely,
then upload `.idx` as the completion marker:

```python
import multistorageclient as msc

msc.upload_file(
    "msc://profile/sft/training_4096.sft.bin",
    "/data/packed/training_4096.sft.bin",
)
msc.upload_file(
    "msc://profile/sft/training_4096.sft.idx",
    "/data/packed/training_4096.sft.idx",
)
```

Upload the metadata JSON as well when the config references it. Direct packing
or rewriting to object storage is rejected because replacing two remote objects
cannot provide the local pair's reader/writer atomicity.

## Training Constraints

Start with the recipe defaults and change one topology feature at a time.

| Setting | Requirement |
| --- | --- |
| Micro batch | Offline packed SFT requires `micro_batch_size=1` |
| Global batch | `global_batch_size` must be divisible by `micro_batch_size * data_parallel_size`; with packed MBS1 it must be a multiple of and no smaller than DP |
| Sequence length | Keep model, dataset, and packed sequence sizes equal in the standard path |
| Context parallelism | Sequence length must be divisible by `2 * context_parallel_size` |
| SFT with CP | Set `model.calculate_per_token_loss=True` and `ddp.average_in_collective=False` |
| Packing alignment | Use `lcm(2 * CP if CP > 1 else 1, CP * TP if sequence parallelism and TP > 1 else 1)` |
| CUDA graphs | TE-scoped graphs do not support packed SFT because `packed_seq_params` is a non-Tensor input; keep graphs disabled unless a separate local full-iteration path has been validated |
| MTP | Multi-Token Prediction is not supported with offline packed SFT |

The generic launcher synchronizes `model.seq_length` and
`packed_sequence_size` from the resolved dataset length. It also raises the
packing alignment to satisfy the resolved TP, CP, evaluation CP, and sequence
parallel topology. If those values change, rebuild the pair because padding and
pack membership can change.

`pad_cu_seqlens=True`, fixed-width padding, and packing metadata provide static
packed shapes, but they do not by themselves make CUDA graph capture supported.
They are only prerequisites for a separately validated experimental graph path.
See [CUDA Graphs](cuda-graphs.md) for the additional RNG, NaN-check, scope, and
backend constraints.

Offline LLM packing and VLM in-batch packing have opposite micro-batch rules.
Do not apply `micro_batch_size=1` to a VLM path that uses
`enable_in_batch_packing=True`; see [Packed Sequences](packed-sequences.md) for
that path and for model-family opt-outs.

## Migrate from Packed Parquet

Migration is intentionally explicit, so existing jobs do not silently switch
artifacts.

1. Keep the old job on its existing `.parquet` path.
2. Re-run packing from the same normalized JSONL using the same tokenizer,
   preprocessing, seed, sequence length, alignment, and dataset options.
3. Give the new output a `.sft` prefix, which selects indexed storage.
4. Compare every decoded row before changing the production recipe.
5. Run a short loss and throughput smoke on the target filesystem.

The comparison path requires the optional Parquet dependency:

```bash
uv sync --extra parquet
mkdir -p /data/packed-parquet /data/packed-indexed
```

First, prepare only the new indexed artifact from the normalized source and
settings that produced the existing production Parquet:

```bash
uv run python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --train-input-path /data/sft/training.jsonl \
    --packed-train-data-path /data/packed-indexed/training_4096.sft \
    --packed-metadata-path /data/packed-indexed/4096_metadata.json \
    --num-tokenizer-workers 8
```

Compare the existing production Parquet directly with the new indexed output:

```bash
uv run python scripts/training/compare_packed_sft_formats.py \
    --parquet /data/existing-packed/training_4096.idx.parquet \
    --indexed /data/packed-indexed/training_4096.sft
```

This direct comparison is the migration gate. Rebuilding both sides with the
current code could allow a shared behavior change to pass parity while differing
from the artifact used by the old production job.

Optionally, validate the current Parquet writer as a separate A/B check. An
output ending in `.parquet` selects the compatibility writer:

```bash
uv run python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --train-input-path /data/sft/training.jsonl \
    --packed-train-data-path /data/packed-parquet/training_4096.idx.parquet \
    --packed-metadata-path /data/packed-parquet/4096_metadata.json \
    --num-tokenizer-workers 8

uv run python scripts/training/compare_packed_sft_formats.py \
    --parquet /data/packed-parquet/training_4096.idx.parquet \
    --indexed /data/packed-indexed/training_4096.sft
```

Use `--max-rows N` for a quick sample. The command fails on any difference in
`input_ids`, target-aligned `loss_mask`, or `seq_start_id`. It also reports row
rate, token rate, elapsed time, and bytes on disk. These are sequential-read
microbenchmarks; measure end-to-end dataloader and training throughput on the
target filesystem before making capacity decisions.

Success is reported as `Parity validation passed for N rows`. If any field
differs, do not switch the recipe: verify that both outputs were freshly built
or originally built with the same normalized input, tokenizer revision,
preprocessing, seed, sequence length, alignment, and dataset options.

Existing `.parquet` and deprecated `.npy` paths remain readable when configured
explicitly. An output without a Parquet or NumPy suffix selects the indexed
writer; use a canonical `.sft` prefix so the pair is named
`.sft.bin/.sft.idx`.

## On-Disk Schema

Each MCore IndexedDataset item is one complete packed SFT row. Its int32 payload
contains:

1. a magic value and schema version
2. the number of sequence boundaries
3. the `seq_start_id` offsets
4. one int32 word per token

The lower 31 bits of a token word store the token ID. The high bit stores the
binary, target-aligned loss mask. The writer rejects non-binary masks, token IDs
outside the supported range, mismatched lengths, empty rows, and invalid
boundaries. The reader validates the header and boundaries before returning a
sample.

This schema preserves the logical fields used by the Parquet implementation
without requiring PyArrow on the default indexed training path.

## Troubleshooting

- **`Exactly one text-only SFT source must be set`.** Set exactly one of
  `dataset_root` or `hf_dataset`. When changing a built-in HF recipe to local
  JSONL through overrides, clear both `hf_dataset` and its
  `hf_validation_proportion` before setting `dataset_root`.

- **`Packed SFT IndexedDataset is incomplete or missing`.** Pass the logical
  `.sft` prefix and verify that both `<prefix>.bin` and `<prefix>.idx` are
  visible on every node. For shards, verify every resolved pair.

- **Training tries to prepare data again.** Inspect the printed final config.
  The packed prefix, sequence length, and source overrides probably differ from
  the preparation command, or only one member of the pair was uploaded.

- **The training command reads SQuAD instead of custom data.** The named recipe
  has a built-in HF source. Set `dataset.hf_dataset=null`, clear the HF split
  settings, and set `dataset.dataset_root` as shown in the quickstart.

- **`a metadata json file is required when pad_cu_seqlens is enabled`.** Point
  `packed_metadata_path` at the metadata produced with the pair. Do not enable
  padded cumulative sequence lengths without it.

- **Packed micro-batch validation fails.** Set `train.micro_batch_size=1`.
  Increase `global_batch_size` for gradient accumulation rather than increasing
  the offline-packed micro batch.

- **CP or sequence-parallel divisibility fails.** Recompute `pad_seq_to_mult`
  from the final topology and rebuild the pair. Do not reuse data packed for a
  different alignment.

- **Remote reads fail before training starts.** Verify the MSC profile and
  credentials on every node, confirm that `.bin` was uploaded before `.idx`,
  and use a shared writable local index cache.

- **Preparation is slow or exhausts file descriptors.** First validate with one
  tokenizer worker. Then increase `num_tokenizer_workers` gradually and check
  the node's memory and file descriptor limits.

- **Finetuning reports that a checkpoint is required.** Set
  `checkpoint.pretrained_checkpoint` for a new SFT/PEFT run, or
  `checkpoint.load` for a complete native checkpoint resume.

For raw schema details and all `GPTSFTDatasetConfig` knobs, see the
[text-only SFT dataset tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/text-only-sft/README.md).
For packed attention behavior and performance guidance, see
[Packed Sequences](packed-sequences.md).
