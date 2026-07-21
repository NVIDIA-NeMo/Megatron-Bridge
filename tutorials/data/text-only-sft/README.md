# Text-only SFT Dataset Tutorial

Choose this transitional path when you have local text JSONL, or when you want to normalize a Hugging Face text source into reusable JSONL before SFT or PEFT. It also supports offline packed-Parquet processing and finite `num_epochs` training. The planned prepared `.bin`/`.idx` SFT workflow tracked by Issue #4664 will replace packed Parquet as the recommended scalable prepared-data path. See the [data tutorial overview](../README.md#which-sft-path-should-i-use) if you are deciding between this path and direct SFT.

You configure the data with `GPTSFTDatasetConfig`; the training framework uses `GPTSFTDatasetBuilder` to bind the tokenizer, materialize sources when needed, prepare offline packing, and construct `GPTSFTDataset` splits.

Choose exactly one source:

- `dataset_root` for local `training.jsonl`, optional `validation.jsonl`, and optional `test.jsonl` files.
- `hf_dataset` for a declarative Hugging Face source plus an optional registered schema adapter.

## Prepare local JSONL

From the repository root, generate a small prompt/completion dataset:

```bash
uv run python tutorials/data/text-only-sft/prepare_example_data.py \
    --output-dir /tmp/bridge-text-only-sft
```

Each line is a JSON object:

```json
{"input": "What is SFT?", "output": "Supervised fine-tuning."}
```

This is two-column paired text. For standard three-field Alpaca-style rows, combine `instruction` and `input` into the configured prompt column before training so neither field is dropped.

For conversation data, select chat preprocessing explicitly:

```json
{"messages": [{"role": "user", "content": "What is SFT?"}, {"role": "assistant", "content": "Supervised fine-tuning."}]}
```

and set `preprocessing=ChatSFTPreprocessingConfig()`. Check that every file is valid JSONL before training:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

for path in Path("/tmp/bridge-text-only-sft").glob("*.jsonl"):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    print(path.name, len(rows))
PY
```

## Connect local data to training

Assign the config to any SFT or PEFT recipe before calling `finetune()`:

```python
from megatron.bridge.data.builders import GPTSFTDatasetConfig, PromptCompletionSFTPreprocessingConfig

cfg.dataset = GPTSFTDatasetConfig(
    seq_length=4096,
    dataset_root="/tmp/bridge-text-only-sft",
    preprocessing=PromptCompletionSFTPreprocessingConfig(
        prompt_column="input",
        completion_column="output",
        separator=" ",
    ),
    dataloader_type="batch",
    do_validation=True,
    do_test=False,
)
```

The generic launcher has a local-data preset:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --dataset llm-finetune-preloaded \
    dataset.dataset_root=/tmp/bridge-text-only-sft \
    dataset.seq_length=4096 \
    checkpoint.pretrained_checkpoint=/checkpoints/base-model
```

`model.seq_length` and `dataset.seq_length` must match. SFT and PEFT also need pretrained weights unless resuming a complete native checkpoint.

## Start from a Hugging Face source

Use `HFDatasetSourceConfig` instead of `dataset_root`:

```python
from megatron.bridge.data.builders import (
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)

cfg.dataset = GPTSFTDatasetConfig(
    seq_length=2048,
    hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
    preprocessing=PromptCompletionSFTPreprocessingConfig(separator=" "),
    hf_validation_proportion=0.1,
    hf_output_root="/data/materialized-squad",
    hf_rewrite=False,
    do_test=False,
)
```

The builder loads the source, applies the optional registered row adapter, normalizes rows for the selected preprocessing mode, writes the standard split files, and then uses the same text-only SFT construction as local mode. Omit `hf_output_root` to use the NeMo dataset cache. Set `hf_rewrite=True` only when existing normalized files should be replaced; builder-managed packed artifacts are regenerated at the same time. Native rows matching the selected preprocessing schema need no `schema_adapter`.

## Enable offline packing

Offline packing preprocesses examples into packed Parquet plus metadata and reuses it across runs:

```python
from megatron.bridge.data.packing import PackedSequenceSpecs

cfg.dataset.enable_offline_packing = True
cfg.dataset.offline_packing_specs = PackedSequenceSpecs(
    packed_sequence_size=cfg.dataset.seq_length,
    num_tokenizer_workers=8,
    pad_seq_to_mult=1,
)
cfg.train.micro_batch_size = 1
```

The demonstrated path keeps the model, dataset, and packed sequence sizes aligned. Advanced runs may pack multiple shorter source examples into a larger packed sequence, but the model/runtime shape must be configured consistently.

The builder prepares packed files automatically on the first training build. To prepare the generated prompt/completion example explicitly, use the same canonical config and builder as training:

```bash
uv run python - <<'PY'
from megatron.bridge.data.builders import (
    GPTSFTDatasetBuilder,
    GPTSFTDatasetConfig,
    PromptCompletionSFTPreprocessingConfig,
)
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.recipes.llama.llama3 import llama32_1b_sft_config
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

recipe = llama32_1b_sft_config()
dataset = GPTSFTDatasetConfig(
    seq_length=recipe.model.seq_length,
    dataset_root="/tmp/bridge-text-only-sft",
    preprocessing=PromptCompletionSFTPreprocessingConfig(
        prompt_column="input",
        completion_column="output",
        separator=" ",
    ),
    enable_offline_packing=True,
    offline_packing_specs=PackedSequenceSpecs(
        packed_sequence_size=recipe.model.seq_length,
        num_tokenizer_workers=1,
    ),
    do_test=False,
)
tokenizer = build_tokenizer(recipe.tokenizer)
GPTSFTDatasetBuilder(config=dataset, tokenizer=tokenizer).prepare_data()
PY
```

This writes default `.idx.parquet` packed splits and metadata under the dataset root. `scripts/training/prepare_gpt_sft_packed_data.py` is also available when the selected named recipe's dataset schema matches the input files.

Packed SFT requires micro batch size 1. With context parallelism, sequence lengths must be divisible by twice the CP size, `calculate_per_token_loss=True`, and `ddp.average_in_collective=False`. CUDA graphs require padded packed metadata.

For the complete constraints and runtime behavior, see [Packed Sequences](../../../docs/training/packed-sequences.md). Contributors can also use the [sequence-packing validation guide](../../../skills/nemo-mbridge-perf-sequence-packing/SKILL.md) when changing or validating this path.

`train.num_epochs` is supported for this finite dataset only with `dataloader_type="batch"`; Bridge derives the iteration count from the true training split size and keeps the final incomplete global batch.

## Available knobs

| Area | Knobs | Purpose |
| --- | --- | --- |
| Source | `dataset_root`, `hf_dataset` | Exactly one local or Hugging Face source |
| Core | `seq_length`, `seed`, `memmap_workers`, `max_train_samples` | Shape, reproducibility, indexing, and train cap |
| Splits | `do_validation`, `do_test` | Build optional split files |
| Preprocessing | `ChatSFTPreprocessingConfig`, `PromptCompletionSFTPreprocessingConfig` | Chat rendering and assistant loss, or raw paired-text formatting and completion/full loss |
| Dataset behavior | `dataset_kwargs` | Backend-only options such as padding, tool schemas, and advanced `GPTSFTDataset` controls |
| Offline packing | `enable_offline_packing`, `offline_packing_specs` | Prepared packed sequences and metadata |
| Loader | `dataloader_type`, `num_workers`, `data_sharding`, `pin_memory`, `drop_last`, `persistent_workers` | DataLoader behavior |
| HF source | `path_or_dataset`, `split`, `subset`, `load_kwargs`, optional registered `schema_adapter` and `adapter_kwargs` | Dataset loading and row normalization |
| HF materialization | `hf_validation_dataset`, `hf_test_dataset`, `hf_output_root`, `hf_validation_proportion`, `hf_rewrite` | Split overrides, cache location, and rematerialization |
| Packing spec | `packed_sequence_size`, `tokenizer_model_name`, `num_tokenizer_workers`, packed paths, metadata path, `pad_cu_seqlens`, `pad_seq_to_mult` | Packed artifact layout and alignment |

Do not repeat config-owned `seed`, `memmap_workers`, `max_num_samples`, or preprocessing fields inside `dataset_kwargs`. Source, adapter, and dataset mappings accept declarative values only; runtime tokenizers and callables belong to the builder.

## Troubleshooting

- “Exactly one text-only SFT source” means both source fields, or neither, were set.
- A missing split file is expected when its `do_*` flag is false; otherwise verify the exact filenames above.
- Chat-template errors usually mean `ChatSFTPreprocessingConfig` was selected but the tokenizer lacks a template.
- Prompt-completion preprocessing never calls `apply_chat_template` and deliberately rejects structured multi-turn rows.
- Packing validation errors identify incompatible micro-batch, CP, metadata, or packed-size settings before training starts.
