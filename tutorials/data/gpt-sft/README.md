# GPT SFT Dataset Tutorial

Use this path for text SFT or PEFT when data already exists as local JSONL, or when a registered Hugging Face maker should first normalize data into reusable JSONL. `GPTSFTDatasetConfig` stores declarative settings; the training framework resolves `GPTSFTDatasetBuilder`, which owns tokenizer binding, Hugging Face materialization, offline packing, and `GPTSFTDataset` construction.

Choose exactly one source:

- `dataset_root` for local `training.jsonl`, optional `validation.jsonl`, and optional `test.jsonl` files.
- `hf_dataset` for a registered maker that materializes those files.

## Prepare local JSONL

From the repository root, generate a small prompt/completion dataset:

```bash
uv run python tutorials/data/gpt-sft/prepare_example_data.py \
    --output-dir /tmp/bridge-gpt-sft
```

Each line is a JSON object:

```json
{"input": "What is SFT?", "output": "Supervised fine-tuning."}
```

For conversation data, use the tokenizer's chat template explicitly:

```json
{"messages": [{"role": "user", "content": "What is SFT?"}, {"role": "assistant", "content": "Supervised fine-tuning."}]}
```

and set `dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True}`. Check that every file is valid JSONL before training:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

for path in Path("/tmp/bridge-gpt-sft").glob("*.jsonl"):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    print(path.name, len(rows))
PY
```

## Connect local data to training

Assign the config to any SFT or PEFT recipe before calling `finetune()`:

```python
from megatron.bridge.training.config import GPTSFTDatasetConfig

cfg.dataset = GPTSFTDatasetConfig(
    seq_length=4096,
    dataset_root="/tmp/bridge-gpt-sft",
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
    dataset.dataset_root=/tmp/bridge-gpt-sft \
    dataset.seq_length=4096 \
    checkpoint.pretrained_checkpoint=/checkpoints/base-model
```

`model.seq_length` and `dataset.seq_length` must match. SFT and PEFT also need pretrained weights unless resuming a complete native checkpoint.

## Materialize a Hugging Face source

Use `HFDatasetSourceConfig` instead of `dataset_root`:

```python
from megatron.bridge.training.config import GPTSFTDatasetConfig, HFDatasetSourceConfig

cfg.dataset = GPTSFTDatasetConfig(
    seq_length=2048,
    hf_dataset=HFDatasetSourceConfig(
        maker_name="squad",
        maker_kwargs={"path_or_dataset": "rajpurkar/squad", "split": "train"},
        val_proportion=0.1,
        output_root="/data/materialized-squad",
        rewrite=False,
    ),
    do_test=False,
)
```

The builder invokes the maker, normalizes chat rows, writes the standard split files, and then uses the same GPT SFT construction as local mode. Omit `output_root` to use the NeMo dataset cache. Set `rewrite=True` only when existing normalized files should be replaced.

## Enable offline packing

Offline packing preprocesses examples into packed Parquet plus metadata and reuses it across runs:

```python
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

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
from megatron.bridge.data.builders.finetuning_dataset import GPTSFTDatasetBuilder
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.recipes.llama.llama3 import llama32_1b_sft_config
from megatron.bridge.training.config import GPTSFTDatasetConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

recipe = llama32_1b_sft_config()
dataset = GPTSFTDatasetConfig(
    seq_length=recipe.model.seq_length,
    dataset_root="/tmp/bridge-gpt-sft",
    dataset_kwargs={"prompt_template": "{input} {output}"},
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

This writes default `.idx.parquet` packed splits and metadata under the dataset root. `scripts/training/pack_sft_data.py` is also available when the selected named recipe's dataset schema matches the input files.

Packed SFT requires micro batch size 1. With context parallelism, sequence lengths must be divisible by twice the CP size, `calculate_per_token_loss=True`, and `ddp.average_in_collective=False`. CUDA graphs require padded packed metadata.

`train.num_epochs` is supported for this finite dataset only with `dataloader_type="batch"`; Bridge derives the iteration count from the true training split size and keeps the final incomplete global batch.

## Available knobs

| Area | Knobs | Purpose |
| --- | --- | --- |
| Source | `dataset_root`, `hf_dataset` | Exactly one local or Hugging Face source |
| Core | `seq_length`, `seed`, `memmap_workers`, `max_train_samples` | Shape, reproducibility, indexing, and train cap |
| Splits | `do_validation`, `do_test` | Build optional split files |
| Dataset behavior | `dataset_kwargs` | Prompt template, BOS/EOS/SEP, truncation, padding, labels/loss masks, chat template, tools, and advanced `GPTSFTDataset` options |
| Offline packing | `enable_offline_packing`, `offline_packing_specs` | Prepared packed sequences and metadata |
| Loader | `dataloader_type`, `num_workers`, `data_sharding`, `pin_memory`, `drop_last`, `persistent_workers` | DataLoader behavior |
| HF source | `maker_name`, `maker_kwargs`, `val_maker_kwargs`, `test_maker_kwargs`, `output_root`, `val_proportion`, `rewrite` | Maker input, split overrides, cache location, and rematerialization |
| Packing spec | `packed_sequence_size`, `tokenizer_model_name`, `num_tokenizer_workers`, packed paths, metadata path, `pad_cu_seqlens`, `pad_seq_to_mult` | Packed artifact layout and alignment |

Do not repeat config-owned `seed`, `memmap_workers`, or `max_num_samples` inside `dataset_kwargs`. Maker and dataset mappings accept declarative values only; runtime tokenizers and callables belong to the builder.

## Troubleshooting

- “Exactly one GPT SFT source” means both source fields, or neither, were set.
- A missing split file is expected when its `do_*` flag is false; otherwise verify the exact filenames above.
- Chat-template errors usually mean the tokenizer lacks a template or chat rows were used without the chat dataset settings.
- Packing validation errors identify incompatible micro-batch, CP, metadata, or packed-size settings before training starts.
