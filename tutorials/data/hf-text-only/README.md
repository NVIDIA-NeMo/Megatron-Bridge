# Hugging Face Text-Only SFT Tutorial

Use this path for text SFT or PEFT when chat or prompt-completion rows should be processed directly from a hosted
Hugging Face dataset or from JSON/JSONL files supported by the Hugging Face `json` loader. All inputs use the
serializable `HFDatasetSourceConfig` and `DirectHFSFTDatasetConfig`; `DirectHFSFTDatasetBuilder` owns runtime
tokenizer, collator, and dataset construction.

For image, video, audio, or omni conversations, use the separate [Hugging Face multimodal tutorial](../hf-multimodal/README.md).
For materialized JSONL, finite `num_epochs`, or offline packed-Parquet behavior, use the transitional
[prepared text-only tutorial](../text-only-sft/README.md).

## Start with a hosted chat dataset

If a hosted dataset already exposes native `messages`, select it and its split directly; no schema adapter is required. Set an instruction-tuned processor with the intended chat template:

```python
from megatron.bridge.recipes.llama.llama3 import llama32_1b_sft_config
from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, HFDatasetSourceConfig, DirectHFSFTDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

cfg = llama32_1b_sft_config()
cfg.checkpoint.pretrained_checkpoint = "/checkpoints/llama32-1b-instruct"
cfg.checkpoint.load = None
cfg.dataset = DirectHFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
    hf_processor_path="meta-llama/Llama-3.2-1B-Instruct",
    source=HFDatasetSourceConfig(
        path_or_dataset="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
    ),
    dataloader_type="single",
    do_validation=False,
    do_test=False,
)

finetune(config=cfg, forward_step_func=forward_step)
```

The base checkpoint must be a native Megatron checkpoint or a local Hugging Face full-model directory accepted by Bridge, and its vocabulary must match the selected processor. Set `hf_processor_path=None` only when the recipe's training tokenizer already defines the intended chat template. Keep `cfg.model.seq_length` equal to `cfg.dataset.seq_length`. The builder repeats normalized examples to the sample counts requested by the training schedule, so iteration-based training remains the supported duration model.

Use a registered `schema_adapter` when hosted columns are not already one of `messages`, `conversation`, or `conversations`. Built-in examples are listed in [Available knobs](#available-knobs). `load_kwargs` belongs to dataset loading; `adapter_kwargs` belongs only to row conversion.

## Load conversation JSON or JSONL through Hugging Face datasets

Generate small `messages` JSONL files from the repository root:

```bash
uv run python tutorials/data/hf-text-only/prepare_example_data.py \
    --output-dir /tmp/bridge-hf-text-only
```

Native chat rows look like this:

```json
{"messages": [{"role": "user", "content": "What belongs in config?"}, {"role": "assistant", "content": "Validated declarative data."}]}
```

Select each file through the Hugging Face `json` dataset builder. The source config stays serializable; `datasets.load_dataset` opens the files and the Direct SFT builder normalizes the resulting rows at runtime:

```python
cfg.dataset = DirectHFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
    hf_processor_path="meta-llama/Llama-3.2-1B-Instruct",
    source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="train",
        load_kwargs={"data_files": {"train": "/tmp/bridge-hf-text-only/training.jsonl"}},
    ),
    validation_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="validation",
        load_kwargs={"data_files": {"validation": "/tmp/bridge-hf-text-only/validation.jsonl"}},
    ),
    test_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="test",
        load_kwargs={"data_files": {"test": "/tmp/bridge-hf-text-only/test.jsonl"}},
    ),
    do_validation=True,
    do_test=True,
)
```

Use a JSON array or one object per line in JSONL. Chat rows may use `messages`, singular `conversation`, or legacy
`conversations` with `from`/`value` turns. Hub and file-backed rows follow the same chat-template, loss-mask, padding,
and collation path after loading.

## Train paired text without a chat template

Select prompt-completion preprocessing when the source is already formatted as paired text and no model chat template should be applied:

```python
from megatron.bridge.data.builders import PromptCompletionSFTPreprocessingConfig

cfg.dataset = DirectHFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    source=HFDatasetSourceConfig(
        path_or_dataset="json",
        load_kwargs={"data_files": {"train": "/data/paired/training.jsonl"}},
    ),
    preprocessing=PromptCompletionSFTPreprocessingConfig(
        prompt_column="prompt",
        completion_column="completion",
        separator="\n",
        loss_mode="completion",
    ),
    do_validation=False,
    do_test=False,
)
```

This mode tokenizes the prompt and completion separately and never calls `apply_chat_template`. It supports completion-only or full-sequence loss. Structured conversations are not flattened into paired text. The schema is intentionally two-column; normalize three-field Alpaca-style `instruction` + `input` + `output` rows first so the `input` field is not dropped.

## In-batch packing and padding

Text chat and prompt-completion collators can pack examples during collation:

```python
cfg.dataset.enable_in_batch_packing = True
cfg.train.micro_batch_size = 4
```

In-batch packing requires micro batch size greater than one. `in_batch_packing_pad_to_multiple_of` is finalized from
CP/SP constraints. Pipeline or expert parallel training automatically enables fixed-length padding through
`pad_to_max_length`. With CP, enable per-token loss and disable collective loss averaging.

## Available knobs

| Area | Knobs | Purpose |
| --- | --- | --- |
| Source | `HFDatasetSourceConfig` | Hub datasets or custom sources accepted by Hugging Face datasets, including its `json` loader |
| Adaptation | `source.schema_adapter`, `source.adapter_kwargs` | Optional conversion to canonical chat or prompt-completion rows; presets own their adapter |
| Preprocessing | `ChatSFTPreprocessingConfig`, `PromptCompletionSFTPreprocessingConfig` | Chat-template loss policy or raw paired-text formatting/loss policy |
| Split overrides | `validation_source`, `test_source`, `do_validation`, `do_test` | Per-split loading and optional split construction |
| Tokenizer | `hf_processor_path`, inherited `trust_remote_code` | Optional instruction tokenizer source and safe remote-code policy |
| Sequence | `seq_length`, `skip_getting_attention_mask_from_dataset` | Training shape and attention-mask handoff |
| Packing | `enable_in_batch_packing`, `defer_in_batch_packing_to_step`, `in_batch_packing_pad_to_multiple_of` | Collate- or model-step packing |
| Padding | `pad_to_max_length`, `pad_to_multiple_of` | Fixed or efficient batch padding |
| Loader | `dataloader_type`, `num_workers`, `data_sharding`, `pin_memory`, `drop_last`, `persistent_workers` | DataLoader behavior |

Built-in text presets and adapters include `squad`, `gsm8k`, and `openmathinstruct2`; native rows matching the selected
preprocessing schema need no adapter. All config mappings must contain serializable declarative values; tokenizers
and collator callables belong to the builder.

## Troubleshooting

- If `hf_processor_path` is unset, the build context must contain a training tokenizer.
- A source must return non-empty normalized rows for every enabled split with a positive requested sample count. File-based sources should provide explicit validation/test `data_files` and split sources, or disable those stages.
- Chat rows should follow the role ordering accepted by the selected model template. Rows with no trainable assistant tokens emit a warning and contribute zero loss.
- Prompt-completion preprocessing accepts text columns only and does not call `apply_chat_template`.
- In-batch packing fails early when the micro batch or CP/SP settings are incompatible.
