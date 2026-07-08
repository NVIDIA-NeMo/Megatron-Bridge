# Direct Hugging Face SFT Dataset Tutorial

Use `DirectHFSFTDatasetConfig` for direct Hugging Face SFT processing. The training framework resolves `DirectHFSFTDatasetBuilder`, which loads a declarative source, applies an optional schema adapter, binds the processor/tokenizer and collator, and constructs schedule-sized `DirectSFTDataset` splits.

This tutorial starts with text chat, but the same Config + Builder path supports paired text, vision, video, audio, and omni rows. Text rows use the same preprocessing configs and tokenization helpers as the materialized text-only SFT path. Multimodal rows use chat preprocessing plus the configured Hugging Face processor and its registered model collator.

Compared with materialized text-only SFT, this path processes adapted rows directly and supports collate-time in-batch packing. It does not materialize reusable text-only SFT JSONL or provide offline packing and finite `num_epochs` semantics.

## Prepare text conversations

Generate local `messages` JSONL files from the repository root:

```bash
uv run python tutorials/data/direct-hf-sft/prepare_example_data.py \
    --output-dir /tmp/bridge-direct-hf-sft
```

Native chat sources may use rows like:

```json
{"messages": [{"role": "user", "content": "What belongs in config?"}, {"role": "assistant", "content": "Validated declarative data."}]}
```

The legacy `conversations` key is also recognized for text. Multimodal schema adapters normally produce a singular `conversation` value with typed content items and media fields.

## Connect text chat to training

Use the Hugging Face `json` loader directly. Set an instruction-tuned processor with a chat template:

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
        path_or_dataset="json",
        load_kwargs={"data_files": {"train": "/tmp/bridge-direct-hf-sft/training.jsonl"}},
    ),
    validation_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="validation",
        load_kwargs={"data_files": {"validation": "/tmp/bridge-direct-hf-sft/validation.jsonl"}},
    ),
    test_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="test",
        load_kwargs={"data_files": {"test": "/tmp/bridge-direct-hf-sft/test.jsonl"}},
    ),
    dataloader_type="single",
    do_validation=True,
    do_test=True,
)

finetune(config=cfg, forward_step_func=forward_step)
```

The base checkpoint must be a native Megatron checkpoint or a local Hugging Face full-model directory accepted by Bridge, and its vocabulary must match the selected processor. Set `hf_processor_path=None` only when the recipe's training tokenizer already defines the intended chat template. Keep `cfg.model.seq_length` equal to `cfg.dataset.seq_length`. The builder repeats normalized examples to the sample counts requested by the training schedule, so iteration-based training remains the supported duration model.

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

This mode tokenizes the prompt and completion separately and never calls `apply_chat_template`. It supports completion-only or full-sequence loss. Structured conversations are not flattened into paired text.

## Use an existing hosted dataset

For a hosted dataset that already exposes native `messages`, select the dataset and split directly; no adapter is required:

```python
cfg.dataset = DirectHFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    preprocessing=ChatSFTPreprocessingConfig(),
    hf_processor_path="meta-llama/Llama-3.2-1B-Instruct",
    source=HFDatasetSourceConfig(
        path_or_dataset="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
    ),
    do_validation=False,
    do_test=False,
)
```

Use `schema_adapter` only when the hosted columns are not already one of `messages`, `conversation`, or `conversations`. `load_kwargs` belongs to dataset loading; `adapter_kwargs` belongs only to row conversion.

## Connect multimodal data

For a built-in VLM or audio dataset, set the model processor path and select its named source preset:

```python
cfg.dataset = DirectHFSFTDatasetConfig(
    seq_length=4096,
    preprocessing=ChatSFTPreprocessingConfig(),
    hf_processor_path="Qwen/Qwen2.5-VL-3B-Instruct",
    source=HFDatasetSourceConfig(dataset_name="cord_v2"),
    enable_in_batch_packing=False,
)
```

Use the corresponding recipe step, such as `vlm_step`, `qwen3_vl_step`, or an audio/omni step. Collators are inferred at runtime from processor type. Runtime callable collator overrides are intentionally not config fields; custom integrations can inject them into `DirectHFSFTDatasetBuilder` without making serialized configs provider-specific.

## In-batch packing and padding

Text and supported model collators can pack examples during collation:

```python
cfg.dataset.enable_in_batch_packing = True
cfg.train.micro_batch_size = 4
```

In-batch packing requires micro batch size greater than 1. `in_batch_packing_pad_to_multiple_of` is finalized from CP/SP constraints. Some model steps set `defer_in_batch_packing_to_step=True` because they need original tensors before generating packed metadata. Pipeline or expert parallel training automatically enables fixed-length padding through `pad_to_max_length`.

## Available knobs

| Area | Knobs | Purpose |
| --- | --- | --- |
| Source | `source.dataset_name` or `source.path_or_dataset`, plus `split` and `load_kwargs` | Built-in preset or custom Hugging Face source |
| Adaptation | `source.schema_adapter`, `source.adapter_kwargs` | Optional conversion to canonical chat or prompt-completion rows; presets own their adapter |
| Preprocessing | `ChatSFTPreprocessingConfig`, `PromptCompletionSFTPreprocessingConfig` | Chat-template loss policy or raw paired-text formatting/loss policy |
| Split overrides | `validation_source`, `test_source`, `do_validation`, `do_test` | Per-split loading and optional split construction |
| Processor | `hf_processor_path`, inherited `trust_remote_code` | Model processor/tokenizer source and safe remote-code policy |
| Sequence | `seq_length`, `skip_getting_attention_mask_from_dataset` | Training shape and attention-mask handoff |
| Packing | `enable_in_batch_packing`, `defer_in_batch_packing_to_step`, `in_batch_packing_pad_to_multiple_of` | Collate- or model-step packing |
| Padding | `pad_to_max_length`, `pad_to_multiple_of` | Fixed or efficient batch padding |
| Loader | `dataloader_type`, `num_workers`, `data_sharding`, `pin_memory`, `drop_last`, `persistent_workers` | DataLoader behavior |

Built-in adapters include `squad`, `gsm8k`, `cord_v2`, and `cv17`; native rows matching the selected preprocessing schema need no adapter. All config mappings must contain serializable declarative values; processors, tokenizers, and collator callables belong to the builder.

## Troubleshooting

- If `hf_processor_path` is unset, the build context must contain a training tokenizer.
- An unknown processor type means its multimodal collator is not registered; text rows select a shared collator from the preprocessing config automatically.
- A source must return non-empty normalized rows for every enabled split with a positive requested sample count.
- Unsupported in-batch packing fails early when a custom/model collator cannot accept packing metadata.
