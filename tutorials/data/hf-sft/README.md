# Direct Hugging Face SFT Dataset Tutorial

Use `HFSFTDatasetConfig` for direct Hugging Face SFT processing. The training framework resolves `HFSFTDatasetBuilder`, which loads a declarative source, applies an optional schema adapter, binds the processor/tokenizer and collator, and constructs schedule-sized `ConversationDataset` splits.

This tutorial starts with text chat, but the same Config + Builder path supports vision, video, audio, and omni rows. Text rows use the same conversation rendering, tokenization, and assistant-only masking as the materialized text-only SFT path. Multimodal rows use the configured Hugging Face processor and its registered model collator.

Compared with materialized text-only SFT, this path processes adapted rows directly and supports collate-time in-batch packing. It does not materialize reusable text-only SFT JSONL or provide offline packing and finite `num_epochs` semantics.

## Prepare text conversations

Generate local `messages` JSONL files from the repository root:

```bash
uv run python tutorials/data/hf-sft/prepare_example_data.py \
    --output-dir /tmp/bridge-hf-sft
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
from megatron.bridge.data.builders import HFDatasetSourceConfig, HFSFTDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

cfg = llama32_1b_sft_config()
cfg.checkpoint.pretrained_checkpoint = "/checkpoints/llama32-1b-instruct"
cfg.checkpoint.load = None
cfg.dataset = HFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    hf_processor_path="meta-llama/Llama-3.2-1B-Instruct",
    source=HFDatasetSourceConfig(
        path_or_dataset="json",
        load_kwargs={"data_files": {"train": "/tmp/bridge-hf-sft/training.jsonl"}},
    ),
    validation_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="validation",
        load_kwargs={"data_files": {"validation": "/tmp/bridge-hf-sft/validation.jsonl"}},
    ),
    test_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="test",
        load_kwargs={"data_files": {"test": "/tmp/bridge-hf-sft/test.jsonl"}},
    ),
    dataloader_type="single",
    do_validation=True,
    do_test=True,
)

finetune(config=cfg, forward_step_func=forward_step)
```

The base checkpoint must be a native Megatron checkpoint or a local Hugging Face full-model directory accepted by Bridge, and its vocabulary must match the selected processor. Set `hf_processor_path=None` only when the recipe's training tokenizer already defines the intended chat template. Keep `cfg.model.seq_length` equal to `cfg.dataset.seq_length`. The builder repeats normalized examples to the sample counts requested by the training schedule, so iteration-based training remains the supported duration model.

## Use an existing hosted dataset

For a hosted dataset that already exposes native `messages`, select the dataset and split directly; no adapter or maker is required:

```python
cfg.dataset = HFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
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

For VLM or audio data, set the model processor path and choose its source adapter:

```python
cfg.dataset = HFSFTDatasetConfig(
    seq_length=4096,
    hf_processor_path="Qwen/Qwen2.5-VL-3B-Instruct",
    source=HFDatasetSourceConfig(
        path_or_dataset="naver-clova-ix/cord-v2",
        split="train",
        schema_adapter="cord_v2",
    ),
    enable_in_batch_packing=False,
)
```

Use the corresponding recipe step, such as `vlm_step`, `qwen3_vl_step`, or an audio/omni step. Collators are inferred at runtime from processor type. Runtime callable collator overrides are intentionally not config fields; custom integrations can inject them into `HFSFTDatasetBuilder` without making serialized configs provider-specific.

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
| Source | `source.path_or_dataset`, `split`, `subset`, `load_kwargs` | Hugging Face dataset and training split |
| Adaptation | `source.schema_adapter`, `source.adapter_kwargs` | Optional conversion for non-native row schemas |
| Split overrides | `validation_source`, `test_source`, `do_validation`, `do_test` | Per-split loading and optional split construction |
| Processor | `hf_processor_path`, inherited `trust_remote_code` | Model processor/tokenizer source and safe remote-code policy |
| Sequence | `seq_length`, `skip_getting_attention_mask_from_dataset` | Training shape and attention-mask handoff |
| Packing | `enable_in_batch_packing`, `defer_in_batch_packing_to_step`, `in_batch_packing_pad_to_multiple_of` | Collate- or model-step packing |
| Padding | `pad_to_max_length`, `pad_to_multiple_of` | Fixed or efficient batch padding |
| Loader | `dataloader_type`, `num_workers`, `data_sharding`, `pin_memory`, `drop_last`, `persistent_workers` | DataLoader behavior |

Built-in adapters include `squad`, `gsm8k`, `cord_v2`, and `cv17`; native chat columns need no adapter. All config mappings must contain serializable declarative values; processors, tokenizers, and collator callables belong to the builder.

## Troubleshooting

- If `hf_processor_path` is unset, the build context must contain a training tokenizer.
- An unknown processor type means its multimodal collator is not registered; text `messages` rows select the shared text collator automatically.
- A source must return non-empty normalized rows for every enabled split with a positive requested sample count.
- Unsupported in-batch packing fails early when a custom/model collator cannot accept packing metadata.
