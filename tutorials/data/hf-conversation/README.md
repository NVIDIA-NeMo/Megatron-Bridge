# Hugging Face Conversation Dataset Tutorial

Use `HFConversationDatasetConfig` for direct Hugging Face conversation processing. The training framework resolves `HFConversationDatasetBuilder`, which loads a registered maker and processor/tokenizer, chooses the runtime collator, and constructs schedule-sized `ConversationDataset` splits.

This tutorial starts with text chat, but the same Config + Builder path supports vision, video, audio, and omni makers. Text rows use the tokenizer's chat template and shared assistant-only loss masking. Multimodal rows use the configured Hugging Face processor and its registered model collator.

Compared with GPT SFT, this path processes maker output directly and supports collate-time in-batch packing. It does not materialize reusable GPT SFT JSONL or provide offline packing and finite `num_epochs` semantics.

## Prepare text conversations

Generate local `messages` JSONL files from the repository root:

```bash
uv run python tutorials/data/hf-conversation/prepare_example_data.py \
    --output-dir /tmp/bridge-hf-conversation
```

The text maker expects chat rows like:

```json
{"messages": [{"role": "user", "content": "What belongs in config?"}, {"role": "assistant", "content": "Validated declarative data."}]}
```

The legacy `conversations` key is also recognized for text. Multimodal makers normally produce a singular `conversation` value with typed content items and media fields.

## Connect text chat to training

Use the Hugging Face `json` loader through the registered `text_chat` maker. Leave `hf_processor_path=None` to reuse the training recipe's tokenizer:

```python
from megatron.bridge.recipes.llama.llama3 import llama32_1b_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.config import HFConversationDatasetConfig
from megatron.bridge.training.gpt_step import forward_step

cfg = llama32_1b_sft_config()
cfg.checkpoint.pretrained_checkpoint = "/checkpoints/llama32-1b"
cfg.checkpoint.load = None
cfg.dataset = HFConversationDatasetConfig(
    seq_length=cfg.model.seq_length,
    hf_processor_path=None,
    maker_name="text_chat",
    maker_kwargs={
        "path_or_dataset": "json",
        "data_files": {"train": "/tmp/bridge-hf-conversation/training.jsonl"},
        "split": "train",
    },
    val_maker_kwargs={
        "data_files": {"validation": "/tmp/bridge-hf-conversation/validation.jsonl"},
        "split": "validation",
    },
    test_maker_kwargs={
        "data_files": {"test": "/tmp/bridge-hf-conversation/test.jsonl"},
        "split": "test",
    },
    dataloader_type="single",
    do_validation=True,
    do_test=True,
)

finetune(config=cfg, forward_step_func=forward_step)
```

The base checkpoint must be a native Megatron checkpoint or a local Hugging Face full-model directory accepted by Bridge. Keep `cfg.model.seq_length` equal to `cfg.dataset.seq_length`. The builder repeats normalized examples to the sample counts requested by the training schedule, so iteration-based training remains the supported duration model.

## Connect multimodal data

For VLM or audio data, set the model processor path and choose its maker:

```python
cfg.dataset = HFConversationDatasetConfig(
    seq_length=4096,
    hf_processor_path="Qwen/Qwen2.5-VL-3B-Instruct",
    maker_name="cord_v2",
    maker_kwargs={"split": "train"},
    enable_in_batch_packing=False,
)
```

Use the corresponding recipe step, such as `vlm_step`, `qwen3_vl_step`, or an audio/omni step. Collators are inferred at runtime from processor type. Runtime callable collator overrides are intentionally not config fields; custom integrations can inject them into `HFConversationDatasetBuilder` without making serialized configs provider-specific.

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
| Source | `maker_name`, `maker_kwargs` | Registered maker and training-split inputs |
| Split overrides | `val_maker_kwargs`, `test_maker_kwargs`, `do_validation`, `do_test` | Per-split loading and optional split construction |
| Processor | `hf_processor_path`, inherited `trust_remote_code` | Model processor/tokenizer source and safe remote-code policy |
| Sequence | `seq_length`, `skip_getting_attention_mask_from_dataset` | Training shape and attention-mask handoff |
| Packing | `enable_in_batch_packing`, `defer_in_batch_packing_to_step`, `in_batch_packing_pad_to_multiple_of` | Collate- or model-step packing |
| Padding | `pad_to_max_length`, `pad_to_multiple_of` | Fixed or efficient batch padding |
| Loader | `dataloader_type`, `num_workers`, `data_sharding`, `pin_memory`, `drop_last`, `persistent_workers` | DataLoader behavior |

Maker kwargs are maker-specific and may select a dataset path, subset, split, data files, or prompt. Registered aliases include `text_chat`, `squad`, `gsm8k`, `cord_v2`, `cv17`, `llava_video_178k`, and `valor32k_avqa`; omni recipes reuse the appropriate vision/audio maker through the same Config + Builder path. All config mappings must contain serializable declarative values; processors, tokenizers, makers, and collator callables belong to the builder.

## Troubleshooting

- If `hf_processor_path` is unset, the build context must contain a training tokenizer.
- An unknown processor type means its multimodal collator is not registered; text `messages` rows select the shared text collator automatically.
- A maker must return a non-empty list of normalized dictionaries for every enabled split with a positive requested sample count.
- Unsupported in-batch packing fails early when a custom/model collator cannot accept packing metadata.
