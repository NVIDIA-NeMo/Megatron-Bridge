# Direct SFT Dataset Tutorial

Choose this path when you want to train directly from hosted Hugging Face rows or local JSON/JSONL conversations without first materializing text-only SFT data. Both source types use `DirectHFSFTDatasetConfig` and `DirectHFSFTDatasetBuilder`. The path supports text, vision, video, audio, and omni examples, as well as collate-time in-batch packing. See the [data tutorial overview](../README.md#which-sft-path-should-i-use) for a comparison with transitional materialized text-only SFT and Energon.

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

## Use local conversation JSON or JSONL

Generate small `messages` JSONL files from the repository root:

```bash
uv run python tutorials/data/direct-hf-sft/prepare_example_data.py \
    --output-dir /tmp/bridge-direct-hf-sft
```

Native chat rows look like this:

```json
{"messages": [{"role": "user", "content": "What belongs in config?"}, {"role": "assistant", "content": "Validated declarative data."}]}
```

Select each local split explicitly. The source config stays serializable; the builder opens and normalizes the file at runtime:

```python
from megatron.bridge.data.builders import LocalConversationDatasetSourceConfig

cfg.dataset = DirectHFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
    hf_processor_path="meta-llama/Llama-3.2-1B-Instruct",
    source=LocalConversationDatasetSourceConfig(path="/tmp/bridge-direct-hf-sft/training.jsonl"),
    validation_source=LocalConversationDatasetSourceConfig(
        path="/tmp/bridge-direct-hf-sft/validation.jsonl"
    ),
    test_source=LocalConversationDatasetSourceConfig(path="/tmp/bridge-direct-hf-sft/test.jsonl"),
    do_validation=True,
    do_test=True,
)
```

JSON files may contain a record list, one record, or a list under `data`, `examples`, or `records`; set `records_key` for another wrapper. JSONL files contain one record per line. Rows may use `messages`, singular `conversation`, or legacy `conversations` with `from`/`value` turns.

For local multimodal rows, use typed content directly or legacy placeholders plus top-level media lists:

```json
{"messages": [{"role": "user", "content": "<image>Describe this receipt."}, {"role": "assistant", "content": "A store receipt."}], "images": ["receipts/0001.png"]}
```

Set `media_root` to resolve relative image, video, and audio paths without embedding machine-specific absolute paths in the data:

```python
source=LocalConversationDatasetSourceConfig(
    path="/data/vlm/training.jsonl",
    media_root="/data/vlm/media",
)
```

Local rows then follow the same chat-template/loss-mask logic and model processor/collator path as hosted rows. CP/SP padding and supported in-batch packing settings remain fields of `DirectHFSFTDatasetConfig`, independent of source type.

### Migrate from `vlm-preloaded`

`PreloadedVLMConversationProvider` and the `vlm-preloaded` launcher selector have been removed. Use the unified builder instead; no deprecated provider alias remains.

| Removed setting | Unified setting |
| --- | --- |
| `--dataset vlm-preloaded` | `--dataset vlm-local` |
| `dataset.train_data_path` | `dataset.source.path` |
| `dataset.valid_data_path` | `dataset.validation_source.path` |
| `dataset.test_data_path` | `dataset.test_source.path` |
| `dataset.image_folder` | `dataset.source.media_root` (and the corresponding split source when needed) |

The replacement keeps chat rendering, assistant loss masking, processor-selected VLM/omni collation, CP/SP padding, and supported in-batch packing in `DirectHFSFTDatasetBuilder`; only source loading moves into `LocalConversationDatasetSourceConfig`.

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

For local multimodal data, replace only the source:

```python
cfg.dataset.source = LocalConversationDatasetSourceConfig(
    path="/data/vlm/training.jsonl",
    media_root="/data/vlm/media",
)
cfg.dataset.do_validation = False
cfg.dataset.do_test = False
```

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
| Source | `HFDatasetSourceConfig` or `LocalConversationDatasetSourceConfig` | Hosted/custom Hugging Face rows or local JSON/JSONL conversations |
| Local files | `path`, optional `media_root` and `records_key` | Split file, relative-media root, and optional JSON wrapper key |
| Adaptation | `source.schema_adapter`, `source.adapter_kwargs` | Optional conversion to canonical chat or prompt-completion rows; presets own their adapter |
| Preprocessing | `ChatSFTPreprocessingConfig`, `PromptCompletionSFTPreprocessingConfig` | Chat-template loss policy or raw paired-text formatting/loss policy |
| Split overrides | `validation_source`, `test_source`, `do_validation`, `do_test` | Per-split loading and optional split construction |
| Processor | `hf_processor_path`, inherited `trust_remote_code` | Model processor/tokenizer source and safe remote-code policy |
| Sequence | `seq_length`, `skip_getting_attention_mask_from_dataset` | Training shape and attention-mask handoff |
| Packing | `enable_in_batch_packing`, `defer_in_batch_packing_to_step`, `in_batch_packing_pad_to_multiple_of` | Collate- or model-step packing |
| Padding | `pad_to_max_length`, `pad_to_multiple_of` | Fixed or efficient batch padding |
| Loader | `dataloader_type`, `num_workers`, `data_sharding`, `pin_memory`, `drop_last`, `persistent_workers` | DataLoader behavior |

Built-in registered adapters include `squad`, `gsm8k`, `cord_v2`, and `cv17`; native rows matching the selected preprocessing schema need no adapter. All config mappings must contain serializable declarative values; processors, tokenizers, and collator callables belong to the builder.

## Troubleshooting

- If `hf_processor_path` is unset, the build context must contain a training tokenizer.
- An unknown processor type means its multimodal collator is not registered; text rows select a shared collator from the preprocessing config automatically.
- A source must return non-empty normalized rows for every enabled split with a positive requested sample count. Local files do not derive validation or test splits; set explicit split sources or disable those stages.
- Chat rows should follow the role ordering accepted by the selected model template. Rows with no trainable assistant tokens emit a warning and contribute zero loss.
- Unsupported in-batch packing fails early when a custom/model collator cannot accept packing metadata.
