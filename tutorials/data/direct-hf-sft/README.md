# Direct SFT Dataset Tutorial

Choose this path when you want to train directly from a Hugging Face dataset, including JSON/JSONL files loaded by the Hugging Face `json` dataset builder, without first materializing text-only SFT data. All inputs use `HFDatasetSourceConfig`, `DirectHFSFTDatasetConfig`, and `DirectHFSFTDatasetBuilder`. The path supports text, vision, video, audio, and omni examples, as well as collate-time in-batch packing. See the [multimodal Direct-HF tutorial](../multimodal-direct/README.md) for a complete local-image Qwen3-VL preparation and training run, and the [data tutorial overview](../README.md#which-sft-path-should-i-use) for workflow selection.

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
uv run python tutorials/data/direct-hf-sft/prepare_example_data.py \
    --output-dir /tmp/bridge-direct-hf-sft
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
    do_validation=True,
    do_test=True,
)
```

Use a JSON array or one object per line in JSONL. Rows may use `messages`, singular `conversation`, or legacy `conversations` with `from`/`value` turns.

For local multimodal rows, encode processor-supported media directly in the conversation. The exact typed-content schema is model-specific; for example, Qwen-VL accepts:

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "/data/vlm/receipts/0001.png"}, {"type": "text", "text": "Describe this receipt."}]}, {"role": "assistant", "content": [{"type": "text", "text": "A store receipt."}]}]}
```

Media references are row data interpreted by the selected model processor, so use a processor-supported schema and paths or URLs that every worker can resolve. The removed preloaded placeholder plus top-level media-list schema is not adapted automatically. For large local media collections that need dataset-managed sharding and asset loading, use Energon. Rows loaded by the Hugging Face `json` builder follow the same chat-template/loss-mask logic and model processor/collator path as Hub rows. CP/SP padding and supported in-batch packing settings remain fields of `DirectHFSFTDatasetConfig`.

### Migrate from `vlm-preloaded`

`PreloadedVLMConversationProvider` and the `vlm-preloaded` launcher selector have been removed. No deprecated provider or replacement local selector remains. Use `vlm-hf` for Hub datasets or files accepted by Hugging Face datasets, and use `vlm-energon` for Energon/WebDataset data.

| Removed setting | Unified setting |
| --- | --- |
| `--dataset vlm-preloaded` | `--dataset vlm-hf` or `--dataset vlm-energon` |
| `dataset.train_data_path` | `source=HFDatasetSourceConfig(path_or_dataset="json", load_kwargs={"data_files": {"train": ...}})` |
| `dataset.valid_data_path` | An explicit HF `validation_source`, or an Energon split |
| `dataset.test_data_path` | An explicit HF `test_source`, or an Energon split |
| `dataset.image_folder` | Encode processor-supported, worker-resolvable media references in conversation content; use an adapter-owned root or Energon where applicable |

The Direct HF path keeps chat rendering, assistant loss masking, processor-selected VLM/omni collation, CP/SP padding, and supported in-batch packing in `DirectHFSFTDatasetBuilder`; Hugging Face datasets owns source loading.

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

For JSON/JSONL accepted by Hugging Face datasets, replace only the source:

```python
cfg.dataset.source = HFDatasetSourceConfig(
    path_or_dataset="json",
    split="train",
    load_kwargs={"data_files": {"train": "/data/vlm/training.jsonl"}},
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

In-batch packing requires micro batch size greater than 1. `in_batch_packing_pad_to_multiple_of` is finalized from CP/SP constraints. Qwen3-VL sets `defer_in_batch_packing_to_step=True` because its step needs original visual tensors before generating packed metadata. Qwen3.5-VL opts out of this packing path, and Qwen3-Omni does not currently emit packed-sequence metadata. Pipeline or expert parallel training automatically enables fixed-length padding through `pad_to_max_length`. With CP, enable per-token loss and disable collective loss averaging.

## Available knobs

| Area | Knobs | Purpose |
| --- | --- | --- |
| Source | `HFDatasetSourceConfig` | Hub datasets or custom sources accepted by Hugging Face datasets, including its `json` loader |
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
- A source must return non-empty normalized rows for every enabled split with a positive requested sample count. File-based sources should provide explicit validation/test `data_files` and split sources, or disable those stages.
- Chat rows should follow the role ordering accepted by the selected model template. Rows with no trainable assistant tokens emit a warning and contribute zero loss.
- Unsupported in-batch packing fails early when a custom/model collator cannot accept packing metadata.
