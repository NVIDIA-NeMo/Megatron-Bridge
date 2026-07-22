# Data Preparation

Megatron Bridge uses different dataset config objects for pretraining, text fine-tuning, and multimodal fine-tuning. Choose the data path by workflow first, then keep the dataset sequence length aligned with `model.seq_length`.

## Data Formats by Workflow

| Workflow | Status | Data format | Config or provider |
|----------|--------|-------------|--------------------|
| LLM pretraining | Recommended | Megatron binary `.bin`/`.idx` prefixes | `GPTDatasetConfig` |
| Text SFT or PEFT, processed at runtime | Recommended | Hosted Hugging Face rows or local JSON/JSONL loaded through Hugging Face datasets | [Hugging Face text-only](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/hf-text-only/README.md) through `DirectHFSFTDatasetConfig` + builder |
| Text SFT or PEFT, prepared data | Planned; not available yet | Pretokenized `.bin`/`.idx` | Future Issue #4664 prepared-SFT builder |
| Text SFT or PEFT, transitional prepared data | Supported until `.bin`/`.idx` replacement | Local/materialized JSONL and optional packed Parquet | `GPTSFTDatasetConfig` |
| Multimodal SFT or PEFT | Recommended | Hosted Hugging Face rows or local conversation JSON/JSONL | [Hugging Face multimodal](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/hf-multimodal/README.md) through `DirectHFSFTDatasetConfig` + builder |
| Large sharded multimodal training | Recommended | WebDataset/Energon | [Multimodal Energon](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/energon/README.md) through `EnergonDatasetConfig` + builder |

All canonical dataset configs expose `seq_length` in Python and through CLI overrides. During finalization,
`GPTDatasetConfig` copies that value to Megatron Core's internal `sequence_length` field.

## LLM Pretraining Data

LLM pretraining uses Megatron binary indexed datasets. Each dataset is represented by a prefix with matching `.bin` and `.idx` files:

```text
/data/dclm/preprocessed_text_document.bin
/data/dclm/preprocessed_text_document.idx
```

Pass the prefix without the `.bin` or `.idx` suffix:

```python
from megatron.bridge.training.config import GPTDatasetConfig

dataset = GPTDatasetConfig(
    seq_length=8192,
    data_path="/data/dclm/preprocessed_text_document",
    split="9999,8,2",
    random_seed=1234,
    reset_attention_mask=False,
    reset_position_ids=False,
    eod_mask_loss=False,
)
```

The CLI-friendly `data_path` field is converted to Megatron Core's `blend` field during config finalization. For weighted multi-dataset training, use either a flattened `data_path` list with weights and prefixes or set `blend`/`blend_per_split` directly.

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_1gpu_h100_bf16_config \
    --mode pretrain --dataset megatron-indexed \
    dataset.data_path=/data/dclm/preprocessed_text_document \
    dataset.seq_length=8192
```

To create Megatron binary data from JSONL text, use the Megatron-LM `tools/preprocess_data.py` workflow. The DCLM tutorial shows a complete download, merge, shuffle, and preprocessing flow: [DCLM Data Preprocessing Tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/dclm/README.md).

## Local JSONL SFT and PEFT Data

Text SFT and PEFT use a directory containing split files named `training.jsonl`, `validation.jsonl`, and optionally `test.jsonl`:

```text
/data/sft_jsonl/
  training.jsonl
  validation.jsonl
  test.jsonl
```

The default materialized backend accepts `input`/`output` prompt-completion rows. Its explicit preprocessing config tokenizes the two fields without calling a chat template:

```json
{"input": "Question: What is Megatron Bridge?", "output": "A PyTorch-native bridge for Megatron-Core workflows."}
```

Configure local JSONL data with `GPTSFTDatasetConfig.dataset_root`:

```python
from megatron.bridge.data.builders import GPTSFTDatasetConfig, PromptCompletionSFTPreprocessingConfig

dataset = GPTSFTDatasetConfig(
    dataset_root="/data/sft_jsonl",
    seq_length=4096,
    preprocessing=PromptCompletionSFTPreprocessingConfig(
        prompt_column="input",
        completion_column="output",
        separator=" ",
        loss_mode="completion",
    ),
)
```

To use local JSONL, select the public `local-jsonl` dataset. This replaces the recipe's dataset object with a
`GPTSFTDatasetConfig` configured for prompt-completion JSONL:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --mode sft --dataset local-jsonl \
    dataset.dataset_root=/data/sft_jsonl \
    dataset.seq_length=4096 \
    checkpoint.pretrained_checkpoint=/checkpoints/base_model
```

For PEFT, use the PEFT recipe or set `cfg.peft`; the data layout stays the same. `checkpoint.pretrained_checkpoint` is required for the frozen base model, and `checkpoint.load` is used only when resuming adapter checkpoints.

For preparation schemas, offline packing, finite epochs, and a complete knob reference, see the [text-only SFT dataset tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/text-only-sft/README.md).

## Hugging Face Datasets for SFT and PEFT

Select a Hugging Face dataset with `HFDatasetSourceConfig`. A built-in `dataset_name` preset owns the physical Hub path, subset, and schema adapter. This avoids repeating coupled metadata such as the SQuAD path and adapter in every recipe. Bridge never infers a schema from an arbitrary Hub path because one repository can expose multiple subsets and schemas. For a custom source, set `path_or_dataset`; rows already matching the selected chat or prompt-completion schema need no adapter.

```python
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.data.builders import (
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)

dataset = GPTSFTDatasetConfig(
    seq_length=512,
    hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
    preprocessing=PromptCompletionSFTPreprocessingConfig(separator=" ", loss_mode="completion"),
    hf_validation_proportion=0.1,
    seed=5678,
    do_validation=True,
    do_test=False,
    dataset_kwargs={"pad_to_max_length": True},
    enable_offline_packing=True,
    offline_packing_specs=PackedSequenceSpecs(packed_sequence_size=512),
)
```

If `hf_output_root` is omitted, the generated JSONL is cached under the NeMo datasets cache for the source. Keep `hf_rewrite=False` when later runs should reuse those files. With builder-managed offline packing, `hf_rewrite=True` regenerates both normalized JSONL and packed artifacts; explicit packed output paths are rejected in this mode to avoid stale data.

> **Deprecated compatibility APIs:** `FinetuningDatasetConfig` and `FinetuningDatasetBuilder` remain only for existing callers. New code must use `GPTSFTDatasetConfig` with `GPTSFTDatasetBuilder`; runtime objects such as tokenizers belong to the builder, not the serialized config.

The generic launcher accepts public Hugging Face dataset names directly:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe llama32_1b_peft_1gpu_h100_bf16_config \
    --mode lora --dataset gsm8k \
    checkpoint.pretrained_checkpoint=/checkpoints/base_model
```

## Direct Hugging Face SFT Data

`DirectHFSFTDatasetConfig` is the direct source path shared by text, VLM, and audio/omni recipes. Every source uses `HFDatasetSourceConfig`: select a Hub dataset, a custom Hugging Face dataset, or the Hugging Face `json` loader for JSON/JSONL files. `DirectHFSFTDatasetBuilder` binds the processor/tokenizer and collator and repeats examples to the sample counts requested by the iteration schedule. The source config remains serializable; Hugging Face datasets owns source loading, while processors, collators, and runtime datasets remain builder responsibilities.

```python
from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    DirectHFSFTDatasetConfig,
    HFDatasetSourceConfig,
)

dataset = DirectHFSFTDatasetConfig(
    seq_length=4096,
    preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
    hf_processor_path="meta-llama/Llama-3.2-1B-Instruct",
    source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="train",
        load_kwargs={"data_files": {"train": "/data/chat/training.jsonl"}},
    ),
    validation_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="validation",
        load_kwargs={"data_files": {"validation": "/data/chat/validation.jsonl"}},
    ),
    do_test=False,
    enable_in_batch_packing=True,
)
```

Set `hf_processor_path` for multimodal or audio models and use the corresponding training step. Media references in JSON rows must be resolvable by every worker; for large local media collections that need managed sharding and asset loading, use Energon. Collator callables are runtime builder inputs, not serializable config fields.

For paired text that must not use a model chat template, select prompt-completion preprocessing instead. The prompt and completion are tokenized separately, and `loss_mode="completion"` masks the prompt:

```python
from megatron.bridge.data.builders import PromptCompletionSFTPreprocessingConfig

dataset.preprocessing = PromptCompletionSFTPreprocessingConfig(
    prompt_column="prompt",
    completion_column="completion",
    separator="\n",
    loss_mode="completion",
)
```

Structured multi-turn rows require chat preprocessing; Bridge does not silently flatten them into prompt-completion text.

Known semantic datasets should use their preset name, for example `squad`, `gsm8k`, `openmathinstruct2`, `cord_v2`, `raven`, `rdr`, `medpix`, `cv17`, or `llava_video_178k`. Do not combine `dataset_name` with `path_or_dataset`, `subset`, or `schema_adapter`; a preset owns those coupled fields. `split`, `load_kwargs`, and `adapter_kwargs` remain available for split selection and declarative runtime options such as a video root. Presets validate published split support: `raven`, `rdr`, and `llava_video_178k` are train-only; `medpix` and `squad` have no test split; `gsm8k` has no validation split; and OpenMathInstruct-2 exposes training variants only. Disable unsupported derived validation/test splits or supply explicit compatible sources.

For text chat, `hf_processor_path=None` reuses the training tokenizer only when that tokenizer already defines the intended chat template. Otherwise select a vocabulary-compatible instruction processor explicitly, as above.

### SFT Implementation Layout

The package structure separates declarative construction, runtime storage, source loading, and batch collation:

| Responsibility | Module |
| --- | --- |
| Materialized JSONL/offline-packed config and builder | `megatron.bridge.data.builders.gpt_sft` |
| Direct Hugging Face config and builder | `megatron.bridge.data.builders.direct_hf_sft` |
| GPT SFT runtime datasets | `megatron.bridge.data.datasets.gpt_sft` |
| Direct normalized-example runtime dataset | `megatron.bridge.data.datasets.direct_sft` |
| Hugging Face source loading and schema adapters | `megatron.bridge.data.sources.hf`, `megatron.bridge.data.sources.hf_adapters` |
| Shared text SFT collators | `megatron.bridge.data.collators.sft` |

The direct runtime dataset is named `DirectSFTDataset` because it receives normalized examples and has no Hugging Face loading responsibility. Source-specific work remains in the builder and source modules.

For focused walkthroughs, use the [Hugging Face text-only tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/hf-text-only/README.md) for chat and prompt-completion rows, or the [Hugging Face multimodal tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/hf-multimodal/README.md) for hosted and local media conversations.

## VLM Fine-Tuning Data

VLM recipes use either the canonical Direct Hugging Face SFT Config + Builder path or Energon. No separate preloaded/local conversation provider is supported.

For Energon/WebDataset data, create tar shards plus `.nv-meta` metadata and pass the dataset root to a recipe that already defines `EnergonDatasetConfig` with the appropriate task-encoder config:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe qwen3_vl_8b_peft_1gpu_h100_bf16_energon_config \
    --mode lora --step-func vlm_step \
    dataset.path=/data/vlm_energon \
    dataset.defer_in_batch_packing_to_step=False \
    checkpoint.pretrained_checkpoint=/checkpoints/qwen3_vl_base
```

`EnergonDatasetConfig` contains only serializable data settings. `EnergonDatasetBuilder` loads the HF processor/tokenizer and constructs the model-specific task encoder at runtime. Shipped Qwen-VL and Nemotron Omni recipes use `QwenVLEnergonTaskEncoderConfig` and `NemotronOmniEnergonTaskEncoderConfig`; custom model integrations can use `HFEnergonTaskEncoderConfig`. Override encoder-specific values through the nested config, for example `dataset.task_encoder.max_num_images=4`. Set `dataset.trust_remote_code` for the configured HF assets; an explicit `dataset.task_encoder.trust_remote_code` value takes precedence. Select a recipe that already contains the required Energon task-encoder config; the launcher does not create one for an unrelated recipe.

For JSON or JSONL accepted by the Hugging Face `json` loader, use records with `messages`, `conversation`, or legacy `conversations`. Multimodal content must follow the selected model processor's schema; for example, Qwen-VL accepts an inline typed image with a worker-resolvable path:

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "/data/vlm/receipt_0001.jpg"}, {"type": "text", "text": "Describe the image."}]}, {"role": "assistant", "content": [{"type": "text", "text": "A receipt."}]}]}
```

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe qwen3_vl_8b_peft_1gpu_h100_bf16_config \
    --mode lora --step-func vlm_step \
    --dataset local-vlm \
    dataset.source.load_kwargs.data_files.train=/data/vlm/train.jsonl \
    dataset.validation_source.load_kwargs.data_files.validation=/data/vlm/validation.jsonl \
    dataset.do_validation=true \
    dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct \
    dataset.defer_in_batch_packing_to_step=False \
    checkpoint.pretrained_checkpoint=/checkpoints/qwen3_vl_base
```

The `local-vlm` preset supplies override-ready `HFDatasetSourceConfig(path_or_dataset="json", load_kwargs={"data_files": ...})` objects; it does not restore the removed `vlm-preloaded` provider. Validation and test are disabled by default. Set their source path together with `dataset.do_validation=true` or `dataset.do_test=true` to enable them. There is no `image_folder` compatibility field and the old placeholder plus top-level media-list schema is not rewritten: encode media using the selected processor's supported conversation schema, use an adapter-owned root where available, or use Energon.

For complete Qwen3-VL preparation and launch commands, see the [Hugging Face multimodal tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/hf-multimodal/README.md) or the [multimodal Energon tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/energon/README.md). [VALOR32K-AVQA](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/valor32k-avqa/data-preparation.md) is the larger audio-video example.

## Checkpoint Conversion Reminder

Data preparation and checkpoint preparation are separate. From-scratch pretraining does not require a checkpoint. SFT and PEFT require base model weights through `checkpoint.pretrained_checkpoint` unless you are resuming from a complete native Megatron checkpoint with `checkpoint.load`.

`checkpoint.pretrained_checkpoint` may point to a native Megatron checkpoint directory, a specific native `iter_N` directory, or a local Hugging Face full-model directory. For production and multi-node jobs, converting Hugging Face checkpoints to native Megatron format first is usually more repeatable.
