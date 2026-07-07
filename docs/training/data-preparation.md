# Data Preparation

Megatron Bridge uses different dataset config objects for pretraining, text fine-tuning, and multimodal fine-tuning. Choose the data path by workflow first, then keep the dataset sequence length aligned with `model.seq_length`.

## Data Formats by Workflow

| Workflow | Data format | Config or provider | Required path fields |
|----------|-------------|--------------------|----------------------|
| LLM pretraining | Megatron binary `.bin`/`.idx` prefixes | `GPTDatasetConfig` | `data_path`, `blend`, or `blend_per_split` |
| LLM SFT or PEFT from local files | JSONL split files | `GPTSFTDatasetConfig` | `dataset_root` |
| LLM SFT or PEFT from Hugging Face datasets | Hugging Face rows converted to SFT JSONL, optionally packed | `GPTSFTDatasetConfig` | `hf_dataset.maker_name`, `hf_dataset.maker_kwargs`, optional `hf_dataset.output_root` |
| Direct Hugging Face conversations for text, vision, or audio | Maker-normalized rows processed at runtime | `HFConversationDatasetConfig` | `maker_name`, optional `hf_processor_path`, split maker kwargs |
| VLM SFT or PEFT | Energon/WebDataset, Hugging Face VLM dataset, or preloaded JSON | `HFConversationDatasetConfig`, Energon, or a specialized provider | `maker_name`, `hf_processor_path`, or provider-specific storage fields |

Use `seq_length` in Bridge examples and CLI overrides. `GPTDatasetConfig` also stores this value as Megatron Core's inherited `sequence_length` field internally, while `GPTSFTDatasetConfig` exposes `seq_length` directly.

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
    --dataset llm-pretrain \
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

The default text SFT dataset expects each JSONL record to contain prompt and answer fields compatible with the configured `prompt_template`. The common input/output format is:

```json
{"input": "Question: What is Megatron Bridge?", "output": "A PyTorch-native bridge for Megatron-Core workflows."}
```

Configure local JSONL data with `GPTSFTDatasetConfig.dataset_root`:

```python
from megatron.bridge.training.config import GPTSFTDatasetConfig

dataset = GPTSFTDatasetConfig(
    dataset_root="/data/sft_jsonl",
    seq_length=4096,
)
```

Launch the generic recipe runner with the preloaded local JSONL dataset type:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe llama32_1b_sft_1gpu_h100_bf16_config \
    --dataset llm-finetune-preloaded \
    dataset.dataset_root=/data/sft_jsonl \
    dataset.seq_length=4096 \
    checkpoint.pretrained_checkpoint=/checkpoints/base_model
```

For PEFT, use the PEFT recipe or set `cfg.peft`; the data layout stays the same. `checkpoint.pretrained_checkpoint` is required for the frozen base model, and `checkpoint.load` is used only when resuming adapter checkpoints.

For preparation schemas, offline packing, finite epochs, and a complete knob reference, see the [GPT SFT Dataset Tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/gpt-sft/README.md).

## Hugging Face Datasets for SFT and PEFT

Select a registered Hugging Face maker with `HFDatasetSourceConfig`. The builder downloads or reads the source dataset, converts rows into chat JSONL, and builds the result through the same SFT path used for local files. Maker inputs default to the tokenizer's Hugging Face chat template and support the same offline packed-sequence options.

```python
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.config import GPTSFTDatasetConfig, HFDatasetSourceConfig

dataset = GPTSFTDatasetConfig(
    seq_length=512,
    hf_dataset=HFDatasetSourceConfig(
        maker_name="squad",
        maker_kwargs={"path_or_dataset": "rajpurkar/squad", "split": "train"},
        val_proportion=0.1,
    ),
    seed=5678,
    do_validation=True,
    do_test=False,
    dataset_kwargs={"pad_to_max_length": True},
    enable_offline_packing=True,
    offline_packing_specs=PackedSequenceSpecs(packed_sequence_size=512),
)
```

If `hf_dataset.output_root` is omitted, the generated JSONL is cached under the NeMo datasets cache for the maker source. Keep `hf_dataset.rewrite=False` when later runs should reuse those files.

`FinetuningDatasetConfig`, `FinetuningDatasetBuilder`, and `HFTextSFTDatasetProvider` remain as deprecated compatibility adapters. New code should use `GPTSFTDatasetConfig` with `GPTSFTDatasetBuilder`; runtime objects such as tokenizers belong to the builder, not the serialized config.

The generic launcher provides preset Hugging Face text datasets through `--dataset llm-finetune`:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe llama32_1b_peft_1gpu_h100_bf16_config \
    --dataset llm-finetune \
    dataset.dataset_name=gsm8k \
    checkpoint.pretrained_checkpoint=/checkpoints/base_model
```

## Direct Hugging Face Conversation Data

`HFConversationDatasetConfig` is the direct maker path shared by text chat, VLM, and audio/omni recipes. Unlike the GPT SFT Hugging Face source above, it does not materialize rows into GPT SFT JSONL. `HFConversationDatasetBuilder` loads the maker and processor/tokenizer at runtime, selects the text or model collator, and repeats examples to the sample counts requested by the iteration schedule.

```python
from megatron.bridge.training.config import HFConversationDatasetConfig

dataset = HFConversationDatasetConfig(
    seq_length=4096,
    hf_processor_path="meta-llama/Llama-3.2-1B-Instruct",
    maker_name="text_chat",
    maker_kwargs={
        "path_or_dataset": "json",
        "data_files": {"train": "/data/chat/training.jsonl"},
        "split": "train",
    },
    val_maker_kwargs={
        "data_files": {"validation": "/data/chat/validation.jsonl"},
        "split": "validation",
    },
    do_test=False,
    enable_in_batch_packing=True,
)
```

Set `hf_processor_path` for multimodal or audio models and use the corresponding training step. Collator callables are runtime builder inputs, not serializable config fields. `HFConversationDatasetProvider` remains only as a deprecated compatibility adapter.

For text chat, `hf_processor_path=None` reuses the training tokenizer only when that tokenizer already defines the intended chat template. Otherwise select a vocabulary-compatible instruction processor explicitly, as above.

For text and multimodal schemas, split maker settings, in-batch packing, processor selection, and all available knobs, see the [Hugging Face Conversation Dataset Tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/hf-conversation/README.md).

## VLM Fine-Tuning Data

VLM recipes use either the canonical HF conversation Config + Builder path, Energon, or a specialized compatibility provider. The runtime builder or provider owns the processor needed to turn image, video, audio, and text records into batches.

For Energon/WebDataset data, create tar shards plus `.nv-meta` metadata and pass the dataset root to the recipe provider:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe qwen3_vl_8b_peft_1gpu_h100_bf16_energon_config \
    --dataset vlm-energon \
    --step_func qwen3_vl_step \
    dataset.path=/data/vlm_energon \
    checkpoint.pretrained_checkpoint=/checkpoints/qwen3_vl_base
```

For preloaded VLM JSON or JSONL, use records with `messages` or `conversations` plus media paths. Relative image and video paths are resolved against `dataset.image_folder` by `PreloadedVLMConversationProvider`:

```json
{"messages": [{"role": "user", "content": "<image>Describe the image."}, {"role": "assistant", "content": "A receipt."}], "images": ["receipt_0001.jpg"]}
```

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe qwen3_vl_8b_peft_1gpu_h100_bf16_config \
    --dataset vlm-preloaded \
    --step_func qwen3_vl_step \
    dataset.train_data_path=/data/vlm/train.jsonl \
    dataset.valid_data_path=/data/vlm/validation.jsonl \
    dataset.image_folder=/data/vlm/images \
    dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct \
    checkpoint.pretrained_checkpoint=/checkpoints/qwen3_vl_base
```

For a complete WebDataset/Energon preparation example, see [VALOR32K-AVQA Dataset Preparation Guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/valor32k-avqa/data-preparation.md).

## Checkpoint Conversion Reminder

Data preparation and checkpoint preparation are separate. From-scratch pretraining does not require a checkpoint. SFT and PEFT require base model weights through `checkpoint.pretrained_checkpoint` unless you are resuming from a complete native Megatron checkpoint with `checkpoint.load`.

`checkpoint.pretrained_checkpoint` may point to a native Megatron checkpoint directory, a specific native `iter_N` directory, or a local Hugging Face full-model directory. For production and multi-node jobs, converting Hugging Face checkpoints to native Megatron format first is usually more repeatable.
