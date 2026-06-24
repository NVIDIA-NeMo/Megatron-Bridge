# Data Preparation

Megatron Bridge uses different dataset config objects for pretraining, text fine-tuning, and multimodal fine-tuning. Choose the data path by workflow first, then keep the dataset sequence length aligned with `model.seq_length`.

## Data Formats by Workflow

| Workflow | Data format | Config or provider | Required path fields |
|----------|-------------|--------------------|----------------------|
| LLM pretraining | Megatron binary `.bin`/`.idx` prefixes | `GPTDatasetConfig` | `data_path`, `blend`, or `blend_per_split` |
| LLM SFT or PEFT from local files | JSONL split files | `FinetuningDatasetConfig` | `dataset_root` |
| LLM SFT or PEFT from Hugging Face datasets | Hugging Face rows converted to SFT JSONL, optionally packed | `HFTextSFTDatasetProvider` | `maker_name`, `maker_kwargs` |
| VLM SFT or PEFT | Energon/WebDataset, Hugging Face VLM dataset, or preloaded JSON | VLM `DatasetProvider` | Provider-specific fields such as `path`, `train_data_path`, or `image_folder` |

Use `seq_length` in Bridge examples and CLI overrides. `GPTDatasetConfig` also stores this value as Megatron Core's inherited `sequence_length` field internally, but `FinetuningDatasetConfig` uses `seq_length`.

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
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_config \
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

Configure local JSONL data with `FinetuningDatasetConfig.dataset_root`:

```python
from megatron.bridge.training.config import FinetuningDatasetConfig

dataset = FinetuningDatasetConfig(
    dataset_root="/data/sft_jsonl",
    seq_length=4096,
)
```

Launch the generic recipe runner with the preloaded local JSONL dataset type:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe llama32_1b_sft_config \
    --dataset llm-finetune-preloaded \
    dataset.dataset_root=/data/sft_jsonl \
    dataset.seq_length=4096 \
    checkpoint.pretrained_checkpoint=/checkpoints/base_model
```

For PEFT, use the PEFT recipe or set `cfg.peft`; the data layout stays the same. `checkpoint.pretrained_checkpoint` is required for the frozen base model, and `checkpoint.load` is used only when resuming adapter checkpoints.

## Hugging Face Datasets for SFT and PEFT

`HFTextSFTDatasetProvider` downloads or reads a Hugging Face dataset, converts rows into chat JSONL, and builds the result through the standard SFT dataset builder. This is the text-only Hugging Face path to use when offline packed sequences are needed.

```python
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_datasets import HFTextSFTDatasetProvider

dataset = HFTextSFTDatasetProvider(
    seq_length=512,
    maker_name="squad",
    maker_kwargs={"path_or_dataset": "rajpurkar/squad", "split": "train"},
    val_proportion=0.1,
    do_validation=True,
    do_test=False,
    dataset_kwargs={"pad_to_max_length": True},
    enable_offline_packing=True,
    offline_packing_specs=PackedSequenceSpecs(packed_sequence_size=512),
)
```

The generic launcher provides preset Hugging Face text datasets through `--dataset llm-finetune`:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe llama32_1b_peft_config \
    --dataset llm-finetune \
    dataset.dataset_name=gsm8k \
    checkpoint.pretrained_checkpoint=/checkpoints/base_model
```

## VLM Fine-Tuning Data

VLM recipes usually use a dataset provider instead of `FinetuningDatasetConfig`. The provider owns both the storage format and the processor needed to turn image, video, audio, and text records into batches.

For Energon/WebDataset data, create tar shards plus `.nv-meta` metadata and pass the dataset root to the recipe provider:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe qwen3_vl_8b_peft_energon_config \
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
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe qwen3_vl_8b_peft_config \
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
