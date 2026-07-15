# Hugging Face Multimodal SFT Tutorial

Use this path when image, video, audio, or omni conversations fit naturally in a hosted Hugging Face dataset or in
JSON/JSONL files supported by the Hugging Face `json` loader. `HFDatasetSourceConfig` and
`DirectHFSFTDatasetConfig` remain serializable; `DirectHFSFTDatasetBuilder` owns runtime processor, model collator,
and repeating dataset construction. No preloaded or separate local-file provider is involved.

The examples use Qwen3-VL 8B LoRA on one H100. For text-only rows, use the
[Hugging Face text-only tutorial](../hf-text-only/README.md). For large sharded media collections, use the
[Energon tutorial](../energon/README.md).

## Start with a hosted chat dataset

A hosted multimodal dataset can expose processor-ready conversations directly or use a registered source preset
whose schema adapter produces canonical chat rows. For example, the `medpix` preset owns the physical
[`mmoukouba/MedPix-VQA`](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) Hub path, its train/validation split
names, and its image-question-answer adapter:

```python
from megatron.bridge.data.builders import HFDatasetSourceConfig
from megatron.bridge.recipes.qwen_vl import qwen3_vl_8b_peft_config

cfg = qwen3_vl_8b_peft_config()
cfg.dataset.source = HFDatasetSourceConfig(dataset_name="medpix")
cfg.dataset.do_validation = True
cfg.dataset.do_test = False
cfg.dataset.enable_in_batch_packing = False
cfg.dataset.defer_in_batch_packing_to_step = False
```

The recipe already selects `DirectHFSFTDatasetConfig`, Qwen's processor, and assistant-only loss. Start without
in-batch packing; when packing is useful, use the normal collate-time path described below rather than deferred
packing. Replace only the source for another compatible preset or hosted dataset. When hosted rows are already named
`messages`, `conversation`, or `conversations`, set `path_or_dataset` and `split` directly; otherwise select a
registered `schema_adapter`. `load_kwargs` belongs to dataset loading, while `adapter_kwargs` belongs only to row
conversion.

See [Run the hosted MedPix preset with W&B](#run-the-hosted-medpix-preset-with-wb) for a complete launch.

## Load local JSON or JSONL through Hugging Face datasets

From the repository root:

```bash
export DATA_DIR=/tmp/bridge-hf-multimodal

uv run python tutorials/data/hf-multimodal/prepare_example_data.py \
  --output-dir "$DATA_DIR"
```

The script creates six deterministic 448×448 PNGs plus train, validation, and test JSONL files:

```text
/tmp/bridge-hf-multimodal/
├── images/
│   ├── red.png
│   └── ...
├── training.jsonl
├── validation.jsonl
└── test.jsonl
```

Each row stores media inside the conversation using the selected processor's schema:

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "/absolute/path/red.png"}, {"type": "text", "text": "What is the dominant color?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "The image is red."}]}]}
```

Paths and URLs are interpreted by the model processor and must be resolvable by every data worker. The removed preloaded `<image>` plus top-level `images` schema is not rewritten automatically.

Hosted and file-backed rows enter the same normalization, chat-template, loss-mask, processor, and collator path.
For production files, provide explicit `validation_source` and `test_source` configs for enabled splits, as shown
below.

## Import the model checkpoint

The training processor can come from the Hub, but model weights should be imported once into a native Megatron checkpoint:

```bash
export MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
export WORKSPACE=${WORKSPACE:-/workspace}
export PRETRAINED_CHECKPOINT="$WORKSPACE/models/Qwen3-VL-8B-Instruct"

./scripts/conversion/convert.sh import \
  --hf-model "$MODEL_ID" \
  --megatron-path "$PRETRAINED_CHECKPOINT"
```

The import requires enough host/GPU memory for the 8B checkpoint and access to the model files through `HF_HOME`/`HF_TOKEN` when needed.

## Run a one-GPU local-data LoRA smoke

The Qwen3-VL 8B PEFT recipe is a one-GPU recipe. Start unpacked with one iteration and no evaluation:

```bash
export WANDB_MODE=disabled
export OUTPUT_DIR="$WORKSPACE/results/qwen3-vl-hf-local-smoke"

uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe qwen3_vl_8b_peft_config \
  --dataset vlm-hf \
  --step_func vlm_step \
  --peft_scheme lora \
  --seq_length 1024 \
  checkpoint.pretrained_checkpoint="$PRETRAINED_CHECKPOINT" \
  checkpoint.load=null \
  checkpoint.save="$OUTPUT_DIR/checkpoints" \
  checkpoint.save_interval=1 \
  train.train_iters=1 \
  train.global_batch_size=1 \
  train.micro_batch_size=1 \
  dataset.source.dataset_name=null \
  dataset.source.path_or_dataset=json \
  dataset.source.split=train \
  "dataset.source.load_kwargs={data_files:{train:${DATA_DIR}/training.jsonl}}" \
  dataset.hf_processor_path="$MODEL_ID" \
  dataset.do_validation=False \
  dataset.do_test=False \
  dataset.num_workers=0 \
  dataset.persistent_workers=False \
  dataset.enable_in_batch_packing=False \
  dataset.defer_in_batch_packing_to_step=False \
  logger.log_interval=1
```

Success means the launcher loads the native checkpoint, the Qwen processor opens the local PNGs, and iteration 1 reports a finite loss before writing the adapter checkpoint.

`--dataset vlm-hf` selects the generic `DirectHFSFTDatasetConfig`; the remaining overrides select Qwen's processor and the local JSON source. Passing the selector also makes `--seq_length` update both model and dataset lengths.

## Run the hosted MedPix preset with W&B

`medpix` is a built-in source preset for
[`mmoukouba/MedPix-VQA`](https://huggingface.co/datasets/mmoukouba/MedPix-VQA). It owns the Hub path,
the MedPix schema adapter, and the published train/validation split names. The dataset has no test split, so disable
test data. With `WANDB_API_KEY` configured (or after `wandb login`), this command runs three Qwen3-VL LoRA steps,
one validation step, checkpoint save, and online metric logging:

```bash
export OUTPUT_DIR="$WORKSPACE/results/qwen3-vl-medpix-direct"

uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe qwen3_vl_8b_peft_config \
  --dataset vlm-hf \
  --step_func vlm_step \
  --peft_scheme lora \
  --seq_length 1024 \
  checkpoint.pretrained_checkpoint="$PRETRAINED_CHECKPOINT" \
  checkpoint.load=null \
  checkpoint.save="$OUTPUT_DIR/checkpoints" \
  checkpoint.save_interval=3 \
  train.train_iters=3 \
  train.global_batch_size=1 \
  train.micro_batch_size=1 \
  validation.eval_interval=3 \
  validation.eval_iters=1 \
  validation.eval_micro_batch_size=1 \
  dataset.source.dataset_name=medpix \
  dataset.hf_processor_path="$MODEL_ID" \
  dataset.do_validation=True \
  dataset.do_test=False \
  dataset.num_workers=0 \
  dataset.persistent_workers=False \
  dataset.enable_in_batch_packing=False \
  dataset.defer_in_batch_packing_to_step=False \
  scheduler.lr_warmup_iters=0 \
  scheduler.lr_decay_iters=3 \
  logger.log_interval=1 \
  logger.tensorboard_log_interval=1 \
  logger.log_throughput_to_tensorboard=True \
  logger.log_memory_to_tensorboard=True \
  logger.wandb_project=bridge-qwen3-vl-medpix \
  logger.wandb_exp_name=direct-hf-medpix \
  logger.wandb_save_dir="$OUTPUT_DIR/wandb"
```

The current Direct-HF implementation materializes each requested split before training. The first MedPix run downloads
the complete Hub shards and needs substantially more host memory than the three target training samples alone imply.
For a smaller training-only data-path check, add the following overrides; the nested quotes around the sliced split are
required by Hydra:

```bash
'dataset.source.split="train[:16]"' \
dataset.do_validation=False \
validation.eval_iters=0
```

Use the normal preset-derived validation split for a real training run. A CLI mapping cannot construct an optional
`validation_source` that started as `None`; define custom per-split sources in Python as shown next.

## Configure explicit validation and test files

File-backed sources do not invent missing splits. For a production Python recipe, give every enabled split its own serializable source:

```python
from megatron.bridge.data.builders import DirectHFSFTDatasetConfig, HFDatasetSourceConfig

cfg.dataset = DirectHFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    hf_processor_path="Qwen/Qwen3-VL-8B-Instruct",
    source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="train",
        load_kwargs={"data_files": {"train": "/data/vlm/training.jsonl"}},
    ),
    validation_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="validation",
        load_kwargs={"data_files": {"validation": "/data/vlm/validation.jsonl"}},
    ),
    test_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="test",
        load_kwargs={"data_files": {"test": "/data/vlm/test.jsonl"}},
    ),
)
```

Disable `do_validation` or `do_test` when that split is intentionally absent.

## Enable collate-time in-batch packing

The unified multimodal path normally packs samples in the model collator and uses the generic `vlm_step`. Packing
requires a configured micro batch greater than one:

```bash
train.micro_batch_size=2 \
train.global_batch_size=2 \
dataset.enable_in_batch_packing=True \
dataset.defer_in_batch_packing_to_step=False
```

Do not apply text-only offline packed-SFT settings to this path. With context parallelism, keep `model.calculate_per_token_loss=True` and `ddp.average_in_collective=False`; `ConfigContainer` derives the required CP/SP packing multiple.
Deferred packing is a model-specific compatibility mode for specialized steps such as `qwen3_vl_step`; it is not
needed for the recommended Qwen3-VL Direct-HF workflow.

## Migrate from `vlm-preloaded`

`PreloadedVLMConversationProvider` and the `vlm-preloaded` launcher selector have been removed. No deprecated
provider or replacement local selector remains. Use `vlm-hf` for hosted datasets or files accepted by Hugging Face
datasets, and use `vlm-energon` for Energon/WebDataset data.

| Removed setting | Supported setting |
| --- | --- |
| `--dataset vlm-preloaded` | `--dataset vlm-hf` or `--dataset vlm-energon` |
| `dataset.train_data_path` | `source=HFDatasetSourceConfig(path_or_dataset="json", load_kwargs={"data_files": {"train": ...}})` |
| `dataset.valid_data_path` | An explicit HF `validation_source`, or an Energon split |
| `dataset.test_data_path` | An explicit HF `test_source`, or an Energon split |
| `dataset.image_folder` | Processor-supported, worker-resolvable media references in conversation content; an adapter-owned root; or Energon |

The Hugging Face path keeps chat rendering, assistant loss masking, processor-selected VLM/omni collation, CP/SP
padding, and supported in-batch packing in `DirectHFSFTDatasetBuilder`; Hugging Face datasets owns source loading.
The old placeholder plus top-level media-list schema is not rewritten automatically.

## Config and runtime ownership

| Setting | Meaning | Owner |
| --- | --- | --- |
| `source.*` | Hub/custom/JSON source and optional schema adapter | Serializable config |
| `hf_processor_path` | Processor identity | Serializable config; loaded by builder |
| typed `content` media | Processor-native image/video/audio references | Dataset rows |
| `pad_*`, `enable_in_batch_packing` | Batch shape policy | Serializable config |
| processor, collator, `DirectSFTDataset` | Runtime objects | `DirectHFSFTDatasetBuilder` |

## Troubleshooting

- `Unknown split`: provide an explicit `validation_source`/`test_source`, or disable the missing stage.
- `No VLM collate function is registered`: `hf_processor_path` resolved to a processor type without a registered model collator.
- Image cannot be opened: use absolute paths, shared paths, or URLs reachable from every worker.
- All tokens are masked: verify the conversation has an assistant turn accepted by the processor chat template.
- Packing validation fails: use micro batch size greater than one, keep `defer_in_batch_packing_to_step=False`, and
  launch with `--step_func vlm_step`.
- OOM on one GPU: reduce sequence length first; this tutorial uses LoRA because full Qwen3-VL 8B SFT is a two-GPU recipe.

For hosted text/chat sources and prompt-completion preprocessing, see the
[Hugging Face text-only tutorial](../hf-text-only/README.md).
