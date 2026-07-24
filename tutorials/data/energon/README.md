# Multimodal Energon Tutorial

Use Energon when media should be packaged into sharded WebDataset tar files with resumable, distributed loading. `EnergonDatasetConfig` remains serializable; `EnergonDatasetBuilder` loads the processor and constructs the configured task encoder and dataloaders at runtime.

This tutorial prepares a tiny image/chat dataset and runs the same one-GPU Qwen3-VL 8B LoRA baseline as the
[Hugging Face multimodal tutorial](../hf-multimodal/README.md). The difference is the data source, not the model
step.

## 1. Build and index a tiny dataset

Megatron-Energon 7 or newer is recommended. From the repository root:

```bash
export ENERGON_PATH=/tmp/bridge-energon-qwen

uv run python tutorials/data/energon/prepare_example_data.py \
  --output-dir "$ENERGON_PATH" \
  --num-workers 2
```

The script performs the complete preparation pipeline:

1. Writes `train-shard-000000.tar` and `val-shard-000000.tar`.
2. Calls Energon's preparation API with explicit train/val filename regexes.
3. Writes `.nv-meta/dataset.yaml` for Bridge's `ChatMLWebdataset` loader.

The result is directly consumable by `EnergonDatasetBuilder`:

```text
/tmp/bridge-energon-qwen/
├── train-shard-000000.tar
├── train-shard-000000.tar.idx
├── val-shard-000000.tar
├── val-shard-000000.tar.idx
└── .nv-meta/
    ├── dataset.yaml
    ├── split.yaml
    ├── .info.json (Energon 7.4+) or .info.yaml (earlier 7.x)
    └── index.sqlite
```

Each WebDataset sample contains matching-key members:

```text
train-000000.image.png
train-000000.conversation.json
```

The conversation uses typed image content without a path because `field_map.imgs` supplies the decoded image:

```json
[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is the dominant color?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "The image is red."}]}]
```

The generated sample loader is:

```yaml
__module__: megatron.bridge.data.energon.task_encoder_utils
__class__: ChatMLWebdataset
field_map:
  imgs: image.png
  conversation: conversation.json
subflavors: {}
```

## 2. Import the Qwen3-VL checkpoint

Reuse the native checkpoint from the Hugging Face multimodal tutorial, or import it once:

```bash
export MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
export WORKSPACE=${WORKSPACE:-/workspace}
export PRETRAINED_CHECKPOINT="$WORKSPACE/models/Qwen3-VL-8B-Instruct"

./scripts/conversion/convert.sh import \
  --hf-model "$MODEL_ID" \
  --megatron-path "$PRETRAINED_CHECKPOINT"
```

## 3. Run a one-GPU LoRA smoke

The Energon recipe already contains `EnergonDatasetConfig` and `QwenVLEnergonTaskEncoderConfig`. The selector validates that model-specific config instead of guessing a runtime encoder:

```bash
export WANDB_MODE=disabled
export OUTPUT_DIR="$WORKSPACE/results/qwen3-vl-energon-smoke"

uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe qwen3_vl_8b_peft_energon_config \
  --dataset vlm-energon \
  --step_func vlm_step \
  --peft_scheme lora \
  --seq_length 1024 \
  checkpoint.pretrained_checkpoint="$PRETRAINED_CHECKPOINT" \
  checkpoint.load=null \
  checkpoint.save="$OUTPUT_DIR/checkpoints" \
  checkpoint.save_interval=1 \
  train.train_iters=1 \
  train.global_batch_size=2 \
  train.micro_batch_size=2 \
  validation.eval_interval=1 \
  validation.eval_iters=1 \
  validation.eval_micro_batch_size=2 \
  dataset.path="$ENERGON_PATH" \
  dataset.micro_batch_size=2 \
  dataset.num_workers=2 \
  dataset.num_val_workers=2 \
  dataset.enable_in_batch_packing=False \
  dataset.defer_in_batch_packing_to_step=False \
  logger.log_interval=1
```

Success means Energon restores both split loaders, the Qwen task encoder decodes and normalizes the PNGs, iteration 1 and validation report finite loss, and the adapter checkpoint is written.

Energon owns its loader micro batch, so `dataset.micro_batch_size`, `train.micro_batch_size`, and `validation.eval_micro_batch_size` must match. Energon currently exposes train and validation iterators, not a test iterator.
The two data workers per split are a realistic starting point for this tiny smoke; tune the counts for storage and
CPU capacity. Set them to zero only while debugging worker-process behavior.

## 4. Convert a MedPix smoke set

To compare Energon against the Hugging Face `medpix` preset with real medical images, package fixed slices of the same
[`mmoukouba/MedPix-VQA`](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) source:

```bash
export ENERGON_PATH=/tmp/bridge-energon-medpix

uv run python tutorials/data/energon/prepare_medpix_data.py \
  --output-dir "$ENERGON_PATH" \
  --train-rows 16 \
  --validation-rows 8 \
  --num-workers 2
```

The helper verifies that MedPix `image_id` values decode as PIL images, writes processor-compatible conversations,
indexes `train` and `val` shards, and records the selected slices in `manifest.json`. Hugging Face may still download
the complete underlying Parquet shards on the first invocation. The small row defaults are intended for correctness
smokes, not meaningful medical-model evaluation.

Run the one-GPU command from the previous section with this `ENERGON_PATH`. For an online three-step comparison, set
`WANDB_MODE=online`, change `train.train_iters=3`, `checkpoint.save_interval=3`, and configure:

```bash
logger.wandb_project=bridge-qwen3-vl-medpix \
logger.wandb_exp_name=energon-medpix \
logger.wandb_save_dir="$OUTPUT_DIR/wandb"
```

The unpacked baseline uses `dataset.enable_in_batch_packing=False`. After it passes, repeat with
`dataset.enable_in_batch_packing=True` to exercise collate-time packing on the same shards.

## 5. Prepare production shards

For your own dataset, write one or more media members plus one conversation member per sample key. Use
split-prefixed tar names, then index them through Bridge's compatibility helper:

```python
from megatron.bridge.data.energon import prepare_webdataset

prepare_webdataset(
    "/data/my_energon_dataset",
    {"train": "train-shard-.*", "val": "val-shard-.*"},
    num_workers=8,
)
```

Split patterns are regular expressions, not shell globs. The helper avoids CLI differences across supported Energon
7.x versions and rejects a split pattern that matches no shards. Write `.nv-meta/dataset.yaml` after indexing because
`ChatMLWebdataset` is a Bridge class rather than a built-in Energon sample type.

Common field mappings are:

| Media stored in each sample | Conversation placeholder | `field_map` |
| --- | --- | --- |
| one decoded image | one `{"type": "image"}` | `imgs: image.jpg` or `image.png` |
| pickled list of image bytes | matching image parts/placeholders | `imgs: jpgs` |
| Qwen/generic video frames, pickled as a list of videos containing encoded JPEG frames | one `{"type": "video"}` per video | `videos: videos` or `videos: mp4s`; Bridge's `videohandler` decodes the frames |
| raw MP4 bytes for `NemotronOmniTaskEncoder` | one `{"type": "video"}` | `videos: video.mp4`; the model-specific encoder decodes the MP4 |
| audio bytes | model-specific audio content | `audio: audio.wav` |

For a real multi-image converter, see `examples/models/qwen/qwen3_vl/prepare_mantis_energon.py`. For a production audio-video example with explicit train/val/test shard construction, see [VALOR32K-AVQA](../valor32k-avqa/data-preparation.md).

## 6. Processor inputs, outputs, and budgets

These similarly named task-encoder settings have different roles:

| Setting | Meaning |
| --- | --- |
| `dataset.task_encoder.visual_keys` | Processor output tensor names retained in `GenericVisualInputs` by generic HF encoders |
| `dataset.task_encoder.min_pixels`, `max_pixels` | Processor input bounds controlling image/frame resize and visual-token cost |
| `dataset.task_encoder.max_num_images`, `max_num_frames` | Qwen sample count/frame limits |
| `dataset.task_encoder.max_visual_tokens` | Qwen post-resize total visual-token budget |

`min_pixels` and `max_pixels` are not visual keys. Qwen has fixed model-owned output keys and exposes the pixel bounds independently.

## 7. Enable collate-time in-batch packing

The recommended Energon path packs in the task encoder's collator and continues to use the generic `vlm_step`.
Keep configured micro batch greater than one:

```bash
dataset.enable_in_batch_packing=True \
dataset.defer_in_batch_packing_to_step=False \
dataset.micro_batch_size=2 \
train.micro_batch_size=2
```

With context parallelism, also use per-token loss and disable collective loss averaging. Do not use text-only offline packed-SFT artifacts with Energon VLM data.
Deferred packing is reserved for explicitly selected model-specific compatibility steps; it is not the normal
Qwen3-VL Energon workflow.

## Troubleshooting

- `dataset.yaml` import error: run from an installed Bridge checkout so `megatron.bridge.data.energon.task_encoder_utils` is importable in every worker.
- Empty split: check the regex in `.nv-meta/split.yaml`; use `.*`, not a glob `*`.
- Dataset micro-batch mismatch: align train, validation, and Energon micro-batch settings.
- Sample skipped: inspect image/frame count and `max_visual_tokens`; the task encoder logs the violated limit.
- Worker hang during preparation: reduce `num_workers`, remove incomplete metadata, and rerun the preparation helper.
- OOM: reduce sequence length or visual pixel/token budgets before changing model parallelism.
