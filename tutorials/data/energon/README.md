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
  --step_func vlm_step \
  --mode lora \
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
  logger.log_interval=1
```

Success means Energon initializes both split loaders, the Qwen task encoder decodes and normalizes the PNGs, iteration 1 and validation report finite loss, and the adapter checkpoint is written.

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

The unpacked baseline leaves both packing modes at their defaults. After it passes, repeat with
`dataset.enable_in_batch_packing=True` to exercise the older collate-time packing path on the same shards.

## 5. Prepare production shards

For a pinned real-dataset example that downloads only one small, self-contained subset, converts the source JSONL and
separate media tar, and launches Qwen3-VL through `scripts/training/train.sh`, see
[Nemotron Image Training v3](nemotron-image-v3.md).

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

## 7. Enable Energon online sequence packing

### Background

An unpacked VLM batch pads every conversation to the longest conversation in that batch. Collate-time packing can
remove that waste, but it only chooses among the few samples already assigned to one micro batch. Energon's native
packing API instead keeps a per-worker candidate buffer and forms multiple packs from that larger window. No offline
packed dataset is written: the WebDataset shards from sections 1 or 5 remain unchanged.

For the Qwen-VL collator path, Bridge implements the Energon lifecycle as follows:

1. `encode_sample` runs the Qwen processor once and measures the exact sequence length after visual tokens expand.
2. `select_samples_to_pack` applies first-fit decreasing to one worker's candidate buffer.
3. `pack_selected_samples` records each selected group while preserving its source restore keys.
4. `batch` emits one physical MBS1 THD row with real and padded sequence boundaries, masks inter-conversation
   attention, and concatenates the visual tensors in conversation order.

`dataset.packing_buffer_size` is a number of candidate samples **per worker**. It is not bytes, tokens, a tar cache,
or a final pack width. A larger value usually improves occupancy at the cost of more host memory and startup latency.
Start with 16, then compare 8, 16, and 32 on the real length distribution. High-resolution images and videos make
each prepared candidate much larger than its compressed shard member, so monitor worker RSS before increasing it.

### Connect an existing dataset

Keep the same `ChatMLWebdataset` preparation and `QwenVLEnergonTaskEncoderConfig` used by the unpacked path. Enable
native packing through the dataset config:

```python
from megatron.bridge.data.builders import EnergonDatasetConfig, QwenVLEnergonTaskEncoderConfig

dataset = EnergonDatasetConfig(
    path="/data/my_energon_dataset",
    seq_length=4096,
    micro_batch_size=1,
    num_workers=2,
    num_val_workers=2,
    packing_buffer_size=16,
    task_encoder=QwenVLEnergonTaskEncoderConfig(
        hf_processor_path="Qwen/Qwen3-VL-8B-Instruct",
    ),
)
```

`packing_buffer_size` is the sole selector for native Energon packing. Leave the legacy `enable_in_batch_packing` and
`defer_in_batch_packing_to_step` settings at their defaults. Explicitly enabling either legacy owner is rejected;
native packs use the generic `vlm_step`.

### Start training

The repository example enables native packing by default:

```bash
export ENERGON_PATH=/data/my_energon_dataset
export PRETRAINED_CHECKPOINT=/models/Qwen3-VL-8B-Instruct
export OUTPUT_DIR=/results/qwen3-vl-energon-native-pack

PACKING_BUFFER_SIZE=16 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=8 \
TRAIN_ITERS=100 \
bash examples/models/qwen/qwen3_vl/peft_energon.sh
```

The equivalent important overrides are:

```bash
--step_func vlm_step \
dataset.packing_buffer_size=16 \
dataset.micro_batch_size=1 \
train.micro_batch_size=1 \
validation.eval_micro_batch_size=1 \
model.calculate_per_token_loss=True \
ddp.average_in_collective=False
```

Native packing requires physical MBS1 because one Energon output sample is already a complete pack. The global batch
size counts physical packs, not source conversations; the number of conversations contributing to an optimizer step
therefore varies with pack membership. Per-token loss keeps that variable membership correctly normalized.

Bridge derives the per-segment padding multiple from CP, TP, and sequence parallelism. Per-token loss requires
`ddp.average_in_collective=False`, including at CP1. All padded gaps receive label `-100` and loss mask zero. Keep the
dataset shards and split metadata, worker counts, shuffle settings and seed, processor, topology, `seq_length`, and
packing-buffer size fixed when resuming; changing any of them can change pack membership. `packing_buffer_size`
applies to validation too, and fixed `eval_iters` therefore evaluates a fixed number of packs, not a fixed number of
source conversations.

### Measure packing quality

Compare the native run with the unpacked command from section 3 at the same sequence length and token-normalized loss.
Useful signals are batch-generator time, samples or tokens per second, GPU utilization, and two separate packing
ratios:

```text
bin fill = sum(aligned segment lengths) / (number of packs * seq_length)
executed-token efficiency = sum(real segment lengths) / sum(aligned segment lengths)
```

A buffer of 1 is a correctness baseline but gives little opportunity to combine samples. Very large buffers can
increase worker memory because prepared token and visual tensors remain resident until their packs are emitted.

The first implementation supports eager Qwen-VL with `vlm_step`, including TP, SP, and CP. Standard eager `alltoall`
expert parallelism has functional coverage for Qwen3.6-35B-A3B at TP1/PP1/EP8 with EP communication overlap disabled;
this does not establish performance. Other EP dispatchers are accepted with fixed-width native packs but do not yet
have equivalent runtime evidence. Logical and physical THD boundaries produce a mask that excludes fixed-width gaps
from MoE auxiliary-loss, z-loss, and expert-bias statistics. Current MCore may still dispatch those padded positions;
expert-capacity/token-dropping configurations do not yet have native-packing runtime coverage. Generic HF, Nemotron
Omni, the legacy `qwen3_vl_step`, MBS greater than one, MTP, CUDA graphs, Qwen3-VL DistTrain, and pipeline parallelism
are unsupported. Requested MoE expert-parallel communication overlap is disabled with a warning so training uses the
non-overlapped path. The checked example uses the Qwen3-VL 8B provider; validate other Qwen-VL variants before
production use.
The older collate-time path remains available by leaving `packing_buffer_size=None`, setting
`enable_in_batch_packing=True`, and using MBS greater than one.

## Troubleshooting

- `dataset.yaml` import error: run from an installed Bridge checkout so `megatron.bridge.data.energon.task_encoder_utils` is importable in every worker.
- Empty split: check the regex in `.nv-meta/split.yaml`; use `.*`, not a glob `*`.
- Dataset micro-batch mismatch: align train, validation, and Energon micro-batch settings.
- Sample skipped: inspect image/frame count and `max_visual_tokens`; the task encoder logs the violated limit.
- Worker hang during preparation: reduce `num_workers`, remove incomplete metadata, and rerun the preparation helper.
- OOM: reduce sequence length or visual pixel/token budgets before changing model parallelism.
