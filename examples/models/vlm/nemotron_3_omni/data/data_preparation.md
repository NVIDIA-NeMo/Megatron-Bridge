# VALOR32K-AVQA Dataset Preparation Guide

This document describes the steps to prepare the VALOR32K-AVQA v2.0 dataset for Nemotron Omni
training, from raw video tar file to Energon WebDataset format.

## Source

- **Videos tar**: Pre-downloaded VALOR-32K video clips (~41 GB)
  - Contains ~32K AudioSet 10-second video clips
  - Filename format inside tar: `{youtube_id}_{start_time}_{end_time}.mp4` (e.g. `CEfOX4fYlsY_350.000_360.000.mp4`)
  - Tar internal structure: `raid/datasets/audioset/valor_videos/*.mp4` (4 path segments, so `--strip-components=4`)
  - Available from [BaiduPan](https://pan.baidu.com/s/1aHWCwUOX1lJi0lSsmJb6Tw?pwd=e3ve) or via YouTube with yt-dlp
- **QA annotations**: Downloaded automatically by `prepare_valor32k_avqa.py` from [inesriahi/valor32k-avqa-2](https://github.com/inesriahi/valor32k-avqa-2)
  - 177,132 train / 22,267 val / 26,088 test QA pairs
  - Each QA has: `video_id` (bare YouTube ID, e.g. `CEfOX4fYlsY`), `question`, `options` (MCQ), `correct_answer_idx`, `modality`

## Prerequisites

```shell
apt-get install -y ffmpeg          # for audio extraction
pip install webdataset tqdm        # for shard building
```

## Step 1: Extract videos from tar

The tar has 4 path segments before the MP4 files (`raid/datasets/audioset/valor_videos/*.mp4`),
so use `--strip-components=4` to extract them directly into the output directory:

```shell
OUTPUT_DIR="/data/valor32k_avqa"
mkdir -p "$OUTPUT_DIR/videos" "$OUTPUT_DIR/audio"

tar xf /path/to/VALOR32K_videos.tar \
  -C "$OUTPUT_DIR/videos/" --strip-components=4
```

**Result**: ~32,327 MP4 files in `/data/valor32k_avqa/videos/`, named `{youtube_id}_{start}_{end}.mp4`.

## Step 2: Download QA annotations and extract audio

```shell
cd /path/to/megatron-bridge

uv run python examples/models/vlm/nemotron_3_omni/data/prepare_valor32k_avqa.py \
  --output_dir /data/valor32k_avqa
```

This script:

1. Downloads the QA annotation ZIP from GitHub (`inesriahi/valor32k-avqa-2`) and extracts
   `combined_dataset_{train,val,test}_flattened.json`
2. Extracts audio from every MP4 in `videos/` using ffmpeg (16 kHz mono WAV):
   `ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav`

Audio files are named after the full video stem, preserving timestamps: `{youtube_id}_{start}_{end}.wav`.

**Result**: ~32,327 WAV files in `/data/valor32k_avqa/audio/`, plus the three annotation JSON files:

```
/data/valor32k_avqa/
  videos/                                    # ~32K MP4 files
    {youtube_id}_{start}_{end}.mp4
  audio/                                     # ~32K WAV files (16 kHz mono)
    {youtube_id}_{start}_{end}.wav
  combined_dataset_train_flattened.json       # 177,132 QA pairs
  combined_dataset_val_flattened.json         # 22,267 QA pairs
  combined_dataset_test_flattened.json        # 26,088 QA pairs
```

> **Note**: Step 2 is slow — extracting audio from ~32K videos with ffmpeg takes roughly 30 minutes on a single machine.

---

## Step 3: Build the Energon dataset

```shell
uv run python examples/models/vlm/nemotron_3_omni/data/build_valor32k_avqa_shards.py \
  --data_root /data/valor32k_avqa \
  --output_dir /data/valor32k_avqa/energon \
  --samples_per_shard 100
```

This script runs the full pipeline in one shot:

1. **Shard building** — For each QA pair, writes a WebDataset sample containing
   `conversation.json` (ChatML), `video.mp4` (raw MP4), and `audio.wav` (16 kHz WAV).
   The QA JSON stores bare YouTube IDs while the actual files have timestamp suffixes
   (`CEfOX4fYlsY_350.000_360.000.mp4`); the script indexes files by stripping those suffixes.
   Output: ~1,772 train + ~223 val + ~261 test shards in per-split subdirectories.

2. **Restructure** — Moves shards from `energon/{split}/shard-XXXXXX.tar` to the Energon
   flat layout: `energon/{split}-shard-XXXXXX.tar`.

3. **Index** — Creates all Energon metadata in `energon/.nv-meta/`:
   `.info.yaml` (per-shard sample counts), `index.sqlite` (byte-offset index for
   random-access loading), `index.uuid`, and `split.yaml`.

   > **Note on `energon prepare`**: This script bypasses `energon prepare` entirely.
   > `energon prepare` deadlocks in all modes tested on this version of megatron-energon —
   > `AggregatorPool.close()` calls `aggregator_process.join()` which blocks indefinitely
   > because the aggregator process never receives all expected worker-completion signals
   > from the multiprocessing queue. The `index.sqlite` is always fully populated before
   > the hang; the process simply never exits. This should be reported to the energon team.
   > The script uses stdlib `tarfile` to index byte offsets directly instead.

## Step 4: Create dataset.yaml

Create `.nv-meta/dataset.yaml` to tell Energon how to decode samples:

```shell
cat > /data/valor32k_avqa/energon/.nv-meta/dataset.yaml << 'EOF'
__module__: megatron.bridge.data.energon.task_encoder_utils
__class__: ChatMLWebdataset
field_map:
  conversation: conversation.json
  audio: audio.wav
  videos: video.mp4
subflavors: {}
EOF
```

## Final Energon dataset structure

```
/data/valor32k_avqa/energon/
  train-shard-000000.tar                     # ~1,772 train shards
  train-shard-000001.tar
  ...
  val-shard-000000.tar                       # ~223 val shards
  ...
  test-shard-000000.tar                      # ~261 test shards
  ...
  .nv-meta/
    dataset.yaml                             # Sample type + field mapping
    split.yaml                               # Train/val/test shard assignment
    .info.yaml                               # Per-shard sample counts
    index.sqlite                             # Shard byte-offset index
    index.uuid                               # Dataset UUID
```

## Training commands

### Prerequisites

Import the pretrained checkpoint (if not already done):

```shell
uv run python examples/conversion/convert_checkpoints.py import \
  --hf_path <HF_MODEL_PATH> \
  --output_dir /checkpoints/nemotron_omni \
  --trust-remote-code
```

`--trust-remote-code` is required because the HF architecture (`NemotronH_Nano_Omni_Reasoning_V3`)
ships custom modeling code.

### Launch training

```shell
uv run torchrun --nproc-per-node=8 scripts/training/run_recipe.py \
  --recipe nemotron_omni_valor32k_sft_config \
  --step_func nemotron_omni_step \
  checkpoint.pretrained_checkpoint=/checkpoints/nemotron_omni \
  checkpoint.finetune=True \
  dataset.path=/data/valor32k_avqa/energon \
  model.tensor_model_parallel_size=2 \
  model.expert_model_parallel_size=8 \
  model.freeze_language_model=False \
  train.train_iters=4000
```
