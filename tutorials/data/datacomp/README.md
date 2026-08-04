# DataComp Image-Caption Training with Energon

This tutorial helps you download a deterministic metadata slice of
[DataComp-1B](https://huggingface.co/datasets/mlfoundations/datacomp_1b), fetch
its images with the
[official DataComp downloader](https://github.com/mlfoundations/datacomp), and
convert successful image-caption pairs into Megatron Energon data.

The prepared samples use Bridge's generic `ChatMLWebdataset` representation:
an image and prompt appear in the user turn, and the original caption is the
assistant target. You can select this local dataset with `--dataset energon`
for pretraining, SFT, LoRA, or DoRA when the chosen VLM recipe has a compatible
Hugging Face processor.

The concrete sizing target here is 1,000 optimizer steps at global batch size
512, or 512,000 training samples. The converter emits 525,000 valid samples
before a deterministic 99/1 train/validation split and requires at least
512,000 samples in the resulting training split.

This is a causal image-conditioned captioning adaptation. It is not the
contrastive CLIP objective used by the canonical DataComp benchmark.

## What you will run

All executable preparation logic lives beside this document:

| File | Purpose |
| --- | --- |
| [`setup_downloader.sh`](setup_downloader.sh) | Check out the official downloader and create its pinned Python 3.10 environment |
| [`download_metadata.py`](download_metadata.py) | Download and verify the selected metadata files |
| [`download_images.sh`](download_images.sh) | Run the official image downloader with the documented settings |
| [`audit_download.py`](audit_download.py) | Verify completed shards and the usable sample count |
| [`prepare_datacomp_energon.py`](prepare_datacomp_energon.py) | Convert raw WebDataset shards and build Energon indexes |
| [`import_qwen36_example.sh`](import_qwen36_example.sh) | Import the pinned Qwen3.6 checkpoint with the maintained converter |
| [`train_qwen36_example.sh`](train_qwen36_example.sh) | Render or launch the maintained Qwen3.6 training workflow |

Run these commands from the Megatron Bridge repository root. Choose a shared
location with at least 90 GB available:

```bash
export DATACOMP_ROOT=/path/to/datacomp-1b
```

The scripts derive the following layout from that root:

| Content | Location |
| --- | --- |
| Download environment | `$DATACOMP_ROOT/downloader-env` |
| Official DataComp checkout | `$DATACOMP_ROOT/datacomp-upstream` |
| Pinned metadata | `$DATACOMP_ROOT/raw/metadata` |
| Downloaded image shards | `$DATACOMP_ROOT/raw/shards` |
| Prepared Energon dataset | `$DATACOMP_ROOT/energon` |

## 1. Set up the official downloader

Start by creating the pinned download environment:

```bash
bash tutorials/data/datacomp/setup_downloader.sh "$DATACOMP_ROOT"
```

The setup script checks out DataComp commit
`4a8df1992566ef8334773f7152e1855b1f716162`, installs the download-only
dependency subset from [`download_requirements.txt`](download_requirements.txt),
and records the resolved environment in
`$DATACOMP_ROOT/downloader-environment.txt`.

DataComp pins both OpenCV distributions at 4.6.0. On a headless node they
provide the same `cv2` namespace, while the GUI wheel also requires `libGL`.
The setup script removes the GUI distribution and reinstalls the exact pinned
headless wheel. This substitution does not change the downloader's decoding,
resizing, JPEG encoding, or WebDataset settings.

## 2. Download the pinned metadata slice

DataComp-1B metadata contains source URLs rather than image payloads. Public
URLs disappear over time, so the final acceptance gate is based on successful
downloads rather than metadata rows alone.

Download and validate the selected files with:

```bash
uv run --no-project --python "$DATACOMP_ROOT/downloader-env/bin/python" python tutorials/data/datacomp/download_metadata.py --output-dir "$DATACOMP_ROOT/raw/metadata"
```

The script pins metadata revision
`086ebeee20d4cc3b3e7c05ae703fcf278ae3a759` and verifies each file's row
count, byte size, SHA-256, and logical Arrow schema:

| File | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `0035af9f90f581816acf269df5eb37ad.parquet` | 532,229 | 130,506,429 | `e3633f90e78b827c8b667c88b8a1dce542e72feacc85be9e27f4706ed71fe1ce` |
| `003da708d909c8cab24c7dcf4d04c371.parquet` | 517,671 | 126,593,324 | `5d2d4b0adc840b23dd9bbca04ed351f7904a6346b310326d0b34256ea1b8b0a8` |
| `00818e301428c0573aac33fb4c1b5f02.parquet` | 542,935 | 132,871,668 | `d3bb081586d8dcf1da4883a37becb57fa18759ad83a2a1e484528719f42be047` |
| `00aa8e74b038faf4d69ac89e84a318ba.parquet` | 540,499 | 132,277,520 | `9138d1c135e9b3452e3273f8bdf10b95c55e86b7007234c70d8cd36a12441bc4` |

The schema check intentionally uses `schema_arrow.names`. Inspecting physical
Parquet leaf names would expose the list-valued `face_bboxes` column as its
leaf name, `item`.

## 3. Download the images

The image download is network- and CPU-intensive. Run it in an allocation with
64 CPU cores and enough local or shared storage:

```bash
bash tutorials/data/datacomp/download_images.sh "$DATACOMP_ROOT"
```

The script uses the official DataComp settings that affect output format: 16
processes, 128 threads per process, a 512-pixel resize target,
`keep_ratio_largest`, JPEG output, two retries, 10,000 attempts per WebDataset
shard, and face-bounding-box blurring.

Before each initial or resumed image download, the script revalidates every
pinned metadata file and rejects unexpected Parquet files so the upstream
downloader's metadata wildcard cannot silently widen the selected slice.

If a worker pool stops making progress, you can run the same command again.
img2dataset's incremental mode skips shards with completed `*_stats.json`
files and replaces interrupted shards that lack completion stats.
The reference worker pool stopped after 190 complete shards; an identical
second invocation resumed and completed the remaining 26.

Each successful raw sample contains contiguous `<key>.jpg`, `<key>.json`, and
`<key>.txt` members. The JSON records the DataComp UID, source URL, downloader
status, dimensions, original download hash, caption, and face boxes. The
download hash describes the original response bytes, so it need not match the
stored JPEG after resizing, face blurring, and JPEG re-encoding.

## 4. Audit the raw download

Once downloading finishes, verify every shard and enforce the sample budget:

```bash
uv run --no-project --python "$DATACOMP_ROOT/downloader-env/bin/python" python tutorials/data/datacomp/audit_download.py --shards-dir "$DATACOMP_ROOT/raw/shards"
```

The audit requires 216 completed shard/stat pairs, the 2,133,334 attempts
represented by the pinned metadata, at least 525,000 successful downloads, and
an exact match between stats and complete JPG/JSON/TXT groups. URL failures are
expected and are not converted into placeholder samples.

The 2026-07-24 reference download produced:

| Metric | Count |
| --- | ---: |
| Attempted URLs | 2,133,334 |
| Successful JPG/JSON/TXT samples | 1,326,942 |
| Download failures | 756,770 |
| Resize/decode failures | 49,622 |
| Complete raw shards | 216 |
| Serialized raw-tar bytes | 56,103,116,800 |

Public URL availability changes over time, so your success count may differ.
The minimum-count gate is the durable requirement; the reference success rate
of 62.20% is only evidence from one run.

## 5. Convert and index the Energon dataset

Convert the first 525,000 valid samples and build the Energon indexes:

```bash
uv run python tutorials/data/datacomp/prepare_datacomp_energon.py --source-dir "$DATACOMP_ROOT/raw/shards" --output-dir "$DATACOMP_ROOT/energon" --maximum-samples 525000 --minimum-train-samples 512000 --max-samples-per-tar 10000 --validation-fraction 0.01 --num-workers 8
```

The converter reads raw shards in sorted order, fully decodes selected JPEGs,
validates each JSON/TXT pair, deduplicates by immutable DataComp UID, and
writes deterministic tar metadata. Each output sample contains:

```text
<uid>.image.jpg
<uid>.conversation.json
```

The conversation presents the image and `Describe this image.` in the user
turn, then uses the original DataComp caption as the assistant target. The
selected ChatML VLM collator masks user and padding tokens, so loss is applied
only to the assistant caption.

The converter records the source revision, adaptation, counts, skip reasons,
and every output tar's size and SHA-256 in `manifest.json`. Energon writes its
indexes and split metadata under `.nv-meta/`.

The 2026-07-24 reference conversion produced:

| Metric | Value |
| --- | ---: |
| Raw shards opened | 85 of 216 |
| Valid samples emitted | 525,000 |
| Invalid or duplicate samples skipped | 0 |
| Training samples | 519,827 |
| Validation samples | 5,173 |
| Training/validation output tars | 52 / 1 |
| Serialized output-tar bytes | 20,070,686,720 |

All 53 tars received Energon indexes. The preparation manifest SHA-256 was
`6e273a96a756d24c90c004ed1c351280328697bc45362d95d521eb05083ad430`.
Treat these values as reference-run evidence because third-party payload
availability can change even though the metadata revision is immutable.

As a loader check, real train and validation batches were loaded through Qwen
3.6 processor revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. Both token batches had shape
`[1, 384]`; their assistant-only loss masks selected 28 and 43 tokens. Visual
inputs had shapes `[884, 1536]` and `[832, 1536]` with nonempty image-grid
metadata. This verifies the prepared data and collator path, not a model
training step.

Use a new output directory when retrying a failed conversion. Keeping partial
artifacts intact makes a failed minimum-count gate diagnosable.

## 6. Select the dataset for training

The public `energon` selector is training-mode agnostic. It preserves a
recipe-owned `EnergonDatasetConfig`, including its model-specific task
encoder. For a compatible direct-HF VLM recipe, it creates the generic
Hugging Face Energon task encoder from `dataset.hf_processor_path`.

You still choose the training semantics explicitly with `--mode pretrain`,
`--mode sft`, `--mode lora`, or `--mode dora` and the corresponding recipe and
step function. Set both the training and dataset micro-batch sizes explicitly
when overriding them; the runner does not silently synchronize unrelated
configuration fields.

## 7. Worked Qwen3.6 example

The following example uses Qwen3.6 35B-A3B processor revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. Qwen3.5 35B-A3B and Qwen3.6
35B-A3B share the architecture represented by the existing library recipe, so
the example overrides model identity instead of adding a Qwen3.6-only recipe.

Set the deployment-specific values:

```bash
export SLURM_ACCOUNT=ACCOUNT
export SLURM_PARTITION=PARTITION
export CONTAINER_IMAGE=/path/to/megatron-bridge.sqsh
export BRIDGE_ROOT="$(pwd)"
```

Render the checkpoint import with the maintained `convert.sh` launcher:

```bash
bash tutorials/data/datacomp/import_qwen36_example.sh
```

This model conversion genuinely uses all eight requested GPUs for
TP2/PP1/EP4. On clusters where the maintained launcher renders an exclusive
eight-GPU allocation, all allocated GPUs are used.

After inspecting the rendered conversion, submit it explicitly:

```bash
bash tutorials/data/datacomp/import_qwen36_example.sh --launch
```

Render the maintained `train.sh` submission without launching it:

```bash
bash tutorials/data/datacomp/train_qwen36_example.sh
```

Inspect the rendered command and resolved settings. When you are ready to
submit the same workflow, opt in explicitly:

```bash
bash tutorials/data/datacomp/train_qwen36_example.sh --launch
```

The example trains for 1,000 steps at global batch size 512 and explicitly
sets both micro-batch sizes to 1. It uses the existing Qwen3.5 35B-A3B recipe,
the Qwen VLM step, the local Energon path, the pinned Qwen3.6 processor, and
maintained import/training launchers. There is no custom Slurm script and no
partial-node job disguised as an eight-GPU job. Both example jobs genuinely
use all eight requested GPUs.

The launcher dry run validates Slurm-facing arguments and renders a
submission; it does not instantiate the training config. For an executed run,
confirm from the persisted runtime config that the dataset path, processor
revision, model identity, topology, batch sizes, 1,000 steps, checkpoint
destination, and validation cadence all resolved as intended.

## Data responsibility

DataComp metadata points to third-party web content. Availability, copyright,
licenses, privacy expectations, and acceptable use can vary by source URL.
The official face-blurring path is a useful safeguard, but you should still
review the corpus and its intended use before training or redistribution.
