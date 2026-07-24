# DataComp Image-Caption Pretraining with Energon

This tutorial downloads a deterministic metadata slice of
[DataComp-1B](https://huggingface.co/datasets/mlfoundations/datacomp_1b), uses the
[official DataComp downloader](https://github.com/mlfoundations/datacomp), and
converts successful image-caption pairs into the `ChatMLWebdataset` format used
by Megatron Bridge's Qwen-VL Energon loader.

The target is 1,000 optimizer steps at global batch size 512, or 512,000
training samples. The preparation gate emits 525,000 valid samples before the
deterministic 99/1 train/validation split and requires at least 512,000 samples
in the resulting train split.

This is a causal image-conditioned captioning adaptation. It is not the
contrastive CLIP objective used by the canonical DataComp benchmark.

## 1. Pin the sources

The commands below use these immutable revisions:

| Source | Revision |
| --- | --- |
| DataComp downloader repository | `4a8df1992566ef8334773f7152e1855b1f716162` |
| `mlfoundations/datacomp_1b` metadata | `086ebeee20d4cc3b3e7c05ae703fcf278ae3a759` |
| Qwen3.6 35B-A3B model and processor | `995ad96eacd98c81ed38be0c5b274b04031597b0` |

From the Megatron Bridge repository root:

```bash
export DATACOMP_ROOT=/data/datacomp-1b-qwen-vl
export DATACOMP_ENV="$DATACOMP_ROOT/downloader-env"
export DATACOMP_UPSTREAM="$DATACOMP_ROOT/datacomp-upstream"
export DATACOMP_METADATA="$DATACOMP_ROOT/raw/metadata"
export DATACOMP_SHARDS="$DATACOMP_ROOT/raw/shards"
export DATACOMP_ENERGON="$DATACOMP_ROOT/energon"

git clone https://github.com/mlfoundations/datacomp "$DATACOMP_UPSTREAM"
git -C "$DATACOMP_UPSTREAM" checkout 4a8df1992566ef8334773f7152e1855b1f716162
test "$(git -C "$DATACOMP_UPSTREAM" rev-parse HEAD)" = \
  4a8df1992566ef8334773f7152e1855b1f716162
```

The upstream requirements include training and evaluation packages that are
not needed for downloading. Install the tested download-only subset in a
separate Python 3.10 environment:

```bash
uv venv --python 3.10 "$DATACOMP_ENV"
uv pip install \
  --python "$DATACOMP_ENV/bin/python" \
  --requirements examples/models/qwen/qwen3_vl/datacomp_download_requirements.txt
```

The subset retains the upstream image-format versions, including
`img2dataset==1.40.0`. It uses `huggingface-hub==0.14.1` only to fetch the
immutable metadata objects; the row, size, schema, and SHA-256 gates below
verify that compatibility update cannot silently change the selected metadata.

DataComp pins both OpenCV distributions at 4.6.0. On a headless node they
install the same `cv2` namespace, while the GUI wheel also requires `libGL`.
Remove the GUI distribution and reinstall the exact pinned headless wheel:

```bash
uv pip uninstall --python "$DATACOMP_ENV/bin/python" opencv-python
uv pip install \
  --python "$DATACOMP_ENV/bin/python" \
  --reinstall-package opencv-python-headless \
  opencv-python-headless==4.6.0.66

uv run --no-project --python "$DATACOMP_ENV/bin/python" python - <<'PY'
import cv2
from importlib.metadata import version

assert cv2.__version__ == "4.6.0"
assert version("img2dataset") == "1.40.0"
PY

uv pip freeze --python "$DATACOMP_ENV/bin/python" \
  > "$DATACOMP_ROOT/downloader-environment.txt"
```

This headless substitution changes no decoding, resizing, JPEG encoding, or
WebDataset settings.

## 2. Download and verify a metadata slice

DataComp-1B metadata files contain URLs rather than image payloads. Image
download success changes as public URLs disappear, so size the slice from
successful outputs rather than metadata rows. The tested preparation selected
the first four lexically sorted Parquet files at the pinned revision:

| File | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `0035af9f90f581816acf269df5eb37ad.parquet` | 532,229 | 130,506,429 | `e3633f90e78b827c8b667c88b8a1dce542e72feacc85be9e27f4706ed71fe1ce` |
| `003da708d909c8cab24c7dcf4d04c371.parquet` | 517,671 | 126,593,324 | `5d2d4b0adc840b23dd9bbca04ed351f7904a6346b310326d0b34256ea1b8b0a8` |
| `00818e301428c0573aac33fb4c1b5f02.parquet` | 542,935 | 132,871,668 | `d3bb081586d8dcf1da4883a37becb57fa18759ad83a2a1e484528719f42be047` |
| `00aa8e74b038faf4d69ac89e84a318ba.parquet` | 540,499 | 132,277,520 | `9138d1c135e9b3452e3273f8bdf10b95c55e86b7007234c70d8cd36a12441bc4` |

Download and validate those exact files:

```bash
mkdir -p "$DATACOMP_METADATA"

DATACOMP_METADATA="$DATACOMP_METADATA" \
uv run --no-project --python "$DATACOMP_ENV/bin/python" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

repo_id = "mlfoundations/datacomp_1b"
revision = "086ebeee20d4cc3b3e7c05ae703fcf278ae3a759"
output_dir = Path(os.environ["DATACOMP_METADATA"])
expected = {
    "0035af9f90f581816acf269df5eb37ad.parquet": (532229, "e3633f90e78b827c8b667c88b8a1dce542e72feacc85be9e27f4706ed71fe1ce"),
    "003da708d909c8cab24c7dcf4d04c371.parquet": (517671, "5d2d4b0adc840b23dd9bbca04ed351f7904a6346b310326d0b34256ea1b8b0a8"),
    "00818e301428c0573aac33fb4c1b5f02.parquet": (542935, "d3bb081586d8dcf1da4883a37becb57fa18759ad83a2a1e484528719f42be047"),
    "00aa8e74b038faf4d69ac89e84a318ba.parquet": (540499, "9138d1c135e9b3452e3273f8bdf10b95c55e86b7007234c70d8cd36a12441bc4"),
}
required_columns = {"uid", "url", "text", "face_bboxes"}
records = []

for filename, (expected_rows, expected_sha256) in sorted(expected.items()):
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    )
    hasher = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    assert parquet.metadata.num_rows == expected_rows
    assert digest == expected_sha256
    assert required_columns <= set(columns)
    records.append(
        {
            "filename": filename,
            "rows": expected_rows,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "columns": columns,
        }
    )

manifest = {"repo_id": repo_id, "revision": revision, "files": records}
(output_dir / "metadata-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY
```

Use `schema_arrow.names`, not the physical Parquet leaf names: the list-valued
`face_bboxes` column otherwise appears as its physical leaf name, `item`.

## 3. Run the official image downloader

Run this network-heavy command in a batch allocation with 64 CPU cores and
enough local or shared storage. It uses the official defaults that affect the
data format: 16 processes, 128 threads per process, 512-pixel resize target,
`keep_ratio_largest`, JPEG output, two retries, 10,000 attempts per WebDataset
shard, and face-bounding-box blurring. Do not pass
`--skip_bbox_blurring`.

```bash
mkdir -p "$DATACOMP_SHARDS"

uv run --no-project --python "$DATACOMP_ENV/bin/python" \
  python "$DATACOMP_UPSTREAM/download_upstream.py" \
  --scale datacomp_1b \
  --data_dir "$DATACOMP_ROOT/raw" \
  --metadata_dir "$DATACOMP_METADATA" \
  --processes_count 16 \
  --thread_count 128 \
  --image_size 512 \
  --resize_mode keep_ratio_largest \
  --encode_format jpg \
  --output_format webdataset \
  --retries 2
```

The reference run's worker pool stopped making progress after 190 complete
shards. Re-running the identical command resumed the remaining 26 shards:
img2dataset's default incremental mode skips shards with completed
`*_stats.json` files and overwrites interrupted shards without completion
stats. Do not infer completion from tar filenames alone; the audit below is the
acceptance gate after either an uninterrupted run or a restart.

Each successful raw sample has contiguous `<key>.jpg`, `<key>.json`, and
`<key>.txt` members. The JSON contains the DataComp UID, source URL, downloader
status, dimensions, original download hash, caption, and face boxes. The
download hash describes the original response bytes; it need not equal the
stored JPEG hash after resizing, face blurring, and JPEG re-encoding.

Count only complete shards and successful records:

```bash
DATACOMP_SHARDS="$DATACOMP_SHARDS" \
uv run --no-project --python "$DATACOMP_ENV/bin/python" python - <<'PY'
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["DATACOMP_SHARDS"])
stats_paths = sorted(root.glob("*_stats.json"))
tar_paths = sorted(root.glob("*.tar"))
assert stats_paths
assert len(stats_paths) == len(tar_paths) == 216
assert {path.stem for path in tar_paths} == {
    path.name.removesuffix("_stats.json") for path in stats_paths
}

stats = [json.loads(path.read_text()) for path in stats_paths]
attempted = sum(row["count"] for row in stats)
successes = sum(row["successes"] for row in stats)
tar_samples = 0
for path in tar_paths:
    members = defaultdict(set)
    closed_keys = set()
    previous_key = None
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isfile():
                key, extension = Path(member.name).name.rsplit(".", 1)
                assert extension in {"jpg", "json", "txt"}
                if previous_key is not None and key != previous_key:
                    closed_keys.add(previous_key)
                assert key not in closed_keys
                previous_key = key
                assert extension not in members[key]
                members[key].add(extension)
    assert members and all(extensions == {"jpg", "json", "txt"} for extensions in members.values())
    tar_samples += len(members)

assert tar_samples == successes
assert attempted == 2133334
assert successes >= 525000
print(json.dumps({"attempted": attempted, "successes": successes}, sort_keys=True))
PY
```

URL failures are expected and are not converted into placeholder samples.

The 2026-07-24 reference run produced:

| Metric | Count |
| --- | ---: |
| Attempted URLs | 2,133,334 |
| Successful JPG/JSON/TXT samples | 1,326,942 |
| Download failures | 756,770 |
| Resize/decode failures | 49,622 |
| Complete raw shards | 216 |
| Serialized raw-tar bytes | 56,103,116,800 |

The tar-header audit found exactly 1,326,942 complete sample groups, matching
the summed img2dataset success count. Public URL availability changes over
time, so a later run may have a different success count; keep the threshold
gate rather than assuming the reference success rate of 62.20%.

The reference raw and converted tars together occupy 76,173,803,520 bytes.
Provision additional space for metadata, per-shard failure records, downloader
and project environments, caches, indexes, and transient files; 90 GB is a
practical lower bound for this four-file preparation.

## 4. Convert and index the Energon dataset

The Bridge converter reads raw shards in sorted order, validates and fully
decodes each selected JPEG, validates its JSON/TXT pair, deduplicates by
DataComp UID, and writes deterministic tar metadata. It stops after 525,000
valid samples so the training corpus matches the planned sample budget plus
split margin:

```bash
uv run python examples/models/qwen/qwen3_vl/prepare_datacomp_energon.py \
  --source-dir "$DATACOMP_SHARDS" \
  --output-dir "$DATACOMP_ENERGON" \
  --maximum-samples 525000 \
  --minimum-train-samples 512000 \
  --max-samples-per-tar 10000 \
  --validation-fraction 0.01 \
  --num-workers 8
```

The output sample key is the immutable DataComp UID. Each sample contains:

```text
<uid>.image.jpg
<uid>.conversation.json
```

The conversation presents the image and `Describe this image.` in the user
turn, then uses the original DataComp caption as the assistant target. The Qwen
collator masks user and padding tokens, so loss is applied only to the
assistant caption. The converter records this adaptation, source revision,
counts, skip reasons, and every output tar's size and SHA-256 in
`manifest.json`; Energon writes its indexes and split metadata under
`.nv-meta/`.

The 2026-07-24 reference preparation produced:

| Metric | Value |
| --- | ---: |
| Raw shards opened | 85 of 216 |
| Valid samples emitted | 525,000 |
| Invalid or duplicate samples skipped | 0 |
| Training samples | 519,827 |
| Validation samples | 5,173 |
| Training/validation output tars | 52 / 1 |
| Serialized output-tar bytes | 20,070,686,720 |

All 53 tars received Energon indexes, and `.nv-meta/` contains `dataset.yaml`,
`split.yaml`, `index.sqlite`, and the generated dataset UUID. The preparation
manifest SHA-256 was
`6e273a96a756d24c90c004ed1c351280328697bc45362d95d521eb05083ad430`.
Treat these as reference-run evidence rather than fixed download expectations:
the pinned metadata is immutable, but third-party URL availability and payloads
can still change.

Real train and validation batches were then loaded through Qwen3.6 processor
revision `995ad96eacd98c81ed38be0c5b274b04031597b0`. Both token batches had
shape `[1, 384]`; their assistant-only loss masks selected 28 and 43 tokens.
The visual inputs had shapes `[884, 1536]` and `[832, 1536]` with nonempty
image-grid metadata. This validates the prepared data and collator path, not a
model training step.

Never reuse a nonempty output directory. A failed minimum-count gate leaves its
artifacts in place for diagnosis; select a new output directory after fixing
the input.

## 5. Render the maintained Qwen-VL launch

The existing Qwen3.5 35B-A3B recipe is architecture-compatible with Qwen3.6
35B-A3B. The recipe freezes the language and vision towers and trains the
vision projection. Override the checkpoint identity and pin the processor
revision explicitly; no Qwen3.6-specific recipe or custom Slurm training script
is required.

Configure the deployment-specific Slurm and container values once. Keep the
dataset, imported checkpoint, and training output under one shared root, and
mount both that root and the current Bridge checkout into every job:

```bash
export SLURM_ACCOUNT=ACCOUNT
export SLURM_PARTITION=PARTITION
export CONTAINER_IMAGE=/path/to/megatron-bridge.sqsh
export BRIDGE_ROOT="$(pwd)"
export QWEN_HF_ID=Qwen/Qwen3.6-35B-A3B
export QWEN_HF_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0
export QWEN_MEGATRON="$DATACOMP_ROOT/models/qwen3.6-35b-a3b-megatron"
```

First import the pinned checkpoint with the maintained distributed GPU
converter. The eight conversion workers use TP2/PP1/EP4; the distributed
checkpoint is resharded to the training recipe's TP4/PP2/EP4 topology when it
is loaded:

```bash
./scripts/conversion/convert.sh import \
  --executor slurm \
  --device gpu \
  --nodes 1 \
  --gpus-per-node 8 \
  --account "$SLURM_ACCOUNT" \
  --partition "$SLURM_PARTITION" \
  --container-image "$CONTAINER_IMAGE" \
  --mount "$DATACOMP_ROOT" \
  --mount "$BRIDGE_ROOT:/opt/Megatron-Bridge" \
  --hf-model "$QWEN_HF_ID" \
  --hf-revision "$QWEN_HF_REVISION" \
  --megatron-path "$QWEN_MEGATRON" \
  --torch-dtype bfloat16 \
  --tp 2 --pp 1 --ep 4 --etp 1
```

Then render the maintained training submission without launching it:

```bash
./scripts/training/train.sh \
  --nodes 1 --gpus-per-node 8 --dry-run \
  --account "$SLURM_ACCOUNT" \
  --partition "$SLURM_PARTITION" \
  --container-image "$CONTAINER_IMAGE" \
  --mount "$DATACOMP_ROOT" \
  --mount "$BRIDGE_ROOT:/opt/Megatron-Bridge" \
  --recipe qwen35_vl_35b_a3b_pretrain_mock_config \
  --mode pretrain \
  --dataset qwen-vl-energon \
  --step-func qwen3_vl_step \
  --pretrained_checkpoint "$QWEN_MEGATRON/iter_0000000" \
  --max_steps 1000 \
  --save_dir "$DATACOMP_ROOT/training/qwen3.6-35b-a3b/checkpoints" \
  --save_interval 500 \
  train.global_batch_size=512 \
  train.micro_batch_size=1 \
  dataset.path="$DATACOMP_ENERGON" \
  dataset.task_encoder.hf_processor_path="$QWEN_HF_ID" \
  dataset.task_encoder.hf_processor_revision="$QWEN_HF_REVISION" \
  dataset.do_validation=true \
  dataset.do_test=false \
  dataset.enable_in_batch_packing=false \
  dataset.defer_in_batch_packing_to_step=true \
  model.hf_model_id="$QWEN_HF_ID" \
  model.bos_token_id=248044 \
  model.eos_token_id=248044 \
  checkpoint.hf_source_path="$QWEN_HF_ID" \
  checkpoint.load=null \
  logger.save_config_filepath="$DATACOMP_ROOT/training/qwen3.6-35b-a3b/resolved-config.yaml"
```

The launcher dry run validates its Slurm-facing arguments and renders the
submission; it does not instantiate the training config. Inspect the rendered
command, then remove `--dry-run` to execute the same maintained `train.sh`
workflow. Confirm from the persisted runtime config that it resolved the
dataset path, pinned model identity, TP4/PP2/EP4 topology, micro batch 1, global
batch 512 from the explicit overrides, 1,000 steps, checkpoint destination, and
validation cadence.

## Data responsibility

DataComp metadata points to third-party web content. Availability, copyright,
licenses, privacy expectations, and acceptable use can vary by source URL.
Preserving the official face-blurring path is not a substitute for reviewing
the corpus and its intended use before training or redistribution.
