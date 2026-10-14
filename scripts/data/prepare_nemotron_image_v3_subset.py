# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare pinned ECD image conversations with complete, bounded Qwen VL samples.

Run with CUDA_VISIBLE_DEVICES='' in a Bridge environment with VL dependencies.
Pass a local Qwen3.8-Flash-Next processor/checkpoint directory; no model weights
are opened. The default selects the first 8,192 complete samples in source order.
The existing converter owns media resolution, message normalization, WebDataset
writing, and Energon indexing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import multiprocessing
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace


LOG = logging.getLogger(__name__)
DATASET_REVISION = "7656391d4d4cb11ec3722b34f10d499435de0460"
MEDIA_REVISION = "284489f140e56f253ac8c1bd76ce45ba8566c645"
CHECKPOINT_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
EXPECTED_ROWS = 55341
PAYLOADS = (
    (
        "nvidia/Nemotron-Image-Training-v3",
        DATASET_REVISION,
        "ecd/ecd.jsonl",
        441779670,
        "e59a9a576fdc40beabe1b2467bba56547f413940c3df8b6777401ebfcbc0d28a",
    ),
    (
        "ChartFoundation/ECD-10k-Images",
        MEDIA_REVISION,
        "images.tar",
        1331769856,
        "8ae3d9fb5b8345a17f4df420d50b662d5102cc54d46539870b0becad92c09cc1",
    ),
)
SOURCE_FILES = (
    "tutorials/data/energon/prepare_nemotron_image_v3.py",
    "src/megatron/bridge/models/qwen_vl/data/collate_fn.py",
    "src/megatron/bridge/data/conversation_processing.py",
    "src/megatron/bridge/data/energon/task_encoder_utils.py",
    "src/megatron/bridge/data/collators/sequence.py",
    "src/megatron/bridge/data/collators/sequence_padding.py",
)

G_PROCESSOR = None
G_LIMITS = None


def sha256(path: Path) -> str:
    """Hash one explicit file without reading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(*, length: int, supervised: int, alignment: int, limit: int) -> tuple[str, int]:
    """Reject invalid supervision and skip complete overlength samples."""
    if min(length, supervised, alignment, limit) <= 0 or supervised > length:
        raise ValueError("Invalid sequence length/supervision/alignment")
    aligned = ((length + alignment - 1) // alignment) * alignment
    return ("overlength" if aligned > limit else "accepted"), aligned


def _worker_init(checkpoint: str, limits: dict, bridge: str) -> None:
    global G_PROCESSOR, G_LIMITS
    sys.path.insert(0, str(Path(bridge) / "src"))
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(1)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("Dataset preparation requires hidden, uninitialized CUDA")
    G_PROCESSOR = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    G_LIMITS = limits
    image = G_PROCESSOR.image_processor
    if (image.patch_size, image.temporal_patch_size, image.merge_size) != (16, 2, 2):
        raise ValueError("Unexpected Qwen3.8 image processor geometry")


def _measure(item: tuple) -> dict:
    from megatron.bridge.data.collators.sequence import prepare_sequence_batch
    from megatron.bridge.data.conversation_processing import (
        normalize_energon_vlm_sample,
        normalized_vlm_sample_to_hf_example,
    )
    from megatron.bridge.data.energon.task_encoder_utils import videohandler
    from megatron.bridge.models.qwen_vl.data.collate_fn import qwen2_5_collate_fn

    key, messages, media = item
    if len(media) != 1:
        raise ValueError(f"{key}: this training path requires exactly one image")
    # Match ChatMLWebdataset's actual image decoder, including float-tensor/PIL round-trip.
    # Decode errors fail preparation; only complete overlength samples may be skipped.
    images = videohandler("torchrgb")("sample.jpgs", pickle.dumps(media, protocol=4))
    sample = SimpleNamespace(conversation=json.dumps(messages), imgs=images, videos=None, audio=None, tools=None)
    example = normalized_vlm_sample_to_hf_example(normalize_energon_vlm_sample(sample), media_first=True)
    batch = qwen2_5_collate_fn(
        [example],
        G_PROCESSOR,
        min_pixels=G_LIMITS["min_pixels"],
        max_pixels=G_LIMITS["max_pixels"],
        require_assistant_matches=True,
        sequence_length=None,
        pad_to_max_length=False,
        pad_to_multiple_of=G_LIMITS["alignment"],
        enable_in_batch_packing=False,
    )
    length = int(batch["attention_mask"].sum().item())
    supervised = int(batch["loss_mask"].sum().item())
    decision, aligned = classify(
        length=length, supervised=supervised, alignment=G_LIMITS["alignment"], limit=G_LIMITS["max_seq_length"]
    )
    # sequence_length=None above deliberately disables both truncation and padding.
    # The exact native padding helper now adds only the amount of padding the training collator adds.
    if batch["input_ids"].numel() != length:
        raise ValueError(f"{key}: unexpected padding in single-example raw processor output")
    prepare_sequence_batch(
        batch,
        sequence_length=aligned,
        pad_to_max_length=False,
        pad_to_multiple_of=G_LIMITS["alignment"],
        enable_in_batch_packing=False,
        sequence_tensor_pad_values={"mm_token_type_ids": 0},
    )
    if batch["input_ids"].numel() != aligned:
        raise ValueError(f"{key}: collator alignment differs from selection")
    grid = batch["visual_inputs"].image_grid_thw
    if grid is None or int(grid.prod(dim=-1).sum().item()) <= 0:
        raise ValueError(f"{key}: missing actual image features")
    if int(grid.prod(dim=-1).sum().item()) // 4 > 2047:
        raise ValueError(f"{key}: too many visual tokens")
    return {
        "key": key,
        "decision": decision,
        "sequence_length": length,
        "aligned_sequence_length": aligned,
        "supervised_tokens": supervised,
        "image_sha256": [hashlib.sha256(b).hexdigest() for b in media],
        "image_grid_thw": grid.tolist(),
        "visual_tokens": int(grid.prod(dim=-1).sum().item()) // 4,
        "untruncated_collator_tensors": {
            name: {
                "sha256": hashlib.sha256(batch[name].contiguous().numpy().tobytes()).hexdigest(),
                "shape": list(batch[name].shape),
                "dtype": str(batch[name].dtype),
            }
            for name in ("input_ids", "labels", "loss_mask")
        },
    }


def _source_dependencies(bridge: Path) -> dict[str, str]:
    # Record the checked-out implementation without restricting users to one old code snapshot.
    return {name: sha256(bridge / name) for name in SOURCE_FILES}


def _load_converter(bridge: Path):
    path = bridge / SOURCE_FILES[0]
    sys.path.insert(0, str(bridge / "src"))
    spec = importlib.util.spec_from_file_location("nemotron_subset_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the repository converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _emit_sample(converter, writer, item: tuple) -> None:
    key, messages, media = item
    converter._write_sample(writer, key, media, messages)


def prepare(args: argparse.Namespace) -> dict:
    """Download two pinned payloads, authenticate, select whole samples, and index."""
    _validate_args(args)
    from huggingface_hub import hf_hub_download

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before preparation")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source_dependencies = _source_dependencies(args.bridge_root)
    converter = _load_converter(args.bridge_root)
    local = []
    for repo, revision, filename, size, digest in PAYLOADS:
        path = Path(
            hf_hub_download(
                repo, filename, repo_type="dataset", revision=revision, local_dir=args.source_dir / repo.split("/")[0]
            )
        )
        if path.stat().st_size != size or sha256(path) != digest:
            raise ValueError(f"Pinned download integrity mismatch: {repo}/{filename}")
        local.append(path)
    jsonl, media_tar = local
    # Count every source row cheaply, even though expensive image preprocessing stops at the accepted cap.
    rows = []
    identities = set()
    with jsonl.open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            sid = row.get("id")
            if not isinstance(sid, str) or not sid or sid in identities:
                raise ValueError(f"Bad/duplicate source identity at line {number}")
            identities.add(sid)
            rows.append(row)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} source rows, found {len(rows)}")
    # Read authenticated tar members directly, without extraction or loose-file fallback.
    limits = {name: getattr(args, name) for name in ("max_seq_length", "min_pixels", "max_pixels", "alignment")}
    counters = {
        "source_rows": len(rows),
        "scanned": 0,
        "accepted": 0,
        "overlength": 0,
        "eligible_after_cap": 0,
        "unscanned": 0,
        "invalid": 0,
    }
    lengths = []
    with (
        converter.MediaResolver(media_tar.parent, allow_loose_media=False, archive_paths=(media_tar,)) as resolver,
        converter._make_shard_writer(args.output_dir, "train", 256) as writer,
        (args.output_dir / "selection.jsonl").open("x") as selected,
        concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(str(args.checkpoint), limits, str(args.bridge_root)),
        ) as pool,
    ):
        for start in range(0, len(rows), args.num_workers):
            batch = []
            for row in rows[start : start + args.num_workers]:
                key = converter._sample_key("ecd", row["id"])
                messages, media = converter._normalize_messages(row["messages"], resolver=resolver, sample_name=key)
                batch.append((key, messages, media))
            for item, record in zip(batch, pool.map(_measure, batch), strict=True):
                counters["scanned"] += 1
                lengths.append(record["sequence_length"])
                if record["decision"] == "overlength":
                    counters["overlength"] += 1
                elif counters["accepted"] >= args.max_samples:
                    counters["eligible_after_cap"] += 1
                    record["decision"] = "eligible_after_cap"
                else:
                    _emit_sample(converter, writer, item)
                    counters["accepted"] += 1
                selected.write(json.dumps(record, sort_keys=True) + "\n")
            if start % 256 == 0:
                LOG.info("SELECTION %s", counters)
            if counters["accepted"] >= args.max_samples:
                break
    counters["unscanned"] = counters["source_rows"] - counters["scanned"]
    if counters["accepted"] == 0:
        raise RuntimeError("No complete supervised image examples selected")
    converter.prepare_webdataset(args.output_dir, {"train": "train-shard-.*"}, num_workers=args.num_workers)
    metadata = args.output_dir / ".nv-meta"
    metadata.mkdir(exist_ok=True)
    (metadata / "dataset.yaml").write_text(converter.DATASET_YAML + "image_decode_spec: torchrgb\n")
    files = {}
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(args.output_dir))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if sum(f["bytes"] for f in files.values()) > 8 << 30 or len(files) > 256:
        raise RuntimeError("Prepared output exceeded 8 GiB/256 files")
    if _source_dependencies(args.bridge_root) != source_dependencies:
        raise RuntimeError("Preparation source files changed during conversion")
    manifest = {
        "schema": "qwen38-nemotron-ecd-prepared-v1",
        "completed": True,
        "dataset": PAYLOADS[0][0],
        "revision": DATASET_REVISION,
        "subsets": ["ecd"],
        "media_dataset": PAYLOADS[1][0],
        "media_revision": MEDIA_REVISION,
        "reference_checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_revision_verified": False,
        "counts": {"train": counters["accepted"], "val": 0},
        "processor_assets": {
            name: {"bytes": (args.checkpoint / name).stat().st_size, "sha256": sha256(args.checkpoint / name)}
            for name in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "merges.txt",
                "vocab.json",
                "chat_template.jinja",
            )
        },
        "selection": counters,
        "limits": limits,
        "max_samples": args.max_samples,
        "selection_policy": "source order; whole samples only; train-only; no truncation",
        "length_range_scanned": [min(lengths), max(lengths)],
        "source_integrity_verified": True,
        "source_payloads": [
            {"repo": r, "revision": v, "filename": f, "bytes": n, "sha256": h} for r, v, f, n, h in PAYLOADS
        ],
        "source_dependencies": source_dependencies,
        "files": files,
    }
    with (args.output_dir / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    LOG.info("PREPARED %s", counters)
    return manifest


def _validate_args(args: argparse.Namespace) -> None:
    if (args.max_seq_length, args.min_pixels, args.max_pixels, args.alignment) != (2048, 65536, 262144, 8):
        raise ValueError("This subset recipe requires sequence2048, pixels65536..262144 and alignment8")
    if not 1 <= args.max_samples <= 8192 or not 1 <= args.num_workers <= 4:
        raise ValueError("Choose 1..8192 samples and 1..4 CPU workers")
    if not (args.bridge_root / SOURCE_FILES[0]).is_file():
        raise ValueError("--bridge-root must point to a Bridge checkout containing the Energon converter")
    if not args.checkpoint.is_dir():
        raise ValueError("--checkpoint must be a local processor/checkpoint directory")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse portable paths and the validated complete-sample selection bounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="Directory for the two pinned HF downloads")
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh destination for indexed Energon shards")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Local Qwen3.8-Flash-Next processor assets; model weight files are not read",
    )
    parser.add_argument(
        "--bridge-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Bridge checkout (defaults to the checkout containing this example)",
    )
    for name, default in (
        ("max-seq-length", 2048),
        ("min-pixels", 65536),
        ("max-pixels", 262144),
        ("alignment", 8),
        ("max-samples", 8192),
        ("num-workers", 4),
    ):
        parser.add_argument("--" + name, default=default, type=int)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def main() -> None:
    """Prepare the pinned ECD subset using local processor assets and CPU workers."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare(args)


if __name__ == "__main__":
    main()
