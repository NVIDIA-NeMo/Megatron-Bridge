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

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset
from transformers import set_seed

from megatron.bridge.models.bagel.data.energon import (
    BagelEditingTaskEncoder,
    BagelT2ITaskEncoder,
    BagelVLMTaskEncoder,
)
from megatron.bridge.models.bagel.data.external import BagelExternalLoader
from megatron.bridge.models.bagel.data.packing import BagelPacker


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse fixed-topology BAGEL alignment arguments."""
    parser = argparse.ArgumentParser(description="Compare 100 Energon and official BAGEL packed batches")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--official-tensor-digests", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--save-after", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def to_jsonable(value: object) -> object:
    """Recursively convert packed values to JSON-compatible objects."""
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def tensor_digest(value: object) -> object:
    """Record a tensor or tensor list without writing its full payload."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, list):
        return [tensor_digest(item) for item in value]
    raise TypeError(f"cannot digest {type(value).__name__}")


def build_loader(dataset_dir: Path, task_encoder: object) -> object:
    """Build one in-process Energon group stream for the one-worker topology."""
    dataset = get_train_dataset(
        dataset_dir,
        split_part="train",
        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
        batch_size=None,
        shuffle_buffer_size=1,
        max_samples_per_sequence=1024,
        task_encoder=task_encoder,
    )
    return get_savable_loader(dataset)


def build_pipeline(
    dataset_root: Path,
    encoders: list[object],
    special_tokens: dict[str, int],
    *,
    length: int,
) -> tuple[list[object], BagelPacker, BagelExternalLoader]:
    """Build the three readers, packer, and external loader."""
    group_names = ("t2i", "editing", "vlm")
    loaders = [build_loader(dataset_root / name, encoder) for name, encoder in zip(group_names, encoders)]
    packer = BagelPacker(
        [iter(loader) for loader in loaders],
        [1.0, 1.0, 1.0],
        [True, False, True],
        special_tokens,
    )
    return loaders, packer, BagelExternalLoader(packer, length=length, stateful_loaders=loaders)


def validate_batch(batch: dict[str, object], step: int, official_stream: TextIO, digest_stream: TextIO) -> str:
    """Compare one packed batch and return its deterministic trace line."""
    actual_digests = {
        field: tensor_digest(batch[field])
        for field in ("nested_attention_masks", "padded_images", "packed_vit_tokens")
        if field in batch
    }
    expected_digests = json.loads(next(digest_stream))
    if {"step": step, **actual_digests} != expected_digests:
        raise ValueError(f"tensor digest mismatch at packed batch {step}")

    compact_batch = dict(batch)
    for field in actual_digests:
        compact_batch.pop(field)
    actual = to_jsonable({"step": step, **compact_batch})
    expected = json.loads(next(official_stream))
    if actual != expected:
        differing = sorted(key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key))
        raise ValueError(f"packed batch {step} differs in fields: {differing}")

    canonical = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    trace = {
        "step": step,
        "source_ids": actual["source_ids"],
        "sequence_length": actual["sequence_length"],
        "packed_sha256": hashlib.sha256(canonical).hexdigest(),
        "tensor_digests": actual_digests,
    }
    return json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n"


def main() -> None:
    """Cook, pack, and compare the fixed 100-batch Energon trace."""
    args = parse_args()
    if args.save_after is not None and not 0 < args.save_after < args.num_batches:
        raise ValueError("save-after must be between zero and num-batches")
    sys.path.insert(0, str(args.bagel_repo))

    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from modeling.qwen2 import Qwen2Tokenizer

    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer_model, local_files_only=True)
    tokenizer, special_tokens, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(image_stride=16, max_image_size=1024, min_image_size=512)
    editing_vit_transform = ImageTransform(image_stride=14, max_image_size=518, min_image_size=224)
    vlm_transform = ImageTransform(
        image_stride=14,
        max_image_size=980,
        min_image_size=378,
        max_pixels=2_007_040,
    )
    encoders = [
        BagelT2ITaskEncoder(tokenizer, vae_transform, 16),
        BagelEditingTaskEncoder(tokenizer, vae_transform, editing_vit_transform, 16, 14),
        BagelVLMTaskEncoder(tokenizer, vlm_transform, 14),
    ]
    _, _, external_loader = build_pipeline(
        args.dataset_root,
        encoders,
        special_tokens,
        length=args.num_batches,
    )
    set_seed(args.seed)
    saved_state = None
    suffix_lines = []

    with (
        args.official.open(encoding="utf-8") as official_stream,
        args.official_tensor_digests.open(encoding="utf-8") as digest_stream,
        args.output.open("w", encoding="utf-8") as output_stream,
    ):
        for step in range(args.num_batches):
            batch = next(external_loader)
            if args.save_after == step + 1:
                saved_state = external_loader.save_state()
            trace_line = validate_batch(batch, step, official_stream, digest_stream)
            output_stream.write(trace_line)
            if args.save_after is not None and step >= args.save_after:
                suffix_lines.append(trace_line)

    if saved_state is not None:
        _, _, restored_external = build_pipeline(
            args.dataset_root,
            encoders,
            special_tokens,
            length=args.num_batches,
        )
        restored_external.restore_state(saved_state)
        with (
            args.official.open(encoding="utf-8") as official_stream,
            args.official_tensor_digests.open(encoding="utf-8") as digest_stream,
        ):
            for _ in range(args.save_after):
                next(official_stream)
                next(digest_stream)
            restored_lines = [
                validate_batch(next(restored_external), step, official_stream, digest_stream)
                for step in range(args.save_after, args.num_batches)
            ]
        if restored_lines != suffix_lines:
            raise ValueError("restored packed-batch suffix differs from the original suffix")
        logger.info("Restored the exact %d-batch suffix", len(restored_lines))

    logger.info("Matched %d official BAGEL packed batches", args.num_batches)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
