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
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO
from unittest.mock import patch

import numpy as np
import torch
from megatron.energon import WorkerConfig, get_train_dataset
from torch.utils.data._utils.worker import _generate_state
from transformers import set_seed

from megatron.bridge.models.bagel.data.external import BagelExternalLoader
from megatron.bridge.models.bagel.data.order import BagelPlannedLoader, plan_manifest_indices
from megatron.bridge.models.bagel.data.packing import BagelPacker


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse independent full-WDS parity arguments."""
    parser = argparse.ArgumentParser(description="Compare full-WDS Energon and official BAGEL batches")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--save-after", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_worker_rng(torch_initial_seed: int, worker_id: int) -> None:
    """Restore PyTorch DataLoader's initial Python, NumPy, and Torch worker RNGs."""
    base_seed = torch_initial_seed - worker_id
    random.seed(torch_initial_seed)
    np.random.seed(_generate_state(base_seed, worker_id))
    torch.manual_seed(torch_initial_seed)


def identity(sample: dict[str, object]) -> dict[str, object]:
    """Keep one already-packed batch unchanged in the outer DataLoader."""
    return sample


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
    """Record tensor identity without writing its full payload."""
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


def build_raw_dataset(dataset_dir: Path, task_encoder: object, worker_config: WorkerConfig) -> object:
    """Build an Energon dataset used only through deterministic sample restoration."""
    return get_train_dataset(
        dataset_dir,
        split_part="train",
        worker_config=worker_config,
        batch_size=None,
        shuffle_buffer_size=1,
        max_samples_per_sequence=1024,
        task_encoder=task_encoder,
    )


def build_worker_pipeline(
    dataset_root: Path,
    encoders: list[object],
    special_tokens: dict[str, int],
    *,
    seed: int,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
) -> tuple[BagelPacker, list[BagelPlannedLoader]]:
    """Build three full-WDS readers and the official BAGEL packer for one worker."""
    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
    group_dirs = ("t2i", "editing", "vlm")
    num_used_data = (10, 10, 1000)
    datasets = [
        build_raw_dataset(dataset_root / group, encoder, worker_config) for group, encoder in zip(group_dirs, encoders)
    ]
    source_loaders = [
        BagelPlannedLoader(
            dataset,
            plan_manifest_indices(
                dataset_root / group / "manifest.json",
                seed=seed,
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
                num_used_data=used,
            ),
            worker_config,
        )
        for group, used, dataset in zip(group_dirs, num_used_data, datasets)
    ]
    return BagelPacker(source_loaders, [1.0, 1.0, 1.0], [True, False, True], special_tokens), source_loaders


class IndependentBagelWorkerDataset(torch.utils.data.IterableDataset):
    """Plan and pack the complete WDS independently inside each real worker."""

    def __init__(
        self,
        dataset_root: Path,
        encoders: list[object],
        special_tokens: dict[str, int],
        *,
        seed: int,
        rank: int,
        world_size: int,
        num_workers: int,
        batches_per_worker: int,
    ) -> None:
        """Store immutable data, topology, and cooker configuration."""
        super().__init__()
        self.dataset_root = dataset_root
        self.encoders = encoders
        self.special_tokens = special_tokens
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.batches_per_worker = batches_per_worker

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Build independent plans and yield this worker's packed batches."""
        worker = torch.utils.data.get_worker_info()
        if worker is None or worker.num_workers != self.num_workers:
            raise RuntimeError("independent BAGEL verifier requires the configured DataLoader workers")
        worker_id = worker.id
        torch_initial_seed = torch.initial_seed()
        with patch("torch.utils.data.get_worker_info", return_value=None):
            packer, _ = build_worker_pipeline(
                self.dataset_root,
                self.encoders,
                self.special_tokens,
                seed=self.seed,
                rank=self.rank,
                world_size=self.world_size,
                worker_id=worker_id,
                num_workers=self.num_workers,
            )
            set_worker_rng(torch_initial_seed, worker_id)
            for _ in range(self.batches_per_worker):
                yield next(packer)


def verify_restore(
    args: argparse.Namespace,
    encoders: list[object],
    special_tokens: dict[str, int],
) -> None:
    """Verify an exact DP1/worker1 suffix after external-loader state restoration."""
    if args.world_size != 1 or args.num_workers != 1:
        raise ValueError("save/restore verification currently requires DP1/worker1")
    generator = torch.Generator().manual_seed(args.seed)
    torch_initial_seed = torch.empty((), dtype=torch.int64).random_(generator=generator).item()
    prefix = f"official_seed{args.seed}_dp1_w1_rank0"

    with patch("torch.utils.data.get_worker_info", return_value=None):
        packer, source_loaders = build_worker_pipeline(
            args.dataset_root,
            encoders,
            special_tokens,
            seed=args.seed,
            rank=0,
            world_size=1,
            worker_id=0,
            num_workers=1,
        )
        set_worker_rng(torch_initial_seed, 0)
        loader = BagelExternalLoader(
            packer,
            length=args.num_batches,
            stateful_loaders=source_loaders,
        )
        suffix = []
        saved_state = None
        with (
            (args.official_root / f"{prefix}_{args.num_batches}.jsonl").open(encoding="utf-8") as official_stream,
            (args.official_root / f"{prefix}_tensor_digests.jsonl").open(encoding="utf-8") as digest_stream,
        ):
            for step in range(args.num_batches):
                line = validate_batch(next(loader), step, official_stream, digest_stream)
                if step + 1 == args.save_after:
                    saved_state = loader.save_state()
                elif step + 1 > args.save_after:
                    suffix.append(line)

        restored_packer, restored_sources = build_worker_pipeline(
            args.dataset_root,
            encoders,
            special_tokens,
            seed=args.seed,
            rank=0,
            world_size=1,
            worker_id=0,
            num_workers=1,
        )
        restored = BagelExternalLoader(
            restored_packer,
            length=args.num_batches,
            stateful_loaders=restored_sources,
        )
        restored.restore_state(saved_state)
        with (
            (args.official_root / f"{prefix}_{args.num_batches}.jsonl").open(encoding="utf-8") as official_stream,
            (args.official_root / f"{prefix}_tensor_digests.jsonl").open(encoding="utf-8") as digest_stream,
        ):
            for _ in range(args.save_after):
                next(official_stream)
                next(digest_stream)
            restored_suffix = [
                validate_batch(next(restored), step, official_stream, digest_stream)
                for step in range(args.save_after, args.num_batches)
            ]
    if restored_suffix != suffix:
        raise ValueError("independent restored packed-batch suffix differs")
    logger.info("Independently restored the exact %d-batch suffix", len(restored_suffix))


def main() -> None:
    """Run full-WDS Energon independently and compare every packed field."""
    args = parse_args()
    if args.num_batches % args.num_workers:
        raise ValueError("num-batches must be divisible by num-workers")
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
    vlm_transform = ImageTransform(image_stride=14, max_image_size=980, min_image_size=378, max_pixels=2_007_040)

    from megatron.bridge.models.bagel.data.energon import (
        BagelEditingTaskEncoder,
        BagelT2ITaskEncoder,
        BagelVLMTaskEncoder,
    )

    encoders = [
        BagelT2ITaskEncoder(tokenizer, vae_transform, 16),
        BagelEditingTaskEncoder(tokenizer, vae_transform, editing_vit_transform, 16, 14),
        BagelVLMTaskEncoder(tokenizer, vlm_transform, 14),
    ]
    with args.output.open("w", encoding="utf-8") as output_stream:
        for rank in range(args.world_size):
            set_seed(args.seed)
            dataset = IndependentBagelWorkerDataset(
                args.dataset_root,
                encoders,
                special_tokens,
                seed=args.seed,
                rank=rank,
                world_size=args.world_size,
                num_workers=args.num_workers,
                batches_per_worker=args.num_batches // args.num_workers,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                num_workers=args.num_workers,
                collate_fn=identity,
            )
            prefix = f"official_seed{args.seed}_dp{args.world_size}_w{args.num_workers}_rank{rank}"
            with (
                (args.official_root / f"{prefix}_{args.num_batches}.jsonl").open(encoding="utf-8") as official_stream,
                (args.official_root / f"{prefix}_tensor_digests.jsonl").open(encoding="utf-8") as digest_stream,
            ):
                for step, batch in zip(range(args.num_batches), loader):
                    trace = json.loads(validate_batch(batch, step, official_stream, digest_stream))
                    output_stream.write(
                        json.dumps({"rank": rank, **trace}, sort_keys=True, separators=(",", ":")) + "\n"
                    )

    logger.info(
        "Independently matched %d ranks x %d official BAGEL packed batches with %d workers",
        args.world_size,
        args.num_batches,
        args.num_workers,
    )
    if args.save_after is not None:
        verify_restore(args, encoders, special_tokens)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
