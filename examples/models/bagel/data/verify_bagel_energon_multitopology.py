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
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.worker import _generate_state
from verify_bagel_energon_100 import build_pipeline, validate_batch


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse fixed multi-rank and multi-worker alignment arguments."""
    parser = argparse.ArgumentParser(description="Compare fixed-topology Energon and official BAGEL batches")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_worker_rng(torch_initial_seed: int, worker_id: int) -> None:
    """Reproduce PyTorch DataLoader's Python, NumPy, and Torch worker seeds."""
    base_seed = torch_initial_seed - worker_id
    random.seed(torch_initial_seed)
    np.random.seed(_generate_state(base_seed, worker_id))
    torch.manual_seed(torch_initial_seed)


def main() -> None:
    """Replay each official worker independently and merge batches in rank order."""
    args = parse_args()
    if args.num_batches % args.num_workers:
        raise ValueError("num-batches must be divisible by num-workers")
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
            packers = []
            external_loaders = []
            worker_states = []
            for worker_id in range(args.num_workers):
                _, packer, external_loader = build_pipeline(
                    args.dataset_root / f"rank{rank}" / f"worker{worker_id}",
                    encoders,
                    special_tokens,
                    length=args.num_batches // args.num_workers,
                )
                seed_path = args.seed_dir / f"rank{rank}_worker{worker_id}_seed.json"
                worker_seed = json.loads(seed_path.read_text(encoding="utf-8"))["torch_initial_seed"]
                set_worker_rng(worker_seed, worker_id)
                packers.append(packer)
                external_loaders.append(external_loader)
                worker_states.append(packer.state_dict())

            prefix = f"official_seed42_dp{args.world_size}_w{args.num_workers}_rank{rank}"
            official_path = args.official_root / f"{prefix}_{args.num_batches}.jsonl"
            digest_path = args.official_root / f"{prefix}_tensor_digests.jsonl"
            with (
                official_path.open(encoding="utf-8") as official_stream,
                digest_path.open(encoding="utf-8") as digest_stream,
            ):
                for step in range(args.num_batches):
                    worker_id = step % args.num_workers
                    packers[worker_id].load_state_dict(worker_states[worker_id])
                    batch = next(external_loaders[worker_id])
                    worker_states[worker_id] = packers[worker_id].state_dict()
                    trace = json.loads(validate_batch(batch, step, official_stream, digest_stream))
                    output_stream.write(
                        json.dumps({"rank": rank, **trace}, sort_keys=True, separators=(",", ":")) + "\n"
                    )

    logger.info(
        "Matched %d ranks x %d official BAGEL packed batches with %d workers",
        args.world_size,
        args.num_batches,
        args.num_workers,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
