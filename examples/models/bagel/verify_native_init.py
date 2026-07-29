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

"""Verify an official BAGEL initialization through the strict MCore remapper."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from megatron.core import parallel_state

from megatron.bridge.models.bagel.provider import BagelModelProvider


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Build MCore BAGEL and require exact native checkpoint coverage."""
    args = parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    provider = BagelModelProvider(
        bagel_repo=str(args.bagel_repo.resolve()),
        model_path=str(args.model_path.resolve()),
        native_model_checkpoint=str(args.checkpoint.resolve()),
        native_model_seed=args.seed * world_size,
        native_world_size=world_size,
    )
    provider.finalize()
    models = provider.provide_distributed_model(wrap_with_ddp=False)
    logger.info(
        "Verified native initialization for %d MCore parameters", sum(p.numel() for p in models[0].parameters())
    )
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
