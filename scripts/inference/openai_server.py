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

"""Direct MCore-style OpenAI-compatible server using MegatronAsyncLLM."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


G_REPO_ROOT = Path(__file__).resolve().parents[2]
G_MCORE_ROOT = G_REPO_ROOT / "3rdparty" / "Megatron-LM"
if G_MCORE_ROOT.exists() and str(G_MCORE_ROOT) not in sys.path:
    sys.path.append(str(G_MCORE_ROOT))

import torch.distributed as dist
from megatron.core.inference.apis import MegatronAsyncLLM, ServeConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import configure_nvtx_profiling
from megatron.inference.utils import (
    add_inference_args,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)
from megatron.training import get_args, initialize_megatron
from megatron.training.arguments import parse_and_validate_args


logger = logging.getLogger(__name__)


def add_server_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add OpenAI-compatible server arguments."""
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="High-level inference server")
    group.add_argument("--coordinator-host", type=str, default=None, help="Coordinator ZMQ host.")
    group.add_argument("--coordinator-port", type=int, default=None, help="Coordinator ZMQ port.")
    group.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind host.")
    group.add_argument("--port", type=int, default=5000, help="HTTP bind port.")
    group.add_argument("--parsers", type=str, nargs="+", default=[], help="Response parser names.")
    group.add_argument("--verbose", action="store_true", default=False, help="Enable per-request HTTP logging.")
    group.add_argument(
        "--frontend-replicas",
        type=int,
        default=4,
        help="Number of HTTP frontend processes spawned on the primary rank.",
    )
    return parser


async def _serve(args: argparse.Namespace, model: object, tokenizer: object) -> None:
    inference_config = get_inference_config_from_model_and_args(model, args)
    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=True,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
    ) as llm:
        await llm.serve(
            ServeConfig(
                host=args.host,
                port=args.port,
                parsers=args.parsers,
                verbose=args.verbose,
                frontend_replicas=args.frontend_replicas,
            ),
            blocking=True,
        )


def main() -> None:
    """Launch an OpenAI-compatible HTTP server using direct MCore model loading."""
    parse_and_validate_args(
        extra_args_provider=add_server_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )
    initialize_megatron()
    args = get_args()

    logging.basicConfig(level=logging.INFO)
    if args.profile and args.nvtx_ranges:
        configure_nvtx_profiling(True)

    tokenizer = build_tokenizer(args)
    model = get_model_for_inference()

    try:
        asyncio.run(_serve(args, model, tokenizer))
    except KeyboardInterrupt:
        logger.info("Server process interrupted by user.")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
