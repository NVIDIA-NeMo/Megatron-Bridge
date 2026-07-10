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

"""Bridge-backed OpenAI-compatible server using MegatronAsyncLLM.

Imports only from ``megatron.core`` and ``megatron.bridge`` (no Megatron-LM reference layer).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


G_REPO_ROOT = Path(__file__).resolve().parents[2]
G_SRC_ROOT = G_REPO_ROOT / "src"
G_MCORE_ROOT = G_REPO_ROOT / "3rdparty" / "Megatron-LM"
for _path in (G_SRC_ROOT, G_MCORE_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.append(str(_path))

import torch.distributed as dist
from megatron.core.inference.apis import MegatronAsyncLLM, ServeConfig
from megatron.core.utils import configure_nvtx_profiling

from megatron.bridge.inference.text_generation import (
    add_distributed_args,
    add_engine_args,
    add_model_loading_args,
    add_parallelism_args,
    build_inference_config,
    build_tokenizer,
    load_bridge_model,
    resolve_hf_model_path,
)
from megatron.bridge.utils.activation_map import str_to_dtype
from megatron.bridge.utils.common_utils import maybe_initialize_distributed


logger = logging.getLogger(__name__)


def add_server_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Bridge OpenAI-compatible server arguments."""
    add_model_loading_args(parser)
    add_parallelism_args(parser)
    add_engine_args(parser)
    add_distributed_args(parser)

    group = parser.add_argument_group(title="Inference server")
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
    group.add_argument("--return-log-probs", action="store_true", help="Materialize all logits for log probs.")

    profiling = parser.add_argument_group(title="Profiling")
    profiling.add_argument("--profile", action="store_true", help="Enable profiling hooks.")
    profiling.add_argument("--nvtx-ranges", action="store_true", help="Emit NVTX ranges when profiling.")
    return parser


async def _serve(args: argparse.Namespace, model: object, tokenizer: object) -> None:
    inference_config = build_inference_config(
        model=model,
        max_sequence_length=args.max_seq_length,
        max_batch_size=args.max_batch_size,
        num_prompts=None,  # server: let the engine auto-size max_requests when unset
        tp=args.tp,
        block_size_tokens=args.block_size_tokens,
        kv_cache_buffer_size_gb=args.kv_cache_buffer_size_gb,
        max_tokens=args.max_tokens,
        return_log_probs=args.return_log_probs,
        enable_chunked_prefill=args.enable_chunked_prefill,
    )
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
    """Launch a Bridge-backed OpenAI-compatible HTTP server."""
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_server_args(parser).parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.distributed_timeout_minutes <= 0:
        raise ValueError("--distributed-timeout-minutes must be positive.")
    maybe_initialize_distributed(args.distributed_timeout_minutes)
    if args.profile and args.nvtx_ranges:
        configure_nvtx_profiling(True)

    dtype = str_to_dtype(args.dtype)
    hf_model_path = resolve_hf_model_path(args.hf_model_path, args.megatron_model_path)
    tokenizer = build_tokenizer(hf_model_path, args.trust_remote_code)
    model = load_bridge_model(
        hf_model_path=hf_model_path,
        megatron_model_path=args.megatron_model_path,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        sequence_parallel=args.sequence_parallel,
        dtype=dtype,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        attention_backend=args.attention_backend,
        cache_mla_latents=args.cache_mla_latents,
        inference_moe_token_dispatcher_type=args.inference_moe_token_dispatcher_type,
    )

    try:
        asyncio.run(_serve(args, model, tokenizer))
    except KeyboardInterrupt:
        logger.info("Server process interrupted by user.")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
