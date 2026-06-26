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

"""Bridge-backed concurrent async text generation with MegatronAsyncLLM.

Like ``text_generation.py`` but drives the asynchronous MCore engine and generates all
prompts concurrently. Imports only from ``megatron.core`` and ``megatron.bridge``.
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
from megatron.core.inference.apis import MegatronAsyncLLM, SamplingParams

from megatron.bridge.inference.text_generation import (
    HFTokenizerAdapter,
    add_distributed_args,
    add_engine_args,
    add_model_loading_args,
    add_parallelism_args,
    add_prompt_args,
    add_sampling_args,
    build_inference_config,
    build_sampling_params,
    build_tokenizer,
    load_bridge_model,
    load_prompts,
    resolve_hf_model_path,
)
from megatron.bridge.utils.activation_map import str_to_dtype
from megatron.bridge.utils.common_utils import maybe_initialize_distributed, print_rank_0


logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS = ["Megatron async inference is", "Concurrent generation is useful because"]


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Bridge async text generation arguments."""
    add_model_loading_args(parser)
    add_parallelism_args(parser)
    add_prompt_args(parser)
    add_sampling_args(parser)
    add_engine_args(parser)
    add_distributed_args(parser)

    coordinator_group = parser.add_argument_group("Coordinator")
    coordinator_group.add_argument("--coordinator-host", type=str, default=None, help="Coordinator ZMQ host.")
    coordinator_group.add_argument("--coordinator-port", type=int, default=None, help="Coordinator ZMQ port.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_n_logprobs > 0 and not args.return_log_probs:
        raise ValueError("--top-n-logprobs requires --return-log-probs.")
    if args.distributed_timeout_minutes <= 0:
        raise ValueError("--distributed-timeout-minutes must be positive.")


async def _generate(
    args: argparse.Namespace,
    model: object,
    tokenizer: HFTokenizerAdapter,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> None:
    longest_prompt = max(len(tokenizer.tokenize(prompt)) for prompt in prompts)
    # Async path grows the configured window to fit the longest request rather than raising.
    max_sequence_length = max(args.max_seq_length, longest_prompt + args.max_new_tokens)
    inference_config = build_inference_config(
        model=model,
        max_sequence_length=max_sequence_length,
        max_batch_size=args.max_batch_size,
        num_prompts=len(prompts),
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
        if llm.is_primary_rank:
            results = await asyncio.gather(*(llm.generate(prompt, sampling_params) for prompt in prompts))
            print_rank_0("======== ASYNC GENERATED TEXT OUTPUT ========")
            for idx, result in enumerate(results):
                print_rank_0(f"[{idx}] Prompt: {prompts[idx]}")
                print_rank_0(f"[{idx}] Generated: {result.generated_text}")
            print_rank_0("============================================")


def main() -> None:
    """Run Bridge-backed concurrent async text generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()

    logging.basicConfig(level=logging.INFO)
    _validate_args(args)
    maybe_initialize_distributed(args.distributed_timeout_minutes)
    dtype = str_to_dtype(args.dtype)
    hf_model_path = resolve_hf_model_path(args.hf_model_path, args.megatron_model_path)
    prompts = load_prompts(args.prompt, args.prompt_file, args.prompt_file_num_truncate, _DEFAULT_PROMPTS)

    print_rank_0(f"Loading model config/tokenizer from: {hf_model_path}")
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
    sampling_params = build_sampling_params(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        skip_prompt_log_probs=args.skip_prompt_log_probs,
        num_tokens_to_generate=args.max_new_tokens,
        termination_id=args.termination_id if args.termination_id is not None else tokenizer.eod,
        top_n_logprobs=args.top_n_logprobs,
        stop_words=args.stop_words,
    )

    try:
        asyncio.run(_generate(args, model, tokenizer, prompts, sampling_params))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
