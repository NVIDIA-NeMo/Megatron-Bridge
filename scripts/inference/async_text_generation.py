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

"""Direct MCore-style concurrent async text generation with MegatronAsyncLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


G_REPO_ROOT = Path(__file__).resolve().parents[2]
G_MCORE_ROOT = G_REPO_ROOT / "3rdparty" / "Megatron-LM"
if G_MCORE_ROOT.exists() and str(G_MCORE_ROOT) not in sys.path:
    sys.path.append(str(G_MCORE_ROOT))

import torch.distributed as dist
from megatron.core.inference.apis import MegatronAsyncLLM, SamplingParams
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.inference.utils import (
    add_inference_args,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)
from megatron.training import initialize_megatron
from megatron.training.arguments import parse_and_validate_args
from megatron.training.utils import print_rank_0


logger = logging.getLogger(__name__)


def add_async_generation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add high-level async generation arguments."""
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="High-level async inference")
    group.add_argument("--coordinator-host", type=str, default=None, help="Coordinator ZMQ host.")
    group.add_argument("--coordinator-port", type=int, default=None, help="Coordinator ZMQ port.")
    return parser


def _prompt_from_json_line(line: str) -> str | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    prompt = value.get("text")
    return prompt if isinstance(prompt, str) else None


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return list(args.prompts)
    if args.prompt_file:
        prompts = []
        with Path(args.prompt_file).open("r", encoding="utf-8") as prompt_file:
            for line in prompt_file:
                raw_prompt = line.rstrip("\n")
                if not raw_prompt:
                    continue
                prompts.append(_prompt_from_json_line(raw_prompt) or raw_prompt)
                if args.prompt_file_num_truncate is not None and len(prompts) >= args.prompt_file_num_truncate:
                    break
        return prompts
    return ["Megatron async inference is", "Concurrent generation is useful because"]


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_n_logprobs > 0 and not args.return_log_probs:
        raise ValueError("--top-n-logprobs requires --return-log-probs.")


def _build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        skip_prompt_log_probs=args.skip_prompt_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=args.termination_id,
        top_n_logprobs=args.top_n_logprobs,
        stop_words=args.stop_words,
    )


async def _generate(args: argparse.Namespace, model: object, tokenizer: object, prompts: list[str]) -> None:
    inference_config = get_inference_config_from_model_and_args(model, args)
    longest_prompt = max(len(tokenizer.tokenize(prompt)) for prompt in prompts)
    inference_config.max_sequence_length = max(
        inference_config.max_sequence_length,
        longest_prompt + args.num_tokens_to_generate,
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
            sampling_params = _build_sampling_params(args)
            results = await asyncio.gather(*(llm.generate(prompt, sampling_params) for prompt in prompts))
            print_rank_0("======== ASYNC GENERATED TEXT OUTPUT ========")
            for idx, result in enumerate(results):
                print_rank_0(f"[{idx}] Prompt: {prompts[idx]}")
                print_rank_0(f"[{idx}] Generated: {result.generated_text}")
            print_rank_0("============================================")


def main() -> None:
    """Run concurrent async generation using direct MCore model loading."""
    args = parse_and_validate_args(
        extra_args_provider=add_async_generation_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )
    initialize_megatron()

    logging.basicConfig(level=logging.INFO)
    _validate_args(args)
    prompts = _load_prompts(args)
    tokenizer = build_tokenizer(args)
    model = get_model_for_inference()

    try:
        asyncio.run(_generate(args, model, tokenizer, prompts))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
