#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Text generation using HuggingFace models converted to Megatron-FSDP format.

Usage examples:
    python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_fsdp_generate_text.py --trust-remote-code --hf-model-id Qwen/Qwen3-VL-30B-A3B-Instruct --ep 1 --tp 1 --cp 1
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from rich.console import Console
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.
    """

    def __init__(self, input_ids, position_ids):
        self.batch = {"tokens": input_ids, "position_ids": position_ids}
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for text generation.

    Extracts a batch from the data iterator and runs the model forward pass
    with the provided input tokens, position IDs, and attention mask.
    """
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


console = Console()
HF_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _get_world_size() -> int:
    # torchrun exports WORLD_SIZE; fall back to 1 for single-process runs.
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _configure_model_provider(model_provider, tp: int, cp: int, ep: int) -> None:
    world_size = _get_world_size()
    mp_size = tp * cp * ep
    if mp_size <= 0:
        raise ValueError(f"Invalid parallel sizes: tp={tp}, cp={cp}, ep={ep}")
    if world_size % mp_size != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by tp*cp*ep ({mp_size}). Got tp={tp}, cp={cp}, ep={ep}."
        )

    model_provider.tensor_model_parallel_size = tp
    model_provider.context_parallel_size = cp
    model_provider.expert_model_parallel_size = ep
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
    trust_remote_code: bool = False,
) -> None:
    """Perform round-trip conversion between HuggingFace and Megatron-FSDP models."""

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id, trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16
    )

    model_provider = bridge.to_megatron_provider(load_weights=False)
    _configure_model_provider(model_provider, tp=tp, cp=cp, ep=ep)
    model_provider.gradient_accumulation_fusion = False

    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        check_for_nan_in_grad=True,
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads_params",
    )

    megatron_model = model_provider.provide_distributed_model(
        ddp_config=ddp_config,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=False,
    )
    bridge.load_hf_weights(megatron_model)

    model = [m.cuda() for m in megatron_model]

    for m in model:
        m.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "what is reinforcement learning?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    generated_ids = input_ids.clone()

    stop_tokens = [tokenizer.eos_token_id]
    max_new_tokens = 100
    for step in range(max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(input_ids, position_ids)

            output = fwd_bwd_function(
                forward_step_func=text_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]
            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                # All-gather operation
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                # Concatenate along last dimension (dim=2)
                output = torch.cat(gathered_tensors, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                if step < 5:  # Only for first few iterations
                    print_rank_0(f"Step {step}: output shape={output.shape}, var={output.var():.4f}")
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                    print_rank_0(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print_rank_0(
                        f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in stop_tokens:
                break

    generated_text = tokenizer.decode(list(generated_ids[0]))
    print_rank_0(f"Generated text: {generated_text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between HuggingFace and Megatron-FSDP model formats.")
    parser.add_argument("--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID to convert")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size.")
    parser.add_argument("--cp", type=int, default=1, help="Context parallelism size.")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing custom modeling/tokenizer code when loading from a model repository.",
    )

    args = parser.parse_args()
    main(
        hf_model_id=args.hf_model_id,
        tp=args.tp,
        cp=args.cp,
        ep=args.ep,
        trust_remote_code=args.trust_remote_code,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
