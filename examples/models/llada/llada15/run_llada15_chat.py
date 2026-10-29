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

"""End-to-end chat / generation with the Megatron-Bridge LLaDA1.5.

Loads a Megatron distributed checkpoint produced by
``convert_llada15_hf_to_megatron.py`` and runs block-diffusion sampling for
a list of user prompts, printing each prompt and the model's reply.

Usage::

    PYTHONPATH=/opt/Megatron-Bridge/src python3 \\
        examples/models/llada/llada15/run_llada15_chat.py \\
        --ckpt-path /path/to/llada15_megatron_ckpt \\
        --tokenizer-path /path/to/huggingface/hub/models--GSAI-ML--LLaDA-1.5/snapshots/<commit-hash> \\
        --gen-length 64 --block-length 32 --steps 32
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

from megatron.bridge import AutoBridge

# Side effect: registers LLaDA15Bridge so AutoBridge can resolve "LLaDAModelLM".
from megatron.bridge.diffusion.conversion.llada15 import llada15_bridge  # noqa: F401
from megatron.bridge.diffusion.models.llada15 import generate_block_diffusion


# Default prompts exercising different reply lengths and topics.
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about a snowy mountain.",
    "Explain in one sentence what a diffusion language model is.",
    "List three differences between Python and Rust.",
]


def setup_distributed_single_gpu():
    """Initialize a 1-rank NCCL process group on cuda:0; Megatron's model load requires it."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)


def apply_chat_template(tokenizer, prompt: str) -> torch.Tensor:
    """Format ``prompt`` with the LLaDA1.5 (Llama3-style) chat template."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids


def load_megatron_model(ckpt_path: str, tokenizer_path: str):
    """Load the Megatron checkpoint via AutoBridge.

    AutoBridge is bootstrapped from the original HF config so the bridge knows
    which provider to instantiate; only the *weights* come from the on-disk
    Megatron checkpoint.
    """
    print(f"Bootstrapping AutoBridge from HF config: {tokenizer_path}")
    bridge = AutoBridge.from_hf_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"Loading Megatron checkpoint: {ckpt_path}")
    model = bridge.load_megatron_model(ckpt_path, wrap_with_ddp=False)
    if isinstance(model, list):
        assert len(model) == 1
        model = model[0]
    return bridge, model.eval()


def main():
    """Load a Megatron LLaDA-1.5 checkpoint and run block-diffusion sampling on a list of prompts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True, help="Megatron checkpoint root (contains iter_*/).")
    parser.add_argument("--tokenizer-path", required=True, help="HF snapshot dir (for tokenizer + bridge config).")
    parser.add_argument("--prompts", nargs="*", default=None, help="Prompts to run. Defaults to a small built-in set.")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    setup_distributed_single_gpu()

    # Tokenizer — needs trust_remote_code because of the bundled tokenizer config.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    bridge, model = load_megatron_model(args.ckpt_path, args.tokenizer_path)
    print(f"Model ready on cuda:0 — {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    for i, prompt in enumerate(prompts):
        ids = apply_chat_template(tok, prompt).to("cuda:0")
        t0 = time.time()
        out_ids = generate_block_diffusion(
            model=model,
            input_ids=ids,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            temperature=args.temperature,
            mask_token_id=126336,
            eos_token_id=126081,
            eos_early_stop=True,
        )
        elapsed = time.time() - t0
        # Decode only the generated portion (everything after the original prompt tokens).
        reply_ids = out_ids[0, ids.shape[1] :].tolist()
        # Trim at EOS if present.
        if 126081 in reply_ids:
            reply_ids = reply_ids[: reply_ids.index(126081)]
        reply = tok.decode(reply_ids, skip_special_tokens=True)

        print(f"\n--- Prompt {i + 1}/{len(prompts)} ({elapsed:.1f}s) ---")
        print(f"USER: {prompt}")
        print(f"LLaDA1.5 (Megatron Bridge): {reply.strip()}")


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
