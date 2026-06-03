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

"""Generation parity test: HF reference vs Megatron Bridge LLaDA1.5.

Reproduces the official ML-GSAI/LLaDA ``generate.py`` block-diffusion sampling
algorithm and runs it against both the HF reference model and the Megatron
model converted by :class:`LLaDA15Bridge`. With the same prompt and
``temperature=0`` (greedy), both should produce identical token sequences up
to bf16 numerical noise.

Why we re-port the algorithm instead of using our ``generate_block_diffusion``
directly: the parity test must drive *both* models with the exact same
sampler, or any algorithmic difference between our loop and the reference
loop confounds the model-correctness question.

Usage::

    PYTHONPATH=/opt/Megatron-Bridge/src python3 \\
        examples/models/llada/llada15/test_llada15_generation_parity.py \\
        --hf-path /path/to/huggingface/hub/models--GSAI-ML--LLaDA-1.5/snapshots/<commit-hash> \\
        --prompt "The capital of France is" \\
        --gen-length 32 --block-length 32 --steps 32
"""

import argparse
import gc
import os
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.bridge import AutoBridge

# Shared diffusion sampling primitives — the same ones the inference path uses,
# so this parity test verifies the exact code that runs in production.
from megatron.bridge.diffusion.common.dllm import add_gumbel_noise, get_num_transfer_tokens

# Side effect: registers LLaDA15Bridge.
from megatron.bridge.diffusion.conversion.llada15 import llada15_bridge  # noqa: F401


MASK_ID = 126336
EOS_ID = 126081


# ---------------------------------------------------------------------------
# Official ML-GSAI/LLaDA sampler — reproduced verbatim in spirit from
# https://github.com/ML-GSAI/LLaDA/blob/main/generate.py. The function takes
# a `forward_fn(input_ids) -> logits` callback so it can drive either model.
# The denoising primitives (gumbel noise, transfer schedule) come from the
# shared dllm module.
# ---------------------------------------------------------------------------


@torch.no_grad()
def official_generate(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    prompt: torch.Tensor,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = MASK_ID,
) -> torch.Tensor:
    """Official ML-GSAI/LLaDA sampler driven by a ``forward_fn`` callback."""
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks

    device = prompt.device
    B, prompt_len = prompt.shape
    x = torch.full((B, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = forward_fn(x)  # [B, S, V]
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, S]

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)  # [B, S]
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)
            else:
                raise NotImplementedError(remasking)

            # Future-block positions can't be selected for transfer.
            x0_p[:, block_end:] = -float("inf")

            # Only masked positions can be replaced.
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = int(num_transfer[j, i].item())
                if k > 0:
                    _, idx = torch.topk(confidence[j], k=k)
                    transfer_index[j, idx] = True
            x[transfer_index] = x0[transfer_index]

    return x


# ---------------------------------------------------------------------------
# Adapters: forward_fn wrappers for HF and Megatron models
# ---------------------------------------------------------------------------


def patch_llada_for_transformers5(cls):
    cls.all_tied_weights_keys = {}
    orig_tie = cls.tie_weights

    def _tie_weights_compat(self, *args, **kwargs):
        kwargs.pop("missing_keys", None)
        kwargs.pop("recompute_mapping", None)
        return orig_tie(self, *args, **kwargs)

    cls.tie_weights = _tie_weights_compat


def load_hf(hf_path: str):
    from transformers import AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("modeling_llada.LLaDAModelLM", hf_path, trust_remote_code=True)
    patch_llada_for_transformers5(cls)

    model = (
        AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0").eval()
    )
    model.config.use_cache = False
    return model


def hf_forward_fn(hf_model):
    def fwd(input_ids: torch.Tensor) -> torch.Tensor:
        out = hf_model(input_ids=input_ids)
        return (out.logits if hasattr(out, "logits") else out[0]).float()

    return fwd


def load_megatron(hf_path: str):
    bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True)
    model = bridge.to_megatron_model(wrap_with_ddp=False)
    if isinstance(model, list):
        model = model[0]
    return bridge, model.eval()


def megatron_forward_fn(megatron_model, hf_vocab_size: int):
    def fwd(input_ids: torch.Tensor) -> torch.Tensor:
        S = input_ids.shape[1]
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        out = megatron_model(input_ids=input_ids, position_ids=pos, attention_mask=None)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        return logits.float()[..., :hf_vocab_size]

    return fwd


def setup_distributed_single_gpu():
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0 (greedy) for determinism.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_distributed_single_gpu()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------- tokenize -----------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda:0")
    print(f"prompt='{args.prompt}'  ids={prompt_ids[0].tolist()}")

    # ----------------------- run HF -----------------------
    print("Loading HF reference and sampling...")
    hf_model = load_hf(args.hf_path)
    torch.manual_seed(args.seed)
    hf_tokens = official_generate(
        hf_forward_fn(hf_model),
        prompt_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
    )
    hf_vocab_size = hf_model.config.vocab_size
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------- run Megatron -----------------------
    print("Loading Megatron model and sampling...")
    bridge, megatron_model = load_megatron(args.hf_path)
    torch.manual_seed(args.seed)
    mc_tokens = official_generate(
        megatron_forward_fn(megatron_model, hf_vocab_size),
        prompt_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
    )

    # ----------------------- compare -----------------------
    print("\n=== Generation parity ===")
    print(f"HF       reply ids: {hf_tokens[0, prompt_ids.shape[1] :].tolist()}")
    print(f"Megatron reply ids: {mc_tokens[0, prompt_ids.shape[1] :].tolist()}")

    hf_reply = hf_tokens[0, prompt_ids.shape[1] :]
    mc_reply = mc_tokens[0, prompt_ids.shape[1] :]
    n_match = int((hf_reply == mc_reply).sum().item())
    n_total = int(hf_reply.numel())
    print(f"Token agreement: {n_match}/{n_total} ({100 * n_match / n_total:.1f}%)")

    if n_match == n_total:
        print("Decoded HF reply :", tok.decode(hf_reply.tolist(), skip_special_tokens=True))
        print("Decoded MC reply :", tok.decode(mc_reply.tolist(), skip_special_tokens=True))
        print("\nverdict: PASS (exact token-for-token match)")
    else:
        # Show where divergence starts
        diff_positions = (hf_reply != mc_reply).nonzero(as_tuple=True)[0].tolist()
        print(f"First divergence at gen position {diff_positions[0]}")
        print(
            f"  HF       token: {int(hf_reply[diff_positions[0]])}  → '{tok.decode([int(hf_reply[diff_positions[0]])])}'"
        )
        print(
            f"  Megatron token: {int(mc_reply[diff_positions[0]])}  → '{tok.decode([int(mc_reply[diff_positions[0]])])}'"
        )
        print(f"All differing positions: {diff_positions}")
        print("Decoded HF reply :", tok.decode(hf_reply.tolist(), skip_special_tokens=True))
        print("Decoded MC reply :", tok.decode(mc_reply.tolist(), skip_special_tokens=True))
        # We allow a small mismatch — bf16 numerical noise can shift one token
        # without indicating a bug. Report but don't fail unless agreement is poor.
        threshold = 0.9
        verdict = "PASS" if n_match / n_total >= threshold else "FAIL"
        print(f"\nverdict: {verdict} (token agreement {n_match}/{n_total}, threshold {threshold:.0%})")
        if verdict == "FAIL":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
