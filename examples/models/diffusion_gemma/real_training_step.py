#!/usr/bin/env python3
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

"""Run the real 26B DiffusionGemma training step on synthetic text-only data.

The checkpoint is loaded in BF16 and all base weights are frozen; only the new
self-conditioning MLP is optimized. This keeps the smoke test within one H200
while exercising corruption, two-pass self-conditioning, the real MoE decoder,
backward, and an optimizer update.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge import AutoBridge
from megatron.bridge.models.diffusion_gemma.diffusion_gemma_step import forward_step


class _Timer:
    def start(self):
        return None

    def stop(self):
        return None


def fake_state(provider, seed: int):
    """Create the minimal standalone forward-step state."""
    return SimpleNamespace(
        timers=lambda *_args, **_kwargs: _Timer(),
        straggler_timer=nullcontext(),
        train_state=SimpleNamespace(step=0),
        cfg=SimpleNamespace(
            model=provider,
            rng=SimpleNamespace(seed=seed),
            rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=True, check_for_spiky_loss=False),
        ),
    )


def batch(vocab_size: int, *, seed: int = 1234) -> dict[str, torch.Tensor]:
    """Generate deterministic synthetic text targets for the smoke test."""
    generator = torch.Generator().manual_seed(seed)
    prompt_length, canvas_length = 32, 64
    input_ids = torch.randint(10, vocab_size - 10, (1, prompt_length), generator=generator)
    canvas_ids = torch.randint(10, vocab_size - 10, (1, canvas_length), generator=generator)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(prompt_length)[None],
        "mm_token_type_ids": torch.zeros_like(input_ids),
        "canvas_ids": canvas_ids,
        "canvas_mask": torch.ones_like(canvas_ids, dtype=torch.float32),
    }


def main() -> None:
    """Load the real BF16 checkpoint and optimize self-conditioning parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    bridge = AutoBridge.from_hf_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.params_dtype = torch.bfloat16
    provider.pipeline_dtype = torch.bfloat16
    provider.bf16 = True
    provider.fp16 = False
    provider.attention_backend = AttnBackend.auto
    provider.moe_aux_loss_coeff = 0.0
    provider.diffusion_loss_variant = "base-sft"
    provider.diffusion_self_conditioning_prob = 1.0
    provider.diffusion_encoder_loss_weight = 0.0
    provider.freeze_language_model = True
    provider.freeze_vision_model = True
    provider.freeze_vision_projection = True
    provider.freeze_self_conditioning = False
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    model = provider.provide_distributed_model(wrap_with_ddp=False)[0].cuda().train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError("Real smoke test has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    state = fake_state(provider, seed=1234)
    rows = []
    for step in range(args.steps):
        state.train_state.step = step
        optimizer.zero_grad(set_to_none=True)
        payload, loss_function = forward_step(state, iter([batch(provider.vocab_size)]), model)
        loss_sum, count, metrics = loss_function(payload)
        loss = loss_sum / count
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        rows.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "num_examples": int(count),
                "finite": bool(torch.isfinite(loss) and torch.isfinite(grad_norm)),
            }
        )
        if not rows[-1]["finite"]:
            raise AssertionError(f"Non-finite real training update: {rows[-1]}")
    report = {
        "model_path": args.model_path,
        "dtype": "bfloat16",
        "frozen_base": True,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "steps": rows,
        "passed": len(rows) == args.steps and all(row["finite"] for row in rows),
    }
    if dist.get_rank() == 0:
        print(json.dumps(report, indent=2), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
