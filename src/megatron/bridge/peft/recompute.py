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

"""Helpers for PEFT-specific activation recompute fixes."""

from __future__ import annotations

from functools import wraps
from typing import Iterable, Set

import torch
from megatron.core.utils import unwrap_model

from megatron.bridge.utils.common_utils import print_rank_0


PEFT_RECOMPUTE_PATCHED: Set[int] = set()


def _iter_unwrapped_models(model) -> Iterable[torch.nn.Module]:
    """Yield unwrapped Megatron modules regardless of list/list-like inputs."""
    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, list):
        for module in unwrapped:
            if module is not None:
                yield module
    else:
        if unwrapped is not None:
            yield unwrapped


def maybe_enable_recompute_inputs_grad(model, peft_recompute_patched: Set[int] | None = None) -> Set[int]:
    """Enable grad on TransformerBlock inputs when only adapters are trainable.

    Root cause analysis:

    - Megatron's CheckpointFunction.backward() is only invoked by PyTorch autograd
      when at least one input tensor requires grad.
    - With PP>1, received tensors from other stages have requires_grad=True, so
      checkpoint backward is always called.
    - With PP=1 and frozen base model, embedding outputs have requires_grad=False.
      This means CheckpointFunction.backward() is never called, and LoRA gradients
      inside the checkpoint are never computed.

    Solution: Hook TransformerBlock.forward to ensure hidden_states.requires_grad=True
    before it enters checkpointed computation. This doesn't unfreeze any parameters;
    it just ensures the autograd machinery calls checkpoint's backward.

    Borrowed (with modifications) from
    https://github.com/HollowMan6/verl/blob/4285f0601028aee7ddcb9ec5a15198ebfc69bba3/verl/utils/megatron_peft_utils.py
    """

    from megatron.core.transformer.transformer_block import TransformerBlock

    patched_registry = peft_recompute_patched or PEFT_RECOMPUTE_PATCHED

    try:
        for unwrapped_model in _iter_unwrapped_models(model):
            cfg = getattr(unwrapped_model, "config", None)
            if cfg is None or getattr(cfg, "recompute_method", None) is None:
                continue

            if id(unwrapped_model) in patched_registry:
                continue

            params = list(unwrapped_model.named_parameters())
            trainable_adapter = any(p.requires_grad and ".adapter." in n.lower() for n, p in params)
            trainable_base = any(
                p.requires_grad and (".to_wrap." not in n.lower() and ".adapter." not in n.lower()) for n, p in params
            )

            if not (trainable_adapter and not trainable_base):
                continue  # Not adapter-only training, no fix needed

            def _patch_transformer_block(module: torch.nn.Module) -> bool:
                if isinstance(module, TransformerBlock):
                    original_forward = module.forward

                    @wraps(original_forward)
                    def patched_forward(hidden_states, *args, _original_forward=original_forward, **kwargs):
                        # Ensure hidden_states requires grad so checkpoint backward is called
                        if (
                            torch.is_tensor(hidden_states)
                            and not hidden_states.requires_grad
                            and hidden_states.is_floating_point()
                        ):
                            hidden_states = hidden_states.detach().requires_grad_(True)
                        return _original_forward(hidden_states, *args, **kwargs)

                    module.forward = patched_forward
                    return True
                return False

            patched = False
            for module in unwrapped_model.modules():
                if _patch_transformer_block(module):
                    patched = True
            if patched:
                patched_registry.add(id(unwrapped_model))
                print_rank_0(
                    "[PEFT+Recompute] Patched TransformerBlock.forward to enable grad on "
                    "hidden_states input. This ensures checkpoint backward is called when "
                    "only adapters are trainable (PP=1 with frozen base model).",
                )
    except Exception as exc:  # pragma: no cover - best effort logging
        # Log but don't fail - user will see grad_norm=0 and can debug
        print_rank_0(f"[PEFT+Recompute] Warning: Failed to patch TransformerBlock: {exc}")

    return patched_registry


__all__ = ["maybe_enable_recompute_inputs_grad", "PEFT_RECOMPUTE_PATCHED"]
