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

"""GLM-5.3-Flash vision-tower numerics alignment (fp32 intermediate reductions).

The GLM-5.3-Flash training run uses the vanilla HF ``Glm5NextVisionModel``
(instantiated via ``Glm5NextVisionModel._from_config`` in
``glm53flash_bridge.py``). The rollout side computes the vision forward with
six fp32-vs-bf16 numerics refinements on top of the same HF building blocks;
replicating them here keeps the train-side vision forward numerically
aligned with the rollout side. Importing this module patches the HF classes
in place so any bridge that uses them gets the aligned behavior without a
new config flag.

The patches are strictly more accurate (fp32 intermediate reductions) and
are no-ops for text-only runs (the vision tower is never invoked).

Patches applied:
1. q_norm/k_norm eps hardcoded to 1e-6.
2. RMSNorm fp32 weight multiply.
3. fp32 residual add in VisionBlock.
4. fp32 swiglu clamp in VisionMLP + VisionPatchMerger.
5. PatchEmbed GEMM via F.linear instead of nn.Conv3d.
6. inv_freq fp32 guard via _apply override.
"""

import functools

import torch
import torch.nn.functional as F
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextRMSNorm,
    Glm5NextVisionBlock,
    Glm5NextVisionMLP,
    Glm5NextVisionModel,
    Glm5NextVisionPatchEmbed,
    Glm5NextVisionPatchMerger,
)


# ---------------------------------------------------------------------------
# Patched forwards
# ---------------------------------------------------------------------------


def _rmsnorm_forward(self, hidden_states):
    """fp32 weight multiply: ``(x.float() * rsqrt(...) * weight.float()).to(dtype)``.

    HF ``Glm5NextRMSNorm.forward`` normalizes in fp32 but does the
    ``self.weight * hidden_states.to(input_dtype)`` multiply in the *input*
    dtype (bf16). The aligned path keeps the multiply in fp32 and only
    downcasts at the end.
    """
    input_dtype = hidden_states.dtype
    x = hidden_states.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    return (x * self.weight.float()).to(input_dtype)


def _vision_block_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    """fp32 residual add: norm2 sees the fp32 sum, residual stored in model dtype.

    The residual branch is stored in the model dtype, but the sum fed to
    ``norm2`` is computed in fp32 and then downcast.
    """
    attention = self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    residual = hidden_states.float() + attention.float()
    normalized = self.norm2(residual).to(hidden_states.dtype)
    return residual.to(hidden_states.dtype) + self.mlp(normalized)


def _vision_mlp_forward(self, hidden_state):
    """fp32 swiglu clamp: gate/up clamped in fp32, silu(gate)*up downcast last."""
    gate = self.gate_proj(hidden_state).float().clamp(max=self.swiglu_limit)
    up = self.up_proj(hidden_state).float().clamp(-self.swiglu_limit, self.swiglu_limit)
    return self.down_proj((F.silu(gate) * up).to(hidden_state.dtype))


def _vision_patch_merger_forward(self, hidden_state):
    """fp32 swiglu clamp on the post-projection merger branch."""
    hidden_state = self.proj(hidden_state)
    hidden_state = self.act1(self.post_projection_norm(hidden_state))
    gate = self.gate_proj(hidden_state).float().clamp(max=self.swiglu_limit)
    up = self.up_proj(hidden_state).float().clamp(-self.swiglu_limit, self.swiglu_limit)
    return self.down_proj((F.silu(gate) * up).to(hidden_state.dtype))


def _patch_embed_forward(self, hidden_states):
    """GEMM via F.linear instead of cuDNN BF16 Conv3d fallback.

    Every input is exactly one non-overlapping Conv3d patch, so the
    contraction is equivalent to a matmul against the flattened kernel.
    """
    patches = hidden_states.reshape(-1, self.proj.weight[0].numel()).to(self.proj.weight.dtype)
    return F.linear(patches, self.proj.weight.flatten(1), self.proj.bias)


# ---------------------------------------------------------------------------
# VisionModel: q/k norm eps + inv_freq fp32 guard
# ---------------------------------------------------------------------------


_original_vision_init = Glm5NextVisionModel.__init__
_original_vision_apply = Glm5NextVisionModel._apply


@functools.wraps(_original_vision_init)
def _vision_init(self, *args, **kwargs):
    _original_vision_init(self, *args, **kwargs)
    # Hardcode per-head q/k norm eps to 1e-6.
    for block in getattr(self, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        for norm_name in ("q_norm", "k_norm"):
            norm = getattr(attn, norm_name, None)
            if norm is not None and hasattr(norm, "variance_epsilon"):
                norm.variance_epsilon = 1e-6


@functools.wraps(_original_vision_apply)
def _vision_apply(self, fn, recurse=True):
    # Frequencies are not checkpoint parameters. Preserve their original
    # FP32 values when Megatron casts the model weights to BF16.
    rotary = getattr(self, "rotary_pos_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    result = _original_vision_apply(self, fn, recurse=recurse)
    if rotary is not None and inv_freq is not None:
        current = getattr(rotary, "inv_freq", None)
        if current is not None:
            rotary.inv_freq = inv_freq.to(device=current.device)
    return result


# ---------------------------------------------------------------------------
# Apply patches at import
# ---------------------------------------------------------------------------

Glm5NextRMSNorm.forward = _rmsnorm_forward
Glm5NextVisionBlock.forward = _vision_block_forward
Glm5NextVisionMLP.forward = _vision_mlp_forward
Glm5NextVisionPatchMerger.forward = _vision_patch_merger_forward
Glm5NextVisionPatchEmbed.forward = _patch_embed_forward
Glm5NextVisionModel.__init__ = _vision_init
Glm5NextVisionModel._apply = _vision_apply
