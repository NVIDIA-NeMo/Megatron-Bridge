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
Layer spec for the ERNIE 4.5 VL Megatron-native Vision Transformer (ViT).

Provides ``get_ernie_vit_layer_spec()`` which returns a ``ModuleSpec`` for a
single ViT transformer layer using Transformer Engine modules.

The spec is identical to the standard MCore ViT spec from
``megatron.core.models.vision.vit_layer_specs.get_vit_layer_with_transformer_engine_spec``
except that ``self_attention.module`` is overridden with ``ErnieVLSelfAttention``
to handle absolute 2D RoPE (non-interleaved rotate_half style).

Architecture details:
    - Attention: TELayerNormColumnParallelLinear (fused QKV + LN)
                 + TEDotProductAttention
                 + TERowParallelLinear
    - MLP:       TELayerNormColumnParallelLinear (fused fc1 + LN)
                 + TERowParallelLinear
    - Mask type: AttnMaskType.no_mask (bidirectional attention for ViT)
    - pre_mlp_layernorm: IdentityOp (LN is fused into TE linear layers)
"""

from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

from megatron.bridge.models.ernie_vl.vision_attention import ErnieVLSelfAttention


def get_ernie_vit_layer_spec():
    """Return a TransformerLayer ModuleSpec for ERNIE ViT.

    This reuses the standard MCore ViT TE spec and only overrides the
    self-attention module with ``ErnieVLSelfAttention`` to apply absolute
    2D RoPE embeddings instead of the standard relative RoPE.

    Returns:
        ModuleSpec: Spec for one ERNIE ViT transformer layer.
    """
    spec = get_vit_layer_with_transformer_engine_spec()
    # Override self-attention module with ERNIE's absolute 2D RoPE variant
    spec.submodules.self_attention.module = ErnieVLSelfAttention
    return spec
