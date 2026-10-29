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

"""LLaDA15ModelProvider: GPTModelProvider with bidirectional core attention."""

from dataclasses import dataclass
from typing import Callable, Union

from megatron.core.transformer import ModuleSpec

from megatron.bridge.diffusion.models.llada15.llada15_attention import LLaDA15TEDotProductAttention
from megatron.bridge.models.gpt_provider import GPTModelProvider, transformer_engine_layer_spec


def _llada15_layer_spec(config: "LLaDA15ModelProvider") -> ModuleSpec:
    """Build a TE GPT layer spec and swap in ``LLaDA15TEDotProductAttention``.

    Injecting the custom core attention is the single change required versus
    a vanilla TE GPT layer: it forces bidirectional attention (``no_mask``)
    so the model matches LLaDA1.5's reference implementation, which uses a
    zero attention bias at every forward pass.
    """
    spec = transformer_engine_layer_spec(config)
    self_attention = spec.submodules.self_attention
    self_attention.submodules.core_attention = LLaDA15TEDotProductAttention
    return spec


@dataclass
class LLaDA15ModelProvider(GPTModelProvider):
    """GPTModelProvider for LLaDA1.5 masked-diffusion dense models.

    Differences vs a vanilla Llama GPTModelProvider:

    * ``transformer_layer_spec`` is overridden so every self-attention layer
      uses ``LLaDA15TEDotProductAttention``, which forces bidirectional
      attention (the reference implementation uses zero attention bias
      everywhere — see ``modeling_llada.py:get_bidirectional_attention_bias``).
    * ``share_embeddings_and_output_weights`` defaults to ``False`` because
      LLaDA1.5 has ``weight_tying: false`` in its HF config — the LM head
      lives at ``model.transformer.ff_out``.

    Full RoPE is handled by Megatron's standard rotary path (the HF reference
    rotates the full ``head_dim`` with a standard rotate-half pattern), so
    ``position_embedding_type`` stays at its default ``"rope"`` value.
    """

    transformer_layer_spec: Union[ModuleSpec, Callable] = _llada15_layer_spec
    share_embeddings_and_output_weights: bool = False
    # LLaDA1.5 uses full standard RoPE — let Megatron handle it. The base
    # default is "learned_absolute" which would create a phantom
    # ``embedding.position_embeddings`` parameter with no HF counterpart
    # and crash weight loading.
    position_embedding_type: str = "rope"
