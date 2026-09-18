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

"""GLM adaptations for Core's split attention/MLP Hybrid stack."""

import copy
from contextlib import nullcontext
from typing import Any

import torch
from megatron.core import tensor_parallel
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.spec_utils import ModuleSpec


def _forward_glm_stack(stack: HybridStack, hidden_states: Any, attention_mask: Any, **kwargs: Any) -> torch.Tensor:
    def run(value: torch.Tensor) -> torch.Tensor:
        # Native DSA reads its cache from packed metadata, the attention mask,
        # or the layer config (in that order). Bind one shared cache for this
        # invocation to every possible carrier, then restore previous bindings.
        # A whole-stage checkpoint invokes this again with fresh caches during
        # replay, so outstanding microbatches never reuse each other's indices.
        holders = {DSAttention._HOLDER_ATTR: {}, DSAttention._LENGTH_HOLDER_ATTR: {}}
        carriers = [attention_mask, kwargs.get("packed_seq_params")]
        carriers.extend((kwargs.get("packed_seq_params_by_layout") or {}).values())
        carriers.extend(module.config for module in stack.modules() if isinstance(module, DSAttention))
        missing = object()
        previous = []
        seen = set()
        try:
            for carrier in carriers:
                if carrier is None or id(carrier) in seen:
                    continue
                seen.add(id(carrier))
                for name, holder in holders.items():
                    previous.append((carrier, name, getattr(carrier, name, missing)))
                    setattr(carrier, name, holder)
            return HybridStack.forward(stack, value, attention_mask, **kwargs)
        finally:
            for carrier, name, old in reversed(previous):
                if old is missing:
                    delattr(carrier, name)
                else:
                    setattr(carrier, name, old)

    if stack.config.recompute_granularity == "full" and stack.training:
        # Replay each sharing group together with fresh native DSA cache bindings.
        def recompute(value: torch.Tensor) -> torch.Tensor:
            original = stack.config
            stack.config = copy.copy(original)
            stack.config.recompute_granularity = None
            try:
                return run(value)
            finally:
                stack.config = original

        return tensor_parallel.checkpoint(recompute, stack.config.distribute_saved_activations, hidden_states)
    return run(hidden_states)


class GLMHybridStack(HybridStack):
    """Isolate top-k state by forward and replay a complete stage on recomputation."""

    def __init__(
        self,
        config: Any,
        submodules: Any,
        *,
        layer_config_list: Any = None,
        layer_type_list: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if layer_config_list is None and layer_type_list is not None:
            layer_config_list = validate_segment_layers("".join(layer_type_list), config)
            layer_type_list = None
        if layer_config_list is not None:
            converted = []
            for layer_config in layer_config_list:
                layer_config = copy.copy(layer_config)
                if isinstance(layer_config, DSALayerConfig) and (layer_config.dsa_indexer_topk_freq or 1) > 1:
                    # DSA occupies physical layers 1, 3, 5, ... in D-/DE pairs.
                    # Express the HF cadence in those coordinates without changing
                    # provider/HF metadata or subclassing Core's DSAttention.
                    layer_config.dsa_indexer_topk_freq *= 2
                    layer_config.dsa_indexer_skip_topk_offset = (
                        2 * max(layer_config.dsa_indexer_skip_topk_offset or 0, 1) - 1
                    )
                converted.append(layer_config)
            layer_config_list = converted
        super().__init__(
            config, submodules, layer_config_list=layer_config_list, layer_type_list=layer_type_list, **kwargs
        )

    def forward(self, hidden_states: Any, attention_mask: Any, **kwargs: Any) -> torch.Tensor:
        """Run the stage with fresh state, including during backward recomputation."""
        return _forward_glm_stack(self, hidden_states, attention_mask, **kwargs)


class GLMHybridMTPLayer(MultiTokenPredictionLayer):
    """Use the same forward-scoped state for Core's nested MTP HybridStack."""

    def _proj_and_transformer_layer(
        self, hidden_states: torch.Tensor, decoder_input: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        # BF16 GLM only: retain Core's projection/RNG/post-norm behavior, but use
        # the GLM stage runner because Core constructs nested stacks directly.
        rng = tensor_parallel.get_cuda_rng_tracker().fork() if self.config.sequence_parallel else nullcontext()
        with rng:
            hidden_states = self._concat_embeddings(hidden_states, decoder_input)
            hidden_states = _forward_glm_stack(
                self.mtp_model_layer,
                hidden_states,
                kwargs.get("attention_mask"),
                padding_mask=kwargs.get("padding_mask"),
                rotary_pos_emb=kwargs.get("rotary_pos_emb"),
                inference_context=kwargs.get("inference_params"),
                packed_seq_params=kwargs.get("packed_seq_params"),
                packed_seq_params_by_layout=kwargs.get("packed_seq_params_by_layout"),
                cp_layout_plan=kwargs.get("cp_layout_plan"),
            )
        return self._postprocess(hidden_states)


def glm_hybrid_stack_spec(config: Any) -> ModuleSpec:
    """Return a private GLM spec without mutating Core's shared specification."""
    spec = copy.deepcopy(hybrid_stack_spec)
    spec.module = GLMHybridStack
    for layer_spec in spec.submodules.mtp_block_spec.submodules.layer_specs:
        layer_spec.module = GLMHybridMTPLayer
    return spec
