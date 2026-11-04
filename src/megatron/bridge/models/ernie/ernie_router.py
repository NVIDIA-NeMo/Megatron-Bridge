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

"""ERNIE 4.5 MoE router: softmax gate with an expert-bias selection correction."""

from dataclasses import replace
from functools import partial

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron.bridge.models.conversion.param_mapping import AutoMapping


# Keep probabilities below 1 so the log-odds stay finite.
_MAX_PROB = 1.0 - 1e-6


def _ernie45_log_odds(logits: torch.Tensor) -> torch.Tensor:
    """Convert router logits to the log-odds of their softmax probabilities."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    log_odds = log_probs - torch.log1p(-log_probs.exp().clamp(max=_MAX_PROB))
    return log_odds.type_as(logits)


class Ernie45TopKRouter(TopKRouter):
    """Router that reproduces ERNIE 4.5 softmax routing with ``e_score_correction_bias``.

    ERNIE 4.5 (Hugging Face ``Ernie4_5_MoeTopKRouter``, Paddle ``moe_gate_act="softmax"``) computes
    ``p = softmax(logits)``, selects experts by ``topk(p + e_score_correction_bias)``, and weights them
    by ``p`` renormalized over the selected experts. Megatron Core applies ``expert_bias`` only on its
    sigmoid score path, so ``gating`` returns the log-odds of ``p``: ``sigmoid(log p - log(1 - p)) == p``.
    With ``moe_router_score_function="sigmoid"`` the existing biased top-k and renormalization then
    match ERNIE exactly. Auxiliary losses see the softmax probabilities, as in ERNIE; a router z-loss
    would act on the log-odds, but ERNIE does not use one.
    """

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """Return the log-odds of the softmax gate probabilities."""
        return _ernie45_log_odds(super().gating(input))


class Ernie45InferenceTopKRouter(InferenceTopKRouter):
    """ERNIE softmax gating with Megatron-Core's optimized inference router."""

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """Return the log-odds of the softmax gate probabilities."""
        return _ernie45_log_odds(super().gating(input))


AutoMapping.register_module_type("Ernie45TopKRouter", "replicated")
AutoMapping.register_module_type("Ernie45InferenceTopKRouter", "replicated")


def _ernie45_moe_submodules(submodules: MoESubmodules) -> MoESubmodules:
    """Replace the MCore router while retaining its inference or training path."""
    router = submodules.router
    ernie_router: type[TopKRouter]
    if router is TopKRouter:
        ernie_router = Ernie45TopKRouter
    elif router is InferenceTopKRouter:
        ernie_router = Ernie45InferenceTopKRouter
    elif router is Ernie45TopKRouter or router is Ernie45InferenceTopKRouter:
        # Multiple layers can share the same MCore layer spec.
        return submodules
    else:
        raise TypeError(f"ERNIE 4.5 expects TopKRouter or InferenceTopKRouter in the MoE spec, got {router!r}.")
    return replace(submodules, router=ernie_router)


def ernie45_decoder_block_spec(
    config: TransformerConfig | GPTModelConfig,
    use_transformer_engine: bool | None = None,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
    **kwargs: object,
) -> TransformerBlockSubmodules:
    """Build a GPT block spec that honors ``moe_layer_freq`` and uses the ERNIE router.

    A builder passes its model config; the spec is then built from the nested transformer config.
    """
    if isinstance(config, GPTModelConfig):
        config = config.transformer
    if use_transformer_engine is None:
        use_transformer_engine = config.transformer_impl == "transformer_engine"
    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=use_transformer_engine,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
        **kwargs,
    )

    for layer_spec in block_spec.layer_specs:
        mlp_spec = layer_spec.submodules.mlp
        if isinstance(mlp_spec, partial) and isinstance(mlp_spec.func, type) and issubclass(mlp_spec.func, MoELayer):
            mlp_kwargs = dict(mlp_spec.keywords or {})
            mlp_kwargs["submodules"] = _ernie45_moe_submodules(mlp_kwargs["submodules"])
            layer_spec.submodules.mlp = partial(mlp_spec.func, *mlp_spec.args, **mlp_kwargs)
        elif isinstance(getattr(mlp_spec, "module", None), type) and issubclass(mlp_spec.module, MoELayer):
            layer_spec.submodules.mlp = replace(mlp_spec, submodules=_ernie45_moe_submodules(mlp_spec.submodules))

    return block_spec
