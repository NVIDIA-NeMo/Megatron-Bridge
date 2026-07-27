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
"""Reviewable H100 expert runtime used by the Qwen3.5 performance recipe."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_swiglu import WeightedSwiGLUFunction
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


def _grouped_mm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Call the Hopper-capable PyTorch grouped-MM entry point."""
    grouped_mm = getattr(F, "grouped_mm", None)
    if grouped_mm is None:
        grouped_mm = getattr(torch, "_grouped_mm", None)
    if grouped_mm is None:
        raise RuntimeError(
            "The Qwen3.5 H100 performance recipe requires PyTorch grouped_mm support."
        )
    return grouped_mm(lhs, rhs, offs=offsets)


def _consolidate_expert_weights(
    linear: torch.nn.Module,
    *,
    label: str,
) -> torch.nn.Parameter:
    """Replace discrete TE expert weights with one grouped-MM parameter."""
    if getattr(linear, "single_grouped_weight", False):
        raise RuntimeError(f"{label}: single_grouped_weight is not supported")

    weight_names = [f"weight{index}" for index in range(linear.num_gemms)]
    weights = [linear.get_parameter(name) for name in weight_names]
    if not weights or any(weight.dtype != torch.bfloat16 for weight in weights):
        raise RuntimeError(f"{label}: the H100 grouped-MM path requires BF16 weights")

    with torch.no_grad():
        stacked_data = torch.stack([weight.detach() for weight in weights], dim=0)
    stacked_weight = torch.nn.Parameter(
        stacked_data,
        requires_grad=any(weight.requires_grad for weight in weights),
    )
    # Preserve expert-parallel and optimizer metadata attached by Megatron.
    stacked_weight.__dict__.update(weights[0].__dict__)

    for name in weight_names:
        linear.register_parameter(name, None)
    linear.register_parameter("_torch_grouped_weight", stacked_weight)
    return stacked_weight


class _Qwen35H100TorchGroupedMLP(TEGroupedMLP):
    """BF16 grouped experts with GPU-resident offsets on Hopper."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = kwargs.get("config")
        if config is None and len(args) >= 2:
            config = args[1]
        if config is None:
            raise RuntimeError("Could not resolve the Qwen3.5 grouped expert config")

        # Static HybridEP capacity currently uses the op-fuser flag for its
        # validation and buffer alignment. H100 cannot instantiate the SM100
        # expert fuser, so construct the normal TE linears before replacing
        # their discrete weights with the Hopper grouped-MM parameters below.
        use_op_fuser = config.use_transformer_engine_op_fuser
        config.use_transformer_engine_op_fuser = False
        try:
            super().__init__(*args, **kwargs)
        finally:
            config.use_transformer_engine_op_fuser = use_op_fuser

        if self.config.add_bias_linear:
            raise RuntimeError("The H100 grouped-MM expert path does not support expert bias")
        if self.config.fp8 or self.config.fp4:
            raise RuntimeError("The H100 grouped-MM expert path is BF16-only")
        if not self.config.gated_linear_unit or self.config.activation_func is not F.silu:
            raise RuntimeError("The H100 grouped-MM expert path requires SwiGLU")
        if self.config.moe_mlp_glu_interleave_size is not None:
            raise RuntimeError("The H100 grouped-MM expert path does not support GLU interleaving")
        if self.config.moe_apply_probs_on_input:
            raise RuntimeError(
                "The H100 grouped-MM expert path applies router probabilities after SwiGLU"
            )
        if self.offload_expert_fc1 or self.offload_moe_act:
            raise RuntimeError(
                "The H100 grouped-MM expert path does not support fine-grained expert offload"
            )

        _consolidate_expert_weights(self.linear_fc1, label="linear_fc1")
        _consolidate_expert_weights(self.linear_fc2, label="linear_fc2")

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Run the routed experts without materializing split sizes on the host."""
        if not tokens_per_expert.is_cuda:
            raise RuntimeError(
                "The H100 grouped-MM expert path requires GPU-resident expert counts; "
                "enable static HybridEP rank capacity."
            )
        if tokens_per_expert.dtype not in (torch.int32, torch.int64):
            raise RuntimeError(
                "The H100 grouped-MM expert path requires integer expert counts, got "
                f"{tokens_per_expert.dtype}."
            )

        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        fc1_output = _grouped_mm(
            permuted_local_hidden_states,
            self.linear_fc1._torch_grouped_weight.transpose(1, 2),
            offsets,
        )
        activation_output = WeightedSwiGLUFunction.apply(
            fc1_output.view(-1, fc1_output.shape[-1]),
            permuted_probs.unsqueeze(-1),
            self.config.activation_func_fp8_input_store,
            self.config.activation_func_clamp_value,
        )
        output = _grouped_mm(
            activation_output,
            self.linear_fc2._torch_grouped_weight.transpose(1, 2),
            offsets,
        )
        return output, None


class _Qwen35H100MoELayer(MoELayer):
    """Fail closed if the static HybridEP budget drops a routed token."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the MoE layer and assert that its static rank budget was sufficient."""
        output = super().forward(*args, **kwargs)
        over_budget = self.token_dispatcher.check_over_budget()
        if over_budget is not None:
            torch._assert_async(
                ~over_budget,
                "Qwen3.5 HybridEP static rank capacity overflowed and dropped routed tokens",
            )
        return output


def _replace_moe_runtime(moe_builder: Any) -> partial:
    """Replace a stock MoE builder with the H100 grouped expert runtime."""
    if not isinstance(moe_builder, partial) or moe_builder.func is not MoELayer:
        raise RuntimeError("Unexpected Qwen3.5 MoE layer builder")

    moe_submodules = moe_builder.keywords["submodules"]
    expert_builder = moe_submodules.experts
    if not isinstance(expert_builder, partial) or expert_builder.func is not TEGroupedMLP:
        raise RuntimeError("Unexpected Qwen3.5 grouped expert builder")

    expert_kwargs = dict(expert_builder.keywords)
    custom_experts = partial(
        _Qwen35H100TorchGroupedMLP,
        *expert_builder.args,
        **expert_kwargs,
    )
    custom_moe_submodules = replace(moe_submodules, experts=custom_experts)
    moe_kwargs = {**moe_builder.keywords, "submodules": custom_moe_submodules}
    return partial(_Qwen35H100MoELayer, *moe_builder.args, **moe_kwargs)


def qwen35_h100_transformer_block_spec(
    config: TransformerConfig,
    vp_stage: int | None = None,
) -> TransformerBlockSubmodules:
    """Build Qwen3.5 layers with the reviewable Hopper grouped expert path."""
    block_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config,
        vp_stage=vp_stage,
    )
    replacements: dict[int, partial] = {}
    for layer_spec in block_spec.layer_specs:
        moe_builder = layer_spec.submodules.mlp
        builder_id = id(moe_builder)
        if builder_id not in replacements:
            replacements[builder_id] = _replace_moe_runtime(moe_builder)
        layer_spec.submodules.mlp = replacements[builder_id]
    return block_spec
