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
"""Model-specific Qwen3.5 runtime patches for the measured H100 performance path."""

from __future__ import annotations

import inspect
from dataclasses import replace
from functools import partial
from importlib import metadata
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_swiglu import WeightedSwiGLUFunction
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


_H100_HYBRIDEP_BF16_ALIGNMENT = 16
_FLASH_LINEAR_ATTENTION_VERSION = "0.4.2"
_WEIGHTED_SWIGLU_FORWARD_HAS_CLAMP_VALUE = (
    "clamp_value" in inspect.signature(WeightedSwiGLUFunction.forward).parameters
)


def _load_fused_gated_rms_norm() -> Any:
    """Load the measured FLA fused gated-RMSNorm kernel lazily."""
    try:
        from fla.modules.fused_norm_gate import rms_norm_gated

        fla_version = metadata.version("flash-linear-attention")
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise ImportError(
            f"The Qwen3.5 H100 performance recipe requires flash-linear-attention=={_FLASH_LINEAR_ATTENTION_VERSION}."
        ) from exc

    if fla_version != _FLASH_LINEAR_ATTENTION_VERSION:
        raise ImportError(
            "The Qwen3.5 H100 performance recipe requires "
            f"flash-linear-attention=={_FLASH_LINEAR_ATTENTION_VERSION}; found {fla_version}."
        )
    return rms_norm_gated


def _configure_fused_gated_rms_norm(module: GatedDeltaNet) -> None:
    """Require and install the measured Qwen3.5 gated-RMSNorm fusion."""
    out_norm = module.out_norm
    if module.activation not in ("silu", "swish"):
        raise RuntimeError("The Qwen3.5 H100 fused gated-RMSNorm path requires SiLU activation")
    if module.config.normalization != "RMSNorm":
        raise RuntimeError("The Qwen3.5 H100 fused gated-RMSNorm path requires RMSNorm")
    if getattr(out_norm, "weight", None) is None or getattr(out_norm, "bias", None) is not None:
        raise RuntimeError("The Qwen3.5 H100 fused gated-RMSNorm path requires a norm weight and no bias")
    module._qwen35_fused_gated_rms_norm = _load_fused_gated_rms_norm()


def _apply_fused_gated_rms_norm(
    module: GatedDeltaNet,
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse the GDN output RMSNorm and SiLU gate."""
    x = x.reshape(-1, x.shape[-1])
    gate = gate.reshape(-1, gate.shape[-1])
    out_norm = module.out_norm
    weight = out_norm.weight
    if getattr(
        out_norm,
        "zero_centered_gamma",
        module.config.layernorm_zero_centered_gamma,
    ):
        weight = weight + 1.0
    return module._qwen35_fused_gated_rms_norm(
        x,
        gate,
        weight,
        None,
        activation="swish",
        eps=getattr(out_norm, "eps", module.config.layernorm_epsilon),
    )


class _Qwen35H100GatedDeltaNet(GatedDeltaNet):
    """Gated Delta Net using MCore's FLA backend and the measured output fusion."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.cp_size != 1:
            raise RuntimeError("The Qwen3.5 H100 GDN path requires context parallel size 1")
        if self.key_head_dim != 128 or self.value_head_dim != 128:
            raise RuntimeError("The Qwen3.5 H100 GDN path requires 128-dimensional QK/V heads")
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        if capability != (9, 0):
            raise RuntimeError(f"The Qwen3.5 H100 GDN path requires SM90, got {capability}")
        _configure_fused_gated_rms_norm(self)

    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply the measured fused output norm and gate."""
        return _apply_fused_gated_rms_norm(self, x, gate)


def _setup_h100_static_hybridep_metadata(
    manager: Any,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Set static BF16 HybridEP metadata with its native H100 alignment."""
    config = manager.config
    if config.fp8 or config.fp4:
        raise RuntimeError("The Qwen3.5 H100 static HybridEP path is BF16-only")
    if manager.drop_and_pad:
        raise RuntimeError("The Qwen3.5 H100 rank-capacity path does not support per-expert drop-and-pad")
    if manager.moe_expert_rank_capacity_factor is None:
        raise RuntimeError("The Qwen3.5 H100 HybridEP path requires static rank capacity")

    num_tokens = routing_map.shape[0]
    manager.routing_map = routing_map.reshape(num_tokens, manager.num_experts)
    manager.token_probs = probs.reshape(num_tokens, manager.num_experts)
    budget = int(num_tokens * config.moe_router_topk * manager.moe_expert_rank_capacity_factor)
    budget += -budget % _H100_HYBRIDEP_BF16_ALIGNMENT
    manager.num_permuted_tokens = budget


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
        raise RuntimeError("The Qwen3.5 H100 performance recipe requires PyTorch grouped_mm support.")
    return grouped_mm(lhs, rhs, offs=offsets)


def _apply_weighted_swiglu(
    input_tensor: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    fp8_input_store: bool,
    clamp_value: float | None,
) -> torch.Tensor:
    """Apply weighted SwiGLU across pinned and newer MCore APIs."""
    if _WEIGHTED_SWIGLU_FORWARD_HAS_CLAMP_VALUE:
        return WeightedSwiGLUFunction.apply(
            input_tensor,
            token_weights,
            fp8_input_store,
            clamp_value,
        )
    return WeightedSwiGLUFunction.apply(
        input_tensor,
        token_weights,
        fp8_input_store,
    )


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
        if self.config.activation_func_clamp_value is not None:
            raise RuntimeError("The H100 grouped-MM expert path does not support activation clamping")
        if self.config.moe_apply_probs_on_input:
            raise RuntimeError("The H100 grouped-MM expert path applies router probabilities after SwiGLU")
        if self.offload_expert_fc1 or self.offload_moe_act:
            raise RuntimeError("The H100 grouped-MM expert path does not support fine-grained expert offload")

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
                f"The H100 grouped-MM expert path requires integer expert counts, got {tokens_per_expert.dtype}."
            )

        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        fc1_output = _grouped_mm(
            permuted_local_hidden_states,
            self.linear_fc1._torch_grouped_weight.transpose(1, 2),
            offsets,
        )
        activation_output = _apply_weighted_swiglu(
            fc1_output.view(-1, fc1_output.shape[-1]),
            permuted_probs.unsqueeze(-1),
            fp8_input_store=self.config.activation_func_fp8_input_store,
            clamp_value=self.config.activation_func_clamp_value,
        )
        output = _grouped_mm(
            activation_output,
            self.linear_fc2._torch_grouped_weight.transpose(1, 2),
            offsets,
        )
        return output, None


class _Qwen35H100MoELayer(MoELayer):
    """Fail closed if the static HybridEP budget drops a routed token."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        manager = self.token_dispatcher._comm_manager
        if self.config.moe_flex_dispatcher_backend != "hybridep":
            raise RuntimeError("The Qwen3.5 H100 runtime requires the HybridEP dispatcher")
        manager.setup_metadata = MethodType(
            _setup_h100_static_hybridep_metadata,
            manager,
        )

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


def _replace_gdn_runtime(attention_spec: Any) -> Any:
    """Replace stock GDN attention with the measured H100 runtime."""
    if not isinstance(attention_spec, ModuleSpec) or attention_spec.module is not GatedDeltaNet:
        return attention_spec
    return replace(attention_spec, module=_Qwen35H100GatedDeltaNet)


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
        layer_spec.submodules.self_attention = _replace_gdn_runtime(layer_spec.submodules.self_attention)
    return block_spec
