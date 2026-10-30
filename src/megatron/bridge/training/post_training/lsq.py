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

"""Scale learning (LSQ) for ModelOpt-quantized models.

ModelOpt's LSQ algorithm makes the per-block amax of static-block weight quantizers learnable
(``_amax_pre`` / ``_amax_post``). This module gives expert scales the expert reduction group, an
optional optimizer parameter group for the scales, and a pre-wrap hook for scale-only learning.
"""

from dataclasses import dataclass
from functools import wraps

import torch
from megatron.core.optimizer import ParamKey
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.training.config import OptimizerConfigOverrideProvider, OptimizerConfigOverrideProviderContext
from megatron.bridge.utils.common_utils import print_rank_0, warn_rank_0


LSQ_SCALE_PARAM_PATTERNS = ("*_amax_pre", "*_amax_post")
"""Name patterns (``fnmatch``) of the ModelOpt LSQ scale parameters."""

_LSQ_SCALE_PARAM_KEY = ParamKey(name=LSQ_SCALE_PARAM_PATTERNS)


def propagate_expert_allreduce_to_lsq_params(model: torch.nn.Module) -> int:
    """Copy each quantized weight's ``allreduce`` attribute onto its LSQ scale parameters.

    Megatron-Core reduces and shards every parameter without ``allreduce=False`` over the
    data-parallel group, so expert scales need the flag of their expert weight. Run after the
    ModelOpt state is restored and before DDP wrapping.

    Args:
        model: Unwrapped model chunk.

    Returns:
        Number of scale parameters updated.
    """
    num_updated = 0
    for module in model.modules():
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if not isinstance(weight_quantizer, torch.nn.Module):
            continue
        scale_params = [
            param for name, param in weight_quantizer.named_parameters() if _LSQ_SCALE_PARAM_KEY.matches(param, name)
        ]
        if not scale_params:
            continue
        # Grouped-GEMM experts own weight0..weightN behind one quantizer container; all share one flag.
        allreduce_flags = {
            param.allreduce
            for name, param in module.named_parameters(recurse=False)
            if name.startswith("weight") and hasattr(param, "allreduce")
        }
        if not allreduce_flags:
            continue
        if len(allreduce_flags) > 1:
            raise ValueError(
                f"Weights of {type(module).__name__} disagree on the 'allreduce' attribute; "
                "cannot choose a reduction group for its LSQ scale parameters."
            )
        allreduce = allreduce_flags.pop()
        for param in scale_params:
            param.allreduce = allreduce
        num_updated += len(scale_params)
    return num_updated


def freeze_all_but_lsq_scales(model: list[MegatronModule]) -> list[MegatronModule]:
    """Pre-wrap hook for scale-only learning: freeze every parameter except the LSQ scales.

    Frozen weights take no DDP gradient buffer or optimizer state, and MoE router expert biases stop
    updating. The hook may run before or after the ModelOpt state is restored; the scales the restore
    creates stay trainable. Gradients through frozen quantized weights need a ModelOpt version that
    computes them.

    Args:
        model: Unwrapped model chunks.

    Returns:
        The same model chunks.

    Raises:
        ValueError: If the restored ModelOpt state has no LSQ scale parameters.
    """
    import modelopt.torch.opt as mto

    num_scales = num_frozen = 0
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            is_scale = _LSQ_SCALE_PARAM_KEY.matches(param, name)
            num_scales += is_scale
            num_frozen += param.requires_grad and not is_scale
            param.requires_grad_(is_scale)
        for module in model_chunk.modules():
            if hasattr(module, "expert_bias"):
                module.frozen_expert_bias = True
        _require_grad_on_recompute_inputs(model_chunk)
    if num_scales == 0 and any(mto.ModeloptStateManager.is_converted(chunk) for chunk in model):
        raise ValueError(f"No LSQ scale parameters {LSQ_SCALE_PARAM_PATTERNS}; quantize with an LSQ recipe.")
    print_rank_0(f"LSQ scale-only learning: froze {num_frozen} parameters")
    return model


def _require_grad_on_recompute_inputs(model: torch.nn.Module) -> None:
    """Make recomputed transformer blocks with trainable parameters take a grad-requiring input.

    Megatron-Core's reentrant activation recompute skips a chunk's backward when no input requires
    grad, as for the first block once the embedding is frozen (same fix as PEFT's
    ``maybe_enable_recompute_inputs_grad``). Trainability is checked when the block runs, since the
    scales may be created after this hook; a distillation teacher's blocks run unchanged.
    """
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.transformer.transformer_block import TransformerBlock

    for module in model.modules():
        if (
            not isinstance(module, (TransformerBlock, HybridStack))
            or module.config.recompute_granularity is None
            or getattr(module.forward, "_lsq_requires_grad_input", False)
        ):
            continue

        @wraps(module.forward)
        def forward(hidden_states, *args, _forward=module.forward, _block=module, **kwargs):
            # Later pipeline stages pass None; inference may pass a WrappedTensor.
            if (
                torch.is_grad_enabled()
                and torch.is_tensor(hidden_states)
                and hidden_states.is_floating_point()
                and not hidden_states.requires_grad
                and any(param.requires_grad for param in _block.parameters())
            ):
                hidden_states = hidden_states.detach().requires_grad_(True)
            return _forward(hidden_states, *args, **kwargs)

        forward._lsq_requires_grad_input = True
        module.forward = forward


@dataclass
class LSQOptimizerConfigOverrideProvider(OptimizerConfigOverrideProvider):
    """Put ModelOpt LSQ scale parameters in their own optimizer parameter group.

    The group follows the base learning-rate schedule with its own peak and minimum learning rate.
    Only settings that differ from the base config are overridden: a scale group identical to the
    base group would be conflated with it when optimizer state is loaded from a checkpoint.

    Args:
        scale_lr: Peak learning rate of the scales.
        scale_min_lr: Minimum learning rate of the scales; ``None`` uses ``optimizer.min_lr``.
        scale_wd_mult: Weight-decay multiplier of the scales; ``0.0`` keeps weight decay from pulling
            them toward zero.
    """

    scale_lr: float
    scale_min_lr: float | None
    scale_wd_mult: float

    def build_config_overrides(
        self, context: OptimizerConfigOverrideProviderContext
    ) -> dict[ParamKey, ParamGroupOverride] | None:
        """Standard overrides plus a parameter group for the LSQ scales.

        Args:
            context: Scheduler, optimizer, and model context for override construction.

        Returns:
            Mapping from ``ParamKey`` matchers to per-group optimizer overrides, or ``None``.
        """
        config_overrides = super().build_config_overrides(context) or {}
        optimizer_config = context.optimizer_config
        scale_override = ParamGroupOverride()
        if self.scale_lr != optimizer_config.lr:
            scale_override["max_lr"] = self.scale_lr
        if self.scale_min_lr is not None and self.scale_min_lr != optimizer_config.min_lr:
            scale_override["min_lr"] = self.scale_min_lr
        if self.scale_wd_mult != 1.0:
            scale_override["wd_mult"] = self.scale_wd_mult
        if scale_override:
            config_overrides[_LSQ_SCALE_PARAM_KEY] = scale_override

        model_chunks = context.model if isinstance(context.model, list) else [context.model]
        if not any(
            param.requires_grad and _LSQ_SCALE_PARAM_KEY.matches(param, name)
            for chunk in model_chunks
            for name, param in chunk.named_parameters()
        ):
            warn_rank_0(f"No trainable LSQ scale parameters {LSQ_SCALE_PARAM_PATTERNS}; quantize with an LSQ recipe.")
        return config_overrides or None
