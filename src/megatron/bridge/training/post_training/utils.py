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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import modelopt.torch.distill as mtd
import torch
from megatron.core import parallel_state
from megatron.core.transformer import MegatronModule
from modelopt.torch.distill.plugins.megatron import (
    DistillationConfig,
    adjust_distillation_model_for_mcore,
    setup_distillation_config,
)

from megatron.bridge.models.conversion import AutoBridge
from megatron.bridge.utils.vocab_utils import validate_and_set_vocab_size


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.state import GlobalState


@dataclass
class ModelOptDistillConfig:
    """Configuration settings for Model Optimizer distillation."""

    teacher_path_or_id: str
    """Path to the teacher checkpoint or HF model ID."""

    logit_layers: tuple[str, str] = ("output_layer", "output_layer")
    """Layer names to use for logit distillation."""

    intermediate_layer_pairs: list[tuple[str, ...]] = field(default_factory=list)
    """Layer names to use for intermediate distillation."""

    skip_lm_loss: bool = True
    """Whether to skip the original LM loss computation."""

    kd_loss_scale: float = 1.0
    """Scale for weighing the KD loss, if original LM loss is not skipped."""

    logit_kl_temperature: float = 1.0
    """Temperature for the logit KL divergence."""


def create_modelopt_pre_wrap_hook(
    cfg: "ConfigContainer", state: "GlobalState"
) -> Callable[[list[MegatronModule]], list[MegatronModule]]:
    """Create a pre-wrap hook that handles ModelOpt logic.

    This hook is executed before the model is wrapped with parallelism wrappers.
    """

    def modelopt_pre_wrap_hook(model: list[MegatronModule]) -> list[MegatronModule]:
        """Pre-wrap hook that handles ModelOpt transformation(s).

        Args:
            model: List of base model modules before distributed wrapping

        Returns:
            List of potentially ModelOpt-transformed model modules
        """
        # Only apply ModelOpt logic if ModelOpt is configured
        if cfg.modelopt is None:
            return model

        # Knowledge Distillation case
        if cfg.modelopt.kd is not None:
            if len(model) > 1:
                raise ValueError("ModelOpt KD currently does not support virtual-pipeline parallel.")
            student_model = model[0]
            teacher_model = _load_teacher_model(cfg.modelopt.kd.teacher_path_or_id, cfg, state)

            kd_cfg = DistillationConfig(
                logit_layers=cfg.modelopt.kd.logit_layers,
                intermediate_layer_pairs=cfg.modelopt.kd.intermediate_layer_pairs,
                skip_lm_loss=cfg.modelopt.kd.skip_lm_loss,
                kd_loss_scale=cfg.modelopt.kd.kd_loss_scale,
                logit_kl_temperature=cfg.modelopt.kd.logit_kl_temperature,
            )
            kd_cfg = setup_distillation_config(kd_cfg, student_model.config, teacher_model.config)
            modelopt_cfg = {
                "teacher_model": teacher_model,
                "criterion": kd_cfg.criterion,
                "loss_balancer": kd_cfg.loss_balancer,
            }
            mtd.convert(student_model, mode=[("kd_loss", modelopt_cfg)])
            adjust_distillation_model_for_mcore(student_model, kd_cfg)

        return model

    return modelopt_pre_wrap_hook


def _load_teacher_model(teacher_model_path_or_id: str, cfg: "ConfigContainer", state: "GlobalState") -> MegatronModule:
    """Load the teacher model from a Megatron Bridge checkpoint path or HF model ID.

    Args:
        teacher_model_path_or_id: Path to the teacher checkpoint or HF model ID.
        cfg: Main config instance.
        state: Global state object.

    Returns:
        GPTModelProvider or MambaProvider: The teacher model provider.
    """
    from megatron.bridge.training.checkpointing import _load_checkpoint_from_path
    from megatron.bridge.training.model_load_save import load_model_config
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists

    # Obtain provider one way or another
    is_megatron_ckpt = checkpoint_exists(teacher_model_path_or_id)
    if is_megatron_ckpt:
        provider, _ = load_model_config(teacher_model_path_or_id)
    else:
        bridge = AutoBridge.from_hf_pretrained(teacher_model_path_or_id, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=True)

    # Additional setup adjustments
    cfg.mixed_precision.setup(provider)
    provider.vocab_size, provider.should_pad_vocab = validate_and_set_vocab_size(
        model_vocab_size=provider.vocab_size,
        tokenizer_vocab_size=build_tokenizer(cfg.tokenizer).vocab_size,
    )

    # Instantiate model from provider
    provider.finalize()
    model = provider.provide(
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )

    # Directly call load_checkpoint_from path in order to avoid
    # the load directory overriding the pretrained checkpoint path
    if is_megatron_ckpt:
        _load_checkpoint_from_path(
            load_dir=teacher_model_path_or_id,
            state=state,
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
        )

    return model


def _mask_loss(output_tensor: torch.Tensor, loss_mask: torch.Tensor):
    if isinstance(output_tensor, tuple):
        # Special distillation flags indicating whether to perform additional tensor-parallel adjustments.
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
    else:
        tp_reduce, is_sequence_parallel = False, False
    tp_group = parallel_state.get_tensor_model_parallel_group()

    if is_sequence_parallel:
        # Sequence-parallel tensor derived from intermediate activation - need to split loss mask.
        idx = tp_group.rank()
        loss_mask = torch.tensor_split(loss_mask, tp_group.size(), dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)

    if tp_reduce or is_sequence_parallel:
        # Losses on parallel tensors require extra all-reduce to sync across MP ranks.
        torch.distributed.all_reduce(loss, group=tp_group)

    return loss


def loss_func_kd(
    output_tensor: torch.Tensor, loss_mask: torch.Tensor, original_loss_fn: Callable, model: MegatronModule
):
    """Loss function (with KD Loss support).

    Args:
        output_tensor (Tensor): The tensor with the losses
        loss_mask (Tensor): Used to mask out some portions of the loss
        original_loss_fn (Callable): The original loss function
        model (GPTModel): The model (can be wrapped)
    """
    assert isinstance(model, mtd.DistillationModel), "Model must be a ModelOpt DistillationModel"

    # Standard lm loss
    loss_lm, num_tokens, report = original_loss_fn(output_tensor)

    # Handle knowledge distillation
    losses_kd = model.compute_kd_loss(
        student_loss=loss_lm,
        loss_reduction_fn=lambda x: _mask_loss(x, loss_mask),
    )

    report["total loss"] = torch.cat([losses_kd["kd_loss"].clone().detach().view(1), num_tokens.view(1)])
    report["logits distillation loss"] = torch.cat(
        [losses_kd["logits_loss"].clone().detach().view(1), num_tokens.view(1)]
    )
    report["intermediate distillation loss"] = torch.cat(
        [losses_kd["intermediate_loss"].clone().detach().view(1), num_tokens.view(1)]
    )

    # Validation loss remains unchanged
    if model.training:
        loss = losses_kd["kd_loss"]
    else:
        loss = loss_lm

    return loss, num_tokens, report
