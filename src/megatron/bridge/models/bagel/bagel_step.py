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

"""BAGEL batch preparation, forward step, and official-style loss."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import torch

from megatron.bridge.models.bagel.data.batch import bagel_packed_batch_to_mimo
from megatron.bridge.models.bagel.diffusion import BagelDiffusionScheduler
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection


def bagel_loss(
    ce_loss: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
    mse_loss: torch.Tensor | None,
    mse_loss_mask: torch.Tensor | None,
    dp_cp_group: torch.distributed.ProcessGroup,
    ce_weight: float,
    mse_weight: float,
    ce_loss_reweighting: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce CE and MSE with official BAGEL token normalization."""
    world_size = torch.distributed.get_world_size(dp_cp_group)
    loss = torch.zeros((), device=ce_loss.device)
    metrics: dict[str, torch.Tensor] = {}

    ce_values = ce_loss.float()
    ce_weights = loss_mask.flatten()[loss_mask.flatten() > 0].float()
    if ce_values.numel() != ce_weights.numel():
        raise RuntimeError("BAGEL CE loss and weight counts differ")
    if ce_loss_reweighting:
        ce_sum = (ce_values * ce_weights).sum()
        ce_denominator = ce_weights.sum()
    else:
        ce_sum = ce_values.sum()
        ce_denominator = torch.tensor(ce_values.numel(), dtype=torch.float, device=ce_values.device)
    ce_denominator_global = ce_denominator.clone()
    torch.distributed.all_reduce(ce_denominator_global, group=dp_cp_group)
    ce_term = ce_sum * world_size / ce_denominator_global.clamp_min(1)
    loss = loss + ce_weight * ce_term
    metrics["ce"] = torch.stack((ce_sum.detach(), ce_denominator.detach()))

    if mse_loss is not None and mse_loss_mask is not None:
        mse_sum = mse_loss.float().mean(dim=-1).sum()
        mse_denominator = mse_loss_mask.sum().float()
        mse_denominator_global = mse_denominator.clone()
        torch.distributed.all_reduce(mse_denominator_global, group=dp_cp_group)
        mse_term = mse_sum * world_size / mse_denominator_global.clamp_min(1)
        loss = loss + mse_weight * mse_term
        metrics["mse"] = torch.stack((mse_sum.detach(), mse_denominator.detach()))

    metrics["loss"] = torch.stack((loss.detach(), torch.ones_like(loss)))
    return loss, torch.ones((), dtype=torch.int, device=loss.device), metrics


class BagelForwardStep:
    """Run one BAGEL packed batch through PR #3635's MCore model."""

    def __init__(self) -> None:
        self.scheduler: BagelDiffusionScheduler | None = None

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: torch.nn.Module,
    ) -> tuple[torch.Tensor, partial]:
        """Prepare modalities, run MIMO, and bind the BAGEL loss."""
        config = state.cfg.model
        if self.scheduler is None:
            if config.bagel_repo is None or config.vae_path is None:
                raise ValueError("BAGEL training requires model.bagel_repo and model.vae_path")
            self.scheduler = BagelDiffusionScheduler(
                bagel_repo=config.bagel_repo,
                vae_path=config.vae_path,
                latent_patch_size=config.latent_patch_size,
                timestep_shift=config.timestep_shift,
                dtype=config.params_dtype,
            )

        state.timers("batch-generator", log_level=2).start()
        with state.straggler_timer(bdata=True):
            packed_batch = next(data_iterator)
            batch = bagel_packed_batch_to_mimo(
                packed_batch,
                self.scheduler,
                num_attention_heads=config.num_attention_heads,
            )
        state.timers("batch-generator").stop()

        output = model(**batch)
        if not isinstance(output, tuple) or len(output) != 4:
            raise RuntimeError("BAGEL PP=1 model must return CE, MSE, MSE mask, and CE mask")
        ce_loss, mse_loss, mse_loss_mask, loss_mask = output
        pg_collection = get_pg_collection(model)
        return ce_loss, partial(
            bagel_loss,
            loss_mask=loss_mask,
            mse_loss=mse_loss,
            mse_loss_mask=mse_loss_mask,
            dp_cp_group=pg_collection.dp_cp,
            ce_weight=getattr(config, "ce_weight", 1.0),
            mse_weight=getattr(config, "mse_weight", 1.0),
            ce_loss_reweighting=getattr(config, "ce_loss_reweighting", False),
        )
