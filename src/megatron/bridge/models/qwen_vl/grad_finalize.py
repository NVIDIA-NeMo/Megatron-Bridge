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

"""Gradient finalization for the combined Qwen3.5-VL model."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING

import torch
from megatron.core.distributed import DistributedDataParallel, finalize_model_grads
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.utils import get_model_config, unwrap_model
from torch.distributed.tensor import DTensor


if TYPE_CHECKING:
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider, Qwen35VLMoEModelProvider


def configure_qwen35vl_grad_finalize(config: Qwen35VLModelProvider | Qwen35VLMoEModelProvider) -> None:
    """Install the combined-model callback, including after an in-process restart."""
    callback = config.finalize_model_grads_func
    while isinstance(callback, partial):
        callback = callback.func
    if callback not in (None, finalize_qwen35vl_model_grads):
        if config.vision_sequence_parallel or (config.tensor_model_parallel_size > 1 and not config.sequence_parallel):
            raise ValueError(
                "Independent Qwen3.5-VL SP requires finalize_qwen35vl_model_grads; custom finalizers conflict"
            )
        return
    if callback is None:
        config.finalize_model_grads_func = finalize_qwen35vl_model_grads


class _FinalizeModelView(torch.nn.Module):
    """Delegate whole-model operations while reserving selected TP reductions."""

    def __init__(self, wrapped: DistributedDataParallel, owned_parameters: set[int]) -> None:
        super().__init__()
        self.module = wrapped
        self.ddp_config = wrapped.ddp_config
        self.owned_parameters = owned_parameters

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        for name, parameter in self.module.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            if id(parameter) not in self.owned_parameters:
                yield name, parameter

    def finish_grad_sync(self, force_all_reduce: bool = False) -> None:
        self.module.finish_grad_sync(force_all_reduce=force_all_reduce)

    def scale_gradients(self, scaling_factor: torch.Tensor) -> None:
        self.module.scale_gradients(scaling_factor)


def finalize_qwen35vl_model_grads(
    model: list[torch.nn.Module],
    num_tokens: torch.Tensor | None = None,
    pg_collection: ProcessGroupCollection | None = None,
    force_all_reduce: bool = False,
) -> None:
    """Finalize combined-model gradients with independent vision/language SP.

    Core performs DP synchronization, embedding synchronization and token scaling
    once. This callback owns vision SP reductions and, when language SP is off,
    GDN output-norm reductions across TP heads. Parameter views exclude those
    parameters from Core's reductions, preventing duplicate sums independently
    of whether Core gates its SP reduction on the root config.

    This callback supports ordinary Megatron DDP with a shared modality TP group.
    MIMO uses its separate per-module finalization path.

    Args:
        model: DDP-wrapped combined-model chunks.
        num_tokens: Token count forwarded to Core for loss normalization.
        pg_collection: Explicit process groups used by the training schedule.
        force_all_reduce: Forwarded to Core's DP synchronization.
    """
    if pg_collection is None:
        raise ValueError("Qwen3.5-VL gradient finalization requires explicit process groups")
    root_config = get_model_config(model[0])
    tp_group = pg_collection.tp
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for wrapped in model:
        raw = unwrap_model(wrapped)
        vision = getattr(raw, "vision_model", None)
        language = getattr(raw, "language_model", None)
        for module in (vision, language):
            if module is not None and module.pg_collection.tp is not tp_group:
                raise ValueError("Combined Qwen3.5-VL requires a shared TP group; use MIMO for separate groups")
        if tp_group.size() <= 1:
            continue
        candidates: list[torch.nn.Parameter] = []
        if vision is not None and vision.config.sequence_parallel:
            candidates.extend(p for p in vision.parameters() if getattr(p, "sequence_parallel", False))
        if language is not None and not root_config.sequence_parallel:
            for module in language.modules():
                if isinstance(module, GatedDeltaNet):
                    candidates.extend(module.out_norm.parameters())
        for parameter in candidates:
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            if getattr(parameter, "average_gradients_across_tp_domain", False):
                continue
            if not isinstance(wrapped, DistributedDataParallel):
                raise ValueError("Independent Qwen3.5-VL gradient reduction currently requires Megatron DDP")
            grad = getattr(parameter, "main_grad", None)
            if isinstance(grad, DTensor):
                raise ValueError("Independent Qwen3.5-VL gradient reduction does not support DTensor gradients")
            seen.add(id(parameter))
            parameters.append(parameter)

    if parameters and (root_config.gtp_weight_remat_size > 1 or root_config.expert_gtp_weight_remat_size > 1):
        raise ValueError("Independent Qwen3.5-VL gradient reduction does not support GTP weight rematerialization")
    views = [_FinalizeModelView(chunk, seen) for chunk in model] if parameters else model
    finalize_model_grads(views, num_tokens, pg_collection=pg_collection, force_all_reduce=force_all_reduce)

    # All ranks in this TP group use the same token divisor. Summation therefore
    # commutes with Core's scaling, and must happen after DP synchronization.
    # Coalesce by dtype/device without relying on private Core gradient helpers.
    groups: dict[tuple[torch.dtype, torch.device], list[torch.Tensor]] = {}
    for parameter in parameters:
        grad = parameter.main_grad
        if grad is not None:
            groups.setdefault((grad.dtype, grad.device), []).append(grad)
    for grads in groups.values():
        flat = torch.cat([grad.reshape(-1) for grad in grads])
        torch.distributed.all_reduce(flat, group=tp_group)
        offset = 0
        for grad in grads:
            grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
            offset += grad.numel()
