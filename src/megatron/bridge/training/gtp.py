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

"""Generalized Tensor Parallelism helpers for the standard Bridge runtime."""

from typing import Any

import torch
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection


def get_transformer_config(model_config: Any) -> Any:
    """Return the MCore transformer config nested in a Bridge model config."""
    model_fields = getattr(type(model_config), "__dataclass_fields__", {})
    if "transformer" in model_fields:
        return model_config.transformer
    return model_config


def is_gtp_remat_active(model_config: Any) -> bool:
    """Return whether dense or expert GTP weight rematerialization is enabled."""
    transformer_config = get_transformer_config(model_config)
    dense_size = getattr(transformer_config, "gtp_weight_remat_size", 1)
    expert_size = getattr(transformer_config, "expert_gtp_weight_remat_size", 1)
    return any(isinstance(size, int) and size > 1 for size in (dense_size, expert_size))


def configure_gtp_remat(model_config: Any) -> None:
    """Configure process-global GTP state before constructing model modules."""
    if not is_gtp_remat_active(model_config):
        return

    transformer_config = get_transformer_config(model_config)
    try:
        from megatron.core.tensor_parallel import gtp_api
    except ImportError as error:
        raise RuntimeError("GTP requires TransformerEngine >= 2.19.") from error

    if not gtp_api.HAVE_GTP:
        raise RuntimeError("GTP requires TransformerEngine >= 2.19.")

    gtp_api.configure_gtp_remat_from_recipe(
        fp4=transformer_config.fp4 is not None,
        fp8_recipe=transformer_config.fp8_recipe,
        fp8=transformer_config.fp8 is not None,
        calculate_per_token_loss=transformer_config.calculate_per_token_loss,
    )


def classify_gtp_remat_chains(model: list[torch.nn.Module], model_config: Any) -> None:
    """Classify all model chunks after distributed wrapping and before first forward."""
    if not is_gtp_remat_active(model_config):
        return

    transformer_config = get_transformer_config(model_config)
    try:
        from megatron.core.tensor_parallel import gtp_api
    except ImportError as error:
        raise RuntimeError("GTP requires TransformerEngine >= 2.19.") from error

    gtp_api.classify_gtp_remat_chains(
        model,
        cuda_graph_modules=transformer_config.cuda_graph_modules,
        moe_shared_expert_overlap=transformer_config.moe_shared_expert_overlap,
        cuda_graph_impl=transformer_config.cuda_graph_impl,
    )


def get_data_distribution_group(
    pg_collection: ProcessGroupCollection,
    model_config: Any,
    *,
    with_context_parallel: bool = False,
) -> torch.distributed.ProcessGroup:
    """Return the group spanning every rank that consumes distinct input data."""
    if not is_gtp_remat_active(model_config):
        return pg_collection.dp_cp if with_context_parallel else pg_collection.dp
    if with_context_parallel:
        return pg_collection.dp_cp_gtp_remat
    return parallel_state.get_data_parallel_group(with_gtp_remat=True)
