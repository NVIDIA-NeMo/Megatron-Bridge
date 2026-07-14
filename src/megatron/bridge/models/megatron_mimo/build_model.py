# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared model-construction entry point for MegatronMIMO.

Used by both ``setup_megatron_mimo`` (training) and the conversion CLI.
Composes ``provider.finalize`` + ``build_infra`` + per-module RNG init +
distributed-model build + ``parallel_state`` global bridge.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import tensor_parallel


if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.models.mimo import MimoModel

    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
        MegatronMIMOInfra,
        MegatronMIMOProvider,
    )


logger = logging.getLogger(__name__)


def build_megatron_mimo_model(
    provider: "MegatronMIMOProvider",
    *,
    ddp_config: Optional["DistributedDataParallelConfig"] = None,
    fp16: bool = False,
    bf16: bool = True,
    seed: int = 0,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = True,
) -> tuple["MimoModel", "MegatronMIMOInfra"]:
    """Build a distributed MegatronMIMO model and return ``(model, infra)``.

    Side effects: initialises Python/NumPy/torch/Megatron-Core RNG and sets
    ``parallel_state._*_GROUP`` globals from the rank-local pg_collection.
    """
    from megatron.bridge.training.megatron_mimo_parallel_utils import (
        get_active_module_pg,
        validate_no_stub_ranks,
    )

    if provider.megatron_mimo_parallelism_config is None:
        raise ValueError("build_megatron_mimo_model requires provider.megatron_mimo_parallelism_config to be set.")

    if not dist.is_initialized():
        raise RuntimeError(
            "build_megatron_mimo_model requires torch.distributed to be initialised. "
            "Launch with torchrun / torch.distributed.run."
        )

    logger.info("Rank %d: Building MIMO infra", dist.get_rank())
    provider.finalize()
    infra = provider.build_infra()

    world_size = dist.get_world_size()
    validate_no_stub_ranks(infra.module_to_grid_map, world_size)

    _set_per_module_random_seeds(infra, seed=seed)

    logger.info("Rank %d: Building distributed MIMO model", dist.get_rank())
    model_list = provider.provide_distributed_model(
        ddp_config=ddp_config,
        fp16=fp16,
        bf16=bf16,
        wrap_with_ddp=wrap_with_ddp,
        data_parallel_random_init=data_parallel_random_init,
    )
    mimo_model = model_list[0]

    active_module_name, local_pg_collection = get_active_module_pg(infra)
    _bridge_parallel_state_globals(local_pg_collection)
    logger.info(
        "Rank %d: bridged parallel_state globals from module %r",
        dist.get_rank(),
        active_module_name,
    )

    return mimo_model, infra


def _set_per_module_random_seeds(infra: "MegatronMIMOInfra", *, seed: int) -> None:
    """Initialise per-module RNG using TP/PP/EP/ETP ranks from the rank's module.

    This takes a raw ``seed`` int instead of reading ``cfg.rng.seed``, so callers
    outside the training loop (conversion CLI) can use it too.
    """
    current_rank = dist.get_rank()

    tp_rank = 0
    pp_rank = 0
    ep_rank = 0
    etp_rank = 0
    for module_name, grid in infra.module_to_grid_map.items():
        if grid.is_current_rank_in_grid():
            pg_collection = infra.pg_collections.get(module_name)
            if pg_collection is None:
                break
            tp_rank = dist.get_group_rank(pg_collection.tp, current_rank)
            pp_rank = dist.get_group_rank(pg_collection.pp, current_rank)
            ep_rank = dist.get_group_rank(pg_collection.ep, current_rank)
            etp_rank = dist.get_group_rank(pg_collection.expt_tp, current_rank)
            break

    # Different PP stages get different base seeds — matches the standard path.
    pp_seed = seed + (100 * pp_rank)
    random.seed(pp_seed)
    np.random.seed(pp_seed)
    torch.manual_seed(pp_seed)

    if torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(pp_seed, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=etp_rank)


def _bridge_parallel_state_globals(local_pg_collection) -> None:
    """Set ``parallel_state`` globals from the rank-local ``ProcessGroupCollection``."""
    from megatron.core import parallel_state as mpu

    if getattr(mpu, "_GLOBAL_MEMORY_BUFFER", None) is None:
        mpu._set_global_memory_buffer()

    mpu._TENSOR_MODEL_PARALLEL_GROUP = local_pg_collection.tp
    mpu._DATA_PARALLEL_GROUP = local_pg_collection.dp
    mpu._DATA_PARALLEL_GROUP_WITH_CP = getattr(local_pg_collection, "dp_cp", local_pg_collection.dp)
    if getattr(local_pg_collection, "pp", None) is not None:
        mpu._PIPELINE_MODEL_PARALLEL_GROUP = local_pg_collection.pp
    # Some conversion paths read expert groups unconditionally. TP is the
    # dense-model fallback when no dedicated EP/ETP group exists.
    if getattr(local_pg_collection, "ep", None) is not None:
        mpu._EXPERT_MODEL_PARALLEL_GROUP = local_pg_collection.ep
    else:
        mpu._EXPERT_MODEL_PARALLEL_GROUP = local_pg_collection.tp
    if getattr(local_pg_collection, "expt_tp", None) is not None:
        mpu._EXPERT_TENSOR_PARALLEL_GROUP = local_pg_collection.expt_tp
    else:
        mpu._EXPERT_TENSOR_PARALLEL_GROUP = local_pg_collection.tp
    if getattr(local_pg_collection, "expt_dp", None) is not None:
        mpu._EXPERT_DATA_PARALLEL_GROUP = local_pg_collection.expt_dp
    else:
        mpu._EXPERT_DATA_PARALLEL_GROUP = local_pg_collection.dp
    if getattr(local_pg_collection, "intra_expt_dp", None) is not None:
        mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = local_pg_collection.intra_expt_dp
    else:
        mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = mpu._EXPERT_DATA_PARALLEL_GROUP
    if getattr(local_pg_collection, "tp_ep", None) is not None:
        mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = local_pg_collection.tp_ep
    else:
        mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = mpu._EXPERT_TENSOR_PARALLEL_GROUP
    if getattr(local_pg_collection, "tp_ep_pp", None) is not None:
        mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = local_pg_collection.tp_ep_pp
    else:
        mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    if getattr(local_pg_collection, "cp", None) is not None:
        mpu._CONTEXT_PARALLEL_GROUP = local_pg_collection.cp
    else:
        mpu._CONTEXT_PARALLEL_GROUP = local_pg_collection.tp
    if getattr(local_pg_collection, "mp", None) is not None:
        mpu._MODEL_PARALLEL_GROUP = local_pg_collection.mp
    else:
        mpu._MODEL_PARALLEL_GROUP = local_pg_collection.tp
