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

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_model_config
from megatron.training.models.dist_utils import (
    _ddp_wrap as _core_ddp_wrap,
)
from megatron.training.models.dist_utils import (
    _print_num_params,  # noqa: F401
    _wrap_with_mp_wrapper,  # noqa: F401
    build_virtual_pipeline_stages,  # noqa: F401
    to_empty_if_meta_device,  # noqa: F401
    unimodal_build_distributed_models,  # noqa: F401
)


def _ddp_wrap(
    model: list[MegatronModule],
    data_parallel_random_init: bool,
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    *,
    pg_collection: ProcessGroupCollection,
    use_layer_wise_distributed_optimizer: bool = False,
    use_layer_wise_param_layout: bool = True,
) -> list[MegatronModule]:
    """Select mHC units enclosing compute and replay; delegate other models to Core.

    Split mHC layers call inner Transformer helpers directly, bypassing their
    module hooks. Replacing the unit list gives the called mHC wrapper ownership
    of those weights. Core still wraps routed experts on their expert mesh first.
    MTP projections and reused embedding/output weights remain root-owned.
    With selective mHC recomputation, a manager can replay several layers from
    one output hook. The enclosing HybridStack must own their dense weights so
    its backward entry gathers them before that replay begins.
    """
    use_mhc_units = (
        use_megatron_fsdp
        and ddp_config.megatron_fsdp_version == 2
        and any(getattr(get_model_config(chunk), "enable_mhc_connections", False) for chunk in model)
    )
    if not use_mhc_units:
        return _core_ddp_wrap(
            model,
            data_parallel_random_init,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            pg_collection=pg_collection,
            use_layer_wise_distributed_optimizer=use_layer_wise_distributed_optimizer,
            use_layer_wise_param_layout=use_layer_wise_param_layout,
        )
    if use_torch_fsdp2 or use_layer_wise_distributed_optimizer or overlap_param_gather_with_optimizer_step:
        raise ValueError("mHC MFSDP v2 requires its native optimizer and parameter-gather scheduling")
    if data_parallel_random_init:
        raise ValueError("MFSDP v2 does not support data-parallel random initialization")
    if any(get_model_config(chunk).overlap_moe_expert_parallel_comm for chunk in model):
        raise ValueError("mHC MFSDP v2 requires automatic hooks; EP communication overlap is unsupported")

    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer

    wrapped_model = []
    for chunk in model:
        config = get_model_config(chunk)
        mhc_recompute = config.recompute_granularity == "selective" and "mhc" in (config.recompute_modules or [])
        wrapped_model.append(
            FullyShardedDataParallelV2(
                config=config,
                ddp_config=ddp_config,
                module=chunk,
                fsdp_unit_modules=[HybridStack if mhc_recompute else HyperConnectionHybridLayer],
                disable_bucketing=False,
                pg_collection=pg_collection,
            )
        )
    return wrapped_model
