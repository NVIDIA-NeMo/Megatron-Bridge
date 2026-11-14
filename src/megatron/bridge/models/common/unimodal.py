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
    """Dispatch DeepSeek-V4 mHC FSDP v2 wrapping; delegate all other paths to Core."""
    if use_megatron_fsdp and ddp_config.megatron_fsdp_version == 2:
        from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import DeepSeekV4HybridModelProvider

        if model and all(
            isinstance(get_model_config(chunk), DeepSeekV4HybridModelProvider)
            and get_model_config(chunk).enable_mhc_connections
            for chunk in model
        ):
            from megatron.bridge.models.deepseek.deepseek_v4_fsdp import wrap_deepseek_v4_fsdp

            return wrap_deepseek_v4_fsdp(
                model=model,
                data_parallel_random_init=data_parallel_random_init,
                ddp_config=ddp_config,
                overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
                use_torch_fsdp2=use_torch_fsdp2,
                pg_collection=pg_collection,
                use_layer_wise_distributed_optimizer=use_layer_wise_distributed_optimizer,
            )

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
