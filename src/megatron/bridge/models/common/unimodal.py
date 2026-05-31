# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import Any, Callable

import torch
from megatron.core import tensor_parallel
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    FullyShardedDataParallel,
)
from megatron.core.enums import ModelType
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_model_config


try:
    from megatron.core.distributed import TorchFullyShardedDataParallel

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

try:
    from megatron.core.fp8_utils import correct_amax_history_if_needed
except ImportError:
    correct_amax_history_if_needed = None


logger = logging.getLogger(__name__)


def unimodal_build_distributed_models(
    build_model_func: Callable,
    transformer_config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
    ddp_config: DistributedDataParallelConfig | None = None,
    overlap_param_gather_with_optimizer_step: bool = False,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = False,
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
    pre_wrap_hook: Callable[[list[MegatronModule]], list[MegatronModule]] | None = None,
    model_type: ModelType = ModelType.encoder_or_decoder,
) -> list[MegatronModule]:
    """Build model stages and wrap them for distributed training."""
    if wrap_with_ddp and not ddp_config:
        raise ValueError("ddp_config is required when wrap_with_ddp is True")

    vp_size = transformer_config.virtual_pipeline_model_parallel_size
    init_model_with_meta_device = transformer_config.init_model_with_meta_device
    if init_model_with_meta_device:
        with torch.device("meta"):
            model_list = build_virtual_pipeline_stages(build_model_func, pg_collection, vp_size, model_type)
    else:
        model_list = build_virtual_pipeline_stages(build_model_func, pg_collection, vp_size, model_type)

    if pre_wrap_hook is not None:
        if not callable(pre_wrap_hook):
            raise TypeError("pre_wrap_hook must be a callable")
        new_model_list = pre_wrap_hook(model_list)
        if new_model_list is not None:
            model_list = new_model_list
        else:
            logger.warning("Final pre-wrap hook returned None; keeping original model list.")

    for model_module in model_list:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    _print_num_params(model_list, pg_collection=pg_collection)

    use_cpu_initialization = transformer_config.use_cpu_initialization
    if not use_torch_fsdp2 and not use_cpu_initialization and not init_model_with_meta_device:
        for model_module in model_list:
            model_module.cuda(torch.cuda.current_device())

    model_list = _wrap_with_mp_wrapper(model_list, transformer_config, mixed_precision_wrapper)

    if init_model_with_meta_device and not use_torch_fsdp2 and not use_megatron_fsdp:
        model_list = [
            to_empty_if_meta_device(model_module, device=torch.device("cuda")) for model_module in model_list
        ]

    if correct_amax_history_if_needed is not None:
        correct_amax_history_if_needed(model_list)

    if wrap_with_ddp:
        model_list = _ddp_wrap(
            model_list,
            data_parallel_random_init,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            pg_collection=pg_collection,
        )

    return model_list


def _print_num_params(model: list[MegatronModule], pg_collection: ProcessGroupCollection) -> None:
    """Print model parameter count on data-parallel and context-parallel rank 0."""
    if (pg_collection.dp.rank() == 0) and (pg_collection.cp.rank() == 0):
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                pg_collection.tp.rank(),
                pg_collection.pp.rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )


def _wrap_with_mp_wrapper(
    model_list: list[MegatronModule],
    transformer_config: TransformerConfig,
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
) -> list[MegatronModule]:
    """Apply mixed-precision wrapper when configured."""
    if (transformer_config.fp16 or transformer_config.bf16) and mixed_precision_wrapper is not None:
        model_list = [mixed_precision_wrapper(transformer_config, model_module) for model_module in model_list]

        for model_module in model_list:
            for submodule in model_module.modules():
                if hasattr(submodule, "_maintain_float32_expert_bias"):
                    submodule._maintain_float32_expert_bias()

    return model_list


def _ddp_wrap(
    model: list[MegatronModule],
    data_parallel_random_init: bool,
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    *,
    pg_collection: ProcessGroupCollection,
) -> list[MegatronModule]:
    """Wrap model chunks with DDP, Megatron FSDP, or Torch FSDP2."""
    if use_megatron_fsdp:
        data_parallel_cls = FullyShardedDataParallel
        if use_torch_fsdp2:
            raise ValueError("Using use_megatron_fsdp and use_torch_fsdp2 at the same time is not supported.")
    elif use_torch_fsdp2:
        assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
        data_parallel_cls = TorchFullyShardedDataParallel
    else:
        data_parallel_cls = DistributedDataParallel

    if not use_torch_fsdp2:
        if ddp_config.num_buckets is not None:
            num_parameters = sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])
            ddp_config.bucket_size = num_parameters // ddp_config.num_buckets

        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(40000000, 1000000 * pg_collection.dp_cp.size())
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

    ddp_stream = torch.cuda.Stream()
    ddp_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(ddp_stream):
        dp_init_kwargs = {}
        if not use_torch_fsdp2:
            dp_init_kwargs["pg_collection"] = pg_collection

        wrapped_model = []
        for model_chunk_idx, model_chunk in enumerate(model):
            chunk_kwargs = dict(dp_init_kwargs)
            disable_bucketing = model_chunk_idx > 0 or overlap_param_gather_with_optimizer_step

            if ddp_config.use_distributed_optimizer and data_parallel_cls is DistributedDataParallel:
                all_params = [p for p in model_chunk.parameters() if p.requires_grad]
                pp_rank = pg_collection.pp.rank()
                effective_bucket_size = None if disable_bucketing or pp_rank > 0 else ddp_config.bucket_size
                chunk_kwargs["full_param_layout"] = DistributedOptimizer.compute_full_param_layout(
                    all_params,
                    effective_bucket_size,
                    pg_collection.dp_cp.size(),
                    ddp_config,
                    expert_data_parallel_world_size=pg_collection.expt_dp.size(),
                )

            wrapped_chunk = data_parallel_cls(
                config=get_model_config(model_chunk),
                ddp_config=ddp_config,
                module=model_chunk,
                disable_bucketing=disable_bucketing,
                **chunk_kwargs,
            )
            wrapped_model.append(wrapped_chunk)
        model = wrapped_model

    torch.cuda.current_stream().wait_stream(ddp_stream)

    if data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()

    return model


def build_virtual_pipeline_stages(
    build_model_func: Callable,
    pg_collection: ProcessGroupCollection,
    vp_size: int | None,
    model_type: ModelType = ModelType.encoder_or_decoder,
) -> list[MegatronModule]:
    """Build virtual pipeline stages if virtual pipeline parallelism is enabled."""
    from megatron.core.pipeline_parallel.utils import (
        is_pp_first_stage,
        is_pp_last_stage,
        is_vp_first_stage,
        is_vp_last_stage,
    )

    pp_group = pg_collection.pp
    if pp_group.size() > 1 and vp_size is not None:
        model_list = []
        for vp_stage in range(vp_size):
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pp_group)
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pp_group)
            model = build_model_func(
                pg_collection,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )
            model.model_type = model_type
            model_list.append(model)
    else:
        pre_process = is_pp_first_stage(pp_group)
        post_process = is_pp_last_stage(pp_group)
        model = build_model_func(pg_collection, pre_process=pre_process, post_process=post_process)
        model.model_type = model_type
        model_list = [model]

    return model_list


def to_empty_if_meta_device(module: torch.nn.Module, *, device: torch.device, recurse=True):
    """Move tensors to ``device`` while materializing meta-device tensors with empty storage."""

    def _empty_like_if_meta(tensor: torch.Tensor, *, device: torch.device):
        if tensor.device == torch.device("meta"):
            return torch.empty_like(tensor, device=device)
        return tensor.to(device)

    return module._apply(lambda t: _empty_like_if_meta(t, device=device), recurse=recurse)
