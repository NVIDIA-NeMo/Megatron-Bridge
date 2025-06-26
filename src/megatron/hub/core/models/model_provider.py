# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Model provider utilities for Megatron-Core models.

This module provides infrastructure for creating and configuring models
with various parallelism strategies including tensor parallelism (TP),
pipeline parallelism (PP), virtual pipeline parallelism (VPP), and
data parallelism (DP). It handles model initialization, wrapping,
and configuration for distributed training.
"""

from typing import Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    TorchFullyShardedDataParallel,
)
from megatron.core.enums import ModelType
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.utils import get_model_config


try:
    from megatron.core.fp8_utils import correct_amax_history_if_needed
except ImportError:
    correct_amax_history_if_needed = None


def get_model(
    model_provider: Callable[[bool, bool, int], nn.Module],
    ddp_config: DistributedDataParallelConfig,
    model_type=ModelType.encoder_or_decoder,
    overlap_param_gather_with_optimizer_step: bool = False,
    fp16: bool | None = None,
    bf16: bool | None = None,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = True,
    use_cpu_initialization: None | bool = False,
    init_model_with_meta_device: bool | None = None,
    pre_wrap_hook: Callable[[list[MegatronModule]], list[MegatronModule]] | None = None,
) -> list[MegatronModule]:
    """Create and configure a model for distributed training.

    This function handles the complete model creation pipeline including:
    - Model instantiation with proper pipeline parallel configuration
    - GPU memory allocation
    - Mixed precision (FP16/BF16) wrapping
    - Float8 tensor correction
    - Distributed Data Parallel (DDP) wrapping

    Args:
        model_provider: Callable that creates the model. Should accept optional
            pre_process(bool), post_process(bool), vp_stage(int) arguments for pipeline parallelism
        ddp_config: Configuration for distributed data parallel training
        model_type: Type of model (encoder, decoder, or encoder_and_decoder)
        overlap_param_gather_with_optimizer_step: Whether to overlap parameter
            gathering with optimizer step for performance optimization
        fp16: Enable FP16 mixed precision training. If None, uses model config
        bf16: Enable BF16 mixed precision training. If None, uses model config
        use_torch_fsdp2: Use PyTorch's Fully Sharded Data Parallel v2
        wrap_with_ddp: Whether to wrap the model with DDP
        data_parallel_random_init: Whether to use random initialization for
            data parallel ranks (vs broadcasting from rank 0)
        use_cpu_initialization: Whether to initialize model on CPU to save GPU memory
        init_model_with_meta_device: Whether to initialize the model on the meta device
        pre_wrap_hook: A callable that takes a list of `MegatronModule` and returns a
            modified list, or `None` to clear the hook.

    Returns:
        list[MegatronModule]: List of model modules. Contains multiple modules
            when using virtual pipeline parallelism, otherwise a single module
    """
    model = _create_model(model_provider, model_type, init_model_with_meta_device=init_model_with_meta_device)

    if pre_wrap_hook:
        _model = pre_wrap_hook(model)
        if _model is not None:
            model = _model

    _print_num_params(model)

    model_config = get_model_config(model[0])
    if use_cpu_initialization:
        model_config.use_cpu_initialization = use_cpu_initialization

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if not (use_torch_fsdp2 or model_config.use_cpu_initialization) and not model_config.init_model_with_meta_device:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    if fp16:
        model_config.fp16 = fp16
    if bf16:
        model_config.bf16 = bf16
    if model_config.fp16 or model_config.bf16:
        model = [Float16Module(model_config, model_module) for model_module in model]

    if correct_amax_history_if_needed is not None:
        correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        model = _ddp_wrap(
            model,
            use_torch_fsdp2,
            data_parallel_random_init,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
        )

    return model


def _create_model(
    model_provider: Callable[..., nn.Module],
    model_type: ModelType,
    init_model_with_meta_device: bool = False,
) -> list[MegatronModule]:
    """Create model instances with appropriate pipeline parallel configuration.

    Handles virtual pipeline parallelism (VPP) by creating multiple model
    instances when needed. Sets pre_process and post_process flags based on
    pipeline parallel rank.

    Args:
        model_provider: Callable that creates the model
        model_type: ModelType enum indicating encoder, decoder, or both

    Returns:
        list: List of model instances. Multiple instances for VPP, otherwise single
    """

    def build_model():
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            assert model_type != ModelType.encoder_and_decoder, (
                "Interleaved schedule not supported for model with both encoder and decoder"
            )
            model = []
            for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
                post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
                this_model = model_provider(
                    pre_process=pre_process,
                    post_process=post_process,
                    vp_stage=i,
                )
                this_model.model_type = model_type
                model.append(this_model)
        else:
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            if model_type == ModelType.encoder_and_decoder:
                if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                    rank = parallel_state.get_pipeline_model_parallel_rank()
                    first_decoder_rank = parallel_state.get_pipeline_model_parallel_decoder_start()
                    world_size = parallel_state.get_pipeline_model_parallel_world_size()
                    pre_process = rank == 0 or rank == first_decoder_rank
                    post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
                model = model_provider()
            else:
                model = model_provider(
                    pre_process=pre_process,
                    post_process=post_process,
                )
            model.model_type = model_type
        return model

    if init_model_with_meta_device:
        with torch.device("meta"):
            model = build_model()
    else:
        model = build_model()

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        if init_model_with_meta_device:
            model_module.config.init_model_with_meta_device = True

    return model


def _ddp_wrap(
    model: list[MegatronModule],
    use_torch_fsdp2: bool,
    data_parallel_random_init: bool,
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool,
) -> list[MegatronModule]:
    """Wrap model with Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP).

    Args:
        model: List of model modules to wrap
        use_torch_fsdp2: Whether to use PyTorch FSDP v2 instead of DDP
        data_parallel_random_init: Whether to broadcast parameters from rank 0
        ddp_config: Configuration for distributed data parallel
        overlap_param_gather_with_optimizer_step: Whether to disable bucketing
            for overlapping parameter gathering with optimizer step

    Returns:
        list[MegatronModule]: List of DDP/FSDP wrapped model modules
    """
    if use_torch_fsdp2:
        DP = TorchFullyShardedDataParallel
    else:
        DP = DistributedDataParallel

    model = [
        DP(
            config=get_model_config(model_chunk),
            ddp_config=ddp_config,
            module=model_chunk,
            # Turn off bucketing for model_chunk 2 onwards, since communication for these
            # model chunks is overlapped with compute anyway.
            disable_bucketing=(model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step,
        )
        for (model_chunk_idx, model_chunk) in enumerate(model)
    ]

    # Broadcast params from data parallel src rank to other data parallel ranks.
    if data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()

    return model


def _print_num_params(model: list[MegatronModule]) -> None:
    """Print the number of parameters in the model on rank 0.

    Only prints on data parallel rank 0 to avoid duplicate output.
    Shows parameter count per (tensor parallel, pipeline parallel) rank.

    Args:
        model: List of model modules to count parameters from
    """
    if parallel_state.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )


@runtime_checkable
class ModelProviderProtocol(Protocol):
    """Protocol defining the interface for model providers.

    This protocol ensures that model provider classes implement the required
    get_model method with the correct signature. It's used for type checking
    and documentation of the expected interface.

    Model providers should implement this protocol to create models compatible
    with Megatron-Core's distributed training infrastructure.
    """

    def get_model(
        self,
        ddp_config: DistributedDataParallelConfig,
        model_type=ModelType.encoder_or_decoder,
        overlap_param_gather_with_optimizer_step: bool = False,
        fp16: bool | None = None,
        bf16: bool | None = None,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        use_cpu_initialization: None | bool = False,
    ):
        """Get a configured model for distributed training.

        Args:
            ddp_config: Configuration for distributed data parallel
            model_type: Type of model (encoder, decoder, or both)
            overlap_param_gather_with_optimizer_step: Enable overlapping for performance
            fp16: Enable FP16 mixed precision
            bf16: Enable BF16 mixed precision
            use_torch_fsdp2: Use PyTorch FSDP v2
            wrap_with_ddp: Whether to wrap with DDP
            data_parallel_random_init: Use random init across data parallel ranks
            use_cpu_initialization: Initialize on CPU to save GPU memory

        Returns:
            Configured model(s) ready for distributed training
        """
        ...
