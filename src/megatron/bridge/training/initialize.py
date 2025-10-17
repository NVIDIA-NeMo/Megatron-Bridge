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

import datetime
import warnings
from typing import Callable, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train
from megatron.core.fusions.fused_bias_gelu import bias_gelu
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import configure_nvtx_profiling, get_te_version, is_te_min_version, is_torch_min_version

from megatron.bridge.models import GPTModelProvider, T5ModelProvider
from megatron.bridge.training.config import ConfigContainer, DistributedInitConfig, RerunStateMachineConfig, RNGConfig
from megatron.bridge.utils.common_utils import get_local_rank_preinit, get_rank_safe, get_world_size_safe


def initialize_megatron(
    cfg: ConfigContainer,
    allow_no_cuda: bool = False,
    skip_mpu_initialization: bool = False,
    get_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]] = None,
    restart_store: Optional[torch.distributed.Store] = None,
) -> Optional[Callable[[], None]]:
    """Initialize Megatron core components and distributed setup.

    Sets up logging, initializes distributed environment (torch.distributed),
    configures microbatch calculator, and sets random seeds.

    Args:
        cfg: The main configuration container.
        allow_no_cuda: If True, allows initialization without CUDA.
        skip_mpu_initialization: If True, skips MPU initialization (for external managers).
        get_embedding_ranks: Optional function to determine embedding layer ranks.
        get_position_embedding_ranks: Optional function to determine position embedding ranks.
        restart_store: Optional store for in-process restart.

    Returns:
        An optional callable to finish MPU initialization if lazy_mpu_init is True,
        otherwise None.
    """

    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    model_config = cfg.model
    dist_config = cfg.dist
    rng_config = cfg.rng
    rerun_state_machine_config = cfg.rerun_state_machine
    train_config = cfg.train
    use_inprocess_restart = cfg.inprocess_restart is not None and cfg.inprocess_restart.enabled

    # Configure NVTX profiling if requested
    if cfg.profiling is not None and cfg.profiling.nvtx_ranges:
        configure_nvtx_profiling(enabled=True)

    # Prep for checkpoint conversion.
    # if args.ckpt_convert_format is not None:
    #     assert args.ckpt_convert_save is not None
    #     assert args.load is not None
    #     args.exit_on_missing_checkpoint = True

    # TODO (maanug): determine if we want to support this behavior
    # if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
    #     assert args.load is not None, "--use-checkpoint-args requires --load argument"
    #     load_args_from_checkpoint(args)

    init_num_microbatches_calculator(
        get_rank_safe(),
        train_config.rampup_batch_size,
        train_config.global_batch_size,
        train_config.micro_batch_size,
        cfg.data_parallel_size,
        train_config.decrease_batch_size_if_needed,
    )

    # init rerun global state
    init_rerun_state(rerun_state_machine_config)

    # torch.distributed initialization
    return torch_dist_init(
        model_config=model_config,
        dist_config=dist_config,
        rng_config=rng_config,
        micro_batch_size=train_config.micro_batch_size,
        num_distributed_optimizer_instances=cfg.ddp.num_distributed_optimizer_instances,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        skip_mpu_initialization=skip_mpu_initialization,
        restart_store=restart_store,
        use_inprocess_restart=use_inprocess_restart,
    )


def torch_dist_init(
    model_config: GPTModelProvider | T5ModelProvider,
    dist_config: DistributedInitConfig,
    rng_config: RNGConfig,
    micro_batch_size: int,
    num_distributed_optimizer_instances: int,
    get_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]],
    get_position_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]],
    skip_mpu_initialization: bool,
    restart_store: Optional[torch.distributed.Store] = None,
    use_inprocess_restart: bool = False,
) -> Optional[Callable[[], None]]:
    """Initialize torch.distributed and dependent components.

    Handles the core distributed setup, including process group initialization,
    MPU (Model Parallel Unit) setup, random seed setting, and optional
    compilation/warmup steps.

    Args:
        model_config: Configuration for the specific model (GPTConfig or T5Config).
        dist_config: Configuration for distributed initialization settings.
        rng_config: Configuration for random number generation.
        micro_batch_size: The micro batch size for JIT warmup.
        num_distributed_optimizer_instances: Number of parallel optimizer instances.
        get_embedding_ranks: Optional function to determine embedding layer ranks.
        get_position_embedding_ranks: Optional function to determine position embedding ranks.
        skip_mpu_initialization: If True, returns a function to finish MPU setup later.

    Returns:
        An optional callable to finish MPU initialization if skip_mpu_initialization
        or lazy_mpu_init is True, otherwise None.
    """

    def finish_mpu_init():
        # Pytorch distributed.
        _initialize_distributed(
            model_config=model_config,
            dist_config=dist_config,
            num_distributed_optimizer_instances=num_distributed_optimizer_instances,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
            restart_store=restart_store,
            use_inprocess_restart=use_inprocess_restart,
        )

        # Random seeds for reproducibility.
        if get_rank_safe() == 0:
            print("> setting random seeds to {} ...".format(rng_config.seed))
        _set_random_seed(
            rng_config.seed,
            rng_config.data_parallel_random_init,
            rng_config.te_rng_tracker,
            rng_config.inference_rng_tracker,
            use_cudagraphable_rng=(model_config.cuda_graph_impl != "none"),
        )

        if model_config.num_moe_experts is not None:
            MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

    if skip_mpu_initialization:
        return None

    if dist_config.lazy_init:
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        parallel_state.set_tensor_model_parallel_world_size(model_config.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        parallel_state.set_tensor_model_parallel_rank(get_rank_safe())
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        if model_config.tp_comm_overlap:
            _initialize_tp_communicators(model_config, micro_batch_size)

        # No continuation function
        return None


def init_rerun_state(rerun_state_machine_config: RerunStateMachineConfig) -> None:
    """Initialize the rerun state machine for result validation or stats.

    Sets up state saving and restoration functions, particularly for RNG trackers.

    Args:
        rerun_state_machine_config: Configuration for the rerun state machine.
    """
    from megatron.core.rerun_state_machine import (
        RerunDiagnostic,
        RerunErrorInjector,
        RerunMode,
        initialize_rerun_state_machine,
    )

    def state_save_func():
        return {"rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states()}

    def state_restore_func(state_dict):
        if state_dict["rng_tracker_states"]:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])

    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(rerun_state_machine_config.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=rerun_state_machine_config.error_injection_rate,
            error_injection_type=RerunDiagnostic(rerun_state_machine_config.error_injection_type),
        ),
    )


def set_jit_fusion_options(model_config: GPTModelProvider | T5ModelProvider, micro_batch_size: int) -> None:
    """Set PyTorch JIT layer fusion options and warmup JIT functions.

    Configures the JIT fuser (nvFuser or legacy) based on the PyTorch version
    and warms up common fused kernels like bias_gelu and bias_dropout_add.

    Args:
        model_config: Configuration for the specific model (GPTConfig or T5Config).
        micro_batch_size: The micro batch size used for warmup tensor shapes.
    """
    # flags required to enable jit fusion kernels
    if is_torch_min_version("2.2.0a0"):
        pass  # we're using torch.compile for jit fusion
    elif is_torch_min_version("1.10.0a0"):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function(model_config, micro_batch_size)


def destroy_global_state() -> None:
    """Destroy Megatron global states.

    Cleans up resources used by microbatch calculator, global memory buffer,
    model parallel groups, and the rerun state machine.
    """
    from megatron.core.rerun_state_machine import destroy_rerun_state_machine

    destroy_num_microbatches_calculator()
    parallel_state.destroy_global_memory_buffer()
    parallel_state.destroy_model_parallel()
    destroy_rerun_state_machine()


def _initialize_tp_communicators(model_config: GPTModelProvider | T5ModelProvider, micro_batch_size: int) -> None:
    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""

    try:
        import transformer_engine  # noqa: F401
        import yaml
        from transformer_engine.pytorch import module as te_module

    except ImportError:
        raise RuntimeError(
            "Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and 'transformer_engine' packages"
        )

    if model_config.tp_comm_overlap_cfg is not None:
        if isinstance(model_config.tp_comm_overlap_cfg, str):
            with open(model_config.tp_comm_overlap_cfg, "r") as stream:
                ub_cfgs = yaml.safe_load(stream)
        else:
            ub_cfgs = model_config.tp_comm_overlap_cfg
    else:
        ub_cfgs = {}

    input_shape = [
        (model_config.seq_length * micro_batch_size) // model_config.context_parallel_size,
        model_config.hidden_size,
    ]

    if is_te_min_version("1.9.0"):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=model_config.tensor_model_parallel_size,
            use_fp8=(model_config.fp8 is not None),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=model_config.tp_comm_bootstrap_backend,
        )
    else:
        if model_config.tp_comm_bootstrap_backend != "mpi":
            warnings.warn(f"Transformer Engine v{get_te_version()} supports only MPI bootstrap backend.")
        # Create a MPI process group to help with TP communication overlap bootstrap.
        torch.distributed.new_group(backend="mpi")

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=model_config.tensor_model_parallel_size,
            use_fp8=(model_config.fp8 is not None),
            ub_cfgs=ub_cfgs,
        )


def _initialize_distributed(
    model_config: GPTModelProvider | T5ModelProvider,
    dist_config: DistributedInitConfig,
    num_distributed_optimizer_instances: int,
    get_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]],
    get_position_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]],
    restart_store: Optional[torch.distributed.Store] = None,
    use_inprocess_restart: bool = False,
) -> None:
    """Initialize torch.distributed and core model parallel."""

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if get_rank_safe() == 0:
            print(
                "torch distributed is already initialized, skipping initialization ...",
                flush=True,
            )

    else:
        if get_rank_safe() == 0:
            print("> initializing torch distributed ...", flush=True)

        # Manually set the device ids.
        if device_count > 0:
            if dist_config.external_gpu_device_mapping:
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(get_local_rank_preinit())

        # Set to non-default stream for cudagraph capturing.
        if model_config.cuda_graph_impl == "transformer_engine":
            torch.cuda.set_stream(torch.cuda.Stream())

        # Call the init process
        init_process_group_kwargs = {
            "backend": dist_config.distributed_backend,
            "world_size": get_world_size_safe(),
            "rank": get_rank_safe(),
            "store": restart_store,
            "timeout": datetime.timedelta(minutes=dist_config.distributed_timeout_minutes),
        }

        torch.distributed.init_process_group(**init_process_group_kwargs)

        # Force NCCL backend initialization if using in-process restart
        if use_inprocess_restart:
            force_nccl_backend_init(torch.cuda.current_device())

        if dist_config.external_gpu_device_mapping:
            torch.distributed.barrier(device_ids=[0])
        else:
            torch.distributed.barrier(device_ids=[get_local_rank_preinit()])

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=model_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=model_config.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=model_config.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_comm_backend=model_config.pipeline_model_parallel_comm_backend,
                context_parallel_size=model_config.context_parallel_size,
                hierarchical_context_parallel_sizes=model_config.hierarchical_context_parallel_sizes,
                expert_model_parallel_size=model_config.expert_model_parallel_size,
                num_distributed_optimizer_instances=num_distributed_optimizer_instances,
                expert_tensor_parallel_size=model_config.expert_tensor_parallel_size,
                distributed_timeout_minutes=dist_config.distributed_timeout_minutes,
                nccl_communicator_config_path=dist_config.nccl_communicator_config_path,
                order="tp-cp-ep-dp-pp" if not dist_config.use_tp_pp_dp_mapping else "tp-pp-dp",
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
                create_gloo_process_groups=dist_config.use_gloo_process_groups,
                use_sharp=dist_config.use_sharp,
                high_priority_stream_groups=dist_config.high_priority_stream_groups,
                sharp_enabled_group=dist_config.sharp_enabled_group,
            )
            if get_rank_safe() == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{parallel_state.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{parallel_state.get_pipeline_model_parallel_world_size()}"
                )


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
) -> None:
    """Set random seed for reproducability."""
    assert seed_ is not None and seed_ > 0, f"Seed ({seed_}) should be a positive integer."

    import random

    import numpy as np

    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * parallel_state.get_data_parallel_rank())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(
            seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng
        )


def _warmup_jit_function(model_config: GPTModelProvider | T5ModelProvider, micro_batch_size: int) -> None:
    """Compilie JIT functions before the main training steps"""
    if model_config.bf16:
        dtype = torch.bfloat16
    elif model_config.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Warmup fused bias+gelu
    bias = torch.rand(
        model_config.ffn_hidden_size // model_config.tensor_model_parallel_size,
        dtype=dtype,
        device="cuda",
    )
    input = torch.rand(
        (
            model_config.seq_length // model_config.context_parallel_size,
            micro_batch_size,
            model_config.ffn_hidden_size // model_config.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            if model_config.activation_func == F.silu:
                output = bias_swiglu(input, bias)
            else:
                output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if model_config.sequence_parallel:
        seq_length = model_config.seq_length // parallel_state.get_tensor_model_parallel_world_size()
    else:
        seq_length = model_config.seq_length
    input = torch.rand(
        (
            seq_length // model_config.context_parallel_size,
            micro_batch_size,
            model_config.hidden_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    residual = torch.rand(
        (
            seq_length // model_config.context_parallel_size,
            micro_batch_size,
            model_config.hidden_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.rand((model_config.hidden_size), dtype=dtype, device="cuda").expand_as(residual)
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        input.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train([input, bias], residual, dropout_rate)
    del bias, input, residual, output
    torch.cuda.empty_cache()


def force_nccl_backend_init(device_id: torch.device) -> None:
    """Force NCCL backend initialization for in-process restart compatibility.

    The nvidia-resiliency-ext in-process restart uses destroy_process_group to
    terminate the NCCL backend, which does not terminate NCCL kernels if the NCCL
    backend wasn't fully initialized before additional distributed subgroups are created.

    This function forces full initialization of the NCCL backend by performing
    a simple all_reduce operation.

    Args:
        device_id: CUDA device ID to use for the dummy tensor operation
    """
    tensor = torch.ones(128, device=device_id)
    torch.distributed.all_reduce(tensor)
    torch.cuda.synchronize()
