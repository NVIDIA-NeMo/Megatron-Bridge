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

import inspect
import logging
import time
from functools import partial
from typing import Any, Callable, NamedTuple

import torch
from megatron.core.config import set_experimental_flag
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel as megatron_FSDP
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer import MegatronModule

from megatron.bridge.data.loaders import setup_data_iterators
from megatron.bridge.models import GPTModelProvider, T5ModelProvider
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    _load_checkpoint_from_path,
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
)
from megatron.bridge.training.config import ConfigContainer, runtime_config_update
from megatron.bridge.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.log_utils import append_to_progress_log, barrier_and_log, setup_logging
from megatron.bridge.utils.common_utils import print_rank_0, get_rank_safe



class SetupOutput(NamedTuple):
    """Represents the output of the main setup function.

    Contains all the initialized components necessary for training or evaluation.

    Attributes:
        state: The global state object holding configuration and runtime information.
        model: The initialized Megatron model.
        optimizer: The initialized optimizer.
        scheduler: The initialized learning rate scheduler.
        train_data_iterator: The data iterator for the training dataset, if applicable.
        valid_data_iterator: The data iterator for the validation dataset, if applicable.
        test_data_iterator: The data iterator for the testing dataset, if applicable.
        checkpointing_context: A dictionary holding context for checkpointing operations,
                               especially for non-persistent local checkpointing.
    """

    state: GlobalState
    model: MegatronModule
    optimizer: MegatronOptimizer
    scheduler: OptimizerParamScheduler
    train_data_iterator: RerunDataIterator | list[RerunDataIterator] | None
    valid_data_iterator: RerunDataIterator | list[RerunDataIterator] | None
    test_data_iterator: RerunDataIterator | list[RerunDataIterator] | None
    checkpointing_context: dict[str, Any]

def setup(
    state: GlobalState,
    train_valid_test_datasets_provider: Callable[..., tuple[Any | None, Any | None, Any | None]],
    get_embedding_ranks: Callable[[list[int], int | None], list[int]] | None = None,
    get_position_embedding_ranks: Callable[[list[int], int | None], list[int]] | None = None,
    restart_store: torch.distributed.Store | None = None,
) -> SetupOutput:
    """Initialize the training/evaluation environment using an existing GlobalState.

    Performs all runtime setup using the provided `state` and its attached config (`state.cfg`).
    This includes:
      - enabling Megatron-Core experimental features
      - initializing async checkpoint workers (if enabled)
      - logging setup
      - torch.distributed and model-parallel initialization (via initialize_megatron)
      - tokenizer/model/optimizer/scheduler construction
      - optional checkpoint load
      - dataloader setup

    Args:
        state: The GlobalState instance to populate and use throughout setup.
        train_valid_test_datasets_provider: Callable returning the train/valid/test datasets or iterators.
        get_embedding_ranks: Optional function to determine embedding layer ranks for model-parallel init.
        get_position_embedding_ranks: Optional function to determine positional embedding ranks.
        restart_store: Optional torch.distributed Store used when in-process restart is enabled.

    Returns:
        SetupOutput containing the populated state, model, optimizer, scheduler, dataloaders, and ckpt context.
    """
    cfg = state.cfg

    # Conditionally enable experimental features for Megatron Core
    set_experimental_flag(cfg.dist.enable_megatron_core_experimental)

    # Initialize async checkpoint worker if enabled (idempotent if already initialized)
    state.initialize_async_checkpoint_worker()

    setup_logging(
        logging_level=cfg.logger.logging_level,
        filter_warning=cfg.logger.filter_warnings,
        modules_to_filter=cfg.logger.modules_to_filter,
        set_level_for_all_loggers=cfg.logger.set_level_for_all_loggers,
    )

    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        restart_store=restart_store,
    )

    timers = state.timers

    if cfg.logger.log_progress:
        append_to_progress_log(cfg.checkpoint.save, "Starting job")

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor([state.start_time], dtype=torch.double, device="cuda")
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print_rank_0("time to initialize megatron (seconds): {:.3f}".format(time.time() - state.start_time))
    barrier_and_log("after megatron is initialized")

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    timers("tokenizer-setup", log_level=0).start(barrier=True)
    tokenizer = build_tokenizer(cfg.tokenizer)
    # Handle model vocab_size configuration with proper validation
    cfg.model.vocab_size, cfg.model.should_pad_vocab = _validate_and_set_vocab_size(
        model_vocab_size=cfg.model.vocab_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )

    cfg.dataset.tokenizer = tokenizer
    timers("tokenizer-setup").stop()
    barrier_and_log("after tokenizer is built")

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)

    # Register PEFT pre-wrap hook if PEFT is configured
    if cfg.peft is not None:
        peft_hook = _create_peft_pre_wrap_hook(cfg, state)
        cfg.model.register_pre_wrap_hook(peft_hook)
        print_rank_0("Registered PEFT pre-wrap hook")

    model = cfg.model.provide_distributed_model(
        ddp_config=cfg.ddp,
        use_megatron_fsdp=cfg.dist.use_megatron_fsdp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
    )
    cfg.model.timers = timers
    cfg.optimizer.timers = timers
    optimizer, scheduler = setup_optimizer(
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        model=model,
        use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
    )
    timers("model-and-optimizer-setup").stop()
    barrier_and_log("after model, optimizer, and learning rate scheduler are built")

    # For PEFT, the pretrained checkpoint is loaded in the pre-wrap hook
    if cfg.peft is not None:
        should_load_checkpoint = (cfg.checkpoint.load is not None and checkpoint_exists(cfg.checkpoint.load))
        if should_load_checkpoint:
            # The finetune toggle is explicitly set to True in order to avoid loading optimizer and RNG states
            # This is switched off here in order to load these states from the checkpoint
            cfg.checkpoint.finetune = False
    else:
        should_load_checkpoint = (cfg.checkpoint.load is not None and checkpoint_exists(cfg.checkpoint.load)) or (cfg.checkpoint.pretrained_checkpoint is not None and checkpoint_exists(cfg.checkpoint.pretrained_checkpoint))

    if should_load_checkpoint:
        timers("load-checkpoint", log_level=0).start(barrier=True)
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=cfg.dist.use_torch_fsdp2 or cfg.dist.use_megatron_fsdp,
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])

    _update_model_config_funcs(
        model,
        cfg.model,
        cfg.ddp,
        optimizer,
        align_grad_reduce=cfg.dist.align_grad_reduce,
    )

    # Data stuff.
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
    if "tokenizer" in inspect.signature(train_valid_test_datasets_provider).parameters:
        train_valid_test_datasets_provider = partial(train_valid_test_datasets_provider, tokenizer=tokenizer)

    train_data_iterator, valid_data_iterator, test_data_iterator = setup_data_iterators(
        cfg=cfg,
        train_state=state.train_state,
        model_length=len(model),
        train_valid_test_datasets_provider=train_valid_test_datasets_provider,
    )
    timers("train/valid/test-data-iterators-setup").stop()
    barrier_and_log("after dataloaders are built")

    # if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
    #     ft_integration.get_rank_monitor_client().init_workload_monitoring()
    #     ft_timeouts = ft_integration.get_rank_monitor_client().timeouts
    #     print_rank_0(f"Fault tolerance client initialized. Timeouts: {ft_timeouts}")

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"], barrier=True)
    if get_rank_safe() == 0:
        # Print final resolved/updated/overridden configs
        print("------- Task Configuration -------")
        cfg.print_yaml()
        print("----------------------------------")

    return SetupOutput(
        state,
        model,
        optimizer,
        scheduler,
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
        checkpointing_context,
    )


def _update_model_config_funcs(
    model: MegatronModule,
    model_config: GPTModelProvider | T5ModelProvider,
    ddp_config: DistributedDataParallelConfig,
    optimizer: MegatronOptimizer | None,
    *,
    align_grad_reduce: bool = True,
) -> None:
    """Update model config sync funcs based on initialized model."""
    if isinstance(model[0], (DistributedDataParallel, megatron_FSDP)) and ddp_config.overlap_grad_reduce:
        assert model_config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        model_config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            model_config.no_sync_func = model_config.no_sync_func[0]
        if align_grad_reduce:
            model_config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                model_config.grad_sync_func = model_config.grad_sync_func[0]
    if ddp_config.overlap_param_gather and ddp_config.align_param_gather:
        model_config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            model_config.param_sync_func = model_config.param_sync_func[0]
    if optimizer is not None:
        model_config.finalize_model_grads_func = finalize_model_grads
        model_config.grad_scale_func = optimizer.scale_loss


def _create_peft_pre_wrap_hook(cfg: ConfigContainer, state: GlobalState) -> Callable[[list[MegatronModule]], list[MegatronModule]]:
    """Create a pre-wrap hook that handles PEFT logic.

    This hook is executed before the model is wrapped with DDP/FSDP and handles:
    1. Loading pretrained checkpoints for PEFT
    2. Applying PEFT transformation to the model

    Args:
        cfg: Configuration container
        state: Global state object containing timers and other state

    Returns:
        A callable hook that can be registered with the model provider
    """
    def peft_pre_wrap_hook(model: list[MegatronModule]) -> list[MegatronModule]:
        """Pre-wrap hook that handles PEFT transformation.

        Args:
            model: List of base model modules before distributed wrapping

        Returns:
            List of potentially PEFT-transformed model modules
        """
        # Only apply PEFT logic if PEFT is configured
        if cfg.peft is None:
            return model

        print_rank_0("Applying PEFT pre-wrap hook...")

        # Load pretrained checkpoint if available
        if cfg.checkpoint.pretrained_checkpoint is None or not checkpoint_exists(cfg.checkpoint.pretrained_checkpoint):
            raise ValueError(f"Invalid pretrained checkpoint directory found: {cfg.checkpoint.pretrained_checkpoint}")

        # Explicitly set finetune to avoid loading optimizer and RNG states
        cfg.checkpoint.finetune = True
        state.timers("load-pretrained-checkpoint", log_level=0).start(barrier=True)
        print_rank_0(f"Loading base model weights from: {cfg.checkpoint.pretrained_checkpoint}")

        # Directly call load_checkpoint_from path in order to avoid
        # the load directory overriding the pretrained checkpoint path
        # This is needed to initialize the base model weights first, and then conditionally load adapter states after
        _load_checkpoint_from_path(
            load_dir=cfg.checkpoint.pretrained_checkpoint,
            state=state,
            model=model,
            optimizer=None,  # Don't load optimizer - will be created after PEFT
            opt_param_scheduler=None,  # Don't load scheduler - will be created after PEFT
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        state.timers("load-pretrained-checkpoint").stop(barrier=True)
        state.timers.log(["load-pretrained-checkpoint"])

        # Apply PEFT transformation
        transformed_model = _apply_peft_transformation(cfg.peft, model)

        return transformed_model

    return peft_pre_wrap_hook


def _apply_peft_transformation(peft, base_model: list[MegatronModule]) -> list[MegatronModule]:
    """Apply PEFT transformation to the base model.

    Args:
        peft: PEFT configuration/object
        base_model: Base model before PEFT transformation

    Returns:
        Model with PEFT transformation applied
    """
    print_rank_0("Applying PEFT transformation...")
    transformed_model = peft(base_model, training=True)
    peft.set_params_to_save(transformed_model)

    # Log PEFT statistics
    model_to_analyze = transformed_model[0] if isinstance(transformed_model, list) else transformed_model
    total_params = 0
    trainable_params = 0
    for param in model_to_analyze.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    print_rank_0(f"PEFT Statistics:")
    print_rank_0(f"  Total parameters: {total_params:,}")
    print_rank_0(f"  Trainable parameters: {trainable_params:,}")
    print_rank_0(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    return transformed_model


def _validate_and_set_vocab_size(model_vocab_size: int | None, tokenizer_vocab_size: int) -> tuple[int, bool]:
    """Validate and determine the correct vocab size for the model.

    Args:
        model_vocab_size: Vocab size set in model config (can be None)
        tokenizer_vocab_size: Unpadded tokenizer vocab size

    Returns:
        tuple[int, bool]: The validated unpadded vocab size and padding flag
            - vocab_size: The validated unpadded vocab size to use for the model
            - should_pad_vocab: True if vocab should be padded, False otherwise

    Raises:
        ValueError: If model vocab size is invalid
    """
    if model_vocab_size is None:
        # If model vocab size is not set, use the tokenizer's vocab size
        # Enable padding since this came from tokenizer
        return tokenizer_vocab_size, True
    elif model_vocab_size < tokenizer_vocab_size:
        # Vocab size smaller than tokenizer
        raise ValueError(
            f"Model vocab_size ({model_vocab_size}) cannot be smaller than tokenizer's vocab_size "
            f"({tokenizer_vocab_size})."
        )
    else:
        # Model vocab size is explicitly set and is >= tokenizer vocab size
        # Disable padding since this was explicitly set
        if model_vocab_size > tokenizer_vocab_size:
            logging.info(
                f"Using preset vocab_size: {model_vocab_size} over the tokenizer vocab_size: {tokenizer_vocab_size}, dummy tokens:"
                f" {model_vocab_size - tokenizer_vocab_size}."
            )
        return model_vocab_size, False
