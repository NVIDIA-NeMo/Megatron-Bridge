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

import argparse
import logging
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Literal, Optional, Union

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.utils import get_model_config

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.training.checkpointing import save_checkpoint
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer, LoggerConfig
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer, build_tokenizer
from megatron.bridge.training.utils.checkpoint_utils import file_exists
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


logger = logging.getLogger(__name__)


def torch_dtype_from_mcore_config(config: Any) -> torch.dtype:
    """Convert Megatron-Core config dtype settings to torch dtype.

    Args:
        config: Megatron-Core configuration object with bf16/fp16 flags.

    Returns:
        The corresponding torch dtype.
    """
    if hasattr(config, "bf16") and config.bf16:
        return torch.bfloat16
    elif hasattr(config, "fp16") and config.fp16:
        return torch.float16
    else:
        return torch.float32


@contextmanager
def megatron_cpu_init_context(config: Any) -> Generator[None, None, None]:
    """Context manager to temporarily force CPU initialization for Megatron models.

    This is useful when initializing a model on a system without GPUs or when
    memory constraints prevent GPU initialization.

    Args:
        config: The Megatron model configuration object (e.g., GPTConfig).
            Must have a `use_cpu_initialization` attribute.

    Yields:
        None. The context modifies the config in place.
    """
    original_use_cpu_initialization = config.use_cpu_initialization
    config.use_cpu_initialization = True

    try:
        yield
    finally:
        config.use_cpu_initialization = original_use_cpu_initialization


@contextmanager
def temporary_distributed_context(backend: str = "gloo") -> Generator[None, None, None]:
    """Context manager to temporarily initialize a minimal distributed environment.

    Sets up a single-process distributed backend, initializes Megatron model parallel state,
    yields control, and then cleans up the distributed environment.
    Useful for operations that require Megatron's parallel state but should run
    standalone (e.g., loading distributed checkpoints).

    Args:
        backend: The distributed backend to use ("gloo" for CPU, "nccl" for GPU).

    Yields:
        None.
    """
    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        init_method = None
    else:
        # Find an available port dynamically
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            addr, port = s.getsockname()
        init_method = f"tcp://{addr}:{port}"

    dist.init_process_group(backend=backend, init_method=init_method, world_size=1, rank=0)
    parallel_state.initialize_model_parallel()

    if backend == "nccl":
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)

    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def load_tokenizer(checkpoint_path: str) -> MegatronTokenizer:
    """Create a tokenizer from a training checkpoint.

    Obtains tokenizer configuration from the checkpoint and builds the tokenizer.
    Checkpoint should be in MCore distributed checkpoint format.

    Args:
        checkpoint_path: path to an MCore distributed checkpoint directory
                          (e.g., /path/to/model/checkpoints/iter_0000001).
    """
    from megatron.bridge.training.checkpointing import (
        get_checkpoint_run_config_filename,
        read_run_config,
    )
    from megatron.bridge.training.mlm_compat.arguments import _load_args_from_checkpoint, _tokenizer_config_from_args
    from megatron.bridge.utils.instantiate_utils import instantiate

    run_config_filename = get_checkpoint_run_config_filename(checkpoint_path)

    if file_exists(run_config_filename):
        run_config = read_run_config(run_config_filename)
        mbridge_ckpt = True
    else:
        try:
            mlm_args = _load_args_from_checkpoint(checkpoint_path)
            mbridge_ckpt = False
        except AssertionError:
            raise RuntimeError(f"Checkpoint at {checkpoint_path} is not in a supported format.")

    if mbridge_ckpt:
        cfg = instantiate(run_config["tokenizer"])
    else:
        cfg = _tokenizer_config_from_args(mlm_args)

    return build_tokenizer(cfg)


def load_model_config(
    checkpoint_path: str,
) -> tuple[TransformerConfig, Optional[argparse.Namespace]]:
    """Returns the model config saved in the checkpoint.

    Supports checkpoints saved with either Megatron Bridge or MegatronLM.

    Args:
        checkpoint_path: path to an MCore distributed checkpoint directory
                          (e.g., /path/to/model/checkpoints/iter_0000001).

    Returns:
        - The model config from the checkpoint. The object returned will be a
          model provider (e.g. GPTModelProvider) if using a Megatron Bridge
          checkpoint or a TransformerConfig if using a MegatronLM checkpoint.
        - If the checkpoint is from MegatronLM, returns the argparse.Namespace
          object. Otherwise None.
    """
    from megatron.bridge.training.checkpointing import (
        get_checkpoint_run_config_filename,
        read_run_config,
    )
    from megatron.bridge.training.mlm_compat.arguments import _load_args_from_checkpoint, _transformer_config_from_args
    from megatron.bridge.utils.instantiate_utils import instantiate

    run_config_filename = get_checkpoint_run_config_filename(checkpoint_path)

    if file_exists(run_config_filename):
        run_config = read_run_config(run_config_filename)
        mbridge_ckpt = True
        mlm_args = None
    else:
        try:
            mlm_args = _load_args_from_checkpoint(checkpoint_path)
            mbridge_ckpt = False
        except AssertionError:
            raise RuntimeError(f"Checkpoint at {checkpoint_path} is not in a supported format.")

    if mbridge_ckpt:
        model_cfg = instantiate(run_config["model"])
    else:
        model_cfg = _transformer_config_from_args(mlm_args)

    return model_cfg, mlm_args


def build_and_load_model(
    checkpoint_path: str,
    model_cfg: TransformerConfig,
    model_type: Optional[Literal["gpt", "mamba"]] = None,
    megatron_args: Optional[argparse.Namespace] = None,
    return_state_dict: bool = False,
    use_cpu_init: bool = False,
    skip_temp_dist_context: Optional[bool] = None,
) -> Union[Any, dict[str, torch.Tensor]]:
    """Load a Megatron model from a distributed checkpoint.

    Creates model instances and optionally a minimal distributed environment
    to load the model weights from `checkpoint_path` into the model.
    Automatically selects the appropriate distributed backend (Gloo for CPU, NCCL for GPU).

    Args:
        checkpoint_path: path to an MCore distributed checkpoint directory. Used only for
                    model weights (e.g., /path/to/model/checkpoints/iter_0000001).
        model_cfg: Model config from load_model_config(). Either a TransformerConfig or
            a model provider (e.g. GPTModelProvider) depending on source of checkpoint.
        model_type: If the checkpoint is from MegatronLM, the model type is required. Currently,
            only GPT and Mamba models are supported.
        megatron_args: If the checkpoint is from MegatronLM, this is required.
        return_state_dict: If True, return the state dict instead of model instance. Default: False.
        use_cpu_init: If True, use CPU initialization context for the model and Gloo backend.
                     If False, use GPU initialization and NCCL backend. Default: False.
        skip_temp_dist_context: If True, skip temporary distributed context setup.
                               If None, automatically skip if distributed is already initialized.
                               Default: None.

    Returns:
        The model instance with loaded weights if return_state_dict is False,
        otherwise returns a dictionary containing the full, unsharded model state_dict.
    """
    from megatron.bridge.training.checkpointing import (
        _load_model_weights_from_checkpoint,
    )
    from megatron.bridge.training.mlm_compat.arguments import _tokenizer_config_from_args
    from megatron.bridge.training.mlm_compat.model import _get_model, _gpt_provider, _mamba_provider

    def _call_model_provider(model_cfg):
        """Handles provider call for both MBridge and MLM providers."""
        if isinstance(model_cfg, ModelProviderMixin):
            if hasattr(model_cfg, "finalize"):
                model_cfg.finalize()
            return model_cfg.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=use_cpu_init)
        else:
            assert model_type in ("gpt", "mamba"), f"model type {model_type} not supported."
            assert megatron_args is not None, "megatron_args must be provided if the checkpoint is from MegatronLM."

            # Generate the unpadded vocab size based on the tokenizer from the checkpoint
            cfg = _tokenizer_config_from_args(megatron_args)
            tokenizer = build_tokenizer(cfg)
            vocab_size = tokenizer.vocab_size

            # Re-calculate the padded vocab size based on the model config instead of the args from the checkpoint
            # to accommodate TP overrides
            megatron_args.padded_vocab_size = calculate_padded_vocab_size(
                vocab_size,
                megatron_args.make_vocab_size_divisible_by,
                model_cfg.tensor_model_parallel_size,
            )
            provider = _gpt_provider if model_type == "gpt" else _mamba_provider
            return _get_model(megatron_args, provider, model_cfg)

    # Auto-detect if we should skip temp dist context
    if skip_temp_dist_context is None:
        skip_temp_dist_context = dist.is_available() and dist.is_initialized()

    def _load_checkpoint():
        target_dtype = torch_dtype_from_mcore_config(model_cfg)
        if model_cfg.params_dtype != target_dtype:
            logger.info(f"Converting params_dtype from {model_cfg.params_dtype} to {target_dtype}")
            model_cfg.params_dtype = target_dtype

        if use_cpu_init:
            with megatron_cpu_init_context(model_cfg):
                model = _call_model_provider(model_cfg)
        else:
            model = _call_model_provider(model_cfg)

        maybe_state_dict = _load_model_weights_from_checkpoint(
            checkpoint_path, model, return_state_dict=return_state_dict
        )

        if return_state_dict:
            del model
            return maybe_state_dict
        else:
            return model

    if skip_temp_dist_context:
        return _load_checkpoint()
    else:
        # Use appropriate backend based on initialization type
        backend = "gloo" if use_cpu_init else "nccl"
        with temporary_distributed_context(backend=backend):
            return _load_checkpoint()


def load_megatron_model(
    checkpoint_path: str,
    model_type: Optional[Literal["gpt", "mamba"]] = None,
    return_state_dict: bool = False,
    use_cpu_init: bool = False,
    skip_temp_dist_context: Optional[bool] = None,
) -> Union[Any, dict[str, torch.Tensor]]:
    """Load a Megatron model from a distributed checkpoint.

    Wrapper around load_model_config() and build_and_load_model() for convenience.

    Args:
        checkpoint_path: path to an MCore distributed checkpoint directory
                          (e.g., /path/to/model/checkpoints/iter_0000001).
        model_type: If the checkpoint is from MegatronLM, the model type is required. Currently,
            only GPT and Mamba models are supported.
        return_state_dict: If True, return the state dict instead of model instance. Default: False.
        use_cpu_init: If True, use CPU initialization context for the model and Gloo backend.
                     If False, use GPU initialization and NCCL backend. Default: False.
        skip_temp_dist_context: If True, skip temporary distributed context setup.
                               If None, automatically skip if distributed is already initialized.
                               Default: None.

    Returns:
        The model instance with loaded weights if return_state_dict is False,
        otherwise returns a dictionary containing the full, unsharded model state_dict.
    """

    model_cfg, mlm_args = load_model_config(checkpoint_path)
    return build_and_load_model(
        checkpoint_path, model_cfg, model_type, mlm_args, return_state_dict, use_cpu_init, skip_temp_dist_context
    )


def save_megatron_model(
    model: list[MegatronModule],
    path: Union[str, Path],
    ckpt_format: str = "torch_dist",
    hf_tokenizer_path: Optional[Union[str, Path]] = None,
) -> None:
    """Save a Megatron model in native Megatron checkpoint format without optimizer state.

    This method saves the model in Megatron's native checkpoint format, which
    can be loaded directly by Megatron for training or inference. The checkpoint
    includes the model configuration and weights, NO optimizer state or other
    artifacts.

    Args:
        model: Megatron model instance or list of instances.
        path: Directory path where the checkpoint will be saved.
        ckpt_format: Checkpoint format to use ("torch_dist" or other supported formats).
        hf_tokenizer_path: Optional HuggingFace model ID or path for tokenizer metadata.
            If provided, the tokenizer metadata will be included in the checkpoint.

    Example:
        >>> # Save model checkpoint
        >>> save_megatron_model(megatron_model, "./megatron_checkpoint")

        >>> # Save model checkpoint with tokenizer metadata
        >>> save_megatron_model(
        ...     megatron_model,
        ...     "./megatron_checkpoint",
        ...     hf_tokenizer_path="meta-llama/Llama-3-8B"
        ... )

    Note:
        - This method is collective and must be called by all ranks
        - The saved checkpoint can be loaded with Megatron's checkpoint loading utilities
        - The checkpoint format follows Megatron's standard structure for compatibility
    """
    # Create tokenizer config if tokenizer path is provided
    tokenizer_config = None
    if hf_tokenizer_path is not None:
        from megatron.bridge.training.tokenizers.config import TokenizerConfig

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(hf_tokenizer_path),
        )

    # Get model config from the first model instance
    model_config = get_model_config(model[0])

    # Validate that the model config is a model provider
    if not isinstance(model_config, ModelProviderMixin):
        raise TypeError(
            f"Expected model config to be an instance of ModelProviderMixin, "
            f"but got {type(model_config).__name__}. "
            f"Model configs must inherit from ModelProviderMixin to ensure proper "
            f"model instantiation and configuration handling."
        )

    # Create global state for checkpointing
    state = GlobalState()
    state.cfg = ConfigContainer(
        model=model_config,
        train=None,
        optimizer=OptimizerConfig(use_distributed_optimizer=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        logger=LoggerConfig(),
        tokenizer=tokenizer_config,
        checkpoint=CheckpointConfig(
            async_save=False,
            save=str(path),
            save_optim=False,
            save_rng=False,
            ckpt_format=ckpt_format,
        ),
        dist=None,
    )

    # Save the checkpoint
    save_checkpoint(
        state=state,
        model=model,
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
    )


def dtype_from_str(dtype: str) -> torch.dtype:
    """Convert a string representation of a dtype to a torch.dtype.

    Handles common variations like 'fp16', 'bf16-mixed'. Defaults to float32
    for unrecognized strings.

    Args:
        dtype: The string representation (e.g., "bf16", "fp16", "float32").

    Returns:
        The corresponding torch.dtype.
    """
    if not isinstance(dtype, str):
        raise TypeError(f"Expected str, got {type(dtype)}")

    if dtype in ("float16", "fp16", "16", "16-mixed"):
        return torch.float16
    elif dtype in ("bfloat16", "bf16-mixed"):
        return torch.bfloat16
    else:
        return torch.float32


def dtype_from_hf(config: Any) -> torch.dtype:
    """Extract the torch.dtype from a Hugging Face PretrainedConfig object.

    Args:
        config: A Hugging Face model config object (must have `torch_dtype` attribute).

    Returns:
        The corresponding torch.dtype.

    Raises:
        ValueError: If the `torch_dtype` attribute is not a recognized string or torch.dtype.
        AttributeError: If the config object does not have a `torch_dtype` attribute.
    """
    if not hasattr(config, "torch_dtype"):
        raise AttributeError("Expected config to have attr `torch_dtype`")

    torch_dtype = config.torch_dtype

    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    elif isinstance(torch_dtype, str):
        return dtype_from_str(torch_dtype)
    else:
        raise ValueError(f"torch_dtype is not of type str/torch.dtype, got {type(torch_dtype)}")
