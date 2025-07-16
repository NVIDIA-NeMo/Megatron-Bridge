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

import abc
import os
from pathlib import Path
from typing import Callable, Generic, TypedDict, TypeVar


try:
    from typing import Unpack
except ImportError:
    try:
        from typing_extensions import Unpack
    except ImportError:
        from unittest.mock import MagicMock

        Unpack = MagicMock()


import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.common.config import from_hf_pretrained, save_hf_pretrained
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.utils.instantiate_utils import InstantiationMode


ModelT = TypeVar("ModelT", bound=MegatronModule)


class ModelProviderMixin(abc.ABC, Generic[ModelT]):
    """A mixin that implements the ModelProvider pattern for Megatron-Hub.

    The ModelProvider pattern solves ecosystem fragmentation by providing a standardized
    way to instantiate models. This mixin provides a consistent `provide_models()` method
    that handles the complexity of distributed training setup, along with HuggingFace-inspired
    `.from_hf_pretrained()` and `.save_hf_pretrained()` for interoperability.

    For advanced customization, multiple hooks can be registered via `register_pre_wrap_hook`
    and `register_post_wrap_hook`. These hooks allow modifying the model before and after
    it's wrapped for distributed training (e.g., freezing layers, logging). The composed
    hooks can be accessed via the `pre_wrap_hook` and `post_wrap_hook` properties.

    Subclasses must implement the `provide` method to define the model architecture.
    """

    CONFIG_NAME = "mhub_model.json"
    DEFAULT_CONFIG_FORMAT = "json"

    @abc.abstractmethod
    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> ModelT:
        """Abstract method to provide the model instance.

        Subclasses must implement this method to return the specific Megatron model
        (e.g., `GPTModel`) with its configuration. This method is called by `get_model`
        to obtain the base model before it is wrapped for distributed training.

        Args:
            pre_process (callable, optional): A function to be called before model instantiation.
            post_process (callable, optional): A function to be called after model instantiation.
            vp_stage (int, optional): The virtual pipeline stage of the model.

        Returns:
            ModelT: The Megatron model instance.
        """
        pass

    def provide_models(
        self,
        ddp_config: DistributedDataParallelConfig | None = None,
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
        post_wrap_hook: Callable[[list[MegatronModule]], list[MegatronModule]] | None = None,
    ) -> list[ModelT]:
        """Instantiate and wrap the model for distributed training.

        This method retrieves the model from `provide` and sets up the distributed
        environment, including data-parallel and model-parallel configurations.
        It's the primary entry point for creating a model that's ready for use
        in the Megatron ecosystem.

        Args:
            ddp_config: Configuration for distributed data parallel.
            model_type: Type of model (encoder, decoder, or both).
            overlap_param_gather_with_optimizer_step: Whether to overlap param gathering.
            fp16: Override FP16 setting.
            bf16: Override BF16 setting.
            use_torch_fsdp2: Use PyTorch FSDP2 instead of custom DDP.
            wrap_with_ddp: Whether to wrap model with DDP.
            data_parallel_random_init: Initialize parameters randomly across data parallel ranks.
            use_cpu_initialization: Initialize model on CPU.
            init_model_with_meta_device: Initialize model on meta device.
            pre_wrap_hook: A single callable to modify the model before it's wrapped. If provided,
                this will override all hooks registered via `register_pre_wrap_hook`.
            post_wrap_hook: A single callable to modify the model after it's wrapped. If provided,
                this will override all hooks registered via `register_post_wrap_hook`.

        Returns:
            A list containing the wrapped model instance.
        """
        if wrap_with_ddp and not ddp_config:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")

        if not torch.distributed.is_initialized():
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            torch.distributed.init_process_group("nccl")

        if not parallel_state.is_initialized():
            print("Model parallel not initialized, initializing...")
            self.initialize_model_parallel(seed=0)

        final_pre_wrap_hook = pre_wrap_hook or self.pre_wrap_hook
        final_post_wrap_hook = post_wrap_hook or self.post_wrap_hook

        model = get_model(
            self.provide,
            ddp_config=ddp_config,
            model_type=model_type,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            fp16=fp16,
            bf16=bf16,
            use_torch_fsdp2=use_torch_fsdp2,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
            use_cpu_initialization=use_cpu_initialization,
            init_model_with_meta_device=init_model_with_meta_device,
            pre_wrap_hook=final_pre_wrap_hook,
        )

        if final_post_wrap_hook:
            _model = final_post_wrap_hook(model)
            if _model is not None:
                model = _model

        return model

    def initialize_model_parallel(
        self, seed: int | None = None, seed_kwargs: dict | None = None, **model_parallel_kwargs
    ) -> None:
        """Initializes model parallelism and sets the random seed.

        This is a convenience method that sets up tensor, pipeline, and other
        forms of model parallelism based on the attributes of the provider instance.

        Args:
            seed: The random seed for model parallel RNG.
            seed_kwargs: Additional arguments for `model_parallel_cuda_manual_seed`.
            **model_parallel_kwargs: Additional arguments for `parallel_state.initialize_model_parallel`.
        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
            torch.cuda.set_device(torch.distributed.get_rank())

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=getattr(self, "tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=getattr(self, "pipeline_model_parallel_size", 1),
            virtual_pipeline_model_parallel_size=getattr(self, "virtual_pipeline_model_parallel_size", None),
            context_parallel_size=getattr(self, "context_parallel_size", 1) or 1,
            expert_model_parallel_size=getattr(self, "expert_model_parallel_size", 1) or 1,
            **model_parallel_kwargs,
        )
        if seed is not None:
            model_parallel_cuda_manual_seed(seed, **(seed_kwargs or {}))

    def __call__(self, *args, **kwargs: Unpack["GetModelKwargs"]) -> list[ModelT]:
        """A convenience wrapper around `provide_models`."""
        return self.provide_models(*args, **kwargs)

    @property
    def meta_model(self) -> list[ModelT]:
        """Returns the model instantiated on the meta device for inspection.

        This is useful for examining the model architecture without allocating
        GPU memory.
        """
        return self(wrap_with_ddp=False, init_model_with_meta_device=True)

    @property
    def pre_wrap_hook(self) -> Callable[[list[MegatronModule]], list[MegatronModule]] | None:
        """A composed callable of all registered pre-wrap hooks.

        This read-only property returns a single function that executes all registered
        pre-wrap hooks in order. The hook is applied before the model is passed to the DDP
        wrapper and can be used for tasks like freezing layers or altering model structure.

        Use `register_pre_wrap_hook` to add a hook to the execution chain.

        Returns:
            A callable that executes all registered pre-wrap hooks in order, or None if no
            hooks are registered.
        """
        if not hasattr(self, "_pre_wrap_hooks") or not self._pre_wrap_hooks:
            return None

        def composed_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            for hook in self._pre_wrap_hooks:
                model = hook(model)
            return model

        return composed_hook

    def register_pre_wrap_hook(
        self,
        hook: Callable[[list[MegatronModule]], list[MegatronModule]],
        prepend: bool = False,
    ) -> None:
        """Registers a hook to be executed before the model is wrapped.

        The hook should be a callable that accepts a list of `MegatronModule` instances
        and returns a (potentially modified) list of `MegatronModule` instances.

        Args:
            hook: The hook to register.
            prepend: If True, the hook is inserted at the beginning of the execution
                chain. Otherwise, it is appended to the end.
        """
        if not hasattr(self, "_pre_wrap_hooks"):
            self._pre_wrap_hooks = []
        if prepend:
            self._pre_wrap_hooks.insert(0, hook)
        else:
            self._pre_wrap_hooks.append(hook)

    @property
    def post_wrap_hook(self) -> Callable[[list[MegatronModule]], list[MegatronModule]] | None:
        """A composed callable of all registered post-wrap hooks.

        This read-only property returns a single function that executes all registered
        post-wrap hooks in order. The hook is applied after the model has been wrapped by
        DDP and is useful for tasks like logging or attaching custom attributes.

        Use `register_post_wrap_hook` to add a hook to the execution chain.

        Returns:
            A callable that executes all registered post-wrap hooks in order, or None if no
            hooks are registered.
        """
        if not hasattr(self, "_post_wrap_hooks") or not self._post_wrap_hooks:
            return None

        def composed_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            for hook in self._post_wrap_hooks:
                model = hook(model)
            return model

        return composed_hook

    def register_post_wrap_hook(
        self,
        hook: Callable[[list[MegatronModule]], list[MegatronModule]],
        prepend: bool = False,
    ) -> None:
        """Registers a hook to be executed after the model is wrapped.

        The hook should be a callable that accepts a list of `MegatronModule` instances
        and returns a (potentially modified) list of `MegatronModule` instances.

        Args:
            hook: The hook to register.
            prepend: If True, the hook is inserted at the beginning of the execution
                chain. Otherwise, it is appended to the end.
        """
        if not hasattr(self, "_post_wrap_hooks"):
            self._post_wrap_hooks = []
        if prepend:
            self._post_wrap_hooks.insert(0, hook)
        else:
            self._post_wrap_hooks.append(hook)

    @classmethod
    def from_hf_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        trust_remote_code: bool = False,
        mode: InstantiationMode | None = None,
        config_name: str | None = None,
        **kwargs,
    ):
        """Load a pretrained model configuration from a directory or HuggingFace Hub.

        This method provides a HuggingFace-inspired interface for loading model
        configurations, enabling interoperability.

        Args:
            pretrained_model_name_or_path: The path to a local directory or a
                HuggingFace model identifier.
            trust_remote_code: Whether to trust remote code when loading.
            mode: The instantiation mode (e.g., `LENIENT`).
            config_name: The name of the configuration file (without extension).
            **kwargs: Additional keyword arguments for `from_hf_pretrained`.

        Returns:
            An instance of the model provider with the loaded configuration.
        """
        if config_name is None:
            config_name = cls.CONFIG_NAME.rsplit(".", 1)[0]
        if mode is None:
            mode = InstantiationMode.LENIENT
        return from_hf_pretrained(
            cls,
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            mode=mode,
            config_name=config_name,
            **kwargs,
        )

    def save_hf_pretrained(
        self,
        save_directory: str | Path,
        config_format: str | None = None,
        config_name: str | None = None,
        **kwargs,
    ):
        """Save the model configuration to a directory.

        This method provides a HuggingFace-inspired interface for saving model
        configurations, enabling interoperability.

        Args:
            save_directory: The directory where the configuration will be saved.
            config_format: The format for the configuration file (e.g., `json` or `yaml`).
            config_name: The name of the configuration file (without extension).
            **kwargs: Additional keyword arguments for `save_hf_pretrained`.
        """
        if config_name is None:
            config_name = self.CONFIG_NAME.rsplit(".", 1)[0]
        if config_format is None:
            config_format = self.DEFAULT_CONFIG_FORMAT
        return save_hf_pretrained(self, save_directory, config_format=config_format, config_name=config_name, **kwargs)


class GetModelKwargs(TypedDict, total=False):
    """Keyword arguments for the `provide_models` method.

    Attributes:
        ddp_config: Configuration for distributed data parallel.
        model_type: Type of model (encoder, decoder, or both).
        overlap_param_gather_with_optimizer_step: Whether to overlap param gathering.
        fp16: Override FP16 setting.
        bf16: Override BF16 setting.
        use_torch_fsdp2: Use PyTorch FSDP2 instead of custom DDP.
        wrap_with_ddp: Whether to wrap model with DDP.
        data_parallel_random_init: Initialize parameters randomly across data parallel ranks.
        use_cpu_initialization: Initialize model on CPU.
        init_model_with_meta_device: Initialize model on meta device.
        pre_wrap_hook: A single callable that overrides all registered pre-wrap hooks.
        post_wrap_hook: A single callable that overrides all registered post-wrap hooks.
    """

    ddp_config: DistributedDataParallelConfig | None
    model_type: ModelType
    overlap_param_gather_with_optimizer_step: bool
    fp16: bool | None
    bf16: bool | None
    use_torch_fsdp2: bool
    wrap_with_ddp: bool
    data_parallel_random_init: bool
    use_cpu_initialization: bool | None
    init_model_with_meta_device: bool | None
    pre_wrap_hook: Callable[[list[MegatronModule]], list[MegatronModule]] | None
    post_wrap_hook: Callable[[list[MegatronModule]], list[MegatronModule]] | None
