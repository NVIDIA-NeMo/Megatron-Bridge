import abc
import os
from typing import Generic, TypedDict, TypeVar, Unpack

import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule

from megatron.hub.common.config import ConfigProtocol, from_pretrained, save_pretrained
from megatron.hub.core.models.model_provider import get_model


ModelT = TypeVar("ModelT", bound=MegatronModule)


class GetModelKwargs(TypedDict, total=False):
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


class ModelProviderMixin(abc.ABC, Generic[ModelT]):
    """Mixin class for providing Megatron model instances.

    This abstract base class provides functionality to create and configure
    Megatron models with proper initialization and distributed data parallel
    support. Subclasses must implement the `provide` method to return the
    specific model instance.
    """

    CONFIG_NAME = "mhub_model.json"
    DEFAULT_CONFIG_FORMAT = "json"

    @abc.abstractmethod
    def provide(self, pre_process=None, post_process=None) -> ModelT:
        pass

    def get_model(
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
    ) -> list[ModelT]:
        """Get a wrapped and configured GPT model ready for training.

        Args:
            ddp_config: Configuration for distributed data parallel
            model_type: Type of model (encoder, decoder, or both)
            overlap_param_gather_with_optimizer_step: Whether to overlap param gathering
            fp16: Override FP16 setting
            bf16: Override BF16 setting
            use_torch_fsdp2: Use PyTorch FSDP2 instead of custom DDP
            wrap_with_ddp: Whether to wrap model with DDP
            data_parallel_random_init: Initialize parameters randomly across data parallel ranks
            use_cpu_initialization: Initialize model on CPU

        Returns:
            List of wrapped model modules
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

        return get_model(
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
        )

    def initialize_model_parallel(self, seed: int | None = None, **kwargs) -> None:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
            torch.cuda.set_device(torch.distributed.get_rank())

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=getattr(self, "tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=getattr(self, "pipeline_model_parallel_size", 1),
            virtual_pipeline_model_parallel_size=getattr(self, "virtual_pipeline_model_parallel_size", None),
            context_parallel_size=getattr(self, "context_parallel_size", 1) or 1,
            expert_model_parallel_size=getattr(self, "expert_model_parallel_size", 1) or 1,
            **kwargs,
        )
        if seed is not None:
            model_parallel_cuda_manual_seed(seed)

    def __call__(self, **kwargs: Unpack["GetModelKwargs"]) -> list[ModelT]:
        return self.get_model(**kwargs)

    @property
    def meta_model(self) -> list[ModelT]:
        return self(wrap_with_ddp=False, init_model_with_meta_device=True)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        trust_remote_code: bool = False,
        mode=None,
        config_name: str | None = None,
        **kwargs,
    ):
        """Load a pretrained model configuration from a directory or file."""
        if config_name is None:
            config_name = cls.CONFIG_NAME.rsplit(".", 1)[0]
        if mode is None:
            from megatron.hub.core.utils.instantiate_utils import InstantiationMode
            mode = InstantiationMode.LENIENT
        return from_pretrained(
            cls,
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            mode=mode,
            config_name=config_name,
            **kwargs
        )

    def save_pretrained(
        self,
        save_directory,
        config_format: str | None = None,
        config_name: str | None = None,
        **kwargs,
    ):
        """Save the model configuration to a directory."""
        if config_name is None:
            config_name = self.CONFIG_NAME.rsplit(".", 1)[0]
        if config_format is None:
            config_format = self.DEFAULT_CONFIG_FORMAT
        return save_pretrained(
            self,
            save_directory,
            config_format=config_format,
            config_name=config_name,
            **kwargs
        )