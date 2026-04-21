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

import os

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge.models.mamba import (
    MambaModelProvider1P3B,
    MambaModelProvider2P7B,
    MambaModelProvider130M,
    MambaModelProvider370M,
    MambaModelProvider780M,
    NVIDIAMambaHybridProvider8B,
    NVIDIAMambaModelProvider8B,
)
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class Mamba2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Mamba2 recipe helper functions."""

    # Core identifiers
    model_provider: (
        type[MambaModelProvider130M]
        | type[MambaModelProvider370M]
        | type[MambaModelProvider780M]
        | type[MambaModelProvider1P3B]
        | type[MambaModelProvider2P7B]
        | type[NVIDIAMambaModelProvider8B]
        | type[NVIDIAMambaHybridProvider8B]
    )
    tokenizer_model: str | None
    dir: str | None
    name: str
    # Dataset configuration
    data_paths: list[str] | None
    data_args_path: str | None
    train_data_path: list[str] | None
    valid_data_path: list[str] | None
    test_data_path: list[str] | None
    per_split_data_args_path: str | None
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    sequence_parallel: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    # Tokenizer selection
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


def mamba2_130m_pretrain_config(**user_kwargs: Unpack[Mamba2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Mamba2 130M."""
    recommended: Mamba2CommonKwargs = {
        "model_provider": MambaModelProvider130M,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": False,
    }
    kwargs: Mamba2CommonKwargs = {**recommended, **user_kwargs}
    return _mamba2_common(tokenizer_model=kwargs.get("tokenizer_model"), **kwargs)


def mamba2_370m_pretrain_config(**user_kwargs: Unpack[Mamba2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Mamba2 370M."""
    recommended: Mamba2CommonKwargs = {
        "model_provider": MambaModelProvider370M,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": False,
    }
    kwargs: Mamba2CommonKwargs = {**recommended, **user_kwargs}
    return _mamba2_common(tokenizer_model=kwargs.get("tokenizer_model"), **kwargs)


def mamba2_780m_pretrain_config(**user_kwargs: Unpack[Mamba2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Mamba2 780M."""
    recommended: Mamba2CommonKwargs = {
        "model_provider": MambaModelProvider780M,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": False,
    }
    kwargs: Mamba2CommonKwargs = {**recommended, **user_kwargs}
    return _mamba2_common(tokenizer_model=kwargs.get("tokenizer_model"), **kwargs)


def mamba2_1p3b_pretrain_config(**user_kwargs: Unpack[Mamba2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Mamba2 1.3B."""
    recommended: Mamba2CommonKwargs = {
        "model_provider": MambaModelProvider1P3B,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": False,
    }
    kwargs: Mamba2CommonKwargs = {**recommended, **user_kwargs}
    return _mamba2_common(tokenizer_model=kwargs.get("tokenizer_model"), **kwargs)


def mamba2_2p7b_pretrain_config(**user_kwargs: Unpack[Mamba2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Mamba2 2.7B."""
    recommended: Mamba2CommonKwargs = {
        "model_provider": MambaModelProvider2P7B,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": False,
    }
    kwargs: Mamba2CommonKwargs = {**recommended, **user_kwargs}
    return _mamba2_common(tokenizer_model=kwargs.get("tokenizer_model"), **kwargs)


def mamba2_8b_pretrain_config(**user_kwargs: Unpack[Mamba2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Mamba2 8B."""
    recommended: Mamba2CommonKwargs = {
        "model_provider": NVIDIAMambaModelProvider8B,
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": True,
    }
    kwargs: Mamba2CommonKwargs = {**recommended, **user_kwargs}
    return _mamba2_common(tokenizer_model=None, **kwargs)


def mamba2_hybrid_8b_pretrain_config(**user_kwargs: Unpack[Mamba2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Mamba2 Hybrid 8B."""
    recommended: Mamba2CommonKwargs = {
        "model_provider": NVIDIAMambaHybridProvider8B,
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "use_null_tokenizer": True,
    }
    kwargs: Mamba2CommonKwargs = {**recommended, **user_kwargs}
    return _mamba2_common(tokenizer_model=None, **kwargs)


def _mamba2_common(
    model_provider: (
        type[MambaModelProvider130M]
        | type[MambaModelProvider370M]
        | type[MambaModelProvider780M]
        | type[MambaModelProvider1P3B]
        | type[MambaModelProvider2P7B]
        | type[NVIDIAMambaModelProvider8B]
        | type[NVIDIAMambaHybridProvider8B]
    ),
    tokenizer_model: str | None = None,
    dir: str | None = None,
    name: str = "default",
    # Dataset configuration
    data_paths: list[str] | None = None,
    data_args_path: str | None = None,
    train_data_path: list[str] | None = None,
    valid_data_path: list[str] | None = None,
    test_data_path: list[str] | None = None,
    per_split_data_args_path: str | None = None,
    mock: bool = False,
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int | None = None,
    # Tokenizer selection
    use_null_tokenizer: bool = False,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Mamba 2.x models.

    Args mirror the individual recipe helpers; see those functions for recommended defaults.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
    )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=100,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            use_distributed_optimizer=True,
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            data_sharding=True,
            dataloader_type="single",
            num_workers=8,
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=(
            TokenizerConfig(
                tokenizer_type="NullTokenizer",
                tokenizer_model=None,
                vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
            )
            if use_null_tokenizer
            else TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=tokenizer_model or "EleutherAI/gpt-neox-20b",
            )
        ),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_load=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg


__all__ = [
    "mamba2_130m_pretrain_config",
    "mamba2_370m_pretrain_config",
    "mamba2_780m_pretrain_config",
    "mamba2_1p3b_pretrain_config",
    "mamba2_2p7b_pretrain_config",
    "mamba2_8b_pretrain_config",
    "mamba2_hybrid_8b_pretrain_config",
]
