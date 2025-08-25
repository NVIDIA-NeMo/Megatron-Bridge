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

from typing import List, Optional, Union

import torch

from megatron.bridge.models.nemotron import Nemotron4ModelProvider340B
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.pretrain_utils import (
    create_checkpoint_config,
    create_dataset_config,
    create_ddp_config,
    create_logger_config,
    create_rng_config,
    create_tokenizer_config,
    create_training_config,
    setup_output_dirs,
)
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def model_config(
    tensor_parallelism: int = 8,
    pipeline_parallelism: int = 12,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = 8,
    context_parallelism: int = 2,
    sequence_parallelism: bool = True,
    sequence_length: int = 4096,
) -> Nemotron4ModelProvider340B:
    """
    Configure the Nemotron4 340B model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        sequence_length (int): Sequence length for the model.

    Returns:
        Nemotron4ModelProvider340B: Configuration for the Nemotron4 340B model.
    """
    return Nemotron4ModelProvider340B(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        seq_length=sequence_length,
    )


def pretrain_config(
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_parallelism: int = 8,
    pipeline_parallelism: int = 12,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = 8,
    context_parallelism: int = 2,
    sequence_parallelism: bool = True,
    sequence_length: int = 4096,
    # Training hyperparameters
    train_iters: int = 100000,
    global_batch_size: int = 2304,
    micro_batch_size: int = 1,
    lr: float = 1.0e-4,
    min_lr: float = 1.0e-5,
    lr_warmup_iters: int = 500,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Nemotron4 340B model.

    Args:
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (Optional[List[str]]): List of paths to dataset files. If None, mock data will be used.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism to be passed to model_config.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        sequence_length (int): Sequence length for the model.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration for the model.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    # Set up output directories
    run_output_dir, checkpoint_dir, tensorboard_dir = setup_output_dirs(dir, name)

    # Create model configuration
    model_cfg = model_config(
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
        sequence_length=sequence_length,
    )

    # Create optimizer and scheduler configurations
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        max_lr=lr,
        min_lr=min_lr,
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
    )
    opt_config.use_precision_optimizer = False

    # Create dataset configuration
    dataset_cfg = create_dataset_config(
        sequence_length=sequence_length,
        data_paths=data_paths,
        data_args_path=data_args_path,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        test_data_path=test_data_path,
        per_split_data_args_path=per_split_data_args_path,
        mock=mock,
    )

    # Create communication overlap configuration if not provided
    if comm_overlap_config is None:
        comm_overlap_config = CommOverlapConfig(
            tp_comm_overlap=True,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=22,
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
        )

    final_precision_config = precision_config
    if isinstance(precision_config, str):
        # Mixed precision configuration
        from megatron.bridge.training.mixed_precision import get_mixed_precision_config

        final_precision_config = get_mixed_precision_config(precision_config)

    final_precision_config.grad_reduce_in_fp32 = False

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=create_training_config(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=create_ddp_config(),
        dataset=dataset_cfg,
        logger=create_logger_config(tensorboard_dir=tensorboard_dir),
        tokenizer=create_tokenizer_config(),
        checkpoint=create_checkpoint_config(checkpoint_dir=checkpoint_dir),
        rng=create_rng_config(),
        comm_overlap=comm_overlap_config,
        mixed_precision=final_precision_config,
    )

    return cfg
