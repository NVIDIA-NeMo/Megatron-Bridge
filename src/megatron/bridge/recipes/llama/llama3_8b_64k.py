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

from megatron.bridge.models.llama import Llama3ModelProvider8B
from megatron.bridge.recipes.llama import llama3_8b
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


SEQUENCE_LENGTH_64K: int = 65536


def model_config(
    tensor_parallelism: int = 4,
    pipeline_parallelism: int = 2,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 4,
    sequence_parallelism: bool = True,
) -> Llama3ModelProvider8B:
    """
    Configure the Llama3 8B model for 64k sequence length training.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism. Default optimized for 64k sequences.
        pipeline_parallelism (int): Degree of pipeline model parallelism. Default optimized for 64k sequences.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism. Default optimized for 64k sequences.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism. Default optimized for 64k sequences.
        sequence_parallelism (bool): Whether to use sequence parallelism. Default optimized for 64k sequences.

    Returns:
        Llama3ModelProvider8B: Configuration for the Llama3 8B model optimized for 64k sequences.
    """
    # Get base model config and override sequence length to 64k
    model_cfg = llama3_8b.model_config(
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
    )

    model_cfg.seq_length = SEQUENCE_LENGTH_64K

    return model_cfg


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
    # Model configuration - defaults optimized for 64k sequences
    tensor_parallelism: int = 4,
    pipeline_parallelism: int = 2,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 4,
    sequence_parallelism: bool = True,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
) -> ConfigContainer:
    """
    Create a pre-training configuration for Llama3 8B model with 64k sequence length.

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
        tensor_parallelism (int): Degree of tensor model parallelism. Default optimized for 64k sequences.
        pipeline_parallelism (int): Degree of pipeline model parallelism. Default optimized for 64k sequences.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism. Default optimized for 64k sequences.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism. Default optimized for 64k sequences.
        sequence_parallelism (bool): Whether to use sequence parallelism. Default optimized for 64k sequences.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int) Number of warmup iterations for the learning rate.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision recipe for the model.

    Returns:
        ConfigContainer: Configuration for pre-training.

    Note:
        Sequence length is hardcoded to 65536 (64k) for long sequence training.
        Default parallelism settings are optimized for handling 64k sequences efficiently.
    """
    # Get base configuration from llama3_8b with 64k sequence length
    cfg = llama3_8b.pretrain_config(
        dir=dir,
        name=name,
        data_paths=data_paths,
        data_args_path=data_args_path,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        test_data_path=test_data_path,
        per_split_data_args_path=per_split_data_args_path,
        mock=mock,
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
        train_iters=train_iters,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        seq_length=SEQUENCE_LENGTH_64K,
        lr=lr,
        min_lr=min_lr,
        lr_warmup_iters=lr_warmup_iters,
        precision_config=precision_config,
    )

    # Override the model configuration to use 64k sequence length
    cfg.model = model_config(
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
    )

    return cfg
