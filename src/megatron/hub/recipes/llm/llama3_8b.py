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
from typing import List, Optional

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.hub.data.loaders import get_blend_and_blend_per_split
from megatron.hub.models.llama import Llama3Config8B
from megatron.hub.tokenizers.config import TokenizerConfig
from megatron.hub.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
)


def model_config(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 2,
    sequence_parallelism: bool = False,
) -> Llama3Config8B:
    """
    Configure the Llama3 8B model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.

    Returns:
        Llama3Config8B: Configuration for the Llama3 8B model.
    """
    return Llama3Config8B(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
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
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 2,
    sequence_parallelism: bool = False,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Llama3 8B model.

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
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int) Number of warmup iterations for the learning rate.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Dataset configuration logic based on mock vs real data
    has_any_data_config = any(
        [data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path]
    )

    if mock or not has_any_data_config:
        # Mock data configuration
        mock = True
        blend = None  # Will trigger mock mode automatically
        blend_per_split = None  # Will trigger mock mode automatically
        split = "1,1,1"  # Equal splits for testing
        data_path = None  # No real data path needed
    else:
        # Real data configuration
        mock = False
        blend_weights, blend_per_split_weights = get_blend_and_blend_per_split(
            data_paths=data_paths,
            data_args_path=data_args_path,
            train_data_path=train_data_path,
            valid_data_path=valid_data_path,
            test_data_path=test_data_path,
            per_split_data_args_path=per_split_data_args_path,
        )

        if blend_weights is None and blend_per_split_weights is None:
            # No data provided, fall back to mock mode
            mock = True
            blend = None
            blend_per_split = None
            split = "1,1,1"
            data_path = None
        else:
            # Construct data_path from the inputs
            if data_paths is not None:
                data_path = data_paths
            elif data_args_path is not None:
                data_path = data_args_path
            else:
                data_path = []
                if train_data_path:
                    data_path.extend(train_data_path)
                if valid_data_path:
                    data_path.extend(valid_data_path)
                if test_data_path:
                    data_path.extend(test_data_path)
                if per_split_data_args_path:
                    data_path = per_split_data_args_path

            # Create the tuples expected by BlendedMegatronDatasetConfig
            # Prioritize blend_per_split_weights over blend_weights if both are provided
            if blend_per_split_weights is not None:
                # For per-split, we need to construct the paths for each split
                train_paths = train_data_path or []
                valid_paths = valid_data_path or []
                test_paths = test_data_path or []

                blend_per_split = [
                    (train_paths, blend_per_split_weights[0]) if train_paths else None,
                    (valid_paths, blend_per_split_weights[1]) if valid_paths else None,
                    (test_paths, blend_per_split_weights[2]) if test_paths else None,
                ]
                # When using blend_per_split, split should be None and blend should be None
                split = None
                blend = None
            elif blend_weights is not None:
                blend = (data_path if isinstance(data_path, list) else [data_path], blend_weights)
                blend_per_split = None
                # When using regular blend, we can use split
                split = "9999,8,2"
            else:
                blend = None
                blend_per_split = None
                split = "9999,8,2"

    model_cfg = model_config(
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
    )

    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=lr,
        min_lr=min_lr,
        weight_decay=0.1,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=1.0,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=2000,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        scheduler=SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=2000,
            lr_warmup_init=0.0,
            lr_decay_iters=train_iters,
            override_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
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
            # Dataloader config parameters
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer"),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=True,
        ),
        rng=RNGConfig(seed=1234),
    )

    return cfg
