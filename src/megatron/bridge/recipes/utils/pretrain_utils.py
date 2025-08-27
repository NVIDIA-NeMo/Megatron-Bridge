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

from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import (
    CheckpointConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)


def setup_output_dirs(dir: Optional[str] = None, name: str = "default") -> tuple[str, str, str]:
    """
    Set up output directories for training.

    Args:
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the training run.

    Returns:
        tuple[str, str, str]: (run_output_dir, checkpoint_dir, tensorboard_dir)
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    return run_output_dir, checkpoint_dir, tensorboard_dir


def create_training_config(
    train_iters: int = 1_168_251,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    eval_interval: int = 2000,
    eval_iters: int = 32,
    manual_gc: bool = True,
    manual_gc_interval: int = 100,
    manual_gc_eval: int = 100,
) -> TrainingConfig:
    """
    Create a training configuration.

    Args:
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        eval_interval (int): Evaluation interval.
        eval_iters (int): Number of evaluation iterations.
        manual_gc (bool): Whether to enable manual garbage collection.
        manual_gc_interval (int): Manual GC interval during training.
        manual_gc_eval (int): Manual GC interval during evaluation.

    Returns:
        TrainingConfig: Training configuration.
    """
    return TrainingConfig(
        train_iters=train_iters,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        manual_gc=manual_gc,
        manual_gc_interval=manual_gc_interval,
        manual_gc_eval=manual_gc_eval,
    )


def create_ddp_config(
    check_for_nan_in_grad: bool = True,
    grad_reduce_in_fp32: bool = True,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    average_in_collective: bool = True,
    use_distributed_optimizer: bool = True,
) -> DistributedDataParallelConfig:
    """
    Create a distributed data parallel configuration.

    Args:
        check_for_nan_in_grad (bool): Whether to check for NaN in gradients.
        grad_reduce_in_fp32 (bool): Whether to reduce gradients in FP32.
        overlap_grad_reduce (bool): Whether to overlap gradient reduction.
        overlap_param_gather (bool): Whether to overlap parameter gathering.
        average_in_collective (bool): Whether to average in collective operations.
        use_distributed_optimizer (bool): Whether to use distributed optimizer.

    Returns:
        DistributedDataParallelConfig: DDP configuration.
    """
    return DistributedDataParallelConfig(
        check_for_nan_in_grad=check_for_nan_in_grad,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        average_in_collective=average_in_collective,
        use_distributed_optimizer=use_distributed_optimizer,
    )


def create_dataset_config(
    sequence_length: int = 4096,
    random_seed: int = 1234,
    reset_attention_mask: bool = False,
    reset_position_ids: bool = False,
    eod_mask_loss: bool = False,
    num_dataset_builder_threads: int = 1,
    data_sharding: bool = True,
    dataloader_type: str = "single",
    num_workers: int = 1,
    # Dataset blend parameters
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
) -> GPTDatasetConfig:
    """
    Create a GPT dataset configuration.

    Args:
        sequence_length (int): Sequence length for the dataset.
        random_seed (int): Random seed for dataset operations.
        reset_attention_mask (bool): Whether to reset attention mask.
        reset_position_ids (bool): Whether to reset position IDs.
        eod_mask_loss (bool): Whether to mask loss at end of document.
        num_dataset_builder_threads (int): Number of dataset builder threads.
        data_sharding (bool): Whether to enable data sharding.
        dataloader_type (str): Type of dataloader to use.
        num_workers (int): Number of workers for data loading.
        data_paths (Optional[List[str]]): List of paths to dataset files.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data.

    Returns:
        GPTDatasetConfig: Dataset configuration.
    """
    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    return GPTDatasetConfig(
        random_seed=random_seed,
        reset_attention_mask=reset_attention_mask,
        reset_position_ids=reset_position_ids,
        eod_mask_loss=eod_mask_loss,
        sequence_length=sequence_length,
        num_dataset_builder_threads=num_dataset_builder_threads,
        blend=blend,
        blend_per_split=blend_per_split,
        split=split,
        data_sharding=data_sharding,
        dataloader_type=dataloader_type,
        num_workers=num_workers,
    )


def create_logger_config(
    log_interval: int = 10,
    tensorboard_dir: Optional[str] = None,
    timing_log_level: int = 1,
    log_timers_to_tensorboard: bool = True,
) -> LoggerConfig:
    """
    Create a logger configuration.

    Args:
        log_interval (int): Logging interval.
        tensorboard_dir (Optional[str]): Directory for tensorboard logs.
        timing_log_level (int): Logging level for timing.
        log_timers_to_tensorboard (bool): Whether to log timers to tensorboard.

    Returns:
        LoggerConfig: Logger configuration.
    """
    return LoggerConfig(
        log_interval=log_interval,
        tensorboard_dir=tensorboard_dir,
        timing_log_level=timing_log_level,
        log_timers_to_tensorboard=log_timers_to_tensorboard,
    )


def create_tokenizer_config(
    tokenizer_type: str = "NullTokenizer",
    vocab_size: int = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
) -> TokenizerConfig:
    """
    Create a tokenizer configuration.

    Args:
        tokenizer_type (str): Type of tokenizer to use.
        vocab_size (int): Vocabulary size.

    Returns:
        TokenizerConfig: Tokenizer configuration.
    """
    return TokenizerConfig(
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
    )


def create_checkpoint_config(
    save_interval: int = 2000,
    checkpoint_dir: Optional[str] = None,
    ckpt_format: str = "torch_dist",
    fully_parallel_save: bool = True,
) -> CheckpointConfig:
    """
    Create a checkpoint configuration.

    Args:
        save_interval (int): Interval for saving checkpoints.
        checkpoint_dir (Optional[str]): Directory for saving checkpoints.
        ckpt_format (str): Format for checkpoint files.
        fully_parallel_save (bool): Whether to use fully parallel save.

    Returns:
        CheckpointConfig: Checkpoint configuration.
    """
    return CheckpointConfig(
        save_interval=save_interval,
        save=checkpoint_dir,
        load=checkpoint_dir,
        ckpt_format=ckpt_format,
        fully_parallel_save=fully_parallel_save,
    )


def create_rng_config(seed: int = 1234) -> RNGConfig:
    """
    Create an RNG configuration.

    Args:
        seed (int): Random seed.

    Returns:
        RNGConfig: RNG configuration.
    """
    return RNGConfig(seed=seed)
