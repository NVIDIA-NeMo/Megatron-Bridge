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
from typing import Optional, Union

from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import (
    CheckpointConfig,
    HFDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)


def create_squad_dataset_config(
    seq_length: int,
    packed_sequence: bool = False,
    seed: int = 1234,
    num_workers: int = 1,
    do_validation: bool = False,
    do_test: bool = False,
    val_proportion: Optional[float] = None,
) -> HFDatasetConfig:
    """
    Create a SQuAD dataset configuration for fine-tuning.

    Args:
        seq_length (int): Sequence length for the dataset.
        packed_sequence (bool): Whether to use packed sequences.
        seed (int): Random seed for dataset operations.
        num_workers (int): Number of data loader workers.
        do_validation (bool): Whether to include validation data.
        do_test (bool): Whether to include test data.
        val_proportion (Optional[float]): Proportion of data to use for validation.

    Returns:
        HFDatasetConfig: Configuration for SQuAD dataset.
    """
    return HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=seq_length,
        seed=seed,
        dataloader_type="cyclic" if packed_sequence else "single",
        num_workers=num_workers,
        do_validation=do_validation,
        do_test=do_test,
        val_proportion=val_proportion,
        dataset_kwargs={"pad_to_max_length": True} if packed_sequence else {},
        packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=seq_length) if packed_sequence else None,
    )


def create_ddp_config(
    use_distributed_optimizer: bool = True,
    check_for_nan_in_grad: bool = True,
) -> DistributedDataParallelConfig:
    """
    Create a DDP configuration for fine-tuning.

    Args:
        use_distributed_optimizer (bool): Whether to use distributed optimizer.
        check_for_nan_in_grad (bool): Whether to check for NaN in gradients.

    Returns:
        DistributedDataParallelConfig: DDP configuration.
    """
    return DistributedDataParallelConfig(
        use_distributed_optimizer=use_distributed_optimizer,
        check_for_nan_in_grad=check_for_nan_in_grad,
    )


def create_logger_config(
    tensorboard_dir: str,
    log_interval: int = 1,
) -> LoggerConfig:
    """
    Create a logger configuration for fine-tuning.

    Args:
        tensorboard_dir (str): Directory for TensorBoard logs.
        log_interval (int): Interval for logging.

    Returns:
        LoggerConfig: Logger configuration.
    """
    return LoggerConfig(
        log_interval=log_interval,
        tensorboard_dir=tensorboard_dir,
        timing_log_level=1,
        log_timers_to_tensorboard=True,
    )


def create_tokenizer_config(
    model_id: str,
    tokenizer_type: str = "HuggingFaceTokenizer",
) -> TokenizerConfig:
    """
    Create a tokenizer configuration for a given model.

    Args:
        model_id (str): HuggingFace model ID or path.
        tokenizer_type (str): Type of tokenizer to use.

    Returns:
        TokenizerConfig: Tokenizer configuration.
    """
    return TokenizerConfig(
        tokenizer_type=tokenizer_type,
        tokenizer_model=model_id,
    )


def create_checkpoint_config(
    pretrained_checkpoint: str,
    checkpoint_dir: str,
    save_interval: int = 50,
    ckpt_format: str = "torch_dist",
    fully_parallel_save: bool = True,
    async_save: bool = True,
) -> CheckpointConfig:
    """
    Create a checkpoint configuration for fine-tuning.

    Args:
        pretrained_checkpoint (str): Path to pretrained checkpoint to load from.
        checkpoint_dir (str): Directory to save checkpoints to.
        save_interval (int): Interval for saving checkpoints.
        ckpt_format (str): Checkpoint format to use.
        fully_parallel_save (bool): Whether to use fully parallel save.
        async_save (bool): Whether to save asynchronously.

    Returns:
        CheckpointConfig: Checkpoint configuration.
    """
    return CheckpointConfig(
        save_interval=save_interval,
        save=checkpoint_dir,
        load=checkpoint_dir,
        ckpt_format=ckpt_format,
        fully_parallel_save=fully_parallel_save,
        async_save=async_save,
        pretrained_checkpoint=pretrained_checkpoint,
    )


def create_training_config(
    train_iters: int,
    global_batch_size: int,
    micro_batch_size: int,
    eval_interval: int = 30,
    eval_iters: int = 32,
    manual_gc: bool = True,
    manual_gc_interval: int = 100,
    manual_gc_eval: int = 100,
) -> TrainingConfig:
    """
    Create a training configuration for fine-tuning.

    Args:
        train_iters (int): Number of training iterations.
        global_batch_size (int): Global batch size.
        micro_batch_size (int): Micro batch size.
        eval_interval (int): Interval for evaluation.
        eval_iters (int): Number of evaluation iterations.
        manual_gc (bool): Whether to use manual garbage collection.
        manual_gc_interval (int): Interval for manual garbage collection.
        manual_gc_eval (int): Evaluation interval for manual garbage collection.

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


def create_rng_config(seed: int = 1234) -> RNGConfig:
    """
    Create an RNG configuration for fine-tuning.

    Args:
        seed (int): Random seed.

    Returns:
        RNGConfig: RNG configuration.
    """
    return RNGConfig(seed=seed)


def create_peft_config(
    peft_scheme: Optional[str],
    dim: int = 8,
    alpha: int = 16,
) -> Optional[Union[LoRA, DoRA]]:
    """
    Create a PEFT configuration based on the scheme.

    Args:
        peft_scheme (Optional[str]): PEFT scheme ("lora", "dora", or None).
        dim (int): PEFT dimension.
        alpha (int): PEFT alpha parameter.

    Returns:
        Optional[Union[LoRA, DoRA]]: PEFT configuration or None.
    """
    if peft_scheme is None or peft_scheme.lower() == "none":
        return None
    elif peft_scheme.lower() == "lora":
        return LoRA(dim=dim, alpha=alpha)
    elif peft_scheme.lower() == "dora":
        return DoRA(dim=dim, alpha=alpha)
    else:
        raise ValueError(f"Unrecognized PEFT scheme: {peft_scheme}. Supported schemes: 'lora', 'dora', or None")


def create_optimizer_and_scheduler_config(
    lr: float,
    train_iters: int,
    lr_warmup_iters: int = 50,
    min_lr: float = 0,
    adam_beta2: float = 0.98,
    use_distributed_optimizer: bool = True,
):
    """
    Create optimizer and scheduler configurations for fine-tuning.

    Args:
        lr (float): Learning rate.
        train_iters (int): Number of training iterations for decay schedule.
        lr_warmup_iters (int): Number of warmup iterations.
        min_lr (float): Minimum learning rate for cosine decay.
        adam_beta2 (float): Adam beta2 parameter.
        use_distributed_optimizer (bool): Whether to use distributed optimizer.

    Returns:
        Tuple[OptimizerConfig, SchedulerConfig]: Optimizer and scheduler configurations.
    """
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=lr,
        min_lr=min_lr,
        adam_beta2=adam_beta2,
    )
    opt_config.use_distributed_optimizer = use_distributed_optimizer
    return opt_config, scheduler


def get_default_learning_rate(peft_scheme: Optional[str]) -> float:
    """
    Get the default learning rate based on the PEFT scheme.

    Args:
        peft_scheme (Optional[str]): PEFT scheme ("lora", "dora", or None).

    Returns:
        float: Default learning rate.
    """
    if peft_scheme is None or peft_scheme.lower() == "none":
        return 5e-6  # Full fine-tuning learning rate
    else:
        return 1e-4  # PEFT learning rate


def get_distributed_optimizer_setting(peft_scheme: Optional[str]) -> bool:
    """
    Get the distributed optimizer setting based on the PEFT scheme.

    Args:
        peft_scheme (Optional[str]): PEFT scheme ("lora", "dora", or None).

    Returns:
        bool: Whether to use distributed optimizer.
    """
    if peft_scheme is None or peft_scheme.lower() == "none":
        return True  # Full fine-tuning uses distributed optimizer
    else:
        return False  # PEFT does not use distributed optimizer


def get_packed_sequence_settings(packed_sequence: bool) -> tuple[int, int]:
    """
    Get sequence length and batch size settings based on packed sequence usage.

    Args:
        packed_sequence (bool): Whether to use packed sequences.

    Returns:
        Tuple[int, int]: (seq_length, global_batch_size)
    """
    if packed_sequence:
        return 4096, 8  # Higher seq_length, lower batch_size for packed
    else:
        return 2048, 128  # Standard settings


def setup_output_directories(
    dir: Optional[str],
    name: str,
) -> tuple[str, str, str]:
    """
    Setup output directories for logging and checkpoints.

    Args:
        dir (Optional[str]): Base directory for outputs.
        name (str): Name of the experiment.

    Returns:
        Tuple[str, str, str]: (run_output_dir, checkpoint_dir, tensorboard_dir)
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")
    return run_output_dir, checkpoint_dir, tensorboard_dir
