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
from typing import List, Optional, Union

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
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


class Llama2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Llama2 recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str
    # Dataset configuration
    data_paths: Optional[List[str]]
    data_args_path: Optional[str]
    train_data_path: Optional[List[str]]
    valid_data_path: Optional[List[str]]
    test_data_path: Optional[List[str]]
    per_split_data_args_path: Optional[str]
    mock: bool
    # Model configuration
    tensor_parallelism: int
    pipeline_parallelism: int
    pipeline_parallelism_dtype: Optional[torch.dtype]
    virtual_pipeline_parallelism: Optional[int]
    context_parallelism: int
    sequence_parallelism: bool
    use_megatron_fsdp: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]


def llama2_7b_pretrain_config(**user_kwargs: Unpack[Llama2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama-2 7B.

    See `_llama2_common` for the full list of parameters.
    """
    recommended_kwargs: Llama2CommonKwargs = {
        "hf_path": "meta-llama/Llama-2-7b-hf",
        "tensor_parallelism": 2,
        "pipeline_parallelism": 1,
        "train_iters": 1_168_251,
        "global_batch_size": 512,
        "micro_batch_size": 1,
        "lr_warmup_iters": 2000,
        "eval_interval": 2000,
        "save_interval": 2000,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Llama2CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama2_common(**combined_kwargs)


def _llama2_common(
    hf_path: str,
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
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    use_megatron_fsdp: bool = False,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 2000,
    save_interval: int = 2000,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Llama2 models using a given HuggingFace path.

    Args:
        hf_path (str): HuggingFace model path (e.g., "meta-llama/Llama-2-7b-hf").
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
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (Optional[int]): Number of iterations over which to decay the LR.
        eval_interval (int): Evaluation interval.
        save_interval (int): Save interval.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration for the model.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_parallelism
    model_cfg.pipeline_model_parallel_size = pipeline_parallelism
    model_cfg.pipeline_dtype = pipeline_parallelism_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_parallelism
    model_cfg.context_parallel_size = context_parallelism
    model_cfg.sequence_parallel = sequence_parallelism
    model_cfg.seq_length = seq_length

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

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
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
            num_workers=8,
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer" if use_null_tokenizer else "HuggingFaceTokenizer",
            tokenizer_model=hf_path if not use_null_tokenizer else None,
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE if use_null_tokenizer else None,
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
        )

    return cfg
