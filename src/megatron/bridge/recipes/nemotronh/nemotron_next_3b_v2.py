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

import torch

from megatron.bridge.models.nemotronh import NemotronNanoNext3Bv2Provider
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

def model_config(
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    enable_deepep: bool = True,
    expert_tensor_parallelism: int = 1,
    expert_model_parallelism: int = 8,
) -> NemotronNanoNext3Bv2Provider:
    """
    Configure the Nemotron Next 3B v2 model.

    Args:
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        pipeline_parallelism_dtype: Data type for pipeline parallelism.
        virtual_pipeline_parallelism: Size of virtual pipeline parallelism.
        context_parallelism: Degree of context parallelism.
        sequence_parallelism: Whether to use sequence parallelism.
        expert_tensor_parallelism: Degree of expert tensor parallelism.
        expert_model_parallelism: Degree of expert model parallelism.
    Returns:
        NemotronNanoNext3Bv2Provider: Configuration for the Nemotron Next 3B v2 model.
    """
    #FIXME: we are getting rid of Model Provider. It should come directly from the bridge
    cfg = NemotronNanoNext3Bv2Provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        expert_tensor_parallel_size=expert_tensor_parallelism,
        expert_model_parallel_size=expert_model_parallelism,
        apply_rope_fusion=False,
        async_tensor_model_parallel_allreduce=True,
        attention_backend="fused",
        gradient_accumulation_fusion=True,
        init_method_std=0.0173,
        use_fused_weighted_squared_relu=True,
    )

    if enable_deepep:
        cfg.moe_token_dispatcher_type = "flex"
        cfg.moe_enable_deepep = True
        cfg.moe_shared_expert_overlap = True

    return cfg


def pretrain_config(
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[list[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[list[str]] = None,
    valid_data_path: Optional[list[str]] = None,
    test_data_path: Optional[list[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    path_to_cache: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    expert_tensor_parallelism: int = 1,
    expert_model_parallelism: int = 8,
    # Training hyperparameters
    train_iters: int = 39735,
    global_batch_size: int = 3072,
    micro_batch_size: int = 2,
    seq_length: int = 8192,
    lr: float = 1.6e-3,
    min_lr: float = 1.6e-5,
    lr_warmup_iters: int = 333,
    lr_decay_iters: Optional[int] = None,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # MoE
    enable_deepep: bool = True,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Nemotron Next 3B v2 model.

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
        tensor_model_parallel_size (int): Degree of tensor model parallelism.
        pipeline_model_parallel_size (int): Degree of pipeline model parallelism.
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
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration for the model.
        enable_deepep (bool): Whether to enable DeepEP for MoE.
        expert_tensor_parallelism (int): Degree of expert tensor parallelism.
        expert_model_parallelism (int): Degree of expert model parallelism.
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

    model_cfg = model_config(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
        enable_deepep=enable_deepep,
        expert_tensor_parallelism=expert_tensor_parallelism,
        expert_model_parallelism=expert_model_parallelism,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        lr_decay_style="cosine",
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=1000,
            eval_iters=14,
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
            path_to_cache=path_to_cache,
            # Dataloader config parameters
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
            skip_getting_attention_mask_from_dataset=True,
            mmap_bin_files=False,
        ),
        logger=LoggerConfig(
            log_interval=50,
            tensorboard_dir=tensorboard_dir,
        ),
        #TODO(liding): what is the correct tokenizer for Nemotron Next 3B v2
        tokenizer=TokenizerConfig(tokenizer_type="TikTokenizer", tiktoken_pattern="v2", tokenizer_model="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data-quality/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
            ckpt_assume_constant_structure=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_bootstrap_backend="nccl",
            tp_comm_overlap=True,
        )

    return cfg
