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

import logging
import os
from typing import List, Optional, Union

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from typing_extensions import TypedDict, Unpack

from megatron.bridge.models.kimi import KimiK2Provider
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


logger = logging.getLogger(__name__)


class KimiK2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Kimi-K2 recipe helper functions."""

    # Core identifiers
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
    expert_model_parallel_size: int
    sequence_parallel: bool
    # Recomputation
    recompute_granularity: str
    recompute_modules: list[str] | None
    recompute_method: str | None
    recompute_num_layers: int | None
    # DeePEP and RoPE
    enable_deepep: bool
    apply_rope_fusion: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    optimizer_type: str
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


def kimi_k2_pretrain_config(**user_kwargs: Unpack[KimiK2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Kimi-K2 (1T).

    See `_kimi_k2_common` for the full list of parameters.
    """
    recommended_kwargs: KimiK2CommonKwargs = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 16,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 32,
        "sequence_parallel": True,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: KimiK2CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _kimi_k2_common(**combined_kwargs)


def _kimi_k2_model_config(
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 16,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 32,
    sequence_parallel: bool = True,
    # Recomputation
    recompute_granularity: str = "selective",
    recompute_modules: list[str] | None = None,
    recompute_method: str | None = None,
    recompute_num_layers: int | None = None,
    enable_deepep: bool = False,
    apply_rope_fusion: bool = False,
    layout: Optional[Union[str, List[List[str]]]] = None,
) -> KimiK2Provider:
    """
    Configure the Kimi-K2 (1T) model.

    Args:
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        pipeline_dtype: Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size: Size of virtual pipeline parallelism.
        context_parallel_size: Degree of context parallelism.
        expert_model_parallel_size: Degree of expert model parallelism.
        sequence_parallel: Whether to use sequence parallelism.
        recompute_granularity: Granularity of recomputation.
        recompute_modules: List of modules to recompute.
        recompute_method: Method of recomputation.
        recompute_num_layers: Number of layers to recompute.
        enable_deepep: Whether to use DeePEP.
        apply_rope_fusion: Whether to apply RoPE fusion.

    Returns:
        KimiK2Provider: Configuration for the Kimi-K2 model.
    """
    cfg = KimiK2Provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=sequence_parallel,
        expert_tensor_parallel_size=1,  # Do not use ETP
        # Recomputation
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
    )

    # Pipeline split for asymmetric stages as used in NeMo recipe
    cfg.account_for_embedding_in_pipeline_split = False
    cfg.account_for_loss_in_pipeline_split = False
    cfg.num_layers_in_first_pipeline_stage = None
    cfg.num_layers_in_last_pipeline_stage = None

    # Performance optimization knobs
    cfg.moe_permute_fusion = True
    if apply_rope_fusion:
        cfg.apply_rope_fusion = True

    # Pipeline parallelism configs. We infer PP layout from the provided PP and VP size
    map_pp_vp_to_layout = {
        (1, 1): None,
        (4, 1): [["embedding"] + ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 13 + ["loss"]],
        (8, 1): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + ["loss"]],
        (4, 2): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + ["loss"]],
        (16, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (8, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (4, 4): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
    }
    pp_size = pipeline_model_parallel_size or 1
    vp_size = virtual_pipeline_model_parallel_size or 1
    if layout is None:
        if (pp_size, vp_size) not in map_pp_vp_to_layout:
            raise ValueError(
                f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
                f"for Kimi-K2. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
            )

        layout = map_pp_vp_to_layout[(pp_size, vp_size)]

        if layout is not None:
            layout = list([list(x) for x in layout])  # yield all the elements

    if layout is not None:
        cfg.pipeline_model_parallel_layout = layout

    if enable_deepep:
        cfg.moe_token_dispatcher_type = "flex"
        cfg.moe_enable_deepep = True
        cfg.moe_shared_expert_overlap = False

    return cfg


def _kimi_k2_common(
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
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 16,
    pipeline_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 32,
    sequence_parallel: bool = True,
    # Recomputation
    recompute_granularity: str = "selective",
    recompute_modules: list[str] | None = None,
    recompute_method: str | None = None,
    recompute_num_layers: int | None = None,
    enable_deepep: bool = False,
    apply_rope_fusion: bool = False,
    # Training hyperparameters
    train_iters: int = 1_000_000,
    global_batch_size: int = 4096,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    optimizer_type: str = "muon",
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None = None,
    comm_overlap_config: CommOverlapConfig | None = None,
    layout: Optional[Union[str, List[List[str]]]] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Kimi-K2 (1T) model.

    Args:
        dir (str | None): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (list[str] | None): List of paths to dataset files. If None, mock data will be used.
        data_args_path (str | None): Path to file containing data arguments.
        train_data_path (list[str] | None): List of training data paths.
        valid_data_path (list[str] | None): List of validation data paths.
        test_data_path (list[str] | None): List of test data paths.
        per_split_data_args_path (str | None): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_model_parallel_size (int): Degree of tensor model parallelism.
        pipeline_model_parallel_size (int): Degree of pipeline model parallelism.
        pipeline_dtype (torch.dtype | None): Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size (int | None): Size of virtual pipeline parallelism.
        context_parallel_size (int): Degree of context parallelism.
        expert_model_parallel_size (int): Degree of expert model parallelism.
        sequence_parallel (bool): Whether to use sequence parallelism.
        recompute_granularity (str): Granularity of recomputation.
        recompute_modules (list[str] | None): List of modules to recompute.
        recompute_method (str | None): Method of recomputation.
        recompute_num_layers (int | None): Number of layers to recompute.
        enable_deepep (bool): Whether to use DeePEP.
        apply_rope_fusion (bool): Whether to apply RoPE fusion.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        optimizer_type (str): Type of optimizer ("adam" or "muon").
        precision_config (MixedPrecisionConfig | str | None): Precision configuration for the model.
        comm_overlap_config (CommOverlapConfig | None): Communication overlap configuration.

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

    model_cfg = _kimi_k2_model_config(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=sequence_parallel,
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
        enable_deepep=enable_deepep,
        apply_rope_fusion=apply_rope_fusion,
        layout=layout,
    )

    if optimizer_type == "adam":
        opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
            lr_warmup_iters=lr_warmup_iters,
            lr_decay_iters=train_iters,
            max_lr=lr,
            min_lr=min_lr,
        )

    elif optimizer_type == "muon":
        opt_cfg, scheduler_cfg = distributed_muon_with_cosine_annealing(
            lr_warmup_iters=lr_warmup_iters,
            lr_decay_iters=train_iters,
            max_lr=lr,
            min_lr=min_lr,
        )
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    if precision_config is None:
        precision_config = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=True,
        )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=2000,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=5,
            manual_gc_eval=5,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,  # Muon needs this to be False
            average_in_collective=True,
            use_distributed_optimizer=False,  # Muon needs this to be False
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
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
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=model_cfg.vocab_size),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=False,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # for mla rope fusion

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
        )

    return cfg
