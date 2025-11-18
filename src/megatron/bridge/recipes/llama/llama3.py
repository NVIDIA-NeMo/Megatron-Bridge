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

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
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
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed, get_mixed_precision_config


class Llama3CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Llama3 family recipe helpers."""

    # Core identifiers
    hf_path: str
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
    use_megatron_fsdp: bool
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    adam_eps: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


class Llama3FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Llama3 finetuning recipe helper functions.

    This is separate from Llama3CommonKwargs to avoid confusion - finetuning
    uses SQuAD dataset by default, not the data path fields.
    """

    # Core identifiers
    dir: str | None
    name: str

    # Finetuning-specific
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool

    # Training hyperparameters
    train_iters: int
    global_batch_size: int | None
    micro_batch_size: int
    seq_length: int | None
    eval_interval: int
    save_interval: int

    # Optimizer
    finetune_lr: float | None
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None

    # Precision
    precision_config: MixedPrecisionConfig | str | None


# Sequence length constants
SEQUENCE_LENGTH_16K: int = 16384
SEQUENCE_LENGTH_64K: int = 65536
SEQUENCE_LENGTH_128K: int = 131072


# Llama3.2 models
def llama32_1b_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3.2 1B.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Llama-3.2-1B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama32_3b_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3.2 3B.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Llama-3.2-3B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


# Llama3 8B models
def llama3_8b_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3 8B.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-8B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 2,
        "sequence_parallel": False,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama3_8b_16k_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3 8B 16K.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-8B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "seq_length": SEQUENCE_LENGTH_16K,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama3_8b_64k_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3 8B 64K.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-8B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 4,
        "sequence_parallel": True,
        "seq_length": SEQUENCE_LENGTH_64K,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama3_8b_128k_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3 8B 128K.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-8B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 8,
        "sequence_parallel": True,
        "seq_length": SEQUENCE_LENGTH_128K,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama3_8b_low_precision_pretrain_config(
    mixed_precision_recipe: str, **user_kwargs: Unpack[Llama3CommonKwargs]
) -> ConfigContainer:
    """Return a low precision (FP8 Current Scaling/MXFP8/NVFP4) pre-training config for Llama 3 8B.

    Args:
        mixed_precision_recipe (str): The mixed precision recipe to use. Valid options are:
            - "bf16_with_mxfp8_mixed"
            - "bf16_with_fp8_current_scaling_mixed"
            - "bf16_with_nvfp4_mixed"
        user_kwargs (Unpack[Llama3CommonKwargs]): Additional user-specified configuration options.

    Returns:
        ConfigContainer: The pre-training configuration for Llama 3 8B.

    See `_llama3_common` for the full list of parameters.
    """
    assert mixed_precision_recipe in [
        "bf16_with_mxfp8_mixed",
        "bf16_with_fp8_current_scaling_mixed",
        "bf16_with_nvfp4_mixed",
    ], f"Invalid low precision recipe: {mixed_precision_recipe}. This recipe has not been tested yet."
    precision_config = get_mixed_precision_config(mixed_precision_recipe)
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-8B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 2,
        "sequence_parallel": False,
        "precision_config": precision_config,
        "lr": 6e-4,
        "min_lr": 6e-6,
        "adam_eps": 1e-8,
        "micro_batch_size": 1,
        "global_batch_size": 768,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


# Llama3 70B models
def llama3_70b_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3 70B.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-70B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 4,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": 5,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "comm_overlap_config": CommOverlapConfig(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
        ),
        "precision_config": bf16_mixed(),
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama3_70b_16k_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3 70B 16K.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-70B",
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "seq_length": SEQUENCE_LENGTH_16K,
        "comm_overlap_config": CommOverlapConfig(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
        ),
        "precision_config": bf16_mixed(),
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama3_70b_64k_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3 70B 64K.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-70B",
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 4,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 8,
        "sequence_parallel": True,
        "seq_length": SEQUENCE_LENGTH_64K,
        "comm_overlap_config": CommOverlapConfig(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
        ),
        "precision_config": bf16_mixed(),
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


# Llama3.1 models
def llama31_8b_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3.1 8B.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3.1-8B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 2,
        "sequence_parallel": False,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama31_70b_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3.1 70B.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3.1-70B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 4,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": 5,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "comm_overlap_config": CommOverlapConfig(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
        ),
        "precision_config": bf16_mixed(),
        "seq_length": SEQUENCE_LENGTH_128K,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def llama31_405b_pretrain_config(**user_kwargs: Unpack[Llama3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Llama 3.1 405B.

    See `_llama3_common` for the full list of parameters.
    """
    recommended_kwargs: Llama3CommonKwargs = {
        "hf_path": "meta-llama/Meta-Llama-3.1-405B",
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 8,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": 2,
        "context_parallel_size": 4,
        "sequence_parallel": True,
        "account_for_embedding_in_pipeline_split": True,
        "account_for_loss_in_pipeline_split": True,
        "comm_overlap_config": CommOverlapConfig(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
        ),
        "precision_config": bf16_mixed(),
        "micro_batch_size": 1,
        "seq_length": SEQUENCE_LENGTH_128K,
    }
    combined_kwargs: Llama3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)


def _llama3_common(
    hf_path: str,
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
    use_megatron_fsdp: bool = False,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
    # Training hyperparameters
    train_iters: int = 1168251,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    adam_eps: float = 1e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int | None = None,
    eval_interval: int = 2000,
    save_interval: int = 500,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Llama3 family models using a given HuggingFace path.

    Args:
        hf_path (str): HuggingFace model path (e.g., "meta-llama/Meta-Llama-3-8B").
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
        pipeline_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size (Optional[int]): Size of virtual pipeline parallelism.
        context_parallel_size (int): Degree of context parallelism.
        sequence_parallel (bool): Whether to use sequence parallelism.
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        account_for_embedding_in_pipeline_split (bool): Whether to account for embedding in pipeline split.
        account_for_loss_in_pipeline_split (bool): Whether to account for loss in pipeline split.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        adam_eps (float): AdamW epsilon.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (Optional[int]): Number of iterations over which to decay the LR.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration.

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
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length
    model_cfg.cross_entropy_fusion_impl = "te"

    # Large model specific pipeline split configurations
    if account_for_embedding_in_pipeline_split:
        model_cfg.account_for_embedding_in_pipeline_split = True
    if account_for_loss_in_pipeline_split:
        model_cfg.account_for_loss_in_pipeline_split = True

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=lr,
        min_lr=min_lr,
        adam_eps=adam_eps,
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
            use_megatron_fsdp=use_megatron_fsdp,
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
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
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

    return cfg


# ============================================================================
# Finetuning Configurations
# ============================================================================


def llama32_1b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.2 1B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=1, PP=1, LR=5e-6
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Llama-3.2-1B", **user_kwargs)

    # Model-specific parallelism settings
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    config.model.context_parallel_size = 1
    config.model.sequence_parallel = False

    # PEFT or Full SFT specific settings
    if is_full_sft:
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False

    return config


def llama32_3b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.2 3B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=1, PP=1, LR=5e-6
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Llama-3.2-3B", **user_kwargs)

    # Model-specific parallelism settings (match NeMo pattern)
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    config.model.context_parallel_size = 1
    config.model.sequence_parallel = False

    # PEFT or Full SFT specific settings
    if is_full_sft:
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False

    return config


def llama3_8b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3 8B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=2, PP=1, LR=5e-6
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    if "finetune_lr" not in user_kwargs:
        user_kwargs["finetune_lr"] = 5e-6 if is_full_sft else 1e-4

    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3-8B", **user_kwargs)

    # Parallelism settings
    if is_full_sft:
        config.model.tensor_model_parallel_size = 2
        config.model.pipeline_model_parallel_size = 1
        config.peft = None
    else:
        config.model.tensor_model_parallel_size = 1
        config.model.pipeline_model_parallel_size = 1
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
        else:
            config.peft = peft
        config.optimizer.use_distributed_optimizer = False
        config.model.cross_entropy_loss_fusion = False

    return config


def llama31_8b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.1 8B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=2, PP=1, LR=5e-6
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3.1-8B", **user_kwargs)

    # Parallelism settings
    if is_full_sft:
        config.model.tensor_model_parallel_size = 2
        config.model.pipeline_model_parallel_size = 1
        config.peft = None
    else:
        config.model.tensor_model_parallel_size = 1
        config.model.pipeline_model_parallel_size = 1
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
        else:
            config.peft = peft
        config.optimizer.use_distributed_optimizer = False
        config.model.cross_entropy_loss_fusion = False

    return config


def llama3_70b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3 70B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=8, PP=1, LR=1e-4, dim=16, alpha=32
    - Full SFT (peft=None): TP=8, PP=4, VPP=5, LR=5e-6 (requires 4 nodes)
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3-70B", **user_kwargs)

    # PEFT or Full SFT specific settings
    if is_full_sft:
        config.model.tensor_model_parallel_size = 8
        config.model.pipeline_model_parallel_size = 4
        config.peft = None
    else:
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 16
            config.peft.alpha = 32
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False
        config.model.tensor_model_parallel_size = 8

    return config


def llama31_70b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.1 70B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=8, PP=1, LR=1e-4, dim=16, alpha=32
    - DoRA: TP=8, PP=1, LR=1e-4, dim=16, alpha=32
    - Full SFT (peft=None): TP=8, PP=4, VPP=5, LR=5e-6 (requires 4 nodes)
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3.1-70B", **user_kwargs)

    if is_full_sft:
        config.model.tensor_model_parallel_size = 8
        config.model.pipeline_model_parallel_size = 4
        config.peft = None
    else:
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 16
            config.peft.alpha = 32
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False
        config.model.tensor_model_parallel_size = 8

    return config


def llama31_405b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.1 405B.

    Default configuration: 4 nodes (LoRA) or 16 nodes (Full SFT), 8 GPUs per node
    - LoRA (default): TP=4, PP=8, VPP=8, CP=1, LR=1e-4, dim=16, alpha=32, GBS=32, SP=True
      Total: 32 GPUs (4 nodes)
      Note: 128 effective layers ÷ 8 = 16 layers/rank, VPP=8 splits into 2 layers/virtual stage
    - DoRA: TP=4, PP=8, VPP=8, CP=1, LR=1e-4, dim=16, alpha=32, GBS=32, SP=True
      Total: 32 GPUs (4 nodes)
    - Full SFT (peft=None): TP=8, PP=16, VPP=None, CP=1, LR=5e-6, GBS=6, SP=True
      Total: 128 GPUs (16 nodes)
      Note: 128 effective layers ÷ 16 = 8 layers/rank
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    if "finetune_lr" not in user_kwargs:
        user_kwargs["finetune_lr"] = 5e-6 if is_full_sft else 1e-4

    if "global_batch_size" not in user_kwargs:
        user_kwargs["global_batch_size"] = 16 if is_full_sft else 32

    if "seq_length" not in user_kwargs:
        user_kwargs["seq_length"] = 2048

    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3.1-405B", **user_kwargs)

    # Parallelism settings
    if is_full_sft:
        config.model.tensor_model_parallel_size = 8
        config.model.pipeline_model_parallel_size = 16
        config.model.virtual_pipeline_model_parallel_size = None
        config.model.sequence_parallel = True
        config.peft = None
        config.ddp = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        )
        config.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=True, defer_embedding_wgrad_compute=True, wgrad_deferral_limit=22
        )
    else:
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 16
            config.peft.alpha = 32
            config.peft.target_modules = ["linear_qkv"]
        else:
            config.peft = peft

        config.optimizer.use_distributed_optimizer = False
        config.model.cross_entropy_loss_fusion = False
        config.model.tensor_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 8
        config.model.virtual_pipeline_model_parallel_size = 8
        config.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    config.mixed_precision = get_mixed_precision_config(config.mixed_precision)
    config.mixed_precision.grad_reduce_in_fp32 = False
    config.ddp.grad_reduce_in_fp32 = False
    config.model.sequence_parallel = True

    config.train.manual_gc = True
    config.train.manual_gc_interval = 100
    config.train.manual_gc_eval = 100

    config.optimizer.use_precision_aware_optimizer = False

    config.model.account_for_embedding_in_pipeline_split = True
    config.model.account_for_loss_in_pipeline_split = True

    return config


def _llama3_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    packed_sequence: bool = False,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int | None = None,
    micro_batch_size: int = 1,
    seq_length: int | None = None,
    eval_interval: int = 30,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
) -> ConfigContainer:
    """Minimal common finetuning configuration.

    This function provides only the basic setup. Individual model configs handle parallelism settings
    depending on PEFT or full SFT.
    """

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Auto-select seq_length based on packed_sequence
    # For unpacked sequence, most samples in SQuAD dataset are shorter than 2K
    if seq_length is None:
        seq_length = 4096 if packed_sequence else 2048

    # Auto-select global_batch_size based on packed_sequence
    if global_batch_size is None:
        global_batch_size = 8 if packed_sequence else 128

    # Create basic model config from HF
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.seq_length = seq_length

    # Basic optimizer configuration
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )
    ddp_cfg = DistributedDataParallelConfig(check_for_nan_in_grad=True)
    return ConfigContainer(
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
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=ddp_cfg,
        dataset=default_squad_config(seq_length, packed_sequence),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=5678),
        peft=None,
        comm_overlap=None,
        mixed_precision=precision_config,
    )
