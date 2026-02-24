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
from transformers import AutoTokenizer, Qwen2VLImageProcessor
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets import (
    EnergonProvider,
    HFDatasetConversationProvider,
    MockVLMConversationProvider,
    PreloadedVLMConversationProvider,
)
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.qwen_vl.data.energon.task_encoder import QwenVLTaskEncoder
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config as _default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetProvider,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed


class Qwen3VLCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3 VL MoE recipe helpers."""

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
    expert_model_parallel_size: int | None
    expert_tensor_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    enable_recompute: bool
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None
    moe_flex_dispatcher_backend: str | None
    # Freeze options
    pretrained_checkpoint: str | None
    freeze_language_model: bool
    freeze_vision_model: bool
    freeze_vision_projection: bool
    # Dataset configuration
    dataset_type: str | None
    image_folder: str | None
    tokenizer_model: str | None
    # PEFT options
    peft: str | PEFT | None
    finetune_lr: float


def qwen3_vl_8b_pretrain_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3-VL 8B Instruct.

    See `_qwen3_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3-VL-8B-Instruct",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
    }
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def qwen3_vl_30b_a3b_pretrain_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3-VL-30B-A3B-Instruct.

    See `_qwen3_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 8,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def qwen3_vl_235b_a22b_pretrain_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3-VL-235B-A22B-Instruct.

    See `_qwen3_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 8,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 8,
        "account_for_embedding_in_pipeline_split": True,
        "account_for_loss_in_pipeline_split": True,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def qwen3_vl_8b_finetune_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3-VL 8B Instruct.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=4, PP=1, LR=1e-5

    See `_qwen3_vl_common` for the full list of parameters.
    """
    # Check if user is doing full SFT or PEFT
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3-VL-8B-Instruct",
        "tensor_model_parallel_size": 4 if is_full_sft else 1,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 1e-5 if is_full_sft else 1e-4,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
        "min_lr": 1e-6,
        "lr": 1e-5,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def qwen3_vl_30b_a3b_finetune_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3-VL-30B-A3B-Instruct.

    This is a Mixture-of-Experts model with 128 experts and top-8 routing.
    Recommended to use with expert parallelism (EP) for efficient training.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, EP=8, LR=2e-4
    - Full SFT: TP=1, PP=1, EP=8, LR=2e-5

    See `_qwen3_vl_common` for the full list of parameters.
    """
    # Check if user is doing full SFT or PEFT
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 8,
        "peft": peft_value,
        "finetune_lr": 2e-5 if is_full_sft else 2e-4,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
        "min_lr": 2e-6,
        "lr": 2e-5,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def qwen3_vl_235b_a22b_finetune_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3-VL-235B-A22B-Instruct.

    This is a Mixture-of-Experts model with 128 experts and top-8 routing.
    Recommended to use with expert parallelism (EP) for efficient training.

    Default configuration: 4 nodes, 32 GPUs total
    - LoRA/DoRA: TP=1, PP=1, EP=8, LR=2e-4
    - Full SFT: TP=4, PP=1, EP=8, LR=2e-5

    See `_qwen3_vl_common` for the full list of parameters.
    """
    # Check if user is doing full SFT or PEFT
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "tensor_model_parallel_size": 4 if is_full_sft else 1,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 2e-5 if is_full_sft else 2e-4,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
        "min_lr": 2e-6,
        "lr": 2e-5,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def _qwen3_vl_common(
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
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 2,
    pipeline_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int | None = 4,
    expert_tensor_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    enable_recompute: bool = False,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
    # Training hyperparameters
    train_iters: int = 300000,
    global_batch_size: int = 32,
    micro_batch_size: int = 2,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 500,
    lr_decay_iters: int | None = None,
    eval_interval: int = 500,
    save_interval: int = 500,
    use_null_tokenizer: bool = False,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = None,
    comm_overlap_config: CommOverlapConfig | None = None,
    moe_flex_dispatcher_backend: str | None = None,
    # Freeze options
    pretrained_checkpoint: str | None = None,
    freeze_language_model: bool = True,
    freeze_vision_model: bool = True,
    freeze_vision_projection: bool = False,
    # Dataset configuration
    dataset_type: str | None = None,
    image_folder: str | None = None,
    tokenizer_model: str | None = None,
    # PEFT options
    peft: str | PEFT | None = None,
    finetune_lr: float | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Qwen3 MoE models using a given HuggingFace path.

    Args:
        hf_path (str): HuggingFace model path (e.g., "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B").
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
        context_parallel_size (int): Degree of context parallelism to be passed to model_config.
        expert_model_parallel_size (int | None): Degree of expert parallelism for MoE.
        expert_tensor_parallel_size (int): Expert tensor parallelism for MoE.
        sequence_parallel (bool): Whether to use sequence parallelism.
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        enable_recompute (bool): Whether to enable recompute for memory optimization.
        account_for_embedding_in_pipeline_split (bool): Whether to account for embedding in pipeline split.
        account_for_loss_in_pipeline_split (bool): Whether to account for loss in pipeline split.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (int | None): Number of iterations over which to decay the LR.
        precision_config (MixedPrecisionConfig | str | None): Precision configuration for the model.
        comm_overlap_config (CommOverlapConfig | None): Communication overlap configuration.
        moe_flex_dispatcher_backend (str | None): Token dispatcher type [deepep, hybridep].
        pretrained_checkpoint (str | None): Path to pretrained checkpoint.
        freeze_language_model (bool): Whether to freeze the language model.
        freeze_vision_model (bool): Whether to freeze the vision model.
        freeze_vision_projection (bool): Whether to freeze the vision projection.
        dataset_type (str | None): Type of dataset to use.
        image_folder (str | None): Path to image folder.
        tokenizer_model (str | None): Path to tokenizer model.
        peft (str | PEFT | None): PEFT configuration (e.g., "lora", "dora", or PEFT object).
        finetune_lr (float | None): Learning rate override for fine-tuning.
    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = expert_tensor_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    # Freeze options
    model_cfg.freeze_language_model = freeze_language_model
    model_cfg.freeze_vision_model = freeze_vision_model
    model_cfg.freeze_vision_projection = freeze_vision_projection

    apply_flex_dispatcher_backend(model_cfg, moe_flex_dispatcher_backend)

    if precision_config is None:
        precision_config = bf16_mixed()

    # MoE-specific pipeline split configurations
    if account_for_embedding_in_pipeline_split:
        model_cfg.account_for_embedding_in_pipeline_split = True
    if account_for_loss_in_pipeline_split:
        model_cfg.account_for_loss_in_pipeline_split = True

    # Add recompute settings for memory optimization (used by some MoE models)
    if enable_recompute:
        model_cfg.recompute_granularity = "full"
        model_cfg.recompute_method = "uniform"
        model_cfg.recompute_num_layers = 1
    model_cfg.seq_length = seq_length
    model_cfg.cross_entropy_fusion_impl = "te"

    # Optimizer and scheduler - use finetune_lr if provided, otherwise use lr
    effective_lr = finetune_lr if finetune_lr is not None else lr
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=effective_lr,
        min_lr=min_lr,
    )

    # PEFT config
    peft_config = _default_peft_config(peft)

    # Determine dataset selection strategy.
    _processor_model = tokenizer_model or hf_path
    _dataset_choice = dataset_type or ("mock" if mock else "hf")

    if _dataset_choice == "mock":
        dataset_cfg: DatasetProvider = MockVLMConversationProvider(
            seq_length=seq_length,
            hf_processor_path=_processor_model,
            prompt="Describe this image.",
            num_workers=1,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            create_attention_mask=True,
            pad_to_max_length=True,
        )
    elif _dataset_choice == "preloaded":
        dataset_cfg = PreloadedVLMConversationProvider(
            seq_length=seq_length,
            hf_processor_path=_processor_model,
            train_data_path=train_data_path[0] if isinstance(train_data_path, list) else train_data_path,
            valid_data_path=valid_data_path[0] if isinstance(valid_data_path, list) else valid_data_path,
            test_data_path=test_data_path[0] if isinstance(test_data_path, list) else test_data_path,
            image_folder=image_folder,
            num_workers=2,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
        )
    elif _dataset_choice == "hf":
        dataset_cfg = HFDatasetConversationProvider(
            seq_length=seq_length,
            hf_processor_path=_processor_model,
            maker_name="make_cord_v2_dataset",
            num_workers=2,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
        )
    elif _dataset_choice == "energon":
        tokenizer = AutoTokenizer.from_pretrained(_processor_model)
        # Use from_pretrained to ensure correct normalization (mean/std) and config (min_pixels)
        # matching Preloaded provider behavior.
        image_processor = Qwen2VLImageProcessor.from_pretrained(_processor_model)

        dataset_cfg = EnergonProvider(
            seq_length=seq_length,
            path=train_data_path[0] if isinstance(train_data_path, list) else train_data_path,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=2,
            dataloader_type="external",
            task_encoder=QwenVLTaskEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_padding_length=seq_length,
                min_pixels=200704,
                max_pixels=1003520,
            ),
        )
    else:
        raise ValueError(
            f"Unsupported dataset_type '{_dataset_choice}'. Expected one of ['mock', 'preloaded', 'hf', 'energon']."
        )
    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=32,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,  # qwen3_vl does not support overlap_grad_reduce=True in current implementation
            overlap_param_gather=False,  # qwen3_vl does not support overlap_param_gather=True in current implementation
            average_in_collective=True,  # Not supported for Megatron FSDP for now, need to be set to False if using Megatron FSDP
            data_parallel_sharding_strategy="optim_grads_params",  # For Megatron FSDP only
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
        ),
        dataset=dataset_cfg,
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
            pretrained_checkpoint=pretrained_checkpoint,
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg
