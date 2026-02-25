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
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, get_mixed_precision_config
from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge.diffusion.data.flux.flux_mock_datamodule import FluxMockDataModuleConfig
from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider


def model_config(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    seq_length: int = 1024,
    # FLUX-specific parameters
    num_joint_layers: int = 19,
    num_single_layers: int = 38,
    hidden_size: int = 3072,
    num_attention_heads: int = 24,
    in_channels: int = 64,
    context_dim: int = 4096,
    guidance_embed: bool = False,
    guidance_scale: float = 3.5,
) -> FluxProvider:
    """
    Configure the FLUX model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        seq_length (int): Sequence length for the model.
        num_joint_layers (int): Number of double (joint) transformer blocks.
        num_single_layers (int): Number of single transformer blocks.
        hidden_size (int): Hidden dimension size.
        num_attention_heads (int): Number of attention heads.
        in_channels (int): Number of input channels (latent channels).
        context_dim (int): Text encoder context dimension.
        guidance_embed (bool): Whether to use guidance embedding (for FLUX-dev).
        guidance_scale (float): Classifier-free guidance scale.

    Returns:
        FluxProvider: Configuration for the FLUX model.
    """
    return FluxProvider(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        seq_length=seq_length,
        # FLUX-specific
        num_joint_layers=num_joint_layers,
        num_single_layers=num_single_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        in_channels=in_channels,
        context_dim=context_dim,
        guidance_embed=guidance_embed,
        guidance_scale=guidance_scale,
    )


def pretrain_config(
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    mock: bool = False,
    # Model configuration
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    use_megatron_fsdp: bool = False,
    # FLUX model configuration
    num_joint_layers: int = 19,
    num_single_layers: int = 38,
    hidden_size: int = 3072,
    num_attention_heads: int = 24,
    in_channels: int = 64,
    context_dim: int = 4096,
    guidance_embed: bool = False,
    guidance_scale: float = 3.5,
    # Image configuration
    image_H: int = 1024,
    image_W: int = 1024,
    vae_channels: int = 16,
    vae_scale_factor: int = 8,
    prompt_seq_len: int = 512,
    pooled_prompt_dim: int = 768,
    # Training hyperparameters
    train_iters: int = 10000,
    global_batch_size: int = 4,
    micro_batch_size: int = 1,
    lr: float = 1e-4,
    lr_warmup_iters: int = 1000,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for FLUX model.

    Args:
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (Optional[List[str]]): List of paths to dataset files. If None, mock data will be used.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        num_joint_layers (int): Number of double (joint) transformer blocks.
        num_single_layers (int): Number of single transformer blocks.
        hidden_size (int): Hidden dimension size.
        num_attention_heads (int): Number of attention heads.
        in_channels (int): Number of input channels (latent channels).
        context_dim (int): Text encoder context dimension.
        guidance_embed (bool): Whether to use guidance embedding (for FLUX-dev).
        guidance_scale (float): Classifier-free guidance scale.
        image_H (int): Image height.
        image_W (int): Image width.
        vae_channels (int): Number of VAE latent channels.
        vae_scale_factor (int): VAE downsampling factor.
        prompt_seq_len (int): Sequence length for text prompts (T5).
        pooled_prompt_dim (int): Dimensionality of pooled text embeddings (CLIP).
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        lr (float): Learning rate.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    model_cfg = model_config(
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
        seq_length=1024,
        num_joint_layers=num_joint_layers,
        num_single_layers=num_single_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        in_channels=in_channels,
        context_dim=context_dim,
        guidance_embed=guidance_embed,
        guidance_scale=guidance_scale,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=lr,
    )
    opt_config.use_precision_aware_optimizer = False

    if isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    precision_config.grad_reduce_in_fp32 = False

    if mock:
        dataset = FluxMockDataModuleConfig(
            path=None,
            seq_length=1024,
            image_H=image_H,
            image_W=image_W,
            vae_channels=vae_channels,
            vae_scale_factor=vae_scale_factor,
            prompt_seq_len=prompt_seq_len,
            context_dim=context_dim,
            pooled_prompt_dim=pooled_prompt_dim,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=16,
            packing_buffer_size=None,
        )
    else:
        # Real dataset configuration using Energon WebDataset
        from megatron.bridge.diffusion.data.flux.flux_energon_datamodule import FluxDataModuleConfig

        dataset = FluxDataModuleConfig(
            path=data_paths,  # Path to WebDataset shards directory
            seq_length=1024,
            vae_scale_factor=vae_scale_factor,
            latent_channels=vae_channels,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=16,
            task_encoder_seq_length=None,
            packing_buffer_size=None,  # Disable Sequence Packing for now
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
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,
        ),
        dataset=dataset,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            save_interval=2000,
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
