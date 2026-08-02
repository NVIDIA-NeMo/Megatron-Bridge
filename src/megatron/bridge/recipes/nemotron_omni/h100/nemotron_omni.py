# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Nemotron Omni SFT/PEFT recipes (CORD v2 VL, Valor32k-AVQA audio-visual, temporal video).

All recipes use ``nemotron_omni_step`` (pass ``--step_func nemotron_omni_step``).
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    DirectHFSFTDatasetConfig,
    EnergonDatasetConfig,
    HFDatasetSourceConfig,
    NemotronOmniEnergonTaskEncoderConfig,
)
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_DEFAULT_HF_PATH = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"


def _make_nemotron_omni_energon_dataset(micro_batch_size: int) -> EnergonDatasetConfig:
    """Create the declarative temporal-video Energon config used by Omni recipes."""
    return EnergonDatasetConfig(
        path=None,
        seq_length=4096,
        micro_batch_size=micro_batch_size,
        num_workers=2,
        task_encoder=NemotronOmniEnergonTaskEncoderConfig(
            hf_processor_path=_DEFAULT_HF_PATH,
            max_audio_duration=10.0,
            num_mel_bins=128,
            visual_keys=("pixel_values",),
            temporal_patch_size=2,
            video_fps=1.0,
            video_nframes=8,
            use_temporal_video_embedder=True,
            patch_dim=16,
            trust_remote_code=True,
        ),
        enable_in_batch_packing=False,
    )


def nemotron_omni_cord_v2_sft_4gpu_h100_bf16_config() -> ConfigContainer:
    """Return a VL SFT config for Nemotron Omni on CORD v2.

    Vision-language finetuning on the CORD v2 receipt parsing dataset.
    Sound modules are omitted because this dataset contains only image-text samples.
    Default configuration: 4 GPUs (TP=4).
    Uses nemotron_omni_step (pass --step_func nemotron_omni_step).
    """
    cfg = _nemotron_omni_base()
    cfg.model.temporal_patch_dim = 1
    cfg.model.has_sound = False
    cfg.dataset = DirectHFSFTDatasetConfig(
        seq_length=4096,
        preprocessing=ChatSFTPreprocessingConfig(),
        hf_processor_path=_DEFAULT_HF_PATH,
        trust_remote_code=True,
        source=HFDatasetSourceConfig(dataset_name="cord_v2"),
        num_workers=2,
        dataloader_type="cyclic",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        enable_in_batch_packing=False,
    )

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_omni_cord_v2_long_context_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return an 8K CORD v2 SFT config with in-batch packing and CP2.

    In-batch packing requires a micro batch greater than one. The TP4/CP2
    topology needs at least eight GPUs and aligns every packed row to the
    combined CP/SP multiple. Precision-aware Adam uses FP16 main parameters
    with stored FP32 remainders, BF16 gradients, and BF16 moments so first-step
    optimizer-state initialization fits within 80 GB H100 memory.
    """
    cfg = nemotron_omni_cord_v2_sft_4gpu_h100_bf16_config()
    cfg.model.seq_length = 8192
    cfg.model.context_parallel_size = 2
    cfg.model.calculate_per_token_loss = True
    cfg.train.micro_batch_size = 2
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.main_params_dtype = torch.float16
    cfg.optimizer.store_param_remainders = True
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.dataset.seq_length = 8192
    cfg.dataset.enable_in_batch_packing = True
    cfg.dataset.in_batch_packing_pad_to_multiple_of = 8

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_omni_cord_v2_peft_4gpu_h100_bf16_config() -> ConfigContainer:
    """Return a LoRA PEFT config for Nemotron Omni on CORD v2.

    LoRA adapters are applied to attention, Mamba, and FC1/FC2 projections.
    Vision base modules remain frozen and sound modules are omitted.
    Default configuration: 4 GPUs (TP=4).
    Uses nemotron_omni_step (pass --step_func nemotron_omni_step).
    """
    from megatron.bridge.peft.lora import LoRA

    cfg = _nemotron_omni_base()
    cfg.model.temporal_patch_dim = 1
    cfg.model.has_sound = False
    cfg.peft = LoRA(
        target_modules=["linear_qkv", "linear_proj", "in_proj", "out_proj", "linear_fc1", "linear_fc2"],
        dim=16,
        alpha=32,
    )
    cfg.checkpoint.load = None
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = True

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=None,
        max_lr=1e-4,
        min_lr=1e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    cfg.dataset = DirectHFSFTDatasetConfig(
        seq_length=4096,
        preprocessing=ChatSFTPreprocessingConfig(),
        hf_processor_path=_DEFAULT_HF_PATH,
        trust_remote_code=True,
        source=HFDatasetSourceConfig(dataset_name="cord_v2"),
        num_workers=2,
        dataloader_type="cyclic",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        enable_in_batch_packing=False,
    )

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def _nemotron_omni_base() -> ConfigContainer:
    """Shared model/training config for all Nemotron Omni recipes."""
    cfg = _sft_common_vlm()
    cfg.model = AutoBridge.from_hf_pretrained(_DEFAULT_HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )
    cfg.model.seq_length = 4096

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True

    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = False

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None

    cfg.train.train_iters = 2000
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.validation.eval_interval = 200
    cfg.validation.eval_iters = 0

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=None,
        max_lr=6e-6,
        min_lr=6e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.checkpoint.save_interval = 200
    cfg.mixed_precision = "bf16_mixed"

    return cfg


def nemotron_omni_valor32k_sft_4gpu_h100_bf16_config() -> ConfigContainer:
    """Return an Energon SFT config with temporal video embedder enabled.

    Uses RADIO's ``separate_video_embedder`` to fuse temporal frame pairs
    (2 consecutive frames → 1 vision embedding) instead of discarding every
    other frame.
    The shard path must be set via CLI override: ``dataset.path=<path>``.

    Uses ``nemotron_omni_step`` (pass ``--step_func nemotron_omni_step``).
    """
    cfg = _nemotron_omni_base()

    # Enable temporal video embedder on the model side
    cfg.model.temporal_patch_dim = 2
    cfg.model.separate_video_embedder = True
    cfg.model.temporal_ckpt_compat = True

    cfg.dataset = _make_nemotron_omni_energon_dataset(cfg.train.micro_batch_size)

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_omni_valor32k_peft_4gpu_h100_bf16_config() -> ConfigContainer:
    """LoRA PEFT recipe on temporal-video Energon path (temporal_patch_dim=2).

    Adapters target attention, Mamba, and FC1/FC2 projections. Vision and sound
    base modules remain frozen while matching adapters are trainable.
    """
    from megatron.bridge.peft.lora import LoRA

    cfg = _nemotron_omni_base()

    cfg.model.temporal_patch_dim = 2
    cfg.model.separate_video_embedder = True
    cfg.model.temporal_ckpt_compat = True

    cfg.peft = LoRA(
        target_modules=["linear_qkv", "linear_proj", "in_proj", "out_proj", "linear_fc1", "linear_fc2"],
        dim=16,
        alpha=32,
    )
    cfg.checkpoint.load = None
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = True

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=None,
        max_lr=1e-4,
        min_lr=1e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    cfg.dataset = _make_nemotron_omni_energon_dataset(cfg.train.micro_batch_size)

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "nemotron_omni_cord_v2_long_context_sft_8gpu_h100_bf16_config",
    "nemotron_omni_cord_v2_peft_4gpu_h100_bf16_config",
    "nemotron_omni_cord_v2_sft_4gpu_h100_bf16_config",
    "nemotron_omni_valor32k_peft_4gpu_h100_bf16_config",
    "nemotron_omni_valor32k_sft_4gpu_h100_bf16_config",
]
