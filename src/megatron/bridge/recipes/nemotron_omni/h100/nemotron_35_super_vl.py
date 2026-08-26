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

"""Nemotron 3.5 Super VL training recipes."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni import (
    _make_nemotron_omni_energon_dataset,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_super import (
    _apply_nemotron_3_super_64gpu_h100_training_stack,
)
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


NEMOTRON_35_SUPER_VL_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-SourceOfTruth"


def _nemotron_35_super_vl_base() -> ConfigContainer:
    """Create the model and training base owned by Nemotron 3.5 Super VL."""
    cfg = _sft_common_vlm()
    cfg.model = AutoBridge.from_hf_pretrained(
        NEMOTRON_35_SUPER_VL_HF_MODEL_ID,
        trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)
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


def nemotron_35_super_vl_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return the 64-H100 BF16 performance pretraining config for Super VL.

    The benchmark policy is inherited directly from the corresponding
    Nemotron 3 Super recipe. Only the model, multimodal dataset, and tokenizer
    are replaced with their Super-VL counterparts, and all present model
    stacks remain trainable.

    Returns:
        The Super-VL performance pretraining configuration.
    """
    # Keep this import local so importing the public library recipe package
    # does not eagerly import the separate performance-recipe package.
    from megatron.bridge.perf_recipes._common import _benchmark_common
    from megatron.bridge.perf_recipes.nemotronh.h100.nemotronh import (
        nemotron_3_super_pretrain_64gpu_h100_bf16_config,
    )

    cfg = nemotron_3_super_pretrain_64gpu_h100_bf16_config()
    vl_cfg = nemotron_35_super_vl_sft_64gpu_h100_bf16_config()
    cfg.model = vl_cfg.model
    cfg.dataset = vl_cfg.dataset
    cfg.tokenizer = vl_cfg.tokenizer
    # Reapply model-level benchmark settings after replacing the text-only
    # provider. Non-model benchmark settings are idempotent.
    cfg.model.moe_router_force_load_balancing = True
    _benchmark_common(cfg)
    # The trainable vision stack needs an additional model-parallel shard on
    # 80 GiB H100s. Keep PP/EP unchanged and place TP=2 within each NVLink
    # domain so the 64-GPU layout remains valid with EP=32.
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.ddp.overlap_param_gather = False
    cfg.model.moe_hybridep_num_sms = None
    cfg.checkpoint.async_save = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    return cfg


def nemotron_35_super_vl_sft_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return the 64-H100 BF16 SFT configuration for Nemotron 3.5 Super VL.

    The model and audio-free image/video Energon data path reuse the Nemotron
    Omni stack. The language decoder reuses the measured Nemotron 3 Super
    TP1/PP2/EP32 HybridEP training stack without replacing this checkpoint's
    native one-layer MTP or separate temporal video embedder configuration.

    The Energon shard path must be set with ``dataset.path=<path>``. Samples
    may contain text, images, and videos, but must not contain audio.

    Returns:
        The Super-VL SFT configuration.
    """
    cfg = _nemotron_35_super_vl_base()
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = True
    cfg.model.calculate_per_token_loss = True

    cfg.dataset = _make_nemotron_omni_energon_dataset(
        cfg.train.micro_batch_size,
        hf_processor_path=NEMOTRON_35_SUPER_VL_HF_MODEL_ID,
    )
    return _apply_nemotron_3_super_64gpu_h100_training_stack(cfg)


__all__ = [
    "nemotron_35_super_vl_pretrain_64gpu_h100_bf16_config",
    "nemotron_35_super_vl_sft_64gpu_h100_bf16_config",
]
