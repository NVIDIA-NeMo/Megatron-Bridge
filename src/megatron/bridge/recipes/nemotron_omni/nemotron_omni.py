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
from megatron.bridge.data.vlm_datasets.collate import nemotron_omni_collate_fn
from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer

_DEFAULT_HF_PATH = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

def nemotron_omni_cord_v2_sft_config(hf_path: str = _DEFAULT_HF_PATH) -> ConfigContainer:
    """Return a VL SFT config for Nemotron Omni on CORD v2.

    Vision-language finetuning on the CORD v2 receipt parsing dataset.
    Default configuration: 1 node, 8 GPUs (TP=4).
    Uses nemotron_omni_step (pass --step_func nemotron_omni_step).

    Args:
        hf_path: HuggingFace model ID or local path to the Nemotron Omni model.
    """
    cfg = _nemotron_omni_base_config(hf_path=hf_path)
    cfg.model.temporal_patch_dim = 1
    cfg.dataset = HFDatasetConversationProvider(
        seq_length=4096,
        hf_processor_path=hf_path,
        maker_name="cord_v2",
        collate_impl=nemotron_omni_collate_fn,
        num_workers=2,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pack_sequences_in_batch=False,
    )

    return cfg


def nemotron_omni_cord_v2_peft_config(hf_path: str = _DEFAULT_HF_PATH) -> ConfigContainer:
    """Return a LoRA PEFT config for Nemotron Omni on CORD v2.

    LoRA adapters are applied to language-model attention + Mamba projections.
    Vision encoder/projection and sound encoder/projection are frozen.
    Default configuration: 1 node, 8 GPUs (TP=4).
    Uses nemotron_omni_step (pass --step_func nemotron_omni_step).

    Args:
        hf_path: HuggingFace model ID or local path to the Nemotron Omni model.
    """
    from megatron.bridge.peft.lora import LoRA

    cfg = _nemotron_omni_base_config(hf_path=hf_path)
    cfg.model.temporal_patch_dim = 1
    cfg.peft = LoRA(
        target_modules=["linear_qkv", "linear_proj", "in_proj", "out_proj"],
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

    cfg.dataset = HFDatasetConversationProvider(
        seq_length=4096,
        hf_processor_path=hf_path,
        maker_name="cord_v2",
        collate_impl=nemotron_omni_collate_fn,
        num_workers=2,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pack_sequences_in_batch=False,
    )

    return cfg


def _nemotron_omni_base_config(
    hf_path: str = _DEFAULT_HF_PATH,
) -> ConfigContainer:
    """Shared model/training config for all Nemotron Omni recipes."""
    cfg = _sft_common_vlm()
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096
    # Dynamic-resolution is the native behavior for the Nemotron-3 Omni
    # Reasoning HF processor (variable per-image H×W within [min, max] patches).
    # The collate pre-patchifies pixel_values and emits imgs_sizes/num_frames.
    cfg.model.dynamic_resolution = True

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


def nemotron_omni_valor32k_sft_config(
    hf_path: str = _DEFAULT_HF_PATH,
) -> ConfigContainer:
    """Return an Energon SFT config with temporal video embedder enabled.

    Uses RADIO's ``separate_video_embedder`` to fuse temporal frame pairs
    (2 consecutive frames → 1 vision embedding) instead of discarding every
    other frame. Requires ``dynamic_resolution=True``.
    The shard path must be set via CLI override: ``dataset.path=<path>``.

    Uses ``nemotron_omni_step`` (pass ``--step_func nemotron_omni_step``).

    Args:
        hf_path: HuggingFace model ID or local path to the Nemotron Omni model.
    """
    from transformers import AutoProcessor

    from megatron.bridge.data.energon.energon_provider import EnergonProvider
    from megatron.bridge.data.energon.nemotron_omni_task_encoder import NemotronOmniTaskEncoder

    cfg = _nemotron_omni_base_config(hf_path=hf_path)

    # Enable temporal video embedder on the model side
    cfg.model.dynamic_resolution = True
    cfg.model.temporal_patch_dim = 2
    cfg.model.separate_video_embedder = True
    cfg.model.temporal_ckpt_compat = True

    processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)
    task_encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=4096,
        max_audio_duration=10.0,
        num_mel_bins=128,
        visual_keys=("pixel_values",),
        temporal_patch_size=2,
        video_fps=1.0,
        video_nframes=8,
        use_temporal_video_embedder=True,
        patch_dim=16,
    )

    cfg.dataset = EnergonProvider(
        path="",  # Must be set via CLI override: dataset.path=<path>
        seq_length=4096,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        num_workers=2,
        task_encoder=task_encoder,
        pack_sequences_in_batch=False,
    )

    return cfg

def nemotron_omni_valor32k_peft_config(
    hf_path: str = _DEFAULT_HF_PATH,
) -> ConfigContainer:
    """LoRA PEFT recipe on temporal-video Energon path (temporal_patch_dim=2)."""
    from transformers import AutoProcessor

    from megatron.bridge.data.energon.energon_provider import EnergonProvider
    from megatron.bridge.data.energon.nemotron_omni_task_encoder import NemotronOmniTaskEncoder
    from megatron.bridge.peft.lora import LoRA

    cfg = _nemotron_omni_base_config(hf_path=hf_path)

    cfg.model.dynamic_resolution = True
    cfg.model.temporal_patch_dim = 2
    cfg.model.separate_video_embedder = True
    cfg.model.temporal_ckpt_compat = True

    cfg.peft = LoRA(
        target_modules=["linear_qkv", "linear_proj", "in_proj", "out_proj"],
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

    processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)
    task_encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=4096,
        max_audio_duration=10.0,
        num_mel_bins=128,
        visual_keys=("pixel_values",),
        temporal_patch_size=2,
        video_fps=1.0,
        video_nframes=8,
        use_temporal_video_embedder=True,
        patch_dim=16,
    )

    cfg.dataset = EnergonProvider(
        path="",
        seq_length=4096,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        num_workers=2,
        task_encoder=task_encoder,
        pack_sequences_in_batch=False,
    )

    return cfg
