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

"""Qwen3-Omni thinker training recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider
from megatron.bridge.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def _model_dtype_from_hf_config(hf_config: Qwen3OmniMoeConfig) -> torch.dtype:
    """Resolve the training dtype from the HF config."""
    dtype_value = getattr(hf_config.thinker_config, "torch_dtype", None) or getattr(hf_config, "torch_dtype", None)
    if dtype_value in (torch.float16, "float16", "fp16"):
        return torch.float16
    if dtype_value in (torch.bfloat16, "bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _provider_from_hf_path(hf_path: str) -> Qwen3OmniModelProvider:
    """Build a Qwen3-Omni provider directly from HF config without AutoBridge."""
    hf_config = Qwen3OmniMoeConfig.from_pretrained(hf_path)
    if getattr(hf_config, "enable_audio_output", False):
        raise NotImplementedError(
            "Qwen3-Omni talker/code2wav audio-output checkpoints are not supported yet. "
            "Only thinker-side training is currently implemented."
        )

    thinker_config = hf_config.thinker_config
    talker_config = getattr(hf_config, "talker_config", None)
    code2wav_config = getattr(hf_config, "code2wav_config", None)
    text_config = thinker_config.text_config
    vision_config = thinker_config.vision_config
    rope_scaling = getattr(text_config, "rope_scaling", None) or getattr(text_config, "rope_parameters", None) or {}
    model_dtype = _model_dtype_from_hf_config(hf_config)

    return Qwen3OmniModelProvider(
        thinker_config=thinker_config,
        talker_config=talker_config,
        code2wav_config=code2wav_config,
        num_layers=text_config.num_hidden_layers,
        hidden_size=text_config.hidden_size,
        ffn_hidden_size=text_config.intermediate_size,
        moe_ffn_hidden_size=getattr(text_config, "moe_intermediate_size", None),
        num_attention_heads=text_config.num_attention_heads,
        num_query_groups=text_config.num_key_value_heads,
        init_method_std=text_config.initializer_range,
        layernorm_epsilon=text_config.rms_norm_eps,
        gated_linear_unit=True,
        make_vocab_size_divisible_by=text_config.vocab_size,
        rotary_base=getattr(text_config, "rope_theta", 1000000.0),
        share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
        vocab_size=text_config.vocab_size,
        seq_length=text_config.max_position_embeddings,
        max_position_embeddings=text_config.max_position_embeddings,
        language_max_sequence_length=text_config.max_position_embeddings,
        fp16=(model_dtype == torch.float16),
        bf16=(model_dtype == torch.bfloat16),
        params_dtype=model_dtype,
        add_qkv_bias=getattr(text_config, "attention_bias", False),
        qk_layernorm=True,
        num_moe_experts=getattr(text_config, "num_experts", 128),
        moe_router_topk=getattr(text_config, "num_experts_per_tok", 8),
        image_token_id=getattr(thinker_config, "image_token_id", 151655),
        video_token_id=getattr(thinker_config, "video_token_id", 151656),
        audio_token_id=getattr(thinker_config, "audio_token_id", 151646),
        vision_start_token_id=getattr(thinker_config, "vision_start_token_id", 151652),
        audio_start_token_id=getattr(thinker_config, "audio_start_token_id", 151647),
        position_id_per_seconds=getattr(thinker_config, "position_id_per_seconds", 25),
        seconds_per_chunk=getattr(thinker_config, "seconds_per_chunk", 2),
        patch_size=getattr(vision_config, "patch_size", 16),
        temporal_patch_size=getattr(vision_config, "temporal_patch_size", 2),
        spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
        position_embedding_type="mrope",
        mrope_section=rope_scaling.get("mrope_section", [24, 20, 20]),
    )


def _qwen3_omni_apply_common(
    cfg: "ConfigContainer",
    hf_path: str,
    *,
    tp: int,
    pp: int,
    max_lr: float,
    min_lr: float,
    gbs: int = 32,
) -> None:
    """Apply shared thinker-only training settings for Qwen3-Omni."""
    cfg.model = _provider_from_hf_path(hf_path)
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = pp
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_audio_model = False
    cfg.model.vit_gradient_checkpointing = False
    cfg.model.multimodal_attn_impl = "auto"

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.attention_backend = "auto"

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = gbs
    cfg.train.micro_batch_size = 1

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=1000,
        max_lr=max_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.pack_sequences_in_batch = False

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"


def qwen3_omni_30b_a3b_sft_config(
    hf_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> "ConfigContainer":
    """Return a minimal thinker-only SFT config for Qwen3-Omni 30B-A3B."""

    cfg = _sft_common_vlm()
    _qwen3_omni_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen3_omni_30b_a3b_sft_preloaded_config(
    hf_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> "ConfigContainer":
    """Return a thinker-only SFT config backed by preloaded local JSON/JSONL data."""

    cfg = _sft_common_vlm()
    _qwen3_omni_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=5e-6, min_lr=5e-7)
    cfg.dataset = PreloadedVLMConversationProvider(
        seq_length=cfg.model.seq_length,
        hf_processor_path=hf_path,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        dataloader_type="single",
        num_workers=2,
    )
    cfg.dataset.pack_sequences_in_batch = False
    return cfg
