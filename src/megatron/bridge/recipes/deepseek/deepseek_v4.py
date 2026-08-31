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
# ruff: noqa: F401
"""Hardware-agnostic DeepSeek V4 recipes and compatibility aliases."""

from __future__ import annotations

from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_bf16_config,
    deepseek_v4_flash_pretrain_64gpu_gb200_bf16_muon_config,
    deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_bf16_config as deepseek_v4_flash_pretrain_gb200_config,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_bf16_muon_config as deepseek_v4_flash_pretrain_muon_gb200_config,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config as deepseek_v4_flash_pretrain_mxfp8_gb200_config,
)
from megatron.bridge.recipes.deepseek.gb300.deepseek_v4 import (
    DEEPSEEK_V4_PRO_HF_PATH,
)
from megatron.bridge.recipes.deepseek.gb300.deepseek_v4 import (
    deepseek_v4_pro_pretrain_32gpu_gb300_bf16_config as deepseek_v4_pro_pretrain_config,
)
from megatron.bridge.recipes.deepseek.gb300.deepseek_v4 import (
    deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config as deepseek_v4_pro_pretrain_mxfp8_config,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import (
    DEEPSEEK_V4_FLASH_HF_PATH,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import (
    deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config as deepseek_v4_flash_no_mtp_sft_config,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import (
    deepseek_v4_flash_pretrain_32gpu_h100_bf16_config as deepseek_v4_flash_pretrain_config,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import (
    deepseek_v4_flash_pretrain_32gpu_h100_bf16_muon_config as deepseek_v4_flash_pretrain_muon_config,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import (
    deepseek_v4_flash_pretrain_32gpu_h100_fp8mx_config as deepseek_v4_flash_pretrain_mxfp8_config,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import (
    deepseek_v4_flash_sft_32gpu_h100_bf16_config as deepseek_v4_flash_sft_config,
)
from megatron.bridge.recipes.utils.dataset_utils import default_openmathinstruct2_thinking_config
from megatron.bridge.training.config import ConfigContainer


__all__ = [
    "deepseek_v4_flash_no_mtp_sft_config",
    "deepseek_v4_flash_peft_openmath_thinking_packed_config",
    "deepseek_v4_flash_pretrain_config",
    "deepseek_v4_flash_pretrain_muon_config",
    "deepseek_v4_flash_pretrain_mxfp8_config",
    "deepseek_v4_flash_pretrain_64gpu_gb200_bf16_config",
    "deepseek_v4_flash_pretrain_64gpu_gb200_bf16_muon_config",
    "deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config",
    "deepseek_v4_flash_pretrain_gb200_config",
    "deepseek_v4_flash_pretrain_mxfp8_gb200_config",
    "deepseek_v4_flash_pretrain_muon_gb200_config",
    "deepseek_v4_flash_sft_config",
    "deepseek_v4_flash_sft_openmath_thinking_packed_config",
    "deepseek_v4_pro_pretrain_config",
    "deepseek_v4_pro_pretrain_mxfp8_config",
    "DEEPSEEK_V4_PRO_HF_PATH",
    "DEEPSEEK_V4_FLASH_HF_PATH",
    "set_deepseek_v4_pipeline_model_parallel_layout",
]


_DEEPSEEK_V4_LORA_TARGET_MODULES = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]


def _apply_deepseek_v4_lora(cfg: ConfigContainer) -> None:
    """Apply the DeepSeek V4 LoRA convergence contract to an SFT config."""
    cfg.peft = LoRA(
        target_modules=list(_DEEPSEEK_V4_LORA_TARGET_MODULES),
        dim=32,
        alpha=32,
        dropout=0.0,
        share_expert_adapters=False,
    )
    cfg.optimizer.lr = 1.0e-4
    cfg.optimizer.min_lr = 0.0
    # MCore's unified recompute path cannot replay the frozen grouped-expert graph.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None


def deepseek_v4_flash_sft_openmath_thinking_packed_config() -> ConfigContainer:
    """DSv4 Flash SFT on OpenMathInstruct-2 with thinking channel and offline-packed sequences.

    CoT reasoning goes into the assistant thinking field and the final answer into the
    content field. Uses packed sequences for efficient training.
    Pre-pack data with ``prepare_gpt_sft_packed_data.py`` before running SFT.
    When using CP>1, pass ``model.cp_partition_mode=contiguous`` (required for DSv4 CSA
    attention) and ``pad_seq_to_mult=4`` to ensure divisibility by cp_size.

    For GB200-optimized training with HybridEP dispatcher and DSA kernel fusion,
    use ``deepseek_v4_flash_sft_openmath_thinking_packed_gb200_config`` instead.
    """
    cfg = deepseek_v4_flash_sft_config()
    # DSv4 hybrid attention requires contiguous CP partition when CP > 1;
    # setting it unconditionally is safe (no-op when context_parallel_size=1).
    cfg.model.cp_partition_mode = "contiguous"
    cfg.dataset = default_openmathinstruct2_thinking_config(
        seq_length=cfg.model.seq_length,
        enable_offline_packing=True,
        pad_seq_to_mult=2 * cfg.model.context_parallel_size,
    )
    return cfg


def deepseek_v4_flash_peft_openmath_thinking_packed_config() -> ConfigContainer:
    """DSv4 Flash LoRA on packed OpenMathInstruct-2 thinking data.

    The attention targets follow the DeepSeek MLA projection layout. Both shared
    and routed MLP projections are adapted; grouped routed experts use one adapter
    per local expert to match the verl training layout.
    """
    cfg = deepseek_v4_flash_sft_openmath_thinking_packed_config()
    _apply_deepseek_v4_lora(cfg)
    return cfg
