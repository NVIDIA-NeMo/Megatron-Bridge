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
"""Public Ling 3.0 Tiny Base SFT recipe."""

from __future__ import annotations

import os

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


LING_V3_TINY_BASE_HF_MODEL = "inclusionAI/Ling-3.0-tiny-base"
LING_V3_TINY_BASE_HF_REVISION = "bab7297fa02713af237e378bf21107718b8e0e1a"
_LING_V3_TINY_BASE_SFT_SEQ_LENGTH = 2048


def _ling_v3_tiny_base_hf_path() -> str:
    """Resolve an optional local Tiny Base reference for offline construction."""
    return os.environ.get("LING_V3_TINY_BASE_HF_PATH", LING_V3_TINY_BASE_HF_MODEL)


def ling_v3_tiny_base_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return an eight-GPU BF16 SFT config for the public Ling 3.0 Tiny Base model.

    The model and tokenizer configuration are read from the public Tiny Base
    Hugging Face reference. Set ``checkpoint.pretrained_checkpoint`` with the
    launcher to load the corresponding local Hugging Face weights.
    """
    cfg = _sft_common()
    hf_model_path = _ling_v3_tiny_base_hf_path()

    hf_model_kwargs: dict[str, object] = {"trust_remote_code": True}
    tokenizer_kwargs: dict[str, str] = {}
    if hf_model_path == LING_V3_TINY_BASE_HF_MODEL:
        hf_model_kwargs["revision"] = LING_V3_TINY_BASE_HF_REVISION
        tokenizer_kwargs["revision"] = LING_V3_TINY_BASE_HF_REVISION

    cfg.model = AutoBridge.from_hf_pretrained(hf_model_path, **hf_model_kwargs).to_megatron_provider(
        load_weights=False
    )

    # Keep the tokenizer tied to the same public Base revision as the model.
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = hf_model_path
    cfg.tokenizer.hf_tokenizer_kwargs = tokenizer_kwargs
    cfg.checkpoint.hf_trust_remote_code = True

    # Eight-way expert parallelism places one expert shard on each GPU while
    # retaining a single tensor- and pipeline-parallel model replica.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    # SQuAD's prompt/completion preprocessing and offline packing are provided
    # by _sft_common; keep the model and dataset sequence lengths identical.
    cfg.model.seq_length = _LING_V3_TINY_BASE_SFT_SEQ_LENGTH
    cfg.dataset.seq_length = _LING_V3_TINY_BASE_SFT_SEQ_LENGTH
    cfg.dataset.offline_packing_specs.packed_sequence_size = _LING_V3_TINY_BASE_SFT_SEQ_LENGTH
    cfg.train.global_batch_size = 32

    # Portable training runtime. Architecture fields, including MTP count,
    # MTP pattern, and MTP loss scaling, remain owned by AutoBridge.
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


ling_v3_tiny_base_sft_config = ling_v3_tiny_base_sft_8gpu_h100_bf16_config


__all__ = [
    "LING_V3_TINY_BASE_HF_MODEL",
    "LING_V3_TINY_BASE_HF_REVISION",
    "ling_v3_tiny_base_sft_8gpu_h100_bf16_config",
    "ling_v3_tiny_base_sft_config",
]
