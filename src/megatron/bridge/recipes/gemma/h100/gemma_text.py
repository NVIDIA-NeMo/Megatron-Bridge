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

"""Additional text-only Gemma H100 pretrain recipes."""

import os
from contextlib import contextmanager

from megatron.bridge.models.gemma.gemma_provider import GemmaModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.text_pretrain_utils import apply_text_pretrain_defaults, build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


@contextmanager
def _gemma4_text_conversion_mode():
    previous_mode = os.environ.get("GEMMA4_CONVERSION_MODE")
    os.environ["GEMMA4_CONVERSION_MODE"] = "text"
    try:
        yield
    finally:
        if previous_mode is None:
            os.environ.pop("GEMMA4_CONVERSION_MODE", None)
        else:
            os.environ["GEMMA4_CONVERSION_MODE"] = previous_mode


def gemma_2b_pretrain_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return the original Gemma 2B H100 pretrain config."""
    cfg = _pretrain_common()
    cfg.model = GemmaModelProvider(
        num_layers=18,
        hidden_size=2048,
        ffn_hidden_size=16384,
        num_attention_heads=8,
        num_query_groups=1,
    )
    return apply_text_pretrain_defaults(
        cfg,
        tensor_parallelism=1,
        pipeline_parallelism=1,
    )


def gemma4_26b_a4b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the text-only Gemma 4 26B-A4B MoE H100 pretrain config."""
    with _gemma4_text_conversion_mode():
        cfg = build_text_pretrain_config(
            hf_model_id="google/gemma-4-26B-A4B",
            revision="6b556d30bb65a6ee0bdaec99bab0afc7bf1494fb",  # pragma: allowlist secret
            tensor_parallelism=1,
            pipeline_parallelism=1,
            expert_parallelism=8,
            trust_remote_code=True,
        )
    # The local Gemma 4 MoE implementation creates temporary fp32 RMSNorm
    # buffers. Full recompute keeps enough activation headroom for them.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    return cfg


def gemma4_31b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the text-only Gemma 4 31B dense H100 pretrain config."""
    with _gemma4_text_conversion_mode():
        return build_text_pretrain_config(
            hf_model_id="google/gemma-4-31B",
            revision="d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3",  # pragma: allowlist secret
            tensor_parallelism=4,
            pipeline_parallelism=2,
            trust_remote_code=True,
        )


__all__ = [
    "gemma4_26b_a4b_pretrain_8gpu_h100_bf16_config",
    "gemma4_31b_pretrain_8gpu_h100_bf16_config",
    "gemma_2b_pretrain_1gpu_h100_bf16_config",
]
