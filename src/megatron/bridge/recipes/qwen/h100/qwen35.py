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

"""H100 pretrain recipes for text-only Qwen3.5 models."""

from megatron.bridge.recipes.utils.text_pretrain_utils import build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def qwen35_27b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Qwen3.5 27B dense H100 pretrain config."""
    return build_text_pretrain_config(
        hf_model_id="Qwen/Qwen3.5-27B",
        revision="fc05daec18b0a78c049392ed2e771dde82bdf654",  # pragma: allowlist secret
        tensor_parallelism=4,
        pipeline_parallelism=2,
        trust_remote_code=True,
    )


def qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Qwen3.5 35B-A3B MoE H100 pretrain config."""
    cfg = build_text_pretrain_config(
        hf_model_id="Qwen/Qwen3.5-35B-A3B",
        revision="59d61f3ce65a6d9863b86d2e96597125219dc754",  # pragma: allowlist secret
        tensor_parallelism=2,
        pipeline_parallelism=1,
        expert_parallelism=8,
        trust_remote_code=True,
    )
    # The hybrid delta-rule and MoE token-combine paths need temporary
    # buffers late in the forward pass, after many layer activations exist.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    return cfg


__all__ = [
    "qwen35_27b_pretrain_8gpu_h100_bf16_config",
    "qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config",
]
