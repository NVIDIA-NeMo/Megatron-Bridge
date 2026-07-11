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

"""H100 pretrain recipe for Mistral 7B."""

from megatron.bridge.recipes.utils.text_pretrain_utils import build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def mistral_7b_pretrain_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Mistral 7B H100 pretrain config."""
    return build_text_pretrain_config(
        hf_model_id="mistralai/Mistral-7B-v0.1",
        revision="27d67f1b5f57dc0953326b2601d68371d40ea8da",
        tensor_parallelism=2,
        pipeline_parallelism=1,
    )


__all__ = ["mistral_7b_pretrain_2gpu_h100_bf16_config"]
