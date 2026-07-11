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

"""H100 pretrain recipe for Sarvam 30B."""

from megatron.bridge.recipes.utils.text_pretrain_utils import build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def sarvam_30b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Sarvam 30B MoE H100 pretrain config."""
    return build_text_pretrain_config(
        hf_model_id="sarvamai/sarvam-30b",
        revision="071ae95e933605ca1104a6b4524a36a98488efa4",
        tensor_parallelism=1,
        pipeline_parallelism=1,
        expert_parallelism=8,
        trust_remote_code=True,
    )


__all__ = ["sarvam_30b_pretrain_8gpu_h100_bf16_config"]
