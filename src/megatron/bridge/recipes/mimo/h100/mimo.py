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

"""H100 pretrain recipe for MiMo 7B."""

from megatron.bridge.recipes.utils.text_pretrain_utils import build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def mimo_7b_pretrain_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return the MiMo 7B H100 pretrain config."""
    return build_text_pretrain_config(
        hf_model_id="XiaomiMiMo/MiMo-7B-Base",
        revision="c72df4586cb8bdeebd65f36929cd3385a6566fbe",  # pragma: allowlist secret
        tensor_parallelism=2,
        pipeline_parallelism=1,
        trust_remote_code=True,
    )


__all__ = ["mimo_7b_pretrain_2gpu_h100_bf16_config"]
