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

"""H100 pretrain recipe for Llama Nemotron Nano."""

from megatron.bridge.recipes.utils.text_pretrain_utils import build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def llama31_nemotron_nano_4b_pretrain_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Llama 3.1 Nemotron Nano 4B H100 pretrain config."""
    return build_text_pretrain_config(
        hf_model_id="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        revision="d552708a9d575fa8d4a690b988fd870d65279f98",  # pragma: allowlist secret
        tensor_parallelism=2,
        pipeline_parallelism=1,
    )


__all__ = ["llama31_nemotron_nano_4b_pretrain_2gpu_h100_bf16_config"]
