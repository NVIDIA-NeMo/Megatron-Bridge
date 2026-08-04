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

"""Builder-backed model construction for Qwen hybrid text models."""

from megatron.bridge.models.gpt.model_builder import LayerSpecGPTModelBuilder, mtp_block_spec_from_layer_spec


qwen_hybrid_mtp_block_spec = mtp_block_spec_from_layer_spec


class QwenHybridModelBuilder(LayerSpecGPTModelBuilder):
    """Build Qwen hybrid models while preserving their MTP attention spec."""


__all__ = ["QwenHybridModelBuilder", "qwen_hybrid_mtp_block_spec"]
