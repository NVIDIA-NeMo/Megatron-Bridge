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

"""Provider for native causal D1 vision/language models."""

from dataclasses import dataclass

from transformers import Lfm2VlConfig

from megatron.bridge.models.d1.modeling_d1.vl_model import D1VLModel
from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class D1VLModelProvider(GPTModelProvider):
    """Construct D1 VL using dimensions and perception settings from its HF config."""

    is_vision_language: bool = True
    scatter_embedding_sequence_parallel: bool = False
    d1_hf_config: dict[str, object] | None = None

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> D1VLModel:
        """Build the full native causal language and HF vision composition.

        Args:
            pre_process: Must be omitted or true.
            post_process: Must be omitted or true.
            vp_stage: Must be omitted; virtual pipeline partitioning is unsupported.

        Returns:
            Native trainable causal model with tied input/output weights.
        """
        if pre_process is False or post_process is False or vp_stage is not None:
            raise ValueError("D1 VL requires the complete unpartitioned model")
        if not self.share_embeddings_and_output_weights:
            raise ValueError("D1 VL requires tied input/output weights")
        if not self.is_vision_language or self.d1_hf_config is None:
            raise ValueError("D1 VL requires the complete published HF vision/language configuration")
        self.finalize()
        hf_config = Lfm2VlConfig(**self.d1_hf_config)
        if (
            self.vocab_size != hf_config.text_config.vocab_size
            or self.hidden_size != hf_config.text_config.hidden_size
        ):
            raise ValueError("Native vocabulary/hidden dimensions must match the source HF configuration")
        return D1VLModel(self, hf_config=hf_config)
