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

"""Native Omni provider; model dimensions come from the HF configuration."""

from dataclasses import dataclass

from megatron.bridge.models.d1.modeling_d1.omni_model import D1OmniModel
from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class D1OmniModelProvider(GPTModelProvider):
    """Construct a native Omni backbone/head and the published HF perception modules.

    Omni-specific fields must be supplied from the HF config by its bridge.
    Neither model size nor optional missing perception configs are inferred.
    """

    share_embeddings_and_output_weights: bool = False
    is_vision_language: bool = True
    scatter_embedding_sequence_parallel: bool = False
    d1_layer_types: tuple[str, ...] = ()
    d1_vision_config: dict[str, object] | None = None
    d1_audio_config: dict[str, object] | None = None
    d1_projector_hidden_size: int | None = None
    d1_head_layers: int | None = None
    d1_max_length: int | None = None
    d1_hf_config: dict[str, object] | None = None

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> D1OmniModel:
        """Build the complete model for the currently supported unpartitioned path.

        Args:
            pre_process: Must be omitted or true; this model owns its embeddings.
            post_process: Must be omitted or true; this model owns its head.
            vp_stage: Must be omitted; virtual pipeline partitioning is unsupported.

        Returns:
            Native Omni model with differentiable decision logits and loss.
        """
        if pre_process is False or post_process is False or vp_stage is not None:
            raise ValueError("D1 Omni requires the complete unpartitioned model")
        if self.share_embeddings_and_output_weights:
            raise ValueError("D1 Omni has an independent marker head and requires untied output weights")
        if not self.is_vision_language:
            raise ValueError("D1 Omni includes both image and audio perception modules")
        if (
            not self.d1_layer_types
            or self.d1_vision_config is None
            or self.d1_audio_config is None
            or self.d1_projector_hidden_size is None
            or self.d1_head_layers is None
            or self.d1_max_length is None
        ):
            raise ValueError("All Omni model fields must be supplied from the HF configuration")
        self.finalize()
        return D1OmniModel(
            self,
            vocab_size=self.vocab_size,
            layer_types=tuple(self.d1_layer_types),
            rotary_base=self.rotary_base,
            vision_config=self.d1_vision_config,
            audio_config=self.d1_audio_config,
            projector_hidden_size=self.d1_projector_hidden_size,
            head_layers=self.d1_head_layers,
            max_length=self.d1_max_length,
        )
