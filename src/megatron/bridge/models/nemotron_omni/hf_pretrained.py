# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native HF processor compatibility for the audio-free Super-VL checkpoint."""

import transformers
from packaging.version import Version
from transformers import AutoProcessor, ProcessorMixin

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class Nemotron35VLPretrained(PreTrainedCausalLM):
    """Keep native processor compatibility separate from legacy remote-code loading."""

    def _load_processor(self) -> ProcessorMixin | None:
        if self.trust_remote_code or self.model_name_or_path is None:
            return super()._load_processor()

        version = Version(transformers.__version__)
        if version < Version("5.19.0"):
            raise ValueError("Native Super-VL requires Transformers >=5.19.0; use trust_remote_code=True otherwise.")

        processor_cls = AutoProcessor
        # Bound the workaround to the release whose loading behavior we tested.
        # Future releases use their own loader and surface errors rather than
        # silently dropping a required VLM artifact during conversion/export.
        if version.base_version == "5.19.0" and getattr(self.config, "audio_config", None) is None:
            from megatron.bridge.models.nemotron_omni.processing_super_vl import Nemotron35VLProcessor

            processor_cls = Nemotron35VLProcessor
        return processor_cls.from_pretrained(self.model_name_or_path, trust_remote_code=False, **self.init_kwargs)
