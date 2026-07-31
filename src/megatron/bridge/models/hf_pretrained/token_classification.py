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

from typing import Generic, TypeVar

from transformers import AutoModelForTokenClassification, PreTrainedModel

from megatron.bridge.models.hf_pretrained.masked_lm import PreTrainedMaskedLM


TokenClassificationType = TypeVar("TokenClassificationType", bound=PreTrainedModel)


class PreTrainedTokenClassification(
    PreTrainedMaskedLM[TokenClassificationType],
    Generic[TokenClassificationType],
):
    """Lazy Hugging Face token-classification model wrapper."""

    def _load_model(self) -> TokenClassificationType:
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load model")

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "config": self.config,
            **self.init_kwargs,
        }
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        model = AutoModelForTokenClassification.from_pretrained(self.model_name_or_path, **model_kwargs)
        return model.to(self.device)
