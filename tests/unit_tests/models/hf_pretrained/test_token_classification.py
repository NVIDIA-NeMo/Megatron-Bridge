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

from unittest.mock import Mock, patch

import torch

from megatron.bridge.models.hf_pretrained.token_classification import PreTrainedTokenClassification


def test_load_model_uses_auto_model_for_token_classification() -> None:
    wrapper = PreTrainedTokenClassification(
        model_name_or_path="Qwen/Qwen3.5-token-classification",
        device="cpu",
        torch_dtype=torch.bfloat16,
        revision="test-revision",
    )
    wrapper.config = Mock()
    model = Mock()
    model.to.return_value = model

    with patch(
        "megatron.bridge.models.hf_pretrained.token_classification.AutoModelForTokenClassification.from_pretrained",
        return_value=model,
    ) as from_pretrained:
        assert wrapper.model is model

    from_pretrained.assert_called_once_with(
        "Qwen/Qwen3.5-token-classification",
        trust_remote_code=False,
        config=wrapper.config,
        revision="test-revision",
        torch_dtype=torch.bfloat16,
    )
    model.to.assert_called_once_with("cpu")


def test_multimodal_processors_are_preserved_as_optional_artifacts(tmp_path) -> None:
    wrapper = PreTrainedTokenClassification(
        model_name_or_path="Qwen/Qwen3.5-token-classification",
        device="cpu",
    )
    processor = Mock()
    processor.image_processor = Mock()

    with patch(
        "megatron.bridge.models.hf_pretrained.token_classification.AutoProcessor.from_pretrained",
        return_value=processor,
    ) as from_pretrained:
        assert wrapper.processor is processor
        assert wrapper.image_processor is processor.image_processor

    assert wrapper.OPTIONAL_ARTIFACTS == ["processor", "image_processor"]
    from_pretrained.assert_called_once_with(
        "Qwen/Qwen3.5-token-classification",
        trust_remote_code=False,
    )

    wrapper._config = Mock()
    wrapper._tokenizer = Mock()
    wrapper.save_artifacts(tmp_path)

    processor.save_pretrained.assert_called_once_with(tmp_path)
    processor.image_processor.save_pretrained.assert_called_once_with(tmp_path)
