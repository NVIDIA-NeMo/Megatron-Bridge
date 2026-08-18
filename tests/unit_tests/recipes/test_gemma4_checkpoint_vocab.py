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

"""Regression coverage for Gemma 4 recipe checkpoint vocabulary ownership."""

from unittest.mock import Mock, patch

import pytest

from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider
from megatron.bridge.recipes.gemma.h100 import gemma4 as gemma4_recipe
from megatron.bridge.training.setup import _validate_and_set_vocab_size


@pytest.mark.unit
def test_gemma4_mock_recipe_preserves_checkpoint_embedding_rows() -> None:
    """The mock-data recipe must allocate the converted checkpoint vocabulary."""
    provider = Gemma4DenseProvider(vocab_size=262143, make_vocab_size_divisible_by=128)
    bridge = Mock()
    bridge.to_megatron_provider.return_value = provider

    with patch.object(gemma4_recipe.AutoBridge, "from_hf_pretrained", return_value=bridge):
        config = gemma4_recipe.gemma4_e4b_pretrain_2gpu_h100_bf16_config()

    runtime_vocab_size, _ = _validate_and_set_vocab_size(
        model_vocab_size=config.model.vocab_size,
        tokenizer_vocab_size=config.tokenizer.vocab_size,
        use_tokenizer_vocab_size=config.tokenizer.use_tokenizer_vocab_size,
    )
    checkpoint_rows = 262144
    runtime_rows = (
        (runtime_vocab_size + config.model.make_vocab_size_divisible_by - 1)
        // config.model.make_vocab_size_divisible_by
        * config.model.make_vocab_size_divisible_by
    )

    assert runtime_rows == checkpoint_rows
