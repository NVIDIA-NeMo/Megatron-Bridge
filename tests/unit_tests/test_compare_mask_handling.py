# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for attention_mask handling in compare.py.

Verifies that the Megatron path uses None attention_mask (letting the model
auto-generate its causal mask) and the HF path uses torch.ones_like(input_ids, dtype=torch.bool).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch


# Mock heavy dependencies before importing compare.py.
# compare.py has top-level imports for megatron.core, megatron.bridge, PIL, requests,
# transformers, qwen_vl_utils, and a local debugger module. These are not available
# in a CPU-only test environment, so we pre-populate sys.modules with MagicMock stubs.
_MODULES_TO_MOCK = [
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
    "megatron.core.pipeline_parallel",
    "megatron.core.pipeline_parallel.schedules",
    "megatron.core.dist_checkpointing",
    "megatron.core.dist_checkpointing.mapping",
    "megatron.core.msc_utils",
    "megatron.bridge",
    "megatron.bridge.automodel",
    "megatron.bridge.automodel.auto_bridge",
    "megatron.bridge.models",
    "megatron.bridge.models.hf_pretrained",
    "megatron.bridge.models.hf_pretrained.utils",
    "megatron.bridge.training",
    "megatron.bridge.training.utils",
    "megatron.bridge.training.utils.nemo_utils",
    "megatron.bridge.training.utils.checkpoint_utils",
    "megatron.bridge.utils",
    "megatron.bridge.utils.common_utils",
    "PIL",
    "PIL.Image",
    "requests",
    "debugger",
    "qwen_vl_utils",
    "transformers",
]

for _mod in _MODULES_TO_MOCK:
    sys.modules.setdefault(_mod, MagicMock())

# Add compare.py's directory to sys.path so we can import from it
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "examples", "conversion", "compare_hf_and_megatron"),
)

import compare  # noqa: E402
from compare import (  # noqa: E402
    SingleBatchIterator,
    _run_hf_inference,  # noqa: E402
    vlm_forward_step,
)


@pytest.mark.unit
class TestCompareMaskHandling:
    """Tests for attention_mask handling in compare.py Megatron and HF paths."""

    def test_single_batch_iterator_stores_none_attention_mask(self):
        """Test that SingleBatchIterator preserves None attention_mask in batch dict."""
        input_ids = torch.tensor([[1, 2, 3]])
        position_ids = torch.arange(3).unsqueeze(0)
        attention_mask = None

        iterator = SingleBatchIterator(input_ids, position_ids, attention_mask)
        batch = next(iterator)

        assert batch["attention_mask"] is None
        assert batch["tokens"].equal(input_ids)
        assert batch["position_ids"].equal(position_ids)

    def test_vlm_forward_step_passes_none_attention_mask(self):
        """Test that vlm_forward_step passes None attention_mask to the model."""
        batch = {
            "tokens": torch.tensor([[1, 2, 3]]),
            "position_ids": torch.arange(3).unsqueeze(0),
            "attention_mask": None,
        }
        data_iterator = iter([batch])
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 3, 100)

        vlm_forward_step(data_iterator, mock_model)

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["attention_mask"] is None

    def test_hf_path_receives_ones_like_attention_mask(self):
        """Test that HF model receives torch.ones_like(input_ids, dtype=torch.bool) attention_mask."""
        mock_hf_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 3, 100)
        mock_hf_model.return_value = mock_output

        input_ids = torch.tensor([[1, 2, 3]])
        expected_mask = torch.ones_like(input_ids, dtype=torch.bool)

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "test"

        with (
            patch.object(compare, "_is_rank_0", return_value=True),
            patch.object(compare, "print_rank_0"),
        ):
            _run_hf_inference(
                mock_hf_model,
                input_ids,
                pixel_values=None,
                image_grid_thw=None,
                tokenizer=mock_tokenizer,
            )

        call_kwargs = mock_hf_model.call_args.kwargs
        assert isinstance(call_kwargs["attention_mask"], torch.Tensor)
        assert call_kwargs["attention_mask"].dtype == torch.bool
        assert call_kwargs["attention_mask"].shape == input_ids.shape
        assert torch.equal(call_kwargs["attention_mask"], expected_mask)
