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
from types import SimpleNamespace
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
    "megatron.core.inference",
    "megatron.core.inference.contexts",
    "megatron.core.inference.utils",
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
    "megatron.bridge.utils.safe_url",
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
    _broadcast_hf_results,
    _maybe_gather_tensor_parallel_logits,
    _run_hf_inference,  # noqa: E402
    _run_megatron_forward,
    inference_forward_step,
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
        assert "inference_context" not in call_kwargs
        assert "runtime_gather_output" not in call_kwargs

    def test_text_inference_forward_step_passes_static_context(self):
        """Test that the text path receives the cache context and gathered-logit request."""
        inference_context = object()
        batch = {
            "tokens": torch.tensor([[1, 2, 3]]),
            "position_ids": torch.arange(3).unsqueeze(0),
            "attention_mask": None,
            "inference_context": inference_context,
        }
        mock_model = MagicMock(return_value=torch.randn(1, 1, 100))

        inference_forward_step(iter([batch]), mock_model)

        assert mock_model.call_args.kwargs["inference_context"] is inference_context
        assert mock_model.call_args.kwargs["runtime_gather_output"] is True

    def test_megatron_forward_activates_inference_mode(self):
        """Test that the scheduled forward runs inside MCore inference mode."""
        mock_forward = MagicMock(return_value="output")
        mock_context = MagicMock()

        with patch.object(compare.InferenceMode, "active", return_value=mock_context) as mock_active:
            result = _run_megatron_forward(mock_forward, forward_only=True)

        assert result == "output"
        mock_active.assert_called_once_with()
        mock_context.__enter__.assert_called_once_with()
        mock_context.__exit__.assert_called_once()
        mock_forward.assert_called_once_with(forward_only=True)

    def test_tp_logits_skip_gather_when_runtime_output_is_already_full(self):
        """Test that runtime-gathered text logits are not gathered a second time."""
        full_logits = torch.randn(1, 1, 128)

        with patch.object(compare.dist, "all_gather") as mock_all_gather:
            result = _maybe_gather_tensor_parallel_logits(full_logits, 128, 2, object())

        assert result is full_logits
        mock_all_gather.assert_not_called()

    def test_tp_logits_gather_vlm_shards(self):
        """Test that the existing sharded VLM path still gathers across TP ranks."""
        local_logits = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64)

        def mock_all_gather(outputs, tensor, group):
            assert group is tp_group
            outputs[0].copy_(tensor)
            outputs[1].copy_(tensor + 64)

        tp_group = object()
        with patch.object(compare.dist, "all_gather", side_effect=mock_all_gather) as gather:
            result = _maybe_gather_tensor_parallel_logits(local_logits, 128, 2, tp_group)

        assert result.shape == (1, 1, 128)
        assert torch.equal(result[..., :64], local_logits)
        assert torch.equal(result[..., 64:], local_logits + 64)
        gather.assert_called_once()

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

    def test_hf_model_is_released_before_megatron_model_load(self):
        """The normal comparison path must avoid co-resident full model weights."""
        events = []
        input_ids = MagicMock()
        input_ids.cuda.return_value = input_ids
        args = SimpleNamespace(
            hf_model_path="hf/model",
            trust_remote_code=False,
            hf_revision=None,
            image_path=None,
            prompt="Hello",
            tp=1,
            roundtrip_hf=False,
        )

        def load_megatron(_args):
            events.append("megatron_load")
            raise RuntimeError("stop after load-order check")

        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.cuda, "empty_cache"),
            patch.object(compare, "is_vision_language_model", return_value=False),
            patch.object(compare, "_load_hf_model", side_effect=lambda *_: events.append("hf_load") or object()),
            patch.object(compare, "_setup_tokenizer_and_processor", return_value=(MagicMock(), None)),
            patch.object(compare, "process_inputs", return_value=(input_ids, None, None, None)),
            patch.object(
                compare,
                "_run_hf_inference",
                side_effect=lambda *_: events.append("hf_run") or (MagicMock(), MagicMock(), None, None, None),
            ),
            patch.object(compare, "_load_megatron_model", side_effect=load_megatron),
            pytest.raises(RuntimeError, match="load-order check"),
        ):
            compare.compare_models_one_step(args)

        assert events == ["hf_load", "hf_run", "megatron_load"]

    def test_hf_broadcast_uses_model_output_vocab_size(self):
        """Test that non-rank-0 buffers use the HF logits size instead of tokenizer vocab size."""
        broadcast_shapes = []

        def mock_broadcast(tensor, _source_rank):
            broadcast_shapes.append(tuple(tensor.shape))
            if len(broadcast_shapes) == 1:
                tensor.fill_(163840)

        with (
            patch.object(torch.distributed, "broadcast", side_effect=mock_broadcast),
            patch.object(torch.distributed, "barrier"),
        ):
            hf_logits, hf_next_token = _broadcast_hf_results(None, None, torch.device("cpu"))

        assert hf_logits.shape == (163840,)
        assert hf_logits.dtype == torch.float32
        assert hf_next_token.shape == (1,)
        assert broadcast_shapes == [(1,), (1,), (163840,)]

    @pytest.mark.parametrize("flag", ["--trust_remote_code", "--trust-remote-code"])
    def test_trust_remote_code_accepts_underscore_and_hyphen_flags(self, flag):
        """Test that compare.py accepts both trust_remote_code flag spellings."""
        args = compare.build_parser().parse_args(
            [
                "--hf_model_path",
                "hf",
                "--prompt",
                "Hello",
                flag,
            ]
        )

        assert args.trust_remote_code is True

    def test_hf_revision_is_parsed_and_forwarded(self):
        """Test that an immutable revision reaches every HF loader via shared kwargs."""
        revision = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        args = compare.build_parser().parse_args(
            [
                "--hf_model_path",
                "org/model",
                "--prompt",
                "Hello",
                "--hf-revision",
                revision,
            ]
        )

        assert args.hf_revision == revision
        assert compare._hf_revision_kwargs(args.hf_revision) == {"revision": revision}
        assert compare._hf_revision_kwargs(None) == {}
