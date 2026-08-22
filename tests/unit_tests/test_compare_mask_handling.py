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

_mocked_modules = {_mod: sys.modules[_mod] if _mod in sys.modules else MagicMock() for _mod in _MODULES_TO_MOCK}
_compare_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "conversion", "compare_hf_and_megatron")

# Keep compare.py's heavy-dependency stubs local to this import. Leaving them in
# sys.modules makes later tests import MagicMock placeholders instead of the
# real Bridge modules.
sys.path.insert(0, _compare_dir)
try:
    with patch.dict(sys.modules, _mocked_modules):
        import compare  # noqa: E402
        from compare import (  # noqa: E402
            SingleBatchIterator,
            _broadcast_hf_results,
            _load_hf_reference_logits,
            _maybe_gather_tensor_parallel_logits,
            _run_hf_inference,
            _run_megatron_forward,
            inference_forward_step,
            vlm_forward_step,
        )
finally:
    sys.path.remove(_compare_dir)


@pytest.mark.unit
class TestCompareMaskHandling:
    """Tests for attention_mask handling in compare.py Megatron and HF paths."""

    def test_gemma3_config_is_detected_as_vision_language_model(self):
        """Structured vision and text configs identify Gemma 3 as a VLM."""
        config = SimpleNamespace(
            model_type="gemma3",
            architectures=["Gemma3ForConditionalGeneration"],
            text_config=object(),
            vision_config=object(),
        )

        with patch.object(compare.AutoConfig, "from_pretrained", return_value=config):
            assert compare.is_vision_language_model("google/gemma-3-4b-it") is True

    def test_vlm_inputs_preserve_image_token_types_and_tp_padding(self):
        """VLM preprocessing keeps Gemma image inputs without requiring a grid."""
        input_ids = torch.tensor([[1, 2, 3]])
        token_type_ids = torch.tensor([[0, 1, 0]])
        pixel_values = torch.randn(1, 3, 4, 4)
        processed = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
        }
        processor = MagicMock()
        processor.apply_chat_template.return_value = processed
        tokenizer = SimpleNamespace(pad_token_id=7)
        image = object()

        with patch.object(compare, "load_image", return_value=image):
            actual_input_ids, actual_pixels, image_grid_thw, actual_token_type_ids, actual_mm_token_type_ids = (
                compare.process_inputs(
                    tokenizer,
                    processor,
                    "/tmp/example.png",
                    "Describe this image.",
                    is_vl_model=True,
                    tp_size=2,
                )
            )

        torch.testing.assert_close(actual_input_ids, torch.tensor([[1, 2, 3, 7]]))
        assert actual_pixels is pixel_values
        assert image_grid_thw is None
        torch.testing.assert_close(actual_token_type_ids, torch.tensor([[0, 1, 0, 0]]))
        assert actual_mm_token_type_ids is None
        processor.apply_chat_template.assert_called_once_with(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def test_vlm_inputs_preserve_mm_token_types_and_tp_padding(self):
        """VLM preprocessing keeps M-RoPE token types under their processor key."""
        input_ids = torch.tensor([[1, 2, 3]])
        mm_token_type_ids = torch.tensor([[0, 1, 0]])
        pixel_values = torch.randn(3, 8)
        image_grid_thw = torch.tensor([[1, 2, 2]])
        processor = MagicMock()
        processor.apply_chat_template.return_value = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        tokenizer = SimpleNamespace(pad_token_id=7)

        with patch.object(compare, "load_image", return_value=object()):
            actual_input_ids, actual_pixels, actual_grid, token_type_ids, actual_mm_token_type_ids = (
                compare.process_inputs(
                    tokenizer,
                    processor,
                    "/tmp/example.png",
                    "Describe this image.",
                    is_vl_model=True,
                    tp_size=2,
                )
            )

        torch.testing.assert_close(actual_input_ids, torch.tensor([[1, 2, 3, 7]]))
        assert actual_pixels is pixel_values
        assert actual_grid is image_grid_thw
        assert token_type_ids is None
        torch.testing.assert_close(actual_mm_token_type_ids, torch.tensor([[0, 1, 0, 0]]))

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

    def test_vlm_forward_step_passes_mm_token_type_ids(self):
        """Megatron receives processor-produced M-RoPE token types unchanged."""
        mm_token_type_ids = torch.tensor([[0, 1, 0]])
        iterator = SingleBatchIterator(
            torch.tensor([[1, 2, 3]]),
            torch.arange(3).unsqueeze(0),
            None,
            mm_token_type_ids=mm_token_type_ids,
        )
        mock_model = MagicMock(return_value=torch.randn(1, 3, 100))

        vlm_forward_step(iterator, mock_model)

        assert mock_model.call_args.kwargs["mm_token_type_ids"] is mm_token_type_ids

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

    def test_hf_text_only_path_selects_composite_language_backbone(self):
        """Text-only comparisons bypass a composite model's media-required forward."""
        composite_model = MagicMock()
        language_model = torch.nn.Linear(3, 3)
        composite_model.language_model = language_model

        with patch.object(compare, "print_rank_0"):
            assert compare._get_hf_forward_model(composite_model, pixel_values=None) is language_model
            assert compare._get_hf_forward_model(composite_model, pixel_values=torch.ones(1)) is composite_model

    def test_hf_path_receives_multimodal_token_type_ids(self):
        """Gemma 3 token types reach HF so its image attention mask matches Megatron."""
        mock_hf_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 3, 100)
        mock_hf_model.return_value = mock_output
        input_ids = torch.tensor([[1, 2, 3]])
        token_type_ids = torch.tensor([[0, 1, 0]])
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "test"

        with (
            patch.object(compare, "_is_rank_0", return_value=True),
            patch.object(compare, "print_rank_0"),
        ):
            _run_hf_inference(
                mock_hf_model,
                input_ids,
                pixel_values=torch.randn(1, 3, 4, 4),
                image_grid_thw=None,
                tokenizer=mock_tokenizer,
                token_type_ids=token_type_ids,
            )

        assert mock_hf_model.call_args.kwargs["token_type_ids"] is token_type_ids

    def test_hf_path_receives_mm_token_type_ids(self):
        """Qwen M-RoPE token types reach HF under the exact processor key."""
        mock_hf_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 3, 100)
        mock_hf_model.return_value = mock_output
        input_ids = torch.tensor([[1, 2, 3]])
        mm_token_type_ids = torch.tensor([[0, 1, 0]])
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "test"

        with (
            patch.object(compare, "_is_rank_0", return_value=True),
            patch.object(compare, "print_rank_0"),
        ):
            _run_hf_inference(
                mock_hf_model,
                input_ids,
                pixel_values=torch.randn(3, 8),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                tokenizer=mock_tokenizer,
                mm_token_type_ids=mm_token_type_ids,
            )

        assert mock_hf_model.call_args.kwargs["mm_token_type_ids"] is mm_token_type_ids

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

    def test_memory_bounded_hf_reference_logits_validate_input_ids(self, tmp_path):
        """Test that saved HF logits are accepted only for the exact tokenized input."""
        path = tmp_path / "hf_logits.pt"
        input_ids = torch.tensor([[1, 2, 3]])
        logits = torch.tensor([[0.1, 0.7, 0.2]])
        torch.save({"input_ids": input_ids, "logits": logits}, path)
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "token"

        with patch.object(compare, "_is_rank_0", return_value=True), patch.object(compare, "print_rank_0"):
            loaded, next_token, *_ = _load_hf_reference_logits(path, input_ids, tokenizer)

        assert torch.equal(loaded, logits.reshape(-1))
        assert next_token.item() == 1
        with (
            patch.object(compare, "_is_rank_0", return_value=True),
            pytest.raises(ValueError, match="different input token IDs"),
        ):
            _load_hf_reference_logits(path, torch.tensor([[9]]), tokenizer)

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

    def test_hf_loader_uses_one_device_without_hf_tensor_parallelism(self):
        """Load the HF reference on one device without a Transformers TP plan."""
        args = compare.build_parser().parse_args(
            [
                "--hf_model_path",
                "org/model",
                "--prompt",
                "Hello",
            ]
        )
        loaded_model = MagicMock()
        model_class = MagicMock()
        model_class.__name__ = "MockModel"
        model_class.from_pretrained.return_value = loaded_model
        loaded_model.to.return_value = loaded_model
        loaded_model.eval.return_value = loaded_model

        with (
            patch.object(compare, "_is_rank_0", return_value=True),
            patch.object(compare, "get_model_class", return_value=model_class),
            patch.object(compare, "is_safe_repo", return_value=True),
            patch.object(compare, "print_rank_0"),
        ):
            result = compare._load_hf_model(args, is_vl_model=False)

        assert result is loaded_model
        model_class.from_pretrained.assert_called_once()
        load_kwargs = model_class.from_pretrained.call_args.kwargs
        assert "device_map" not in load_kwargs
        assert "tp_plan" not in load_kwargs
        assert "tp_size" not in load_kwargs
        loaded_model.to.assert_called_once_with("cuda")
