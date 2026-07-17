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

"""Unit tests for VLM generation helpers
in ``examples.conversion.vlm_generate_utils``.

Covers the ``ImportError`` fallback when ``qwen_vl_utils`` is unavailable
and the success paths with mocked processors and ``qwen_vl_utils`` helpers.
"""

import importlib.util
import os
import sys
from unittest import mock

import pytest
import torch


# Load examples/conversion/vlm_generate_utils.py directly from its file path.
# This avoids ambiguity when another `examples` package (e.g. from Megatron-LM)
# shadows the local one on sys.path.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VLM_GEN_UTILS_PATH = os.path.join(_REPO_ROOT, "examples", "conversion", "vlm_generate_utils.py")
_spec = importlib.util.spec_from_file_location("vlm_generate_utils_under_test", _VLM_GEN_UTILS_PATH)
vlm_generate_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vlm_generate_utils)


@pytest.mark.unit
class TestProcessMultiImageInputs:
    """Tests for ``process_multi_image_inputs``."""

    def test_raises_import_error_when_qwen_vl_utils_missing(self):
        """Without qwen_vl_utils, the function must raise ImportError early."""
        with mock.patch.object(vlm_generate_utils, "_HAS_QWEN_VL_UTILS", False):
            with pytest.raises(ImportError, match="qwen-vl-utils required"):
                vlm_generate_utils.process_multi_image_inputs(mock.MagicMock(), ["/a.png"], "describe")

    def test_processes_images_and_returns_tuple(self):
        """Multi-image path: load each image, run process_vision_info, then call processor."""
        rgb_a, rgb_b = mock.MagicMock(name="rgb_a"), mock.MagicMock(name="rgb_b")
        img_a, img_b = mock.MagicMock(name="img_a"), mock.MagicMock(name="img_b")
        img_a.convert.return_value = rgb_a
        img_b.convert.return_value = rgb_b
        load_image_mock = mock.MagicMock(side_effect=[img_a, img_b])

        proc = mock.MagicMock()
        proc.apply_chat_template.return_value = "TEMPLATED"
        proc_call_result = mock.MagicMock()
        proc_call_result.input_ids = torch.tensor([[1, 2, 3]])
        proc_call_result.get.side_effect = lambda key: {
            "pixel_values": "PIXELS",
            "image_grid_thw": "GRID",
        }.get(key)
        proc.return_value = proc_call_result

        process_vision_info_mock = mock.MagicMock(return_value=("IMG_INPUTS", "VID_INPUTS"))

        with (
            mock.patch.object(vlm_generate_utils, "_HAS_QWEN_VL_UTILS", True),
            mock.patch.object(vlm_generate_utils, "load_image", load_image_mock),
            mock.patch.object(vlm_generate_utils, "process_vision_info", process_vision_info_mock, create=True),
        ):
            input_ids, pixel_values, image_grid_thw = vlm_generate_utils.process_multi_image_inputs(
                proc, ["/a.png", "/b.png"], "describe these"
            )

        assert load_image_mock.call_count == 2
        load_image_mock.assert_any_call("/a.png")
        load_image_mock.assert_any_call("/b.png")
        img_a.convert.assert_called_once_with("RGB")
        img_b.convert.assert_called_once_with("RGB")

        (sent_messages,), _ = process_vision_info_mock.call_args
        assert sent_messages[0]["role"] == "user"
        contents = sent_messages[0]["content"]
        assert {"type": "image", "image": rgb_a} in contents
        assert {"type": "image", "image": rgb_b} in contents
        assert {"type": "text", "text": "describe these"} in contents

        proc.apply_chat_template.assert_called_once_with(sent_messages, tokenize=False, add_generation_prompt=True)
        proc.assert_called_once_with(
            text=["TEMPLATED"],
            images="IMG_INPUTS",
            videos="VID_INPUTS",
            padding=True,
            return_tensors="pt",
        )

        torch.testing.assert_close(input_ids, torch.tensor([[1, 2, 3]]))
        assert pixel_values == "PIXELS"
        assert image_grid_thw == "GRID"

    def test_returns_none_when_processor_omits_optional_fields(self):
        """When processor output lacks pixel_values / image_grid_thw, those fields must be None."""
        img = mock.MagicMock()
        img.convert.return_value = mock.MagicMock(name="rgb")
        load_image_mock = mock.MagicMock(return_value=img)

        proc = mock.MagicMock()
        proc.apply_chat_template.return_value = "TEMPLATED"
        proc_call_result = mock.MagicMock()
        proc_call_result.input_ids = torch.tensor([[5]])
        proc_call_result.get.return_value = None
        proc.return_value = proc_call_result

        with (
            mock.patch.object(vlm_generate_utils, "_HAS_QWEN_VL_UTILS", True),
            mock.patch.object(vlm_generate_utils, "load_image", load_image_mock),
            mock.patch.object(vlm_generate_utils, "process_vision_info", return_value=([], []), create=True),
        ):
            ids, px, grid = vlm_generate_utils.process_multi_image_inputs(proc, ["/x.png"], "p")

        assert px is None
        assert grid is None
        torch.testing.assert_close(ids, torch.tensor([[5]]))


@pytest.mark.unit
class TestProcessImageInputs:
    """Tests for ``process_image_inputs``."""

    def test_base_processor_without_chat_template_uses_raw_image_prompt(self):
        """Base VLMs without a chat template should receive an image-token prompt."""
        image = mock.MagicMock(name="image")
        rgb_image = mock.MagicMock(name="rgb_image")
        image.convert.return_value = rgb_image

        inputs = mock.MagicMock()
        inputs.input_ids = torch.tensor([[10, 20]])
        inputs.get.side_effect = lambda key: {
            "pixel_values": "PIXELS",
            "image_sizes": "IMAGE_SIZES",
        }.get(key)

        proc = mock.MagicMock()
        proc.chat_template = None
        proc.image_token = "[IMG]"
        proc.return_value = inputs

        with (
            mock.patch.object(vlm_generate_utils, "_HAS_QWEN_VL_UTILS", False),
            mock.patch.object(vlm_generate_utils, "load_image", return_value=image) as load_image_mock,
        ):
            result = vlm_generate_utils.process_image_inputs(proc, "/image.png", "describe", is_mistral3=True)

        load_image_mock.assert_called_once_with("/image.png")
        image.convert.assert_called_once_with("RGB")
        proc.assert_called_once_with(
            text=["[IMG]\ndescribe"],
            images=[rgb_image],
            padding=True,
            return_tensors="pt",
        )
        assert len(result) == 6
        input_ids, pixel_values, image_grid_thw, image_sizes, mm_token_type_ids, image_position_ids = result
        torch.testing.assert_close(input_ids, torch.tensor([[10, 20]]))
        assert pixel_values == "PIXELS"
        assert image_grid_thw is None
        assert image_sizes == "IMAGE_SIZES"
        assert mm_token_type_ids is None
        assert image_position_ids is None

    def test_base_processor_preserves_existing_image_token(self):
        """Do not inject a second image token when the prompt already contains one."""
        image = mock.MagicMock()
        image.convert.return_value = "RGB"
        inputs = mock.MagicMock()
        inputs.input_ids = torch.tensor([[1]])
        inputs.get.return_value = None

        proc = mock.MagicMock()
        proc.chat_template = None
        proc.image_token = "[IMG]"
        proc.return_value = inputs

        with mock.patch.object(vlm_generate_utils, "load_image", return_value=image):
            vlm_generate_utils.process_image_inputs(proc, "/image.png", "[IMG]\ncaption", is_mistral3=True)

        proc.assert_called_once_with(
            text=["[IMG]\ncaption"],
            images=["RGB"],
            padding=True,
            return_tensors="pt",
        )

    def test_unrecognized_processor_without_chat_template_fails_clearly(self):
        """Do not apply the Ministral raw-prompt convention to other VLMs."""
        proc = mock.MagicMock()
        proc.chat_template = None

        with pytest.raises(ValueError, match="no model-specific raw-prompt fallback"):
            vlm_generate_utils.process_image_inputs(proc, "/image.png", "describe")

    def test_kimi_image_path_returns_full_six_field_contract(self):
        """Kimi image preprocessing must match the VLM generation unpack contract."""
        inputs = mock.MagicMock()
        inputs.input_ids = torch.tensor([[10, 99, 20]])
        inputs.pixel_values = "PIXELS"
        inputs.grid_thws = torch.tensor([[1, 4, 4]])

        proc = mock.MagicMock(return_value=inputs)

        with mock.patch.object(vlm_generate_utils, "load_image", return_value=mock.MagicMock(name="image")):
            result = vlm_generate_utils.process_image_inputs(
                proc,
                "/image.png",
                "describe",
                is_kimi=True,
                image_token_id=99,
            )

        assert len(result) == 6
        input_ids, pixel_values, image_grid_thw, image_sizes, mm_token_type_ids, image_position_ids = result
        expected_input_ids = torch.tensor([[10, 99, 99, 99, 99, 20]])
        torch.testing.assert_close(input_ids, expected_input_ids)
        assert pixel_values == "PIXELS"
        torch.testing.assert_close(image_grid_thw, inputs.grid_thws)
        assert image_sizes is None
        assert mm_token_type_ids is None
        assert image_position_ids is None


@pytest.mark.unit
class TestProcessVideoInputs:
    """Tests for ``process_video_inputs``."""

    def test_raises_import_error_when_qwen_vl_utils_missing(self):
        """Without qwen_vl_utils, the function must raise ImportError early."""
        with mock.patch.object(vlm_generate_utils, "_HAS_QWEN_VL_UTILS", False):
            with pytest.raises(ImportError, match="qwen-vl-utils required"):
                vlm_generate_utils.process_video_inputs(mock.MagicMock(), "/v.mp4", "describe")

    def test_video_path_pre_expands_video_tokens(self):
        """Video path: fetch_video → processor → video_processor → pre-expand <|video_pad|>."""
        # Single <|video_pad|> placeholder (id 151656). With grid_thw=[1, 4, 4] and
        # spatial_merge_size=2, pre_expand_vision_tokens expands it to 1*(4//2)*(4//2)=4 tokens.
        text_input_ids = torch.tensor([[100, 151656, 200]])
        video_grid_thw = torch.tensor([[1, 4, 4]])

        proc = mock.MagicMock()
        proc.apply_chat_template.return_value = "VID_TEMPLATED"
        proc.return_value = {"input_ids": text_input_ids}
        proc.tokenizer.convert_tokens_to_ids.return_value = 151656
        proc.video_processor.return_value = {
            "video_grid_thw": video_grid_thw,
            "pixel_values_videos": "VID_PIXELS",
        }

        fake_qvu = mock.MagicMock()
        fake_qvu.fetch_video = mock.MagicMock(return_value="DECODED_FRAMES")

        with (
            mock.patch.object(vlm_generate_utils, "_HAS_QWEN_VL_UTILS", True),
            mock.patch.dict(sys.modules, {"qwen_vl_utils": fake_qvu}),
        ):
            input_ids, pixel_values_videos, grid = vlm_generate_utils.process_video_inputs(
                proc, "/clip.mp4", "what is happening", fps=3.0
            )

        fake_qvu.fetch_video.assert_called_once_with({"video": "/clip.mp4", "fps": 3.0})
        proc.apply_chat_template.assert_called_once()
        proc.assert_called_once_with(text=["VID_TEMPLATED"], padding=True, return_tensors="pt")
        proc.video_processor.assert_called_once_with(
            videos=["DECODED_FRAMES"], return_tensors="pt", do_sample_frames=False
        )

        expected = torch.tensor([[100, 151656, 151656, 151656, 151656, 200]])
        torch.testing.assert_close(input_ids, expected)
        assert pixel_values_videos == "VID_PIXELS"
        torch.testing.assert_close(grid, video_grid_thw)

    def test_default_fps_is_2(self):
        """Default fps should be 2.0 when not specified."""
        proc = mock.MagicMock()
        proc.apply_chat_template.return_value = "T"
        proc.return_value = {"input_ids": torch.tensor([[1]])}
        proc.video_processor.return_value = {
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "pixel_values_videos": "P",
        }

        fake_qvu = mock.MagicMock()
        fake_qvu.fetch_video = mock.MagicMock(return_value="F")

        with (
            mock.patch.object(vlm_generate_utils, "_HAS_QWEN_VL_UTILS", True),
            mock.patch.dict(sys.modules, {"qwen_vl_utils": fake_qvu}),
        ):
            vlm_generate_utils.process_video_inputs(proc, "/v.mp4", "p")

        fake_qvu.fetch_video.assert_called_once_with({"video": "/v.mp4", "fps": 2.0})

    def test_video_message_structure(self):
        """The user message must contain a video placeholder followed by the prompt text."""
        proc = mock.MagicMock()
        proc.apply_chat_template.return_value = "T"
        proc.return_value = {"input_ids": torch.tensor([[1]])}
        proc.video_processor.return_value = {
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "pixel_values_videos": "P",
        }

        fake_qvu = mock.MagicMock()
        fake_qvu.fetch_video = mock.MagicMock(return_value="F")

        with (
            mock.patch.object(vlm_generate_utils, "_HAS_QWEN_VL_UTILS", True),
            mock.patch.dict(sys.modules, {"qwen_vl_utils": fake_qvu}),
        ):
            vlm_generate_utils.process_video_inputs(proc, "/v.mp4", "narrate this")

        (sent_messages,), kw = proc.apply_chat_template.call_args
        assert sent_messages[0]["role"] == "user"
        content = sent_messages[0]["content"]
        assert {"type": "video"} in content
        assert {"type": "text", "text": "narrate this"} in content
        assert kw == {"tokenize": False, "add_generation_prompt": True}
