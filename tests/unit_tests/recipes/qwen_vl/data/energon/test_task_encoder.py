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

import io
import json
import pickle
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from megatron.energon import SkipSample
from PIL import Image

from megatron.bridge.data.energon.metadata import batch_metadata_kwargs, sample_metadata_kwargs
from megatron.bridge.recipes.qwen_vl.data.energon.task_encoder import (
    ChatMLSample,
    QwenVLTaskBatch,
    QwenVLTaskEncoder,
    QwenVLTaskSample,
    _resolve_hf_mm_token_ids,
    convert_to_qwenvl_content,
    find_pattern_indices,
    get_ltor_masks_and_position_ids,
    process_vision,
    videohandler,
)


@pytest.fixture(autouse=True)
def cleanup_local_folder():
    pass


class TestHelperFunctions(unittest.TestCase):
    def test_find_pattern_indices(self):
        seq = np.array([1, 2, 3, 4, 5])
        pattern = np.array([3, 4])
        start, end = find_pattern_indices(seq, pattern)
        self.assertEqual(start, 2)
        self.assertEqual(end, 4)

        # Test not found
        start, end = find_pattern_indices(seq, np.array([6]))
        self.assertEqual(start, -1)
        self.assertEqual(end, -1)

        # Test empty pattern
        start, end = find_pattern_indices(seq, np.array([]))
        self.assertEqual(start, -1)
        self.assertEqual(end, -1)

    def test_convert_to_qwenvl_content(self):
        text = "Hello <image> world <video>!"
        content = convert_to_qwenvl_content(text)
        # Expected parsing behavior
        self.assertTrue(any(c["type"] == "image" for c in content))
        self.assertTrue(any(c["type"] == "video" for c in content))
        self.assertEqual(content[0]["text"], "Hello")
        self.assertEqual(content[1]["image"], "0")
        self.assertEqual(content[2]["text"], "world")
        self.assertEqual(content[3]["video"], "0")
        self.assertEqual(content[4]["text"], "!")

    def test_get_ltor_masks_and_position_ids(self):
        data = torch.tensor([[1, 2, 3]], dtype=torch.long)
        att_mask, loss_mask, pos_ids = get_ltor_masks_and_position_ids(
            data,
            eod_token=99,
            eod_mask_loss=False,
            reset_attention_mask=False,
            reset_position_ids=False,
        )
        self.assertEqual(att_mask.shape, (1, 1, 3, 3))
        self.assertEqual(loss_mask.shape, (1, 3))
        self.assertEqual(pos_ids.shape, (1, 3))
        self.assertTrue(torch.all(loss_mask == 1.0))


class TestResolveHfMmTokenIds(unittest.TestCase):
    def test_resolves_from_tokenizer_attributes(self):
        tokenizer = MagicMock()
        tokenizer.image_token_id = 100
        tokenizer.video_token_id = 200
        image_id, video_id = _resolve_hf_mm_token_ids(tokenizer)
        self.assertEqual(image_id, 100)
        self.assertEqual(video_id, 200)

    def test_falls_back_to_convert_tokens_to_ids(self):
        tokenizer = MagicMock()
        tokenizer.image_token_id = None
        tokenizer.video_token_id = None
        tokenizer.convert_tokens_to_ids.side_effect = lambda x: {"<image>": 300, "<video>": 400}[x]
        image_id, video_id = _resolve_hf_mm_token_ids(tokenizer)
        self.assertEqual(image_id, 300)
        self.assertEqual(video_id, 400)

    def test_returns_defaults_when_all_fail(self):
        tokenizer = MagicMock()
        tokenizer.image_token_id = None
        tokenizer.video_token_id = None
        tokenizer.convert_tokens_to_ids.side_effect = Exception("not found")
        image_id, video_id = _resolve_hf_mm_token_ids(tokenizer)
        self.assertEqual(image_id, 151655)
        self.assertEqual(video_id, 151656)


class TestVideoHandler(unittest.TestCase):
    def setUp(self):
        self.handler = videohandler("pilrgb")

    def _make_jpeg_bytes(self, color="red"):
        img = Image.new("RGB", (4, 4), color=color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_returns_none_for_non_matching_extension(self):
        result = self.handler("sample.txt", b"data")
        self.assertIsNone(result)

    def test_decodes_jpgs(self):
        images_bytes = [self._make_jpeg_bytes() for _ in range(2)]
        data = pickle.dumps(images_bytes)
        result = self.handler("sample.jpgs", data)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_decodes_mp4s(self):
        frames = [self._make_jpeg_bytes("blue") for _ in range(3)]
        videos = [frames]  # one video with 3 frames
        data = pickle.dumps(videos)
        result = self.handler("sample.mp4s", data)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)


class TestQwenVLTaskEncoder(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        # Setup attributes for _resolve_hf_mm_token_ids
        self.tokenizer.image_token_id = 151655
        self.tokenizer.video_token_id = 151656
        self.tokenizer.convert_tokens_to_ids.side_effect = lambda x: {
            "<image>": 151655,
            "<video>": 151656,
        }.get(x, 10)

        self.image_processor = MagicMock()

        self.encoder = QwenVLTaskEncoder(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            max_padding_length=512,
            patch_size=14,
            spatial_merge_size=2,
        )

    def test_process_vision(self):
        # Mock processor behavior
        self.image_processor.return_value = {
            "image_grid_thw": torch.tensor([[1, 28, 28]]),
            "video_grid_thw": None,
        }
        res = process_vision(self.image_processor, images=[1], videos=None)
        self.assertIn("image_grid_thw", res)
        self.assertIn("video_grid_thw", res)

    def test_encode_sample(self):
        # Mock process_vision return via image_processor
        def processor_side_effect(images=None, videos=None, **kwargs):
            res = {}
            if images:
                res["image_grid_thw"] = np.array([[1, 28, 28]])  # 1 tile, 28x28
                res["pixel_values"] = torch.randn(1, 3, 28, 28)
            if videos:
                res["video_grid_thw"] = np.array([[1, 28, 28]])
                res["pixel_values_videos"] = torch.randn(1, 3, 28, 28)
            return res

        self.image_processor.side_effect = processor_side_effect

        # Mock apply_chat_template
        # The encoder expects numpy array return from apply_chat_template
        # It creates input_ids with placeholders for images/videos
        # <image> is 151655
        self.tokenizer.apply_chat_template.return_value = [
            np.array([10, 11, 151655, 12, 13])  # dummy tokens with image placeholder
        ]

        # Mock encode for finding answer
        self.tokenizer.encode.side_effect = lambda x, **kwargs: [12, 13] if x == "Nice" else [999]

        sample = ChatMLSample(
            **sample_metadata_kwargs(key="key", restore_key="restore_key", subflavors={}),
            imgs=[MagicMock(spec=Image.Image)],
            videos=[],
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Look <image>"},
                    {"role": "assistant", "content": "Nice"},
                ]
            ),
        )

        encoded = self.encoder.encode_sample(sample)

        self.assertIsInstance(encoded, QwenVLTaskSample)
        self.assertTrue(torch.is_tensor(encoded.text))
        self.assertTrue(torch.is_tensor(encoded.target))
        # Check if image mask is set correctly around the placeholder
        # The logic in encode_sample expands the placeholder based on grid size
        # 28x28 with merge_size=2 means (28/14)*(28/14) = 4 patches? No.
        # merge_size=2.
        # Logic: size = image_thw_grids[idx].prod() // merge_length
        # 1*28*28 = 784. merge_length = 2**2 = 4. size = 196.
        # So the single token 151655 should be replaced by 196 tokens.

        # Verify length expansion
        original_len = 5
        expanded_len = original_len - 1 + 196
        self.assertEqual(len(encoded.text), expanded_len)

    def test_encode_sample_from_value_format(self):
        """Test encode_sample with 'from'/'value' conversation format."""

        def processor_side_effect(images=None, videos=None, **kwargs):
            res = {}
            if images:
                res["image_grid_thw"] = np.array([[1, 28, 28]])
                res["pixel_values"] = torch.randn(1, 3, 28, 28)
            if videos:
                res["video_grid_thw"] = np.array([[1, 28, 28]])
                res["pixel_values_videos"] = torch.randn(1, 3, 28, 28)
            return res

        self.image_processor.side_effect = processor_side_effect

        self.tokenizer.apply_chat_template.return_value = [np.array([10, 11, 151655, 12, 13])]
        self.tokenizer.encode.side_effect = lambda x, **kwargs: [12, 13] if x == "Nice" else [999]

        sample = ChatMLSample(
            **sample_metadata_kwargs(key="key", restore_key="restore_key", subflavors={}),
            imgs=[MagicMock(spec=Image.Image)],
            videos=[],
            conversation=json.dumps(
                [
                    {"from": "human", "value": "Look <image>"},
                    {"from": "gpt", "value": "Nice"},
                ]
            ),
        )

        encoded = self.encoder.encode_sample(sample)
        self.assertIsInstance(encoded, QwenVLTaskSample)
        self.assertTrue(torch.is_tensor(encoded.text))
        self.assertTrue(torch.is_tensor(encoded.target))
        # Same expansion as role/content format: 5 - 1 + 196 = 200
        self.assertEqual(len(encoded.text), 200)

    def test_encode_sample_text_only(self):
        """Test encode_sample with no images or videos."""
        self.tokenizer.apply_chat_template.return_value = [np.array([10, 11, 12, 13])]
        self.tokenizer.encode.side_effect = lambda x, **kwargs: [12, 13] if x == "Hello" else [999]

        sample = ChatMLSample(
            **sample_metadata_kwargs(key="key", restore_key="restore_key", subflavors={}),
            imgs=None,
            videos=None,
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            ),
        )

        encoded = self.encoder.encode_sample(sample)
        self.assertIsInstance(encoded, QwenVLTaskSample)
        self.assertEqual(len(encoded.text), 4)  # no expansion
        self.assertEqual(len(encoded.imgs), 0)
        self.assertEqual(len(encoded.videos), 0)

    def test_batch(self):
        # Create dummy encoded samples
        s1 = QwenVLTaskSample(
            __key__="k1",
            __subflavors__={},
            imgs=torch.randn(1, 3, 14, 14),
            videos=torch.tensor([]),
            image_thw_grids=[torch.tensor([1, 14, 14])],
            video_thw_grids=[],
            image_input_mask=torch.tensor([True] * 5),
            video_input_mask=torch.tensor([False] * 5),
            text=torch.tensor([1, 2, 3, 4, 5]),
            target=torch.tensor([1, 2, 3, 4, 5]),
        )
        s2 = QwenVLTaskSample(
            __key__="k2",
            __subflavors__={},
            imgs=torch.tensor([]),
            videos=torch.tensor([]),
            image_thw_grids=[],
            video_thw_grids=[],
            image_input_mask=torch.tensor([False] * 3),
            video_input_mask=torch.tensor([False] * 3),
            text=torch.tensor([1, 2, 3]),
            target=torch.tensor([1, 2, 3]),
        )

        batch = self.encoder.batch([s1, s2])
        self.assertIsInstance(batch, QwenVLTaskBatch)
        self.assertEqual(batch.input_ids.shape, (2, 5))  # padded to max length
        self.assertEqual(batch.labels.shape, (2, 5))

    def test_encode_batch(self):
        # Create a dummy batch
        batch = QwenVLTaskBatch(
            **batch_metadata_kwargs(keys=["k1"]),
            __keys__=["k1"],
            __subflavors__=[{}],
            pixel_values=torch.randn(1, 3, 14, 14),
            pixel_values_videos=None,
            image_grid_thw=torch.tensor([[1, 14, 14]]),
            video_grid_thw=None,
            image_input_mask=torch.randn(1, 5),
            video_input_mask=torch.randn(1, 5),
            input_ids=torch.randn(1, 5),
            attention_mask=torch.randn(1, 1, 5, 5),
            position_ids=torch.randn(1, 5),
            labels=torch.randn(1, 5),
            loss_mask=torch.randn(1, 5),
        )

        encoded_dict = self.encoder.encode_batch(batch)
        self.assertIsInstance(encoded_dict, dict)
        self.assertIn("visual_inputs", encoded_dict)
        self.assertIn("input_ids", encoded_dict)
        # Ensure __subflavors__ is removed
        self.assertNotIn("__subflavors__", encoded_dict)


class TestQwenVLTaskEncoderLimits(unittest.TestCase):
    """Tests for the per-sample budget limits added to QwenVLTaskEncoder."""

    def setUp(self):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        self.tokenizer.image_token_id = 151655
        self.tokenizer.video_token_id = 151656
        self.tokenizer.convert_tokens_to_ids.side_effect = lambda x: {
            "<image>": 151655,
            "<video>": 151656,
        }.get(x, 10)
        self.image_processor = MagicMock()

    def _make_encoder(self, **kwargs):
        defaults = dict(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            max_padding_length=512,
            patch_size=14,
            spatial_merge_size=2,
        )
        defaults.update(kwargs)
        return QwenVLTaskEncoder(**defaults)

    def _make_sample(self, *, n_images=0, n_videos=0, frames_per_video=0, conversation_text="Look <image>"):
        imgs = [Image.new("RGB", (4, 4), color="red") for _ in range(n_images)] or None
        if n_videos:
            videos = [
                [Image.new("RGB", (4, 4), color="blue") for _ in range(frames_per_video)] for _ in range(n_videos)
            ]
        else:
            videos = None
        return ChatMLSample(
            **sample_metadata_kwargs(key="key", restore_key="restore_key", subflavors={}),
            imgs=imgs,
            videos=videos,
            conversation=json.dumps(
                [
                    {"role": "user", "content": conversation_text},
                    {"role": "assistant", "content": "Nice"},
                ]
            ),
        )

    def test_default_limits_set_on_init(self):
        enc = self._make_encoder()
        self.assertEqual(enc.max_num_images, 10)
        self.assertEqual(enc.max_num_frames, 60)
        self.assertEqual(enc.max_visual_tokens, 16384)

    def test_init_accepts_none_to_disable_limits(self):
        enc = self._make_encoder(max_num_images=None, max_num_frames=None, max_visual_tokens=None)
        self.assertIsNone(enc.max_num_images)
        self.assertIsNone(enc.max_num_frames)
        self.assertIsNone(enc.max_visual_tokens)

    def test_max_num_images_skip_when_exceeded(self):
        # Configure the processor so that, IF we got to it, encoding would succeed.
        # The point of this test is that we should *not* get to it.
        self.image_processor.side_effect = AssertionError("processor must not be called when sample is skipped")

        enc = self._make_encoder(max_num_images=2)
        sample = self._make_sample(n_images=3)
        with self.assertRaises(SkipSample):
            enc.encode_sample(sample)

    def test_max_num_images_none_disables_check(self):
        # Even with many images, no SkipSample should be raised here — we expect the
        # call to fail later inside processing instead, which is fine for this test.
        enc = self._make_encoder(max_num_images=None, max_visual_tokens=None)
        sample = self._make_sample(n_images=50)
        # A non-SkipSample error is acceptable; we only assert SkipSample is NOT raised.
        with self.assertRaises(Exception) as cm:
            enc.encode_sample(sample)
        self.assertNotIsInstance(cm.exception, SkipSample)

    def test_max_num_frames_truncates_in_place(self):
        # process_vision routes videos through processor.video_processor, so we
        # capture the videos arg there.
        captured = {}

        def video_processor_side_effect(videos=None, **kwargs):
            captured["videos"] = videos
            return {
                "video_grid_thw": torch.tensor([[1, 14, 14]]),
                "pixel_values_videos": torch.randn(1, 3, 14, 14),
            }

        self.image_processor.video_processor.side_effect = video_processor_side_effect
        self.tokenizer.apply_chat_template.return_value = [np.array([10, 11, 151656, 12, 13])]
        self.tokenizer.encode.side_effect = lambda x, **kwargs: [12, 13] if x == "Nice" else [999]

        enc = self._make_encoder(max_num_frames=4, max_visual_tokens=None)
        sample = self._make_sample(n_videos=1, frames_per_video=10, conversation_text="Watch <video>")
        enc.encode_sample(sample)

        # videos arg passed to processor should have a single clip of 4 frames.
        self.assertIsNotNone(captured["videos"])
        self.assertEqual(len(captured["videos"]), 1)
        self.assertEqual(len(captured["videos"][0]), 4)

    def test_max_num_frames_keeps_short_videos_intact(self):
        captured = {}

        def video_processor_side_effect(videos=None, **kwargs):
            captured["videos"] = videos
            return {
                "video_grid_thw": torch.tensor([[1, 14, 14]]),
                "pixel_values_videos": torch.randn(1, 3, 14, 14),
            }

        self.image_processor.video_processor.side_effect = video_processor_side_effect
        self.tokenizer.apply_chat_template.return_value = [np.array([10, 11, 151656, 12, 13])]
        self.tokenizer.encode.side_effect = lambda x, **kwargs: [12, 13] if x == "Nice" else [999]

        enc = self._make_encoder(max_num_frames=10, max_visual_tokens=None)
        sample = self._make_sample(n_videos=1, frames_per_video=3, conversation_text="Watch <video>")
        enc.encode_sample(sample)

        self.assertEqual(len(captured["videos"][0]), 3)

    def test_max_visual_tokens_skip_when_exceeded(self):
        # 1 image of grid (1, 28, 28); merge_length = 2**2 = 4 -> 1*28*28/4 = 196 tokens.
        def processor_side_effect(images=None, videos=None, **kwargs):
            return {
                "image_grid_thw": torch.tensor([[1, 28, 28]]),
                "pixel_values": torch.randn(1, 3, 28, 28),
            }

        self.image_processor.side_effect = processor_side_effect

        enc = self._make_encoder(max_visual_tokens=100)  # 196 > 100 -> skip
        sample = self._make_sample(n_images=1)
        with self.assertRaises(SkipSample):
            enc.encode_sample(sample)

    def test_max_visual_tokens_none_disables_check(self):
        # 196 visual tokens — would trigger the limit at 100, but None disables it.
        def processor_side_effect(images=None, videos=None, **kwargs):
            return {
                "image_grid_thw": torch.tensor([[1, 28, 28]]),
                "pixel_values": torch.randn(1, 3, 28, 28),
            }

        self.image_processor.side_effect = processor_side_effect
        self.tokenizer.apply_chat_template.return_value = [np.array([10, 11, 151655, 12, 13])]
        self.tokenizer.encode.side_effect = lambda x, **kwargs: [12, 13] if x == "Nice" else [999]

        enc = self._make_encoder(max_visual_tokens=None)
        sample = self._make_sample(n_images=1)
        encoded = enc.encode_sample(sample)
        self.assertIsInstance(encoded, QwenVLTaskSample)

    def test_visual_tokens_exceeding_seq_len_raises_skip(self):
        # The post-expansion length depends on visual tokens, so a small seq_len
        # paired with many visual tokens triggers the SkipSample branch.
        def processor_side_effect(images=None, videos=None, **kwargs):
            return {
                "image_grid_thw": torch.tensor([[1, 28, 28]]),  # 196 visual tokens
                "pixel_values": torch.randn(1, 3, 28, 28),
            }

        self.image_processor.side_effect = processor_side_effect
        self.tokenizer.apply_chat_template.return_value = [np.array([10, 11, 151655, 12, 13])]
        self.tokenizer.encode.side_effect = lambda x, **kwargs: [12, 13] if x == "Nice" else [999]

        # max_visual_tokens=None to bypass that earlier guard, seq_len small so the
        # later branch fires.
        enc = self._make_encoder(max_padding_length=50, max_visual_tokens=None)
        sample = self._make_sample(n_images=1)
        with self.assertRaises(SkipSample):
            enc.encode_sample(sample)


class TestProcessVisionVideoBranch(unittest.TestCase):
    """Verify that process_vision routes videos through processor.video_processor with do_sample_frames=False."""

    def test_video_processor_called_with_do_sample_frames_false(self):
        processor = MagicMock()
        processor.video_processor = MagicMock(
            return_value={
                "video_grid_thw": torch.tensor([[1, 14, 14]]),
                "pixel_values_videos": torch.randn(1, 3, 14, 14),
            }
        )

        videos = [[Image.new("RGB", (4, 4), color="blue") for _ in range(3)]]
        res = process_vision(processor, images=None, videos=videos)

        # Top-level processor must NOT be called for videos in this path.
        processor.assert_not_called()
        # video_processor must be called with do_sample_frames=False.
        processor.video_processor.assert_called_once()
        kwargs = processor.video_processor.call_args.kwargs
        self.assertIs(kwargs.get("do_sample_frames"), False)
        self.assertEqual(kwargs.get("return_tensors"), "pt")
        self.assertIs(kwargs.get("videos"), videos)

        self.assertIn("video_grid_thw", res)
        self.assertIsNotNone(res["video_grid_thw"])


if __name__ == "__main__":
    unittest.main()
