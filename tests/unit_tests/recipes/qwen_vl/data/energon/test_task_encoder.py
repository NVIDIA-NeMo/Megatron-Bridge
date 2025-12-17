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

import json
import pickle
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from megatron.bridge.recipes.qwen_vl.data.energon.task_encoder import (
    ChatMLSample,
    QwenVLTaskBatch,
    QwenVLTaskEncoder,
    QwenVLTaskSample,
    convert_to_qwenvl_content,
    cook_chatml_sample,
    find_pattern_indices,
    get_ltor_masks_and_position_ids,
    process_vision,
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

    def test_cook_chatml_sample(self):
        sample_dict = {
            "__key__": "test_key",
            "__restore_key__": "test_restore_key",
            "__subflavor__": {},
            "__subflavors__": {},
            "json": json.dumps([{"role": "user", "content": "hi"}]),
            "jpgs": pickle.dumps([np.zeros((10, 10, 3), dtype=np.uint8)]),
            "videos": pickle.dumps([]),
        }
        sample = cook_chatml_sample(sample_dict)
        self.assertIsInstance(sample, ChatMLSample)
        self.assertEqual(len(sample.imgs), 1)
        self.assertIsInstance(sample.imgs[0], Image.Image)


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
            max_padding_length=128,
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
            __key__="key",
            __restore_key__="restore_key",
            __subflavor__={},
            __subflavors__={},
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


if __name__ == "__main__":
    unittest.main()
