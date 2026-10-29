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

import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory

from megatron.bridge.data.energon.task_encoder_utils import (
    IGNORE_INDEX,
    ChatMLWebdataset,
    _images_to_pil,
    _tensor_to_pil,
    _videos_to_pil,
    cook_chatml_sample,
    extract_chatml_tools,
    find_pattern_indices,
    get_ltor_masks_and_position_ids,
)


class TestIgnoreIndex(unittest.TestCase):
    def test_value(self):
        self.assertEqual(IGNORE_INDEX, -100)


class TestFindPatternIndices(unittest.TestCase):
    def test_basic_match(self):
        seq = np.array([1, 2, 3, 4, 5])
        start, end = find_pattern_indices(seq, [3, 4])
        self.assertEqual(start, 2)
        self.assertEqual(end, 4)

    def test_not_found(self):
        seq = np.array([1, 2, 3])
        start, end = find_pattern_indices(seq, [6])
        self.assertEqual(start, -1)
        self.assertEqual(end, -1)

    def test_empty_pattern(self):
        seq = np.array([1, 2, 3])
        start, end = find_pattern_indices(seq, [])
        self.assertEqual(start, -1)
        self.assertEqual(end, -1)

    def test_start_offset(self):
        seq = np.array([1, 2, 1, 2, 3])
        start, end = find_pattern_indices(seq, [1, 2], start=2)
        self.assertEqual(start, 2)
        self.assertEqual(end, 4)

    def test_list_input(self):
        start, end = find_pattern_indices([10, 20, 30], [20, 30])
        self.assertEqual(start, 1)
        self.assertEqual(end, 3)


class TestGetLtorMasksAndPositionIds(unittest.TestCase):
    def test_basic(self):
        data = torch.tensor([[1, 2, 3]], dtype=torch.long)
        att_mask, loss_mask, pos_ids = get_ltor_masks_and_position_ids(
            data, eod_token=99, eod_mask_loss=False, reset_attention_mask=False, reset_position_ids=False
        )
        self.assertEqual(att_mask.shape, (1, 1, 3, 3))
        self.assertEqual(loss_mask.shape, (1, 3))
        self.assertEqual(pos_ids.shape, (1, 3))
        self.assertTrue(torch.all(loss_mask == 1.0))
        self.assertTrue(torch.equal(pos_ids[0], torch.tensor([0, 1, 2])))

    def test_eod_mask_loss(self):
        data = torch.tensor([[1, 99, 3]], dtype=torch.long)
        _, loss_mask, _ = get_ltor_masks_and_position_ids(
            data, eod_token=99, eod_mask_loss=True, reset_attention_mask=False, reset_position_ids=False
        )
        self.assertEqual(loss_mask[0, 1].item(), 0.0)
        self.assertEqual(loss_mask[0, 0].item(), 1.0)

    def test_no_attention_mask(self):
        data = torch.tensor([[1, 2]], dtype=torch.long)
        att_mask, _, _ = get_ltor_masks_and_position_ids(
            data,
            eod_token=99,
            eod_mask_loss=False,
            reset_attention_mask=False,
            reset_position_ids=False,
            compute_attention_mask=False,
        )
        self.assertIsNone(att_mask)


class TestTensorToPil(unittest.TestCase):
    def test_conversion(self):
        t = torch.rand(3, 4, 4)
        pil_img = _tensor_to_pil(t)
        self.assertEqual(pil_img.size, (4, 4))


class TestImagesToPil(unittest.TestCase):
    def test_single_3d_tensor(self):
        t = torch.rand(3, 4, 4)
        result = _images_to_pil(t)
        self.assertEqual(len(result), 1)

    def test_4d_tensor(self):
        t = torch.rand(2, 3, 4, 4)
        result = _images_to_pil(t)
        self.assertEqual(len(result), 2)

    def test_list_of_tensors(self):
        imgs = [torch.rand(3, 4, 4), torch.rand(3, 4, 4)]
        result = _images_to_pil(imgs)
        self.assertEqual(len(result), 2)


class TestVideosToPil(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(_videos_to_pil(None))

    def test_4d_tensor_video(self):
        video = torch.rand(3, 3, 4, 4)  # 3 frames
        result = _videos_to_pil([video])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)

    def test_list_of_frames(self):
        frames = [torch.rand(3, 4, 4), torch.rand(3, 4, 4)]
        result = _videos_to_pil([frames])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)


class TestChatMLWebdataset(unittest.TestCase):
    def test_disables_parent_decoder_before_installing_bridge_handlers(self):
        parent_signature = inspect.signature(DefaultDecoderWebdatasetFactory.__init__)
        with (
            patch("megatron.bridge.data.energon.task_encoder_utils.inspect.signature", return_value=parent_signature),
            patch.object(DefaultDecoderWebdatasetFactory, "__init__", return_value=None) as parent_init,
        ):
            dataset = ChatMLWebdataset(EPath("/tmp/unused"))

        expected_switch = {"decoder": None} if "decoder" in parent_signature.parameters else {"auto_decode": False}
        parent_init.assert_called_once_with(EPath("/tmp/unused"), **expected_switch)
        self.assertIsNotNone(dataset._decoder)

    def test_uses_legacy_auto_decode_switch_when_decoder_parameter_is_unavailable(self):
        legacy_parameters = {"self": object(), "path": object(), "auto_decode": object(), "kwargs": object()}
        with (
            patch("megatron.bridge.data.energon.task_encoder_utils.inspect.signature") as signature,
            patch.object(DefaultDecoderWebdatasetFactory, "__init__", return_value=None) as parent_init,
        ):
            signature.return_value.parameters = legacy_parameters
            ChatMLWebdataset(EPath("/tmp/unused"), auto_decode=False)

        parent_init.assert_called_once_with(EPath("/tmp/unused"), auto_decode=False)


class TestCookChatmlSample(unittest.TestCase):
    def test_role_content_format(self):
        conv = json.dumps(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        )
        result = cook_chatml_sample(conv)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "Hello")
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(result[1]["content"], "Hi!")

    def test_from_value_format(self):
        conv = json.dumps(
            [
                {"from": "human", "value": "What is this?"},
                {"from": "gpt", "value": "A cat."},
            ]
        )
        result = cook_chatml_sample(conv)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "What is this?")
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(result[1]["content"], "A cat.")

    def test_system_turn(self):
        conv = json.dumps(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        )
        result = cook_chatml_sample(conv)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "system")

    def test_dict_wrapper(self):
        conv = {
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ]
        }
        result = cook_chatml_sample(conv)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")

    def test_multi_turn(self):
        conv = json.dumps(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        )
        result = cook_chatml_sample(conv)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[2]["role"], "user")
        self.assertEqual(result[3]["content"], "A2")

    def test_preserves_content_string(self):
        """Content with <image>/<video> tags should be left as-is (no Qwen formatting)."""
        conv = json.dumps(
            [
                {"role": "user", "content": "Look at <image> please"},
                {"role": "assistant", "content": "Nice image!"},
            ]
        )
        result = cook_chatml_sample(conv)
        self.assertEqual(result[0]["content"], "Look at <image> please")

    def test_preserves_tool_messages_and_extracts_top_level_tools(self):
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"weather"}'},
            }
        ]
        payload = {
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                {"role": "tool", "content": "sunny", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "It is sunny."},
            ],
            "tools": tools,
        }

        result = cook_chatml_sample(json.dumps(payload))

        self.assertEqual(result, payload["messages"])
        self.assertEqual(extract_chatml_tools(payload), tools)


if __name__ == "__main__":
    unittest.main()
