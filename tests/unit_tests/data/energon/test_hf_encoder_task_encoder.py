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

import json
import unittest
from unittest.mock import MagicMock

import torch

from megatron.bridge.data.energon.hf_encoder_task_encoder import (
    HFEncoderTaskBatch,
    HFEncoderTaskSample,
    HFEncoderVLMTaskEncoder,
)
from megatron.bridge.data.energon.task_encoder_utils import IGNORE_INDEX, ChatMLSample
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


def _make_processor(
    pad_token_id=0,
    eos_token_id=1,
    input_ids=None,
    pixel_values=None,
    apply_chat_template_return="Hello assistant",
    encode_return=None,
):
    """Build a mock HF processor + tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = pad_token_id
    tokenizer.eos_token_id = eos_token_id
    tokenizer.apply_chat_template.return_value = apply_chat_template_return

    if encode_return is None:
        encode_return = [12, 13]
    tokenizer.encode.return_value = encode_return

    processor = MagicMock()
    processor.tokenizer = tokenizer

    if input_ids is None:
        input_ids = torch.tensor([[10, 11, 12, 13]])

    proc_output = {"input_ids": input_ids}
    if pixel_values is not None:
        proc_output["pixel_values"] = pixel_values
    processor.return_value = proc_output

    return processor


def _make_chatml_sample(conversation, imgs=None, videos=None, key="k1"):
    """Create a ChatMLSample with the correct base-class fields."""
    return ChatMLSample(
        __key__=key,
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__={},
        imgs=imgs,
        videos=videos,
        conversation=conversation,
    )


class TestHFEncoderTaskSample(unittest.TestCase):
    def test_fields(self):
        s = HFEncoderTaskSample(
            __key__="k1",
            __subflavors__={},
            input_ids=torch.tensor([1, 2, 3]),
            labels=torch.tensor([2, 3, -100]),
            loss_mask=torch.tensor([1.0, 1.0, 0.0]),
            visual_tensors={"pixel_values": torch.randn(1, 3, 4, 4)},
        )
        self.assertEqual(s.__key__, "k1")
        self.assertEqual(s.input_ids.shape, (3,))
        self.assertIn("pixel_values", s.visual_tensors)


class TestHFEncoderVLMTaskEncoderEncodeSample(unittest.TestCase):
    def test_text_only(self):
        processor = _make_processor(
            input_ids=torch.tensor([[10, 11, 12, 13]]),
            encode_return=[12, 13],
        )
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128)

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            ),
        )

        encoded = encoder.encode_sample(sample)
        self.assertIsInstance(encoded, HFEncoderTaskSample)
        self.assertEqual(encoded.input_ids.shape[0], 4)
        self.assertEqual(encoded.labels.shape[0], 4)
        self.assertEqual(encoded.loss_mask.shape[0], 4)
        self.assertEqual(len(encoded.visual_tensors), 0)

    def test_with_images(self):
        pv = torch.randn(1, 3, 224, 224)
        processor = _make_processor(
            input_ids=torch.tensor([[10, 11, 12, 13]]),
            pixel_values=pv,
            encode_return=[12, 13],
        )
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128, visual_keys=("pixel_values",))

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Describe <image>"},
                    {"role": "assistant", "content": "A photo"},
                ]
            ),
            imgs=[torch.rand(3, 4, 4)],
        )

        encoded = encoder.encode_sample(sample)
        self.assertIn("pixel_values", encoded.visual_tensors)
        self.assertEqual(encoded.visual_tensors["pixel_values"].shape, pv.shape)

    def test_truncation(self):
        long_ids = torch.tensor([list(range(200))])
        processor = _make_processor(input_ids=long_ids, encode_return=[150, 151])
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=50)

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "long prompt"},
                    {"role": "assistant", "content": "answer"},
                ]
            ),
        )
        encoded = encoder.encode_sample(sample)
        self.assertEqual(encoded.input_ids.shape[0], 50)

    def test_loss_mask_only_on_assistant(self):
        # Tokens: [10, 11, 12, 13, 14]
        # Assistant answer tokens: [13, 14]  (at positions 3,4)
        processor = _make_processor(
            input_ids=torch.tensor([[10, 11, 12, 13, 14]]),
            encode_return=[13, 14],
        )
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128)

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A B"},
                ]
            ),
        )
        encoded = encoder.encode_sample(sample)
        self.assertTrue(encoded.loss_mask.sum() > 0, "loss_mask should have nonzero entries for assistant tokens")
        # The user input region (beginning) should have zero loss
        self.assertEqual(encoded.loss_mask[0].item(), 0.0)


class TestHFEncoderVLMTaskEncoderBatch(unittest.TestCase):
    def setUp(self):
        self.processor = _make_processor()
        self.encoder = HFEncoderVLMTaskEncoder(processor=self.processor, seq_length=128)

    def test_padding(self):
        s1 = HFEncoderTaskSample(
            __key__="k1",
            __subflavors__={},
            input_ids=torch.tensor([1, 2, 3, 4, 5]),
            labels=torch.tensor([2, 3, 4, 5, IGNORE_INDEX]),
            loss_mask=torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
            visual_tensors={},
        )
        s2 = HFEncoderTaskSample(
            __key__="k2",
            __subflavors__={},
            input_ids=torch.tensor([1, 2, 3]),
            labels=torch.tensor([2, 3, IGNORE_INDEX]),
            loss_mask=torch.tensor([0.0, 1.0, 0.0]),
            visual_tensors={},
        )

        batch = self.encoder.batch([s1, s2])
        self.assertIsInstance(batch, HFEncoderTaskBatch)
        self.assertEqual(batch.input_ids.shape, (2, 5))
        self.assertEqual(batch.labels.shape, (2, 5))
        self.assertEqual(batch.loss_mask.shape, (2, 5))
        self.assertIsNotNone(batch.attention_mask)
        self.assertEqual(batch.position_ids.shape, (2, 5))

    def test_visual_tensor_aggregation(self):
        pv1 = torch.randn(1, 3, 4, 4)
        pv2 = torch.randn(2, 3, 4, 4)
        s1 = HFEncoderTaskSample(
            __key__="k1",
            __subflavors__={},
            input_ids=torch.tensor([1, 2]),
            labels=torch.tensor([2, IGNORE_INDEX]),
            loss_mask=torch.tensor([1.0, 0.0]),
            visual_tensors={"pixel_values": pv1},
        )
        s2 = HFEncoderTaskSample(
            __key__="k2",
            __subflavors__={},
            input_ids=torch.tensor([3, 4]),
            labels=torch.tensor([4, IGNORE_INDEX]),
            loss_mask=torch.tensor([1.0, 0.0]),
            visual_tensors={"pixel_values": pv2},
        )
        batch = self.encoder.batch([s1, s2])
        self.assertIn("pixel_values", batch.visual_tensors)
        self.assertEqual(batch.visual_tensors["pixel_values"].shape[0], 3)  # 1 + 2


class TestHFEncoderVLMTaskEncoderEncodeBatch(unittest.TestCase):
    def test_encode_batch(self):
        processor = _make_processor()
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128)

        pv = torch.randn(2, 3, 4, 4)
        batch = HFEncoderTaskBatch(
            __keys__=["k1", "k2"],
            __subflavors__=[{}, {}],
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            labels=torch.tensor([[2, -100], [4, -100]]),
            loss_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            attention_mask=torch.randn(2, 1, 2, 2),
            position_ids=torch.tensor([[0, 1], [0, 1]]),
            visual_tensors={"pixel_values": pv},
        )

        result = encoder.encode_batch(batch)
        self.assertIsInstance(result, dict)
        self.assertIn("visual_inputs", result)
        self.assertIsInstance(result["visual_inputs"], GenericVisualInputs)
        self.assertNotIn("visual_tensors", result)
        self.assertNotIn("__subflavors__", result)
        self.assertIn("input_ids", result)

    def test_encode_batch_no_visuals(self):
        processor = _make_processor()
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128)

        batch = HFEncoderTaskBatch(
            __keys__=["k1"],
            __subflavors__=[{}],
            input_ids=torch.tensor([[1, 2]]),
            labels=torch.tensor([[2, -100]]),
            loss_mask=torch.tensor([[1.0, 0.0]]),
            attention_mask=torch.randn(1, 1, 2, 2),
            position_ids=torch.tensor([[0, 1]]),
            visual_tensors={},
        )

        result = encoder.encode_batch(batch)
        self.assertIn("visual_inputs", result)
        vi = result["visual_inputs"]
        self.assertIsInstance(vi, GenericVisualInputs)
        self.assertIsNone(vi.pixel_values)


class TestGenericVisualInputsCompat(unittest.TestCase):
    """Test GenericVisualInputs is compatible with vlm_step.py patterns."""

    def test_as_model_kwargs(self):
        vi = GenericVisualInputs(pixel_values=torch.randn(1, 3, 4, 4))
        kwargs = vi.as_model_kwargs()
        self.assertIn("pixel_values", kwargs)
        self.assertNotIn("image_grid_thw", kwargs)

    def test_normalized_for_model(self):
        vi = GenericVisualInputs(
            pixel_values=torch.randn(1, 3, 4, 4),
            image_sizes=torch.tensor([[4, 4]]),
        )
        result = vi.normalized_for_model()
        self.assertIn("pixel_values", result)
        self.assertIn("image_sizes", result)

    def test_dict_iteration(self):
        """vlm_step.py iterates __dict__ and calls .cuda() on non-None values."""
        vi = GenericVisualInputs(
            pixel_values=torch.randn(1, 3, 4, 4),
            image_grid_thw=None,
        )
        non_none = {k: v for k, v in vi.__dict__.items() if v is not None}
        self.assertIn("pixel_values", non_none)
        self.assertNotIn("image_grid_thw", non_none)


if __name__ == "__main__":
    unittest.main()
