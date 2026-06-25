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

from megatron.bridge.data.energon.hf_task_encoder import (
    HFEnergonBatch,
    HFEnergonSample,
    HFTaskEncoder,
)
from megatron.bridge.data.energon.metadata import batch_metadata_kwargs, sample_metadata_kwargs
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample
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
    processor.image_token_id = 10
    processor.apply_chat_template.return_value = apply_chat_template_return

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
        **sample_metadata_kwargs(key=key, restore_key=(), subflavors={}),
        imgs=imgs,
        videos=videos,
        conversation=conversation,
    )


def _make_collate_fn(pixel_values=None, seen_examples=None):
    """Build a tiny collate function with the same output keys as HF VLM collators."""

    def _collate(
        examples,
        processor,  # noqa: ARG001 - processor is part of the collate contract
        *,
        visual_keys=None,  # noqa: ARG001 - generic HF collate contract
        min_pixels=None,  # noqa: ARG001 - generic HF collate contract
        max_pixels=None,  # noqa: ARG001 - generic HF collate contract
        sequence_length=None,  # noqa: ARG001 - generic HF collate contract
        pad_to_max_length=False,  # noqa: ARG001 - generic HF collate contract
        pad_to_multiple_of=128,  # noqa: ARG001 - generic HF collate contract
        enable_in_batch_packing=False,  # noqa: ARG001 - generic HF collate contract
        in_batch_packing_pad_to_multiple_of=1,  # noqa: ARG001 - generic HF collate contract
    ):
        if seen_examples is not None:
            seen_examples.extend(examples)
        batch_size = len(examples)
        seq_len = 5
        visual_inputs = GenericVisualInputs(pixel_values=pixel_values) if pixel_values is not None else None
        return {
            "input_ids": torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len),
            "labels": torch.full((batch_size, seq_len), -100, dtype=torch.long),
            "loss_mask": torch.zeros(batch_size, seq_len),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "visual_inputs": visual_inputs,
        }

    return _collate


class TestHFEnergonSample(unittest.TestCase):
    def test_fields(self):
        example = {"conversation": [{"role": "user", "content": "Hi"}]}
        s = HFEnergonSample(
            __key__="k1",
            __subflavors__={},
            example=example,
        )
        self.assertEqual(s.__key__, "k1")
        self.assertEqual(s.example, example)


class TestHFTaskEncoderEncodeSample(unittest.TestCase):
    def test_text_only(self):
        processor = _make_processor()
        encoder = HFTaskEncoder(processor=processor, seq_length=128, collate_fn=_make_collate_fn())

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            ),
        )

        encoded = encoder.encode_sample(sample)
        self.assertIsInstance(encoded, HFEnergonSample)
        self.assertEqual(
            encoded.example["conversation"],
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        )

    def test_with_images(self):
        processor = _make_processor()
        encoder = HFTaskEncoder(
            processor=processor,
            seq_length=128,
            visual_keys=("pixel_values",),
            collate_fn=_make_collate_fn(),
        )

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
        user_content = encoded.example["conversation"][0]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image")
        self.assertIn("image", user_content[1])
        processor.assert_not_called()


class TestHFTaskEncoderBatch(unittest.TestCase):
    def setUp(self):
        self.processor = _make_processor()
        self.seen_examples = []
        self.encoder = HFTaskEncoder(
            processor=self.processor,
            seq_length=128,
            collate_fn=_make_collate_fn(seen_examples=self.seen_examples),
        )

    def test_batch_uses_collate_fn(self):
        s1 = HFEnergonSample(
            __key__="k1",
            __subflavors__={},
            example={"conversation": [{"role": "user", "content": "one"}]},
        )
        s2 = HFEnergonSample(
            __key__="k2",
            __subflavors__={},
            example={"conversation": [{"role": "user", "content": "two"}]},
        )

        batch = self.encoder.batch([s1, s2])
        self.assertIsInstance(batch, HFEnergonBatch)
        self.assertEqual(batch.input_ids.shape, (2, 5))
        self.assertEqual(batch.labels.shape, (2, 5))
        self.assertEqual(batch.loss_mask.shape, (2, 5))
        self.assertIsNotNone(batch.attention_mask)
        self.assertEqual(batch.position_ids.shape, (2, 5))
        self.assertEqual(self.seen_examples, [s1.example, s2.example])

    def test_visual_inputs_passthrough(self):
        pv = torch.randn(3, 3, 4, 4)
        encoder = HFTaskEncoder(
            processor=self.processor,
            seq_length=128,
            collate_fn=_make_collate_fn(pixel_values=pv),
        )
        s1 = HFEnergonSample(
            __key__="k1",
            __subflavors__={},
            example={"conversation": [{"role": "user", "content": "one"}]},
        )
        batch = encoder.batch([s1])
        self.assertIsInstance(batch.visual_inputs, GenericVisualInputs)
        self.assertEqual(batch.visual_inputs.pixel_values.shape[0], 3)

    def test_batch_uses_encode_sample_examples(self):
        seen_examples = []
        encoder = HFTaskEncoder(
            processor=self.processor,
            seq_length=128,
            collate_fn=_make_collate_fn(seen_examples=seen_examples),
        )
        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            ),
            key="from-encode-sample",
        )

        encoded = encoder.encode_sample(sample)
        batch = encoder.batch([encoded])

        self.assertEqual(batch.input_ids.shape, (1, 5))
        self.assertEqual(seen_examples, [encoded.example])

    def test_collate_fn_threads_supported_encoder_options(self):
        seen_kwargs = {}

        def _collate(
            examples,
            processor,  # noqa: ARG001 - processor is part of the collate contract
            *,
            visual_keys,
            min_pixels=None,
            max_pixels=None,
            sequence_length=None,
            pad_to_max_length=False,
            pad_to_multiple_of=128,
            enable_in_batch_packing=False,
            in_batch_packing_pad_to_multiple_of=1,
        ):
            del (
                sequence_length,
                pad_to_max_length,
                pad_to_multiple_of,
                enable_in_batch_packing,
                in_batch_packing_pad_to_multiple_of,
            )
            seen_kwargs.update(
                {
                    "examples": examples,
                    "visual_keys": visual_keys,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            )
            return _make_collate_fn()(examples, processor)

        encoder = HFTaskEncoder(
            processor=self.processor,
            seq_length=128,
            visual_keys=("pixel_values", "image_sizes"),
            min_pixels=16,
            max_pixels=128,
            collate_fn=_collate,
        )
        sample = HFEnergonSample(
            __key__="manual",
            __subflavors__={},
            example={"conversation": [{"role": "user", "content": "manual"}]},
        )

        encoder.batch([sample])

        self.assertEqual(seen_kwargs["examples"], [sample.example])
        self.assertEqual(seen_kwargs["visual_keys"], ("pixel_values", "image_sizes"))
        self.assertEqual(seen_kwargs["min_pixels"], 16)
        self.assertEqual(seen_kwargs["max_pixels"], 128)

    def test_collate_fallback_rejects_oversized_batches(self):
        processor = _make_processor()
        encoder = HFTaskEncoder(
            processor=processor,
            seq_length=4,
            collate_fn=_make_collate_fn(),
        )
        sample = HFEnergonSample(
            __key__="manual",
            __subflavors__={},
            example={"conversation": [{"role": "user", "content": "manual"}]},
        )

        with self.assertRaisesRegex(ValueError, "exceeds seq_length"):
            encoder.batch([sample])


class TestHFTaskEncoderEncodeBatch(unittest.TestCase):
    def test_encode_batch(self):
        processor = _make_processor()
        encoder = HFTaskEncoder(processor=processor, seq_length=128, collate_fn=_make_collate_fn())

        pv = torch.randn(2, 3, 4, 4)
        batch = HFEnergonBatch(
            **batch_metadata_kwargs(keys=["k1", "k2"]),
            __keys__=["k1", "k2"],
            __subflavors__=[{}, {}],
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            labels=torch.tensor([[2, -100], [4, -100]]),
            loss_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            attention_mask=torch.randn(2, 1, 2, 2),
            position_ids=torch.tensor([[0, 1], [0, 1]]),
            visual_inputs=GenericVisualInputs(pixel_values=pv),
        )

        result = encoder.encode_batch(batch)
        self.assertIsInstance(result, dict)
        self.assertIn("visual_inputs", result)
        self.assertIsInstance(result["visual_inputs"], GenericVisualInputs)
        self.assertNotIn("__subflavors__", result)
        self.assertIn("input_ids", result)

    def test_encode_batch_no_visuals(self):
        processor = _make_processor()
        encoder = HFTaskEncoder(processor=processor, seq_length=128, collate_fn=_make_collate_fn())

        batch = HFEnergonBatch(
            **batch_metadata_kwargs(keys=["k1"]),
            __keys__=["k1"],
            __subflavors__=[{}],
            input_ids=torch.tensor([[1, 2]]),
            labels=torch.tensor([[2, -100]]),
            loss_mask=torch.tensor([[1.0, 0.0]]),
            attention_mask=torch.randn(1, 1, 2, 2),
            position_ids=torch.tensor([[0, 1]]),
            visual_inputs=None,
        )

        result = encoder.encode_batch(batch)
        self.assertIn("visual_inputs", result)
        self.assertIsNone(result["visual_inputs"])


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
