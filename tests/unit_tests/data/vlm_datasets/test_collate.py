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

import torch

import megatron.bridge.data.vlm_datasets.collate as collate


class _DummyProcessor:
    class _Tok:
        pad_token_id = 0
        added_tokens_decoder = {}

    def __init__(self):
        self.tokenizer = self._Tok()

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):
        if tokenize:
            # Return dict mimicking HF processor output when tokenize=True
            # Minimal keys used by default_collate_fn
            input_ids = torch.tensor([[1, 2, 3]])
            pixel_values = torch.randn(1, 1, 3, 4, 4)
            return {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
        # Non-tokenized: just a string
        return "dummy"

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
        # Minimal shape/value outputs used by qwen2_5_collate_fn
        input_ids = torch.tensor([[1, 2, 3]])
        out = {"input_ids": input_ids}
        if images is not None:
            # Create 1-batch, N images = len(images)
            n = len(images)
            out["pixel_values"] = torch.randn(1, n, 3, 4, 4)
            out["image_grid_thw"] = torch.tensor([[[1, 2, 2]] * n])
        return out


def test_default_collate_builds_visual_inputs(monkeypatch):
    # Force HAVE_QWEN_VL_UTILS True
    monkeypatch.setattr(collate, "HAVE_QWEN_VL_UTILS", True)
    proc = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    ]
    batch = collate.default_collate_fn(examples, proc)
    assert "visual_inputs" in batch
    vi = batch["visual_inputs"]
    # normalized_for_model called in training path; here we just assert fields present
    assert hasattr(vi, "pixel_values")


def test_qwen2_5_collate_fn_handles_no_images(monkeypatch):
    monkeypatch.setattr(collate, "HAVE_QWEN_VL_UTILS", True)
    # Stub process_vision_info to return (None, None)
    monkeypatch.setattr(collate, "process_vision_info", lambda conv: (None, None))
    proc = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
    ]
    batch = collate.qwen2_5_collate_fn(examples, proc)
    assert "input_ids" in batch and "labels" in batch and "loss_mask" in batch
    assert "visual_inputs" in batch


def test_qwen2_audio_collate_fn_uses_audio_inputs_key(monkeypatch):
    """qwen2_audio_collate_fn should store Qwen2AudioInputs under 'audio_inputs', not 'visual_inputs'."""

    class _AudioProcessor:
        class _Tok:
            pad_token_id = 0
            padding_side = "right"
            added_tokens_decoder = {}

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2]}

        def __init__(self):
            self.tokenizer = self._Tok()

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):
            return "dummy"

        def __call__(self, text=None, audio=None, return_tensors="pt", padding=True, **kwargs):
            n = len(text)
            return {
                "input_ids": torch.tensor([[1, 2, 3]] * n),
                "input_features": torch.randn(n, 80, 16),
                "feature_attention_mask": torch.ones(n, 16),
            }

    # Stub _gather_assistant_text_segments to return a findable text
    monkeypatch.setattr(collate, "_gather_assistant_text_segments", lambda ex: ["dummy"])

    proc = _AudioProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    ]
    batch = collate.qwen2_audio_collate_fn(examples, proc)

    # Must use 'audio_inputs', not 'visual_inputs'
    assert "audio_inputs" in batch, f"Expected 'audio_inputs' key, got keys: {list(batch.keys())}"
    assert "visual_inputs" not in batch
    ai = batch["audio_inputs"]
    assert hasattr(ai, "input_features")
    assert hasattr(ai, "feature_attention_mask")
    # Raw keys should be cleaned up
    assert "input_features" not in batch
    assert "feature_attention_mask" not in batch


def test_qwen2_5_collate_fn_handles_with_images(monkeypatch):
    monkeypatch.setattr(collate, "HAVE_QWEN_VL_UTILS", True)

    # Return list of N fake images for first example, None for second
    def _fake_pvi(conv):
        # Push 2 images for first, no images for second
        text = str(conv)
        if "hi" in text:
            return ([object(), object()], None)
        return (None, None)

    monkeypatch.setattr(collate, "process_vision_info", _fake_pvi)
    proc = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
    ]
    batch = collate.qwen2_5_collate_fn(examples, proc)
    assert "visual_inputs" in batch
    vi = batch["visual_inputs"]
    # Ensure fields exist when images present
    assert hasattr(vi, "pixel_values")
