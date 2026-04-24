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
        pad_token = "<pad>"
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


def test_expand_image_tokens_handles_multiple_images_and_temporal_grids():
    image_token_id = 163605
    input_ids = torch.tensor([11, image_token_id, 22, image_token_id, 33])
    attention_mask = torch.ones_like(input_ids)
    grid_thws = torch.tensor([[1, 4, 4], [2, 6, 4]])

    expanded_input_ids, expanded_attention_mask = collate._expand_image_tokens(
        input_ids,
        attention_mask,
        grid_thws,
        image_token_id,
    )

    expected = [11] + [image_token_id] * 4 + [22] + [image_token_id] * 12 + [33]
    assert expanded_input_ids.tolist() == expected
    assert expanded_attention_mask.tolist() == [1] * len(expected)


# ---------------------------------------------------------------------------
# kimi_k25_vl_collate_fn tests
# ---------------------------------------------------------------------------

MEDIA_TOKEN_ID = 163605  # default Kimi K2.5 media placeholder


class _KimiDummyTokenizer:
    """Minimal tokenizer mock for kimi_k25_vl_collate_fn tests."""

    pad_token_id = 0
    added_tokens_decoder = {}

    def convert_tokens_to_ids(self, token):
        return MEDIA_TOKEN_ID

    def __call__(self, text, add_special_tokens=True, **kwargs):
        # Return a fixed token sequence so loss-mask search can find the span.
        return {"input_ids": [10, 11, 12]}


class _KimiDummyProcessor:
    """Minimal processor mock that mimics KimiK25Processor behaviour."""

    media_placeholder_token_id = MEDIA_TOKEN_ID

    def __init__(self, *, include_image: bool = False):
        self.tokenizer = _KimiDummyTokenizer()
        self._include_image = include_image

    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=False, **kwargs):
        return "dummy text"

    def __call__(self, text=None, medias=None, return_tensors="pt", **kwargs):
        # Build minimal processor output with or without image data.
        seq = [1, 2, MEDIA_TOKEN_ID, 10, 11, 12, 3] if self._include_image else [1, 10, 11, 12, 3]
        input_ids = torch.tensor([seq])
        attention_mask = torch.ones_like(input_ids)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._include_image and medias:
            out["pixel_values"] = torch.randn(1, 3, 4, 4)
            out["grid_thws"] = torch.tensor([[1, 2, 2]])  # expands to 1 token
        return out


def test_kimi_k25_vl_collate_fn_text_only():
    """Text-only batch: no pixel_values / grid_thws in result."""
    proc = _KimiDummyProcessor(include_image=False)
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            ]
        },
    ]
    batch = collate.kimi_k25_vl_collate_fn(examples, proc)

    assert "input_ids" in batch
    assert "labels" in batch
    assert "loss_mask" in batch
    assert "position_ids" in batch
    assert "visual_inputs" in batch
    # No image data → visual_inputs fields should be None
    vi = batch["visual_inputs"]
    assert vi.pixel_values is None
    assert vi.image_grid_thw is None
    # Shapes consistent
    B, L = batch["input_ids"].shape
    assert batch["labels"].shape == (B, L)
    assert batch["loss_mask"].shape == (B, L)
    assert batch["position_ids"].shape == (B, L)


def test_kimi_k25_vl_collate_fn_with_image():
    """Image batch: pixel_values and grid_thws forwarded to visual_inputs."""
    proc = _KimiDummyProcessor(include_image=True)
    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "dummy.jpg"},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "it's a cat"}]},
            ]
        },
    ]
    batch = collate.kimi_k25_vl_collate_fn(examples, proc)

    vi = batch["visual_inputs"]
    assert vi.pixel_values is not None
    assert vi.image_grid_thw is not None
    # input_ids should not contain raw pixel_values / grid_thws keys
    assert "pixel_values" not in batch
    assert "grid_thws" not in batch


def test_kimi_k25_vl_collate_fn_pads_to_max_length():
    """max_length is respected: short sequences padded, long ones truncated."""
    proc = _KimiDummyProcessor(include_image=False)
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            ]
        },
    ]
    max_length = 20
    batch = collate.kimi_k25_vl_collate_fn(examples, proc, max_length=max_length)

    assert batch["input_ids"].shape[1] == max_length
    assert batch["attention_mask"].shape[1] == max_length


def test_kimi_k25_vl_collate_fn_multi_sample_batch():
    """Multiple samples are batched correctly with equal sequence lengths."""
    proc = _KimiDummyProcessor(include_image=False)
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "q1"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "q2"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
            ]
        },
    ]
    batch = collate.kimi_k25_vl_collate_fn(examples, proc)

    assert batch["input_ids"].shape[0] == 2
    # All sequences must have the same length after collation
    assert batch["input_ids"].shape[1] == batch["labels"].shape[1]


# ---------------------------------------------------------------------------
# Gemma4 collate — registration and image_position_ids passthrough
# ---------------------------------------------------------------------------


def test_gemma4_processor_registered_in_collate_fns():
    """Gemma4Processor must be registered in COLLATE_FNS."""
    assert "Gemma4Processor" in collate.COLLATE_FNS


def test_gemma4_vl_collate_fn_is_ministral3_alias():
    """gemma4_vl_collate_fn is an alias for ministral3_collate_fn."""
    assert collate.gemma4_vl_collate_fn is collate.ministral3_collate_fn


def test_gemma4_registered_fn_matches_alias():
    """The registered function for Gemma4Processor equals the alias."""
    assert collate.COLLATE_FNS["Gemma4Processor"] is collate.gemma4_vl_collate_fn


class _Gemma4ProcessorBase:
    """Minimal Gemma4Processor stub for ministral3_collate_fn tests.

    create_multiturn_loss_mask_by_search calls tokenizer(text, add_special_tokens=False)
    so _Tok must be callable.
    """
    chat_template = "dummy"

    class _Tok:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        eos_token = "<eos>"

        def __call__(self, text, add_special_tokens=True, **kwargs):
            # Return minimal tokenized output: each word → one token id
            ids = list(range(1, len(text.split()) + 1))
            return {"input_ids": ids if ids else [1]}

    def __init__(self, include_position_ids=True):
        self.tokenizer = self._Tok()
        self._include_position_ids = include_position_ids

    def apply_chat_template(self, conversations, tokenize=False, **kwargs):
        if not tokenize:
            return "dummy text"
        seq_len = 8
        batch_size = len(conversations)
        result = {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
        }
        if self._include_position_ids:
            result["image_position_ids"] = torch.zeros(batch_size, 196, 2, dtype=torch.long)
        return result


def test_ministral3_collate_wraps_image_position_ids_in_visual_inputs():
    """image_position_ids returned by processor ends up inside GenericVisualInputs."""
    proc = _Gemma4ProcessorBase(include_position_ids=True)
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ]
        }
    ]
    batch = collate.ministral3_collate_fn(examples, proc)

    assert "visual_inputs" in batch
    vi = batch["visual_inputs"]
    assert vi is not None
    assert hasattr(vi, "image_position_ids")
    assert vi.image_position_ids is not None


def test_ministral3_collate_no_image_position_ids_excluded():
    """When processor returns no image_position_ids, the field stays None in visual_inputs."""
    proc = _Gemma4ProcessorBase(include_position_ids=False)
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ]
        }
    ]
    batch = collate.ministral3_collate_fn(examples, proc)

    assert "visual_inputs" in batch
    vi = batch["visual_inputs"]
    assert vi is not None
    assert vi.image_position_ids is None
