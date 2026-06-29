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

import pytest
import torch

import megatron.bridge.data.vlm_datasets.collate as collate
import megatron.bridge.models.gemma_vl.data.collate_fn as gemma_vl_collate
import megatron.bridge.models.kimi_vl.data.collate_fn as kimi_collate
import megatron.bridge.models.nemotron_omni.data.collate_fn as nemotron_omni_collate
import megatron.bridge.models.qwen_audio.data.collate_fn as qwen_audio_collate
import megatron.bridge.models.qwen_vl.data.collate_fn as qwen_vl_collate
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.vlm_processing import build_assistant_loss_mask as canonical_build_assistant_loss_mask


pytestmark = pytest.mark.unit


def test_vlm_collate_reexports_assistant_loss_mask_for_compatibility():
    assert collate.build_assistant_loss_mask is canonical_build_assistant_loss_mask


def test_vlm_collate_keeps_qwen_vl_registration():
    assert collate.COLLATE_FNS["Qwen2_5_VLProcessor"] is collate.qwen2_5_collate_fn


class _DummyProcessor:
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    class _Tok:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def encode(self, text, add_special_tokens=False):
            return self(text, add_special_tokens=add_special_tokens)["input_ids"]

        def __call__(self, text, add_special_tokens=False):
            mapping = {
                "<|im_start|>assistant\n": [102],
                "<|im_end|>": [103],
            }
            return {"input_ids": mapping.get(text, [1])}

    def __init__(self):
        self.tokenizer = self._Tok()
        self.template_kwargs = []
        self.processor_kwargs = []

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):
        self.template_kwargs.append(kwargs)
        if tokenize:
            # Return dict mimicking HF processor output when tokenize=True
            # Minimal keys used by gemma3_vl_collate_fn
            input_ids = torch.tensor([[1, 2, 3]])
            output = {
                "input_ids": input_ids,
            }
            if kwargs.get("return_assistant_tokens_mask"):
                output["input_ids"] = [1, 2, 3]
                output["assistant_masks"] = [0, 0, 0]
                return output
            pixel_values = torch.randn(1, 1, 3, 4, 4)
            output["pixel_values"] = pixel_values
            output["image_grid_thw"] = torch.tensor([[[1, 2, 2]]])
            output["image_sizes"] = torch.tensor([[4, 4]])
            return output
        # Non-tokenized: just a string
        return "dummy"

    def __call__(self, text=None, images=None, videos=None, padding=True, return_tensors="pt", **kwargs):
        self.processor_kwargs.append(kwargs)
        # Minimal shape/value outputs used by qwen2_5_collate_fn
        input_ids = torch.tensor([[1, 2, 3]])
        out = {"input_ids": input_ids}
        if images is not None:
            # Create 1-batch, N images = len(images)
            n = len(images)
            out["pixel_values"] = torch.randn(1, n, 3, 4, 4)
            out["image_grid_thw"] = torch.tensor([[[1, 2, 2]] * n])
        if videos is not None:
            n = len(videos)
            out["pixel_values_videos"] = torch.randn(1, n, 3, 4, 4)
            out["video_grid_thw"] = torch.tensor([[[2, 2, 2]] * n])
        return out


def test_gemma3_vl_collate_builds_visual_inputs():
    proc = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    ]
    batch = collate.gemma3_vl_collate_fn(examples, proc)
    assert "visual_inputs" in batch
    vi = batch["visual_inputs"]
    # normalized_for_model called in training path; here we just assert fields present
    assert vi.pixel_values is not None
    assert vi.image_grid_thw is not None


def test_gemma3_vl_collate_honors_visual_keys_and_pixel_constraints():
    proc = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    ]

    batch = collate.gemma3_vl_collate_fn(
        examples,
        proc,
        visual_keys=("pixel_values", "image_sizes"),
        min_pixels=16,
        max_pixels=128,
    )

    collate_template_kwargs = next(kwargs for kwargs in proc.template_kwargs if kwargs.get("return_tensors") == "pt")
    assert collate_template_kwargs["min_pixels"] == 16
    assert collate_template_kwargs["max_pixels"] == 128
    assert batch["visual_inputs"].pixel_values is not None
    assert batch["visual_inputs"].image_sizes is not None
    assert batch["visual_inputs"].image_grid_thw is None
    assert "image_grid_thw" not in batch
    assert "image_sizes" not in batch


def test_qwen2_5_collate_fn_handles_no_images(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)
    # Stub process_vision_info to return (None, None)
    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", lambda conv: (None, None))
    proc = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
    ]
    batch = collate.qwen2_5_collate_fn(examples, proc)
    assert "input_ids" in batch and "labels" in batch and "loss_mask" in batch
    assert "visual_inputs" in batch


def test_qwen2_5_collate_fn_uses_shared_pixel_defaults(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)
    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", lambda conv: ([object()], None))

    proc = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    ]

    collate.qwen2_5_collate_fn(examples, proc)

    assert proc.processor_kwargs[-1]["min_pixels"] == qwen_vl_collate.QWEN_VL_MIN_PIXELS
    assert proc.processor_kwargs[-1]["max_pixels"] == qwen_vl_collate.QWEN_VL_MAX_PIXELS


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

    # Stub assistant text extraction to return a findable text.
    monkeypatch.setattr(qwen_audio_collate, "gather_assistant_text_segments", lambda ex: ["dummy"])

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


def test_qwen2_audio_collate_fn_defers_packing_to_audio_step(monkeypatch):
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

    monkeypatch.setattr(qwen_audio_collate, "gather_assistant_text_segments", lambda ex: ["dummy"])

    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
    ]

    batch = collate.qwen2_audio_collate_fn(
        examples, _AudioProcessor(), sequence_length=128, enable_in_batch_packing=True
    )

    assert batch["input_ids"].shape == (2, 128)
    assert "cu_seqlens" not in batch
    assert "max_seqlen" not in batch


def test_qwen2_5_collate_fn_handles_with_images(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)

    # Return list of N fake images for first example, None for second
    def _fake_pvi(conv):
        # Push 2 images for first, no images for second
        text = str(conv)
        if "hi" in text:
            return ([object(), object()], None)
        return (None, None)

    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", _fake_pvi)
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


def test_qwen2_5_collate_fn_handles_with_videos(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)

    def _fake_pvi(conv):
        text = str(conv)
        if "watch" in text:
            return (None, [[object(), object()]])
        return (None, None)

    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", _fake_pvi)
    proc = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "watch"}]}]},
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
    ]

    batch = collate.qwen2_5_collate_fn(examples, proc)

    vi = batch["visual_inputs"]
    assert vi.pixel_values_videos is not None
    assert vi.video_grid_thw is not None
    assert "pixel_values_videos" not in batch
    assert "video_grid_thw" not in batch


def test_qwen2_5_collate_fn_preserves_attention_mask_for_mixed_image_text_batch(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)

    class _PadAwareProcessor:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        class _Tok:
            pad_token_id = 99
            pad_token = "<pad>"
            added_tokens_decoder = {}
            chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1]}

        def __init__(self):
            self.tokenizer = self._Tok()

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):
            rendered = conversation[0]["content"][-1]["text"]
            if tokenize and kwargs.get("return_assistant_tokens_mask"):
                length = 3 if "short" in rendered else 5
                return {
                    "input_ids": list(range(1, length + 1)),
                    "assistant_masks": [0] * (length - 1) + [1],
                }
            return rendered

        def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
            texts = text if isinstance(text, list) else [text]
            lengths = [3 if "short" in item else 5 for item in texts]
            max_len = max(lengths)
            input_ids = torch.full((len(texts), max_len), self.tokenizer.pad_token_id)
            attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)
            for row, length in enumerate(lengths):
                input_ids[row, :length] = torch.arange(1, length + 1)
                attention_mask[row, :length] = 1
            out = {"input_ids": input_ids, "attention_mask": attention_mask}
            if images is not None:
                out["pixel_values"] = torch.randn(1, len(images), 3, 4, 4)
                out["image_grid_thw"] = torch.tensor([[[1, 2, 2]] * len(images)])
            return out

    def _fake_pvi(conv):
        if "short image" in str(conv):
            return ([object()], None)
        return (None, None)

    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", _fake_pvi)

    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "short image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "long text only"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        },
    ]

    batch = collate.qwen2_5_collate_fn(examples, _PadAwareProcessor())

    assert batch["input_ids"].tolist() == [[1, 2, 3, 99, 99], [1, 2, 3, 4, 5]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_qwen2_5_collate_fn_uses_declared_chatml_boundary_config_without_generation_template(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)
    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", lambda conv: (None, None))

    class _ChatMLProcessor:
        chat_template = "<|im_start|>user\n{{ content }}<|im_end|><|im_start|>assistant\n{{ content }}<|im_end|>"

        class _Tok:
            pad_token_id = 0
            pad_token = "<pad>"
            added_tokens_decoder = {}
            chat_template = "<|im_start|>user\n{{ content }}<|im_end|><|im_start|>assistant\n{{ content }}<|im_end|>"

            def __call__(self, text, add_special_tokens=False):
                mapping = {
                    "<|im_start|>assistant\n": [102],
                    "<|im_end|>": [103],
                }
                return {"input_ids": mapping.get(text, [42])}

        def __init__(self):
            self.tokenizer = self._Tok()

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):
            return "rendered"

        def __call__(self, text=None, padding=True, return_tensors="pt", **kwargs):
            return {"input_ids": torch.tensor([[100, 7, 101, 102, 3, 4, 103]])}

    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
    ]

    batch = collate.qwen2_5_collate_fn(examples, _ChatMLProcessor())

    assert batch["loss_mask"].tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 3, 4, 103, -100]]


def test_qwen2_5_collate_fn_packs_vlm_batch(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)
    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", lambda conv: (None, None))

    class _PackableProcessor:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        class _Tok:
            pad_token_id = 99
            pad_token = "<pad>"
            added_tokens_decoder = {}
            chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1]}

        def __init__(self):
            self.tokenizer = self._Tok()

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):
            rendered = conversation[0]["content"][-1]["text"]
            if tokenize and kwargs.get("return_assistant_tokens_mask"):
                length = 3 if "short" in rendered else 5
                return {
                    "input_ids": list(range(1, length + 1)),
                    "assistant_masks": [0] * (length - 1) + [1],
                }
            return rendered

        def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
            texts = text if isinstance(text, list) else [text]
            lengths = [3 if "short" in item else 5 for item in texts]
            max_len = max(lengths)
            input_ids = torch.full((len(texts), max_len), self.tokenizer.pad_token_id)
            attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)
            for row, length in enumerate(lengths):
                input_ids[row, :length] = torch.arange(1, length + 1)
                attention_mask[row, :length] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "short"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "long"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        },
    ]

    batch = collate.qwen2_5_collate_fn(
        examples,
        _PackableProcessor(),
        sequence_length=16,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 0, 0]]
    assert batch["attention_mask"] is None
    assert batch["cu_seqlens"].tolist() == [[0, 4, 12]]
    assert batch["max_seqlen"].tolist() == [[8]]
    assert batch["visual_inputs"] is not None


def test_expand_image_tokens_handles_multiple_images_and_temporal_grids():
    image_token_id = 163605
    input_ids = torch.tensor([11, image_token_id, 22, image_token_id, 33])
    attention_mask = torch.ones_like(input_ids)
    grid_thws = torch.tensor([[1, 4, 4], [2, 6, 4]])

    expanded_input_ids, expanded_attention_mask = kimi_collate._expand_image_tokens(
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
KIMI_IM_ASSISTANT_ID = 601
KIMI_ASSISTANT_TEXT_ID = 602
KIMI_IM_MIDDLE_ID = 603
KIMI_IM_END_ID = 604
KIMI_THINK_OPEN_ID = 605
KIMI_THINK_CLOSE_ID = 606
KIMI_ASSISTANT_HEADER_IDS = [KIMI_IM_ASSISTANT_ID, KIMI_ASSISTANT_TEXT_ID, KIMI_IM_MIDDLE_ID]


class _KimiDummyTokenizer:
    """Minimal tokenizer mock for kimi_k25_vl_collate_fn tests."""

    pad_token_id = 0
    added_tokens_decoder = {}
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    def convert_tokens_to_ids(self, token):
        mapping = {
            "<|im_assistant|>": KIMI_IM_ASSISTANT_ID,
            "<|im_end|>": KIMI_IM_END_ID,
            "<|media_pad|>": MEDIA_TOKEN_ID,
            "<think>": KIMI_THINK_OPEN_ID,
            "</think>": KIMI_THINK_CLOSE_ID,
        }
        return mapping.get(token, MEDIA_TOKEN_ID)

    def __call__(self, text, add_special_tokens=True, **kwargs):
        mapping = {
            "<|im_assistant|>assistant<|im_middle|>": KIMI_ASSISTANT_HEADER_IDS,
            "<|im_end|>": [KIMI_IM_END_ID],
            "<think>": [KIMI_THINK_OPEN_ID],
            "</think>": [KIMI_THINK_CLOSE_ID],
        }
        return {"input_ids": mapping.get(text, [10, 11, 12])}


class _KimiDummyProcessor:
    """Minimal processor mock that mimics KimiK25Processor behaviour."""

    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
    media_placeholder_token_id = MEDIA_TOKEN_ID

    def __init__(self, *, include_image: bool = False):
        self.tokenizer = _KimiDummyTokenizer()
        self._include_image = include_image
        self.template_kwargs = []
        self.processor_kwargs = []

    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=False, **kwargs):
        self.template_kwargs.append(kwargs)
        if tokenize and kwargs.get("return_assistant_tokens_mask"):
            if self._include_image:
                return {
                    "input_ids": [1, 2, MEDIA_TOKEN_ID, 10, 11, 12, 3],
                    "assistant_masks": [0, 0, 0, 1, 1, 1, 0],
                }
            return {"input_ids": [1, 10, 11, 12, 3], "assistant_masks": [0, 1, 1, 1, 0]}
        return "dummy text"

    def __call__(self, text=None, medias=None, return_tensors="pt", **kwargs):
        self.processor_kwargs.append({"text": text, "medias": medias, "return_tensors": return_tensors, **kwargs})
        # Build minimal processor output with or without image data.
        seq = [1, 2, MEDIA_TOKEN_ID, 10, 11, 12, 3] if self._include_image else [1, 10, 11, 12, 3]
        input_ids = torch.tensor([seq])
        attention_mask = torch.ones_like(input_ids)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._include_image and medias:
            out["pixel_values"] = torch.randn(1, 3, 4, 4)
            out["grid_thws"] = torch.tensor([[1, 2, 2]])  # expands to 1 token
        return out


class _KimiScenarioTokenizer:
    """Tokenizer mock with Kimi marker tokenization semantics."""

    pad_token_id = 0
    added_tokens_decoder = {}
    chat_template = "<|im_assistant|>assistant<|im_middle|>{{ content }}<|im_end|>"

    def convert_tokens_to_ids(self, token):
        mapping = {
            "<|im_assistant|>": KIMI_IM_ASSISTANT_ID,
            "<|im_end|>": KIMI_IM_END_ID,
            "<|media_pad|>": MEDIA_TOKEN_ID,
            "<think>": KIMI_THINK_OPEN_ID,
            "</think>": KIMI_THINK_CLOSE_ID,
        }
        return mapping[token]

    def __call__(self, text, add_special_tokens=False, **kwargs):
        mapping = {
            "<|im_assistant|>assistant<|im_middle|>": KIMI_ASSISTANT_HEADER_IDS,
            "<|im_end|>": [KIMI_IM_END_ID],
            "<think>": [KIMI_THINK_OPEN_ID],
            "</think>": [KIMI_THINK_CLOSE_ID],
        }
        return {"input_ids": mapping.get(text, [999])}


class _KimiScenarioProcessor:
    """Processor mock returning caller-provided token streams."""

    media_placeholder_token_id = MEDIA_TOKEN_ID

    def __init__(self, rows, grid_thws=None):
        self.tokenizer = _KimiScenarioTokenizer()
        self.rows = rows
        self.grid_thws = grid_thws or [None] * len(rows)
        self.template_kwargs = []
        self.processor_kwargs = []
        self._call_idx = 0

    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=False, **kwargs):
        self.template_kwargs.append(kwargs)
        return f"rendered-{len(self.template_kwargs) - 1}"

    def __call__(self, text=None, medias=None, return_tensors="pt", **kwargs):
        row_idx = self._call_idx
        self._call_idx += 1
        self.processor_kwargs.append({"text": text, "medias": medias, "return_tensors": return_tensors, **kwargs})

        input_ids = torch.tensor([self.rows[row_idx]], dtype=torch.long)
        out = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        if self.grid_thws[row_idx] is not None and medias:
            out["pixel_values"] = torch.ones(len(medias), 3, 4, 4)
            out["grid_thws"] = self.grid_thws[row_idx]
        return out


def _kimi_target_ids(batch, row=0):
    target = batch["labels"][row][batch["loss_mask"][row].bool()]
    assert torch.all(target != IGNORE_INDEX)
    return target.tolist()


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
    """max_length is respected for short sequences that need padding."""
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
    assert batch["loss_mask"].shape[1] == max_length


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


def test_kimi_k25_vl_collate_fn_forwards_tools_to_chat_template():
    proc = _KimiDummyProcessor(include_image=False)
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ],
            "tools": tools,
        },
    ]

    collate.kimi_k25_vl_collate_fn(examples, proc)

    assert proc.template_kwargs[0]["tools"] == tools


def test_kimi_k25_vl_collate_fn_preserves_thinking_and_passes_empty_medias():
    proc = _KimiDummyProcessor(include_image=False)
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {"role": "assistant", "reasoning_content": "think", "content": [{"type": "text", "text": "a"}]},
            ],
        },
    ]

    collate.kimi_k25_vl_collate_fn(examples, proc)

    assert proc.template_kwargs[0]["preserve_thinking"] is True
    assert proc.processor_kwargs[0]["medias"] == []


def test_kimi_k25_vl_collate_fn_keeps_loss_mask_selected_special_tokens():
    proc = _KimiDummyProcessor(include_image=False)
    proc.tokenizer.added_tokens_decoder = {10: "<|im_end|>"}
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ],
        },
    ]

    batch = collate.kimi_k25_vl_collate_fn(examples, proc)

    assert batch["labels"][0, 0].item() == 10


def test_kimi_k25_vl_collate_fn_trains_thinking_but_skips_empty_think_markers():
    proc = _KimiScenarioProcessor(
        rows=[
            [
                11,
                *KIMI_ASSISTANT_HEADER_IDS,
                KIMI_THINK_OPEN_ID,
                31,
                32,
                KIMI_THINK_CLOSE_ID,
                41,
                KIMI_IM_END_ID,
            ],
            [
                12,
                *KIMI_ASSISTANT_HEADER_IDS,
                KIMI_THINK_OPEN_ID,
                KIMI_THINK_CLOSE_ID,
                51,
                KIMI_IM_END_ID,
            ],
        ]
    )
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "q1"}]},
                {
                    "role": "assistant",
                    "reasoning_content": "reasoning",
                    "content": [{"type": "text", "text": "answer"}],
                },
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "q2"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ],
        },
    ]

    batch = collate.kimi_k25_vl_collate_fn(examples, proc)

    assert _kimi_target_ids(batch, row=0) == [
        KIMI_THINK_OPEN_ID,
        31,
        32,
        KIMI_THINK_CLOSE_ID,
        41,
        KIMI_IM_END_ID,
    ]
    assert _kimi_target_ids(batch, row=1) == [51, KIMI_IM_END_ID]


def test_kimi_k25_vl_collate_fn_trains_tool_calls_but_masks_tool_responses():
    tool_call_begin = 71
    tool_name = 72
    tool_call_end = 73
    tool_response = 81
    final_answer = 91
    proc = _KimiScenarioProcessor(
        rows=[
            [
                10,
                *KIMI_ASSISTANT_HEADER_IDS,
                tool_call_begin,
                tool_name,
                tool_call_end,
                KIMI_IM_END_ID,
                tool_response,
                *KIMI_ASSISTANT_HEADER_IDS,
                final_answer,
                KIMI_IM_END_ID,
            ],
        ]
    )
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "call"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
                },
                {"role": "tool", "content": [{"type": "text", "text": "result"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    ]

    batch = collate.kimi_k25_vl_collate_fn(examples, proc)

    target_ids = _kimi_target_ids(batch)
    assert target_ids == [
        tool_call_begin,
        tool_name,
        tool_call_end,
        KIMI_IM_END_ID,
        final_answer,
        KIMI_IM_END_ID,
    ]
    assert tool_response not in target_ids


def test_kimi_k25_vl_collate_fn_masks_expanded_media_tokens():
    answer = 91
    proc = _KimiScenarioProcessor(
        rows=[
            [
                10,
                MEDIA_TOKEN_ID,
                *KIMI_ASSISTANT_HEADER_IDS,
                answer,
                KIMI_IM_END_ID,
            ],
        ],
        grid_thws=[torch.tensor([[1, 4, 4]])],
    )
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
                {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            ],
        },
    ]

    batch = collate.kimi_k25_vl_collate_fn(examples, proc)

    media_positions = (batch["input_ids"][0] == MEDIA_TOKEN_ID).nonzero(as_tuple=True)[0]
    assert media_positions.numel() == 4
    assert torch.all(batch["loss_mask"][0, media_positions] == 0)
    assert _kimi_target_ids(batch) == [answer, KIMI_IM_END_ID]
    assert batch["visual_inputs"].image_grid_thw.tolist() == [[1, 4, 4]]


def test_kimi_k25_vl_collate_fn_does_not_treat_user_marker_literal_as_assistant_turn():
    user_marker_literal_payload = 71
    assistant_marker_literal = KIMI_IM_ASSISTANT_ID
    assistant_answer = 91
    proc = _KimiScenarioProcessor(
        rows=[
            [
                10,
                KIMI_IM_ASSISTANT_ID,
                user_marker_literal_payload,
                *KIMI_ASSISTANT_HEADER_IDS,
                assistant_marker_literal,
                assistant_answer,
                KIMI_IM_END_ID,
            ],
        ]
    )
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "<|im_assistant|> leak"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "use <|im_assistant|> here"}]},
            ],
        },
    ]

    batch = collate.kimi_k25_vl_collate_fn(examples, proc)

    target_ids = _kimi_target_ids(batch)
    assert target_ids == [assistant_marker_literal, assistant_answer, KIMI_IM_END_ID]
    assert user_marker_literal_payload not in target_ids


def test_kimi_k25_vl_collate_fn_refuses_to_truncate_oversized_records():
    proc = _KimiScenarioProcessor(
        rows=[
            [
                10,
                *KIMI_ASSISTANT_HEADER_IDS,
                91,
                KIMI_IM_END_ID,
            ],
        ]
    )
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ],
        },
    ]

    with pytest.raises(ValueError, match="refuses to truncate"):
        collate.kimi_k25_vl_collate_fn(examples, proc, max_length=4)


# ---------------------------------------------------------------------------
# Gemma collates — registration and image_position_ids passthrough
# ---------------------------------------------------------------------------


def test_gemma3_processor_registered_in_collate_fns():
    """Gemma3Processor must be registered in COLLATE_FNS."""
    assert "Gemma3Processor" in collate.COLLATE_FNS


def test_gemma3_registered_fn_matches_collate_fn():
    """The registered function for Gemma3Processor is Gemma3-VL specific."""
    assert collate.COLLATE_FNS["Gemma3Processor"] is collate.gemma3_vl_collate_fn


def test_gemma4_processor_registered_in_collate_fns():
    """Gemma4Processor must be registered in COLLATE_FNS."""
    assert "Gemma4Processor" in collate.COLLATE_FNS


def test_gemma4_vl_collate_fn_declares_gemma4_boundaries(monkeypatch):
    """Gemma4 wraps Ministral3 collation with explicit Gemma4 assistant boundaries."""
    captured = {}

    def _fake_ministral3_collate_fn(examples, processor, *, assistant_mask_boundary_config=None, **kwargs):
        captured["examples"] = examples
        captured["processor"] = processor
        captured["boundary_config"] = assistant_mask_boundary_config
        captured["kwargs"] = kwargs
        return {"input_ids": torch.tensor([[1]])}

    class _Processor:
        class _Tok:
            def __call__(self, text, add_special_tokens=False):
                mapping = {
                    "<|turn>model\n": [202],
                    "<turn|>": [203],
                }
                return {"input_ids": mapping[text]}

        tokenizer = _Tok()

    examples = [{"conversation": []}]
    processor = _Processor()
    monkeypatch.setattr(gemma_vl_collate, "ministral3_collate_fn", _fake_ministral3_collate_fn)

    batch = collate.gemma4_vl_collate_fn(
        examples,
        processor,
        sequence_length=256,
        pad_to_max_length=True,
        pad_to_multiple_of=32,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=8,
    )

    assert batch["input_ids"].tolist() == [[1]]
    assert captured["examples"] == examples
    assert captured["processor"] is processor
    assert captured["boundary_config"].role_start_tokens == {"assistant": [202]}
    assert captured["boundary_config"].role_end_tokens == {"assistant": [203]}
    assert captured["kwargs"]["sequence_length"] == 256
    assert captured["kwargs"]["pad_to_max_length"] is True
    assert captured["kwargs"]["pad_to_multiple_of"] == 32
    assert captured["kwargs"]["enable_in_batch_packing"] is True
    assert captured["kwargs"]["in_batch_packing_pad_to_multiple_of"] == 8


def test_gemma4_registered_fn_matches_alias():
    """The registered function for Gemma4Processor equals the alias."""
    assert collate.COLLATE_FNS["Gemma4Processor"] is collate.gemma4_vl_collate_fn


class _Ministral3InstructionProcessor:
    """Minimal Ministral3 processor stub without HF generation mask support."""

    chat_template = "{{ messages }}"

    class _Tok:
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        added_tokens_decoder = {}
        chat_template = "{{ messages }}"

        def encode(self, text, add_special_tokens=False):
            return self(text, add_special_tokens=add_special_tokens)["input_ids"]

        def __call__(self, text, add_special_tokens=False, **kwargs):
            mapping = {
                "[/INST]": [30],
                "</s>": [2],
            }
            return {"input_ids": mapping.get(text, [99])}

    def __init__(self):
        self.tokenizer = self._Tok()

    def apply_chat_template(self, conversations, tokenize=False, **kwargs):
        if not tokenize:
            return "<s>[INST]question[/INST]answer</s>"
        return {"input_ids": torch.tensor([[1, 11, 30, 31, 2]], dtype=torch.long)}


def test_ministral3_collate_uses_declared_instruction_boundaries_without_generation_template():
    """Ministral3 templates lack HF generation blocks, so the collator must declare boundaries."""
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
    ]

    batch = collate.ministral3_collate_fn(examples, _Ministral3InstructionProcessor())

    assert batch["loss_mask"].tolist() == [[0.0, 0.0, 1.0, 1.0, 0.0]]
    assert batch["labels"].tolist() == [[-100, -100, 31, 2, -100]]


class _Gemma4ProcessorBase:
    """Minimal Gemma4Processor stub for ministral3_collate_fn tests."""

    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    class _Tok:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        eos_token = "<eos>"
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

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
        if kwargs.get("return_assistant_tokens_mask"):
            return {"input_ids": [1] * seq_len, "assistant_masks": [0, 0, 0, 1, 1, 1, 1, 0]}
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


# ---------------------------------------------------------------------------
# Nemotron Omni collate — audio and video paths
# ---------------------------------------------------------------------------

NEMO_SO_TOKEN_ID = 90
NEMO_VIDEO_TOKEN_ID = 91
NEMO_IMAGE_TOKEN_ID = 92
NEMO_IMG_START_TOKEN_ID = 93
NEMO_IMG_END_TOKEN_ID = 94


class _NemotronOmniTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    audio_token = "<so_embedding>"
    added_tokens_decoder = {}

    def __init__(self, tokenized_rows: list[list[int]] | None = None):
        self.tokenized_rows = tokenized_rows or [[5, NEMO_SO_TOKEN_ID, 6, 7]]
        self.tokenized_texts = []

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        return "user <|audio_1|> assistant"

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt", **kwargs):
        if isinstance(texts, str):
            marker_tokens = {
                "<|im_start|>assistant\n": [101],
                "<|im_end|>": [102],
            }
            return {"input_ids": marker_tokens.get(texts, [1])}
        self.tokenized_texts = list(texts)
        max_len = max(len(row) for row in self.tokenized_rows)
        out = torch.full((len(self.tokenized_rows), max_len), self.pad_token_id, dtype=torch.long)
        for i, row in enumerate(self.tokenized_rows):
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        return {"input_ids": out}

    def convert_tokens_to_ids(self, token):
        mapping = {
            "<so_embedding>": NEMO_SO_TOKEN_ID,
            "<video>": NEMO_VIDEO_TOKEN_ID,
            "<image>": NEMO_IMAGE_TOKEN_ID,
            "<img>": NEMO_IMG_START_TOKEN_ID,
            "</img>": NEMO_IMG_END_TOKEN_ID,
        }
        return mapping[token]


class _NemotronOmniProcessor:
    def __init__(self, tokenized_rows: list[list[int]] | None = None):
        self.tokenizer = _NemotronOmniTokenizer(tokenized_rows)
        self.image_processor = type("ImageProcessor", (), {"max_num_tiles": 4})()
        self.calls = []

    def apply_chat_template(self, conversations, tokenize=False, **kwargs):
        self.calls.append(("apply_chat_template", conversations, kwargs))
        return "video prompt"

    def __call__(self, **kwargs):
        self.calls.append(("processor", kwargs))
        if "videos" in kwargs:
            return {
                "input_ids": torch.tensor([[1, NEMO_VIDEO_TOKEN_ID, 7, 8]], dtype=torch.long),
                "pixel_values_videos": torch.ones(1, 3, 16, 16),
            }
        return {"input_ids": torch.tensor(self.tokenizer.tokenized_rows, dtype=torch.long)}


def _zero_assistant_loss_mask(
    example,
    input_ids,
    processor,
    skipped_tokens,
    **kwargs,
):  # noqa: ARG001 - test helper signature
    return torch.zeros(int(input_ids.shape[0]), dtype=torch.float32)


def test_nemotron_omni_collate_replaces_audio_placeholder_with_computed_token_count(monkeypatch):
    import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils

    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)
    monkeypatch.setattr(omni_utils, "compute_mel_features", lambda waveform, sampling_rate=16000: torch.ones(9, 128))

    proc = _NemotronOmniProcessor(tokenized_rows=[[5, NEMO_SO_TOKEN_ID, 6, 7]])
    examples = [
        {
            "conversation": [
                {"role": "user", "content": "<|audio_1|> What is spoken?"},
                {"role": "assistant", "content": "hello"},
            ],
            "audio": ([0.0, 0.1, -0.1], 16000),
        }
    ]

    batch = collate.nemotron_omni_collate_fn(examples, proc)

    assert "<so_embedding>" in proc.tokenizer.tokenized_texts[0]
    assert batch["input_ids"].tolist() == [[5, NEMO_SO_TOKEN_ID, NEMO_SO_TOKEN_ID, 6, 7]]
    assert batch["sound_clips"].shape == (1, 9, 128)
    assert batch["sound_length"].tolist() == [9]
    assert batch["visual_inputs"] is None


def test_nemotron_omni_collate_loads_audio_path_when_no_placeholder_exists(monkeypatch):
    import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils

    loaded_paths = []
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)
    monkeypatch.setattr(
        omni_utils,
        "load_audio",
        lambda path, target_sr=16000: loaded_paths.append((path, target_sr)) or [0.0, 0.1],
    )
    monkeypatch.setattr(omni_utils, "compute_mel_features", lambda waveform, sampling_rate=16000: torch.ones(1, 128))

    proc = _NemotronOmniProcessor(tokenized_rows=[[5, 6, 7]])
    examples = [
        {
            "conversation": [
                {"role": "user", "content": "What is spoken?"},
                {"role": "assistant", "content": "hello"},
            ],
            "audio_path": "/tmp/audio.wav",
            "max_audio_duration": 1.0,
        }
    ]

    batch = collate.nemotron_omni_collate_fn(examples, proc)

    assert loaded_paths == [("/tmp/audio.wav", 16000)]
    assert batch["input_ids"].tolist() == [[5, NEMO_SO_TOKEN_ID, 6, 7]]
    assert batch["sound_clips"].shape == (1, 1, 128)
    assert batch["sound_length"].tolist() == [1]


def test_nemotron_omni_collate_video_path_wraps_visual_inputs(monkeypatch):
    import megatron.bridge.models.nemotron_vl.nemotron_vl_utils as vl_utils

    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)
    monkeypatch.setattr(vl_utils, "maybe_path_or_url_to_data_urls", lambda *args, **kwargs: (["frame-1"], {"fps": 1}))
    monkeypatch.setattr(vl_utils, "pil_image_from_base64", lambda data_url: f"decoded-{data_url}")

    proc = _NemotronOmniProcessor()
    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": "/tmp/video.mp4"},
                        {"type": "text", "text": "What happens?"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "an event"}]},
            ]
        }
    ]

    batch = collate.nemotron_omni_collate_fn(examples, proc)

    processor_call = [call for call in proc.calls if call[0] == "processor"][0][1]
    assert processor_call["videos"] == [["decoded-frame-1"]]
    assert processor_call["videos_kwargs"] == {"video_metadata": {"fps": 1}}
    assert batch["input_ids"].tolist() == [[1, NEMO_IMAGE_TOKEN_ID, 7, 8]]
    assert batch["visual_inputs"].pixel_values.dtype == torch.bfloat16
    assert batch["visual_inputs"].pixel_values.shape == (1, 3, 16, 16)
