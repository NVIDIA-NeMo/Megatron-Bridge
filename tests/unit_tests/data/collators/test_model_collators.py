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

from types import SimpleNamespace

import pytest
import torch

import megatron.bridge.models.gemma_vl.data.collate_fn as gemma_vl_collate
import megatron.bridge.models.glm_vl.data.collate_fn as glm_vl_collate
import megatron.bridge.models.kimi_vl.data.collate_fn as kimi_collate
import megatron.bridge.models.ministral3.data.collate_fn as ministral3_collate
import megatron.bridge.models.nemotron_omni.data.collate_fn as nemotron_omni_collate
import megatron.bridge.models.nemotron_vl.data.collate_fn as nemotron_vl_collate
import megatron.bridge.models.qwen_audio.data.collate_fn as qwen_audio_collate
import megatron.bridge.models.qwen_vl.data.collate_fn as qwen_vl_collate
from megatron.bridge.data.collators.registry import model_collate_required_for_all_examples, resolve_model_collate
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


pytestmark = pytest.mark.unit

collate = SimpleNamespace(
    gemma3_vl_collate_fn=gemma_vl_collate.gemma3_vl_collate_fn,
    gemma4_vl_collate_fn=gemma_vl_collate.gemma4_vl_collate_fn,
    glm4v_collate_fn=glm_vl_collate.glm4v_collate_fn,
    kimi_k25_vl_collate_fn=kimi_collate.kimi_k25_vl_collate_fn,
    ministral3_collate_fn=ministral3_collate.ministral3_collate_fn,
    nemotron_nano_v2_vl_collate_fn=nemotron_vl_collate.nemotron_nano_v2_vl_collate_fn,
    nemotron_omni_collate_fn=nemotron_omni_collate.nemotron_omni_collate_fn,
    qwen2_5_collate_fn=qwen_vl_collate.qwen2_5_collate_fn,
    qwen2_audio_collate_fn=qwen_audio_collate.qwen2_audio_collate_fn,
)


def test_model_collate_registry_rejects_unknown_processor():
    with pytest.raises(ValueError, match="No VLM collate function"):
        resolve_model_collate("UnknownProcessor")


def test_only_nemotron_omni_requires_model_collate_for_all_examples():
    assert model_collate_required_for_all_examples("NemotronH_Nano_Omni_Reasoning_V3Processor")
    assert not model_collate_required_for_all_examples("Qwen3VLProcessor")
    assert not model_collate_required_for_all_examples("UnknownProcessor")


def test_vlm_collate_keeps_qwen_vl_registration():
    assert resolve_model_collate("Qwen2_5_VLProcessor") is collate.qwen2_5_collate_fn


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
                "<|im_end|>\n": [103, 104],
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


def test_gemma3_vl_collate_forwards_shared_tools_to_chat_template():
    proc = _DummyProcessor()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    examples = [
        {
            "conversation": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "Sunny."},
            ],
            "tools": tools,
        }
    ]

    collate.gemma3_vl_collate_fn(examples, proc)

    assert proc.template_kwargs[0]["tools"] == tools


def test_gemma3_packed_collate_processes_unpadded_rows_directly():
    processor = _DummyProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": text}]}]}
        for text in ("first", "second")
    ]

    batch = collate.gemma3_vl_collate_fn(
        examples,
        processor,
        sequence_length=8,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    processor_calls = [kwargs for kwargs in processor.template_kwargs if kwargs.get("return_tensors") == "pt"]
    assert [kwargs["padding"] for kwargs in processor_calls] == [False, False]
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 1, 2, 3, 0]]
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert batch["visual_inputs"].pixel_values.shape[0] == 2


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


def test_qwen2_audio_collate_fn_forwards_source_sampling_rate(monkeypatch):
    class _AudioProcessor:
        class _Tok:
            pad_token_id = 0
            padding_side = "right"
            added_tokens_decoder = {}

            def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
                return {"input_ids": [1, 2]}

        def __init__(self):
            self.tokenizer = self._Tok()
            self.sampling_rate = None

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):  # noqa: ARG002
            return "dummy"

        def __call__(
            self,
            text=None,
            audio=None,
            *,
            sampling_rate,
            return_tensors="pt",
            padding=True,
        ):
            self.sampling_rate = sampling_rate
            n = len(text)
            return {
                "input_ids": torch.tensor([[1, 2, 3]] * n),
                "input_features": torch.randn(n, 80, 16),
                "feature_attention_mask": torch.ones(n, 16),
            }

    monkeypatch.setattr(qwen_audio_collate, "gather_assistant_text_segments", lambda ex: ["dummy"])

    processor = _AudioProcessor()
    examples = [
        {
            "conversation": [{"role": "user", "content": [{"type": "text", "text": "transcribe"}]}],
            "audio": (torch.zeros(8_000), 8_000),
        }
    ]

    collate.qwen2_audio_collate_fn(examples, processor)

    assert processor.sampling_rate == 8_000


@pytest.mark.parametrize(
    ("audio_values", "error"),
    [
        ([(torch.zeros(1), 8_000), (torch.zeros(1), 16_000)], "single sampling rate"),
        ([(torch.zeros(1), 8_000), torch.zeros(1)], "known and unknown sampling rates"),
    ],
)
def test_qwen2_audio_collate_fn_rejects_ambiguous_sampling_rates(audio_values, error):
    class _AudioProcessor:
        class _Tok:
            pad_token_id = 0
            padding_side = "right"
            added_tokens_decoder = {}

        def __init__(self):
            self.tokenizer = self._Tok()

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):  # noqa: ARG002
            return "dummy"

        def __call__(self, **kwargs):  # noqa: ARG002
            raise AssertionError("ambiguous audio rates must be rejected before processor invocation")

    examples = [
        {
            "conversation": [{"role": "user", "content": [{"type": "text", "text": "transcribe"}]}],
            "audio": audio,
        }
        for audio in audio_values
    ]

    with pytest.raises(ValueError, match=error):
        collate.qwen2_audio_collate_fn(examples, _AudioProcessor())


def test_qwen2_audio_collate_fn_preserves_raw_audio_compatibility(monkeypatch):
    class _AudioProcessor:
        class _Tok:
            pad_token_id = 0
            padding_side = "right"
            added_tokens_decoder = {}

            def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
                return {"input_ids": [1, 2]}

        def __init__(self):
            self.tokenizer = self._Tok()
            self.sampling_rate = "unset"

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):  # noqa: ARG002
            return "dummy"

        def __call__(
            self,
            text=None,
            audio=None,
            *,
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
        ):
            self.sampling_rate = sampling_rate
            n = len(text)
            return {
                "input_ids": torch.tensor([[1, 2, 3]] * n),
                "input_features": torch.randn(n, 80, 16),
                "feature_attention_mask": torch.ones(n, 16),
            }

    monkeypatch.setattr(qwen_audio_collate, "gather_assistant_text_segments", lambda ex: ["dummy"])

    processor = _AudioProcessor()
    examples = [
        {
            "conversation": [{"role": "user", "content": [{"type": "text", "text": "transcribe"}]}],
            "audio": torch.zeros(8_000),
        }
    ]

    collate.qwen2_audio_collate_fn(examples, processor)

    assert processor.sampling_rate is None


def test_qwen2_audio_collate_fn_rejects_non_native_whisper_sampling_rate():
    from transformers import WhisperFeatureExtractor

    class _AudioProcessor:
        class _Tok:
            pad_token_id = 0
            padding_side = "right"
            added_tokens_decoder = {}

        def __init__(self):
            self.tokenizer = self._Tok()
            self.feature_extractor = WhisperFeatureExtractor()

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):  # noqa: ARG002
            return "dummy"

        def __call__(
            self,
            text=None,
            audio=None,
            *,
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
        ):
            return self.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                return_tensors=return_tensors,
            )

    examples = [
        {
            "conversation": [{"role": "user", "content": [{"type": "text", "text": "transcribe"}]}],
            "audio": (torch.zeros(8_000).numpy(), 8_000),
        }
    ]

    with pytest.raises(ValueError, match="8000"):
        collate.qwen2_audio_collate_fn(examples, _AudioProcessor())


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

    with pytest.warns(UserWarning, match="defers in-batch packing to audio_lm_step"):
        batch = collate.qwen2_audio_collate_fn(
            examples, _AudioProcessor(), sequence_length=128, enable_in_batch_packing=True
        )

    assert batch["input_ids"].shape == (2, 128)
    assert "cu_seqlens" not in batch
    assert "max_seqlen" not in batch
    assert "cu_seqlens_q" not in batch
    assert "max_seqlen_q" not in batch


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


def test_qwen2_5_collate_fn_normalizes_video_path_without_mutating_example(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)
    seen_conversations = []

    class _PathCheckingProcessor(_DummyProcessor):
        def apply_chat_template(self, conversation, tokenize=False, **kwargs):
            assert conversation[0]["content"][0] == {"type": "video", "video": "/videos/clip.mp4"}
            return super().apply_chat_template(conversation, tokenize=tokenize, **kwargs)

    def _fake_pvi(conversation):
        seen_conversations.append(conversation)
        assert conversation[0]["content"][0] == {"type": "video", "video": "/videos/clip.mp4"}
        return (None, [[object()]])

    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", _fake_pvi)
    example = {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": "/videos/clip.mp4"},
                    {"type": "text", "text": "What happens?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "An event."}]},
        ]
    }

    batch = collate.qwen2_5_collate_fn([example], _PathCheckingProcessor())

    assert seen_conversations
    assert batch["visual_inputs"].pixel_values_videos is not None
    assert example["conversation"][0]["content"][0] == {"type": "video", "path": "/videos/clip.mp4"}


def test_qwen2_5_collate_fn_preserves_attention_mask_for_mixed_image_text_batch(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)

    class _PadAwareProcessor:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        class _Tok:
            pad_token_id = 99
            pad_token = "<pad>"
            padding_side = "left"
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
        chat_template = "<|im_start|>user\n{{ content }}<|im_end|>\n<|im_start|>assistant\n{{ content }}<|im_end|>\n"

        class _Tok:
            pad_token_id = 0
            pad_token = "<pad>"
            added_tokens_decoder = {103: "<|im_end|>"}
            chat_template = (
                "<|im_start|>user\n{{ content }}<|im_end|>\n<|im_start|>assistant\n{{ content }}<|im_end|>\n"
            )

            def __call__(self, text, add_special_tokens=False):
                mapping = {
                    "<|im_start|>assistant\n": [102],
                    "<|im_end|>": [103],
                    "<|im_end|>\n": [103, 104],
                }
                return {"input_ids": mapping.get(text, [42])}

        def __init__(self):
            self.tokenizer = self._Tok()

        def apply_chat_template(self, conversation, tokenize=False, **kwargs):
            return "rendered"

        def __call__(self, text=None, padding=True, return_tensors="pt", **kwargs):
            return {"input_ids": torch.tensor([[100, 7, 103, 104, 102, 3, 4, 103, 104]])}

    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
    ]

    batch = collate.qwen2_5_collate_fn(examples, _ChatMLProcessor())

    assert batch["loss_mask"].tolist() == [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, 3, 4, 103, 104, -100]]


def test_qwen2_5_collate_fn_packs_vlm_batch(monkeypatch):
    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)
    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", lambda conv: (None, None))

    class _PackableProcessor:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        class _Tok:
            pad_token_id = 99
            pad_token = "<pad>"
            padding_side = "left"
            added_tokens_decoder = {}
            chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1]}

        def __init__(self):
            self.tokenizer = self._Tok()
            self.padding_values = []

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
            self.padding_values.append(padding)
            texts = text if isinstance(text, list) else [text]
            lengths = [3 if "short" in item else 5 for item in texts]
            max_len = max(lengths)
            input_ids = torch.full((len(texts), max_len), self.tokenizer.pad_token_id)
            attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)
            for row, length in enumerate(lengths):
                if self.tokenizer.padding_side == "left":
                    input_ids[row, max_len - length :] = torch.arange(1, length + 1)
                    attention_mask[row, max_len - length :] = 1
                else:
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

    processor = _PackableProcessor()
    batch = collate.qwen2_5_collate_fn(
        examples,
        processor,
        sequence_length=16,
        pad_to_max_length=True,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 0, 0]]
    assert batch["input_ids"].shape[1] != 16
    assert processor.padding_values == [False, False]
    assert processor.tokenizer.padding_side == "left"
    assert batch["attention_mask"] is None
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 8]
    assert batch["cu_seqlens_kv"].tolist() == [0, 3, 8]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 12]
    assert batch["cu_seqlens_kv_padded"].tolist() == [0, 4, 12]
    assert batch["max_seqlen_q"].item() == 8
    assert batch["max_seqlen_kv"].item() == 8
    assert "cu_seqlens" not in batch
    assert "cu_seqlens_unpadded" not in batch
    assert batch["visual_inputs"] is not None


def test_qwen2_5_packed_collate_preserves_flat_media_and_video_timing(monkeypatch):
    image_a, image_b, video = object(), object(), object()

    def _process_vision_info(conversation):
        marker = conversation[0]["content"][0]["text"]
        if marker == "images":
            return [image_a, image_b], None
        return None, [video]

    class _MediaProcessor:
        class _Tokenizer:
            padding_side = "left"
            pad_token_id = 0

        def __init__(self):
            self.tokenizer = self._Tokenizer()

        def apply_chat_template(self, conversation, **kwargs):
            return conversation[0]["content"][0]["text"]

        def __call__(self, *, text, padding, return_tensors, images=None, videos=None, **kwargs):
            assert padding is False
            output = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }
            if images is not None:
                assert images == [image_a, image_b]
                output["pixel_values"] = torch.ones(2, 3, 4, 4)
                output["image_grid_thw"] = torch.tensor([[1, 2, 2], [1, 2, 2]])
            if videos is not None:
                assert videos == [video]
                output["pixel_values_videos"] = torch.ones(1, 3, 4, 4)
                output["video_grid_thw"] = torch.tensor([[2, 2, 2]])
                output["second_per_grid_ts"] = torch.tensor([0.5])
            return output

    monkeypatch.setattr(qwen_vl_collate, "HAVE_QWEN_VL_UTILS", True)
    monkeypatch.setattr(qwen_vl_collate, "process_vision_info", _process_vision_info)
    monkeypatch.setattr(
        qwen_vl_collate, "extract_skipped_token_ids", lambda processor: torch.empty(0, dtype=torch.long)
    )
    monkeypatch.setattr(qwen_vl_collate, "assistant_mask_boundary_config_from_markers", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        qwen_vl_collate,
        "build_assistant_loss_mask",
        lambda example, input_ids, *args, **kwargs: torch.ones_like(input_ids, dtype=torch.float32),
    )
    examples = [
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "images"}]}]},
        {"conversation": [{"role": "user", "content": [{"type": "text", "text": "video"}]}]},
    ]

    batch = qwen_vl_collate.qwen2_5_collate_fn(
        examples,
        _MediaProcessor(),
        sequence_length=8,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    visual_inputs = batch["visual_inputs"]
    assert visual_inputs.pixel_values.shape == (2, 3, 4, 4)
    assert visual_inputs.pixel_values_videos.shape == (1, 3, 4, 4)
    assert visual_inputs.second_per_grid_ts.tolist() == [0.5]


def test_glm4v_collate_packs_mm_token_type_ids_and_restores_padding(monkeypatch):
    class _GlmProcessor:
        class _Tokenizer:
            padding_side = "left"

        def __init__(self):
            self.tokenizer = self._Tokenizer()
            self.padding_values = []

        def apply_chat_template(self, conversations, **kwargs):
            assert self.tokenizer.padding_side == "right"
            self.padding_values.append(kwargs["padding"])
            marker = conversations[0][0]["content"]
            if marker == "short":
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                    "mm_token_type_ids": torch.tensor([[0, 1]]),
                }
            return {
                "input_ids": torch.tensor([[3, 4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                "mm_token_type_ids": torch.tensor([[0, 2, 2, 0]]),
            }

    monkeypatch.setattr(
        glm_vl_collate, "extract_skipped_token_ids", lambda processor: torch.empty(0, dtype=torch.long)
    )
    monkeypatch.setattr(glm_vl_collate, "infer_assistant_mask_boundary_config", lambda processor: None)
    monkeypatch.setattr(
        glm_vl_collate,
        "build_assistant_loss_mask",
        lambda example, input_ids, *args, **kwargs: (input_ids != 0).to(dtype=torch.float32),
    )
    examples = [
        {"conversation": [{"role": "user", "content": "short"}]},
        {"conversation": [{"role": "user", "content": "long"}]},
    ]
    processor = _GlmProcessor()

    batch = glm_vl_collate.glm4v_collate_fn(
        examples,
        processor,
        sequence_length=8,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 0, 0, 3, 4, 5, 6]]
    assert batch["visual_inputs"].mm_token_type_ids.tolist() == [[0, 1, 0, 0, 0, 2, 2, 0]]
    assert processor.padding_values == [False, False]
    assert processor.tokenizer.padding_side == "left"


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


def test_kimi_k25_packed_collate_builds_direct_rows():
    processor = _KimiDummyProcessor(include_image=False)
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": text}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
        for text in ("first", "second")
    ]

    batch = collate.kimi_k25_vl_collate_fn(
        examples,
        processor,
        sequence_length=8,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["input_ids"].shape == (1, 16)
    assert batch["cu_seqlens_q"].tolist() == [0, 5, 10]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 8, 16]


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
    """Gemma3Processor must resolve to its model-owned collator."""
    assert resolve_model_collate("Gemma3Processor") is collate.gemma3_vl_collate_fn


def test_gemma3_registered_fn_matches_collate_fn():
    """The registered function for Gemma3Processor is Gemma3-VL specific."""
    assert resolve_model_collate("Gemma3Processor") is collate.gemma3_vl_collate_fn


def test_gemma4_processor_registered_in_collate_fns():
    """Gemma4Processor must resolve to its model-owned collator."""
    assert resolve_model_collate("Gemma4Processor") is collate.gemma4_vl_collate_fn


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
    assert resolve_model_collate("Gemma4Processor") is collate.gemma4_vl_collate_fn


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
        self.padding_values = []

    def apply_chat_template(self, conversations, tokenize=False, **kwargs):
        if not tokenize:
            return "<s>[INST]question[/INST]answer</s>"
        if kwargs.get("return_tensors") == "pt":
            self.padding_values.append(kwargs["padding"])
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


def test_ministral3_packed_collate_processes_unpadded_rows_directly():
    processor = _Ministral3InstructionProcessor()
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": text}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
        for text in ("first", "second")
    ]

    batch = collate.ministral3_collate_fn(
        examples,
        processor,
        sequence_length=8,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert processor.padding_values == [False, False]
    assert batch["input_ids"].shape == (1, 16)
    assert batch["cu_seqlens_q"].tolist() == [0, 5, 10]


def test_ministral3_nonpacked_collate_supervises_each_rows_last_real_token(monkeypatch):
    class _MixedLengthProcessor(_Ministral3InstructionProcessor):
        def apply_chat_template(self, conversations, tokenize=False, **kwargs):
            if not tokenize:
                return "rendered"
            return {
                "input_ids": torch.tensor([[1, 11, 30, 31, 2], [1, 30, 2, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            }

    monkeypatch.setattr(
        ministral3_collate, "extract_skipped_token_ids", lambda processor: torch.empty(0, dtype=torch.long)
    )
    monkeypatch.setattr(ministral3_collate, "_has_generation_chat_template", lambda processor: True)
    monkeypatch.setattr(ministral3_collate, "infer_assistant_mask_boundary_config", lambda processor: None)
    monkeypatch.setattr(
        ministral3_collate,
        "build_assistant_loss_mask",
        lambda example, input_ids, *args, **kwargs: torch.zeros_like(input_ids, dtype=torch.float32),
    )
    examples = [{"conversation": [{"role": "user", "content": text}]} for text in ("long", "short")]

    batch = ministral3_collate.ministral3_collate_fn(examples, _MixedLengthProcessor())

    assert batch["loss_mask"].tolist() == [[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
    assert batch["labels"][1].tolist() == [-100, 2, -100, -100, -100]


def test_nemotron_vl_video_collate_rejects_in_batch_packing():
    examples = [{"conversation": [{"role": "user", "content": [{"type": "video", "path": "video.mp4"}]}]}]

    with pytest.raises(ValueError, match="does not support in-batch packing"):
        collate.nemotron_nano_v2_vl_collate_fn(examples, object(), enable_in_batch_packing=True)


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
NEMO_SO_START_TOKEN_ID = 95
NEMO_SO_END_TOKEN_ID = 96


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
                "<|im_end|>\n": [102, 103],
            }
            return {"input_ids": marker_tokens.get(texts, [1])}
        self.tokenized_texts = list(texts)
        max_len = max(len(row) for row in self.tokenized_rows)
        out = torch.full((len(self.tokenized_rows), max_len), self.pad_token_id, dtype=torch.long)
        for i, row in enumerate(self.tokenized_rows):
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask = torch.zeros_like(out)
        for i, row in enumerate(self.tokenized_rows):
            attention_mask[i, : len(row)] = 1
        return {"input_ids": out, "attention_mask": attention_mask}

    def convert_tokens_to_ids(self, token):
        mapping = {
            "<so_embedding>": NEMO_SO_TOKEN_ID,
            "<video>": NEMO_VIDEO_TOKEN_ID,
            "<image>": NEMO_IMAGE_TOKEN_ID,
            "<img>": NEMO_IMG_START_TOKEN_ID,
            "</img>": NEMO_IMG_END_TOKEN_ID,
            "<so_start>": NEMO_SO_START_TOKEN_ID,
            "<so_end>": NEMO_SO_END_TOKEN_ID,
        }
        return mapping[token]


class _NemotronOmniProcessor:
    def __init__(
        self,
        tokenized_rows: list[list[int]] | None = None,
    ):
        self.tokenizer = _NemotronOmniTokenizer(tokenized_rows)
        self.image_processor = type("DynamicImageProcessor", (), {"max_num_patches": 64})()
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
        input_ids = torch.tensor(self.tokenizer.tokenized_rows, dtype=torch.long)
        output = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.long),
        }
        if kwargs.get("images") is not None:
            num_patches = [1] * len(kwargs["images"])
            output["num_patches"] = torch.tensor(num_patches, dtype=torch.long)
            output["pixel_values"] = torch.ones(sum(num_patches), 3, 32, 32)
        return output


def _zero_assistant_loss_mask(
    example,
    input_ids,
    processor,
    skipped_tokens,
    **kwargs,
):  # noqa: ARG001 - test helper signature
    return torch.zeros(int(input_ids.shape[0]), dtype=torch.float32)


def test_nemotron_omni_collate_keeps_chatml_turn_end_token():
    proc = _NemotronOmniProcessor(
        tokenized_rows=[
            [
                100,
                NEMO_IMG_START_TOKEN_ID,
                NEMO_IMAGE_TOKEN_ID,
                NEMO_IMG_END_TOKEN_ID,
                10,
                102,
                103,
                101,
                21,
                22,
                102,
                103,
            ]
        ]
    )
    proc.tokenizer.added_tokens_decoder = {102: "<|im_end|>"}
    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "receipt"},
                        {"type": "text", "text": "question"},
                    ],
                },
                {"role": "assistant", "content": "answer"},
            ],
        }
    ]

    batch = collate.nemotron_omni_collate_fn(examples, proc)

    assert batch["loss_mask"][0, -5:].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert batch["labels"][0, -5:].tolist() == [21, 22, 102, 103, -100]


def test_nemotron_omni_collate_rejects_unsupported_visual_keys():
    proc = _NemotronOmniProcessor(tokenized_rows=[[5, 6]])

    with pytest.raises(ValueError, match=r"visual_keys must be exactly \('pixel_values',\)"):
        collate.nemotron_omni_collate_fn(
            [{"conversation": [{"role": "user", "content": "text"}]}],
            proc,
            visual_keys=("pixel_values", "image_sizes"),
        )


def test_nemotron_omni_hf_collate_packs_heterogeneous_image_rows_at_post_merge_boundaries(monkeypatch):
    processor = _DynamicNemotronOmniProcessor()
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)

    batch = collate.nemotron_omni_collate_fn(
        _heterogeneous_nemotron_examples(),
        processor,
        enable_in_batch_packing=True,
        sequence_length=24,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["input_ids"].tolist() == [
        [
            10,
            11,
            2,
            0,
            20,
            NEMO_IMG_START_TOKEN_ID,
            NEMO_IMAGE_TOKEN_ID,
            NEMO_IMG_END_TOKEN_ID,
            21,
            2,
            0,
            0,
            30,
            NEMO_IMG_START_TOKEN_ID,
            NEMO_IMAGE_TOKEN_ID,
            NEMO_IMG_END_TOKEN_ID,
            32,
            NEMO_IMG_START_TOKEN_ID,
            NEMO_IMAGE_TOKEN_ID,
            NEMO_IMG_END_TOKEN_ID,
            31,
            2,
        ]
    ]
    assert batch["attention_mask"] is None
    assert batch["num_image_tiles"].tolist() == [1, 2, 2]
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 9, 21]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 12, 24]
    assert batch["max_seqlen_q"].item() == 12
    assert batch["total_tokens"] == 24
    assert batch["loss_mask"].nonzero(as_tuple=False).tolist() == [[0, 0], [0, 7], [0, 19]]
    assert batch["labels"][0, 0].item() == 11
    assert batch["labels"][0, 7].item() == 21
    assert batch["labels"][0, 19].item() == 31
    assert torch.all(batch["loss_mask"][0, [3, 10, 11]] == 0)
    assert torch.all(batch["labels"][0, [3, 10, 11]] == IGNORE_INDEX)


def test_nemotron_omni_hf_collate_rejects_packed_post_merge_total_overflow(monkeypatch):
    processor = _DynamicNemotronOmniProcessor()
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)

    with pytest.raises(
        ValueError,
        match=r"aligned lengths \[4, 8, 12\], and total 24 with sequence_length=23",
    ):
        collate.nemotron_omni_collate_fn(
            _heterogeneous_nemotron_examples(),
            processor,
            enable_in_batch_packing=True,
            sequence_length=23,
            in_batch_packing_pad_to_multiple_of=4,
        )


def test_nemotron_omni_hf_collate_fixed_packing_matches_pipeline_parallel_merge_width(monkeypatch):
    from types import SimpleNamespace

    from megatron.core.models.multimodal.llava_model import LLaVAModel

    from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params

    processor = _DynamicNemotronOmniProcessor()
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)
    batch = collate.nemotron_omni_collate_fn(
        _heterogeneous_nemotron_examples(),
        processor,
        enable_in_batch_packing=True,
        sequence_length=32,
        pad_to_max_length=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["input_ids"].shape == (1, 30)
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 9, 21]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 12, 32]
    assert batch["total_tokens"] == 32
    assert get_packed_seq_params(batch).seq_idx.shape == (1, 32)

    hidden_size = 4
    pp_model = SimpleNamespace(
        add_decoder=True,
        pre_process=True,
        post_process=True,
        _language_is_pipeline_parallel=True,
        _language_max_sequence_length=32,
        context_parallel_lm=1,
    )
    final_embedding, final_labels, final_loss_mask = LLaVAModel._preprocess_data(
        pp_model,
        image_embeddings=torch.ones(1, 5, hidden_size),
        language_embeddings=torch.ones(1, batch["input_ids"].shape[1], hidden_size),
        input_ids=batch["input_ids"],
        loss_mask=batch["loss_mask"],
        labels=batch["labels"],
        use_inference_kv_cache=False,
        inference_context=None,
        image_token_index=NEMO_IMAGE_TOKEN_ID,
        num_image_tiles=batch["num_image_tiles"],
        is_packed_dynamic_res=True,
    )

    assert final_embedding.shape == (32, 1, hidden_size)
    assert final_labels.shape == final_loss_mask.shape == (1, 32)


def test_nemotron_omni_hf_collate_fixed_packing_rejects_misaligned_sequence_length(monkeypatch):
    processor = _DynamicNemotronOmniProcessor()
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)

    with pytest.raises(
        ValueError,
        match=r"sequence_length to be divisible.*got 30 and 4",
    ):
        collate.nemotron_omni_collate_fn(
            _heterogeneous_nemotron_examples(),
            processor,
            enable_in_batch_packing=True,
            sequence_length=30,
            pad_to_max_length=True,
            in_batch_packing_pad_to_multiple_of=4,
        )


def test_nemotron_omni_collate_replaces_audio_placeholder_with_computed_token_count(monkeypatch):
    import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils

    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)
    monkeypatch.setattr(
        omni_utils,
        "compute_mel_features",
        lambda waveform, sampling_rate=16000, num_mel_bins=128: torch.ones(9, num_mel_bins),
    )

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
    assert batch["input_ids"].tolist() == [
        [5, NEMO_SO_START_TOKEN_ID, NEMO_SO_TOKEN_ID, NEMO_SO_TOKEN_ID, NEMO_SO_END_TOKEN_ID, 6, 7]
    ]
    assert batch["sound_clips"].shape == (1, 9, 128)
    assert batch["sound_length"].tolist() == [9]
    assert batch["visual_inputs"] is None


def test_nemotron_omni_collate_does_not_duplicate_existing_audio_framing(monkeypatch):
    import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils

    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)
    monkeypatch.setattr(
        omni_utils,
        "compute_mel_features",
        lambda waveform, sampling_rate=16000, num_mel_bins=128: torch.ones(9, num_mel_bins),
    )
    proc = _NemotronOmniProcessor(
        tokenized_rows=[[5, NEMO_SO_START_TOKEN_ID, NEMO_SO_TOKEN_ID, NEMO_SO_END_TOKEN_ID, 6, 7]]
    )
    examples = [
        {
            "conversation": [{"role": "user", "content": "framed audio"}],
            "audio": ([0.0, 0.1, -0.1], 16000),
        }
    ]

    batch = collate.nemotron_omni_collate_fn(examples, proc)

    assert batch["input_ids"].tolist() == [
        [5, NEMO_SO_START_TOKEN_ID, NEMO_SO_TOKEN_ID, NEMO_SO_TOKEN_ID, NEMO_SO_END_TOKEN_ID, 6, 7]
    ]


@pytest.mark.parametrize(
    "tokenized_row",
    (
        [5, NEMO_SO_START_TOKEN_ID, NEMO_SO_TOKEN_ID, 6, 7],
        [5, NEMO_SO_TOKEN_ID, NEMO_SO_END_TOKEN_ID, 6, 7],
    ),
)
def test_nemotron_omni_collate_rejects_partially_framed_audio_placeholder(monkeypatch, tokenized_row):
    import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils

    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)
    monkeypatch.setattr(
        omni_utils,
        "compute_mel_features",
        lambda waveform, sampling_rate=16000, num_mel_bins=128: torch.ones(9, num_mel_bins),
    )
    proc = _NemotronOmniProcessor(tokenized_rows=[tokenized_row])
    examples = [
        {
            "conversation": [{"role": "user", "content": "malformed audio"}],
            "audio": ([0.0, 0.1, -0.1], 16000),
        }
    ]

    with pytest.raises(ValueError, match="must have both <so_start> and <so_end>"):
        collate.nemotron_omni_collate_fn(examples, proc)


def test_nemotron_omni_collate_rejects_disjoint_audio_placeholder_runs(monkeypatch):
    import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils

    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)
    monkeypatch.setattr(
        omni_utils,
        "compute_mel_features",
        lambda waveform, sampling_rate=16000, num_mel_bins=128: torch.ones(9, num_mel_bins),
    )
    proc = _NemotronOmniProcessor(tokenized_rows=[[5, NEMO_SO_TOKEN_ID, 6, NEMO_SO_TOKEN_ID, 7]])
    examples = [
        {
            "conversation": [{"role": "user", "content": "two audio placeholders"}],
            "audio": ([0.0, 0.1, -0.1], 16000),
        }
    ]

    with pytest.raises(ValueError, match="one contiguous audio placeholder block"):
        collate.nemotron_omni_collate_fn(examples, proc)


def test_nemotron_omni_collate_rejects_mixed_audio_and_no_audio_samples():
    proc = _NemotronOmniProcessor(tokenized_rows=[[5, NEMO_SO_TOKEN_ID, 6], [7, 8, 9]])
    examples = [
        {
            "conversation": [{"role": "user", "content": "audio"}],
            "audio": ([0.0, 0.1], 16000),
        },
        {"conversation": [{"role": "user", "content": "text"}]},
    ]

    with pytest.raises(ValueError, match="does not support mixing audio and no-audio samples"):
        collate.nemotron_omni_collate_fn(examples, proc)


def test_nemotron_omni_collate_loads_audio_path_when_no_placeholder_exists(monkeypatch):
    import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils

    loaded_paths = []
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)
    monkeypatch.setattr(
        omni_utils,
        "load_audio",
        lambda path, target_sr=16000: loaded_paths.append((path, target_sr)) or [0.0, 0.1],
    )
    monkeypatch.setattr(
        omni_utils,
        "compute_mel_features",
        lambda waveform, sampling_rate=16000, num_mel_bins=128: torch.ones(1, num_mel_bins),
    )

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
    assert batch["input_ids"].tolist() == [[5, NEMO_SO_START_TOKEN_ID, NEMO_SO_TOKEN_ID, NEMO_SO_END_TOKEN_ID, 6, 7]]
    assert batch["sound_clips"].shape == (1, 1, 128)
    assert batch["sound_length"].tolist() == [1]


def test_nemotron_omni_video_collate_requires_temporal_embedder():
    proc = _NemotronOmniProcessor()
    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Watch this."},
                        {"type": "video", "path": "/tmp/video.mp4"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "an event"}]},
            ]
        }
    ]

    with pytest.raises(ValueError, match="requires use_temporal_video_embedder=True"):
        collate.nemotron_omni_collate_fn(examples, proc)


@pytest.mark.parametrize("use_temporal_video_embedder", [False, True])
def test_nemotron_omni_collate_rejects_legacy_fixed_tile_processor(use_temporal_video_embedder):
    proc = _NemotronOmniProcessor(tokenized_rows=[[5, 6]])
    proc.image_processor = type("LegacyImageProcessor", (), {"max_num_tiles": 4})()

    with pytest.raises(ValueError, match="dynamic-resolution image processor"):
        collate.nemotron_omni_collate_fn(
            [{"conversation": [{"role": "user", "content": "text"}]}],
            proc,
            use_temporal_video_embedder=use_temporal_video_embedder,
        )


def _heterogeneous_nemotron_examples():
    return [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "text only"}]},
                {"role": "assistant", "content": "row zero answer"},
            ]
        },
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "image-1"}, {"type": "text", "text": "one image"}],
                },
                {"role": "assistant", "content": "row one answer"},
            ]
        },
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "image-2"},
                        {"type": "image", "image": "image-3"},
                        {"type": "text", "text": "two images"},
                    ],
                },
                {"role": "assistant", "content": "row two answer"},
            ]
        },
    ]


def _sentinel_assistant_loss_mask(example, input_ids, processor, skipped_tokens, **kwargs):  # noqa: ARG001
    return torch.isin(input_ids, torch.tensor([11, 21, 31], device=input_ids.device)).to(dtype=torch.float32)


class _DynamicNemotronOmniProcessor:
    def __init__(self):
        self.tokenizer = _NemotronOmniTokenizer()
        self.image_processor = type("DynamicImageProcessor", (), {"max_num_patches": 64})()
        self.calls = []
        self.rows = [
            [10, 11, 2],
            [20, 93, 92, 92, 92, 94, 21, 2],
            [30, 93, 92, 92, 94, 32, 93, 92, 92, 92, 92, 94, 31, 2],
        ]

    def __call__(self, **kwargs):
        row_index = len(self.calls)
        self.calls.append(kwargs)
        assert kwargs["return_tensors"] is None
        output = {"input_ids": [self.rows[row_index]]}
        if kwargs.get("images") is not None:
            if row_index == 1:
                output["pixel_values"] = torch.ones(1, 3, 32, 32)
            else:
                output["pixel_values"] = [torch.ones(3, 32, 64), torch.ones(3, 64, 32)]
        return output


class _ExpandingDynamicNemotronOmniProcessor:
    def __init__(self):
        self.tokenizer = _NemotronOmniTokenizer()
        self.image_processor = type("DynamicImageProcessor", (), {})()

    def __call__(self, **kwargs):
        assert kwargs["return_tensors"] is None
        return {
            "input_ids": [[20, NEMO_IMG_START_TOKEN_ID, *([NEMO_IMAGE_TOKEN_ID] * 4), NEMO_IMG_END_TOKEN_ID, 21]],
            "pixel_values": [torch.ones(3, 64, 64)],
        }


def test_nemotron_omni_dynamic_collate_handles_mixed_shapes_within_one_sample(monkeypatch):
    processor = _DynamicNemotronOmniProcessor()
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)

    batch = collate.nemotron_omni_collate_fn(_heterogeneous_nemotron_examples(), processor)

    assert batch["input_ids"].shape[0] == 3
    assert [(row == NEMO_IMAGE_TOKEN_ID).sum().item() for row in batch["input_ids"]] == [0, 1, 2]
    assert [(row == NEMO_IMG_START_TOKEN_ID).sum().item() for row in batch["input_ids"]] == [0, 1, 2]
    assert [(row == NEMO_IMG_END_TOKEN_ID).sum().item() for row in batch["input_ids"]] == [0, 1, 2]
    assert batch["imgs_sizes"].tolist() == [[32, 32], [32, 64], [64, 32]]
    assert batch["num_image_tiles"].tolist() == [1, 2, 2]
    assert batch["visual_inputs"].pixel_values.shape == (1, 20, 768)
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["loss_mask"].shape == batch["input_ids"].shape
    assert torch.all(batch["loss_mask"][batch["attention_mask"] == 0] == 0)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX)


def test_nemotron_omni_dynamic_collate_reserves_fixed_width_for_model_merge(monkeypatch):
    from types import SimpleNamespace

    from megatron.core.models.multimodal.llava_model import LLaVAModel

    processor = _DynamicNemotronOmniProcessor()
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)

    batch = collate.nemotron_omni_collate_fn(
        _heterogeneous_nemotron_examples(),
        processor,
        sequence_length=32,
        pad_to_max_length=True,
        pad_to_multiple_of=1,
    )

    assert batch["input_ids"].shape == (3, 30)
    assert batch["attention_mask"].sum(dim=1).tolist() == [3, 6, 10]
    assert torch.all(batch["input_ids"][:, 10:] == processor.tokenizer.pad_token_id)
    assert torch.all(batch["loss_mask"][batch["attention_mask"] == 0] == 0)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX)

    hidden_size = 4
    pp_model = SimpleNamespace(
        add_decoder=True,
        pre_process=True,
        post_process=True,
        _language_is_pipeline_parallel=True,
        _language_max_sequence_length=32,
        context_parallel_lm=1,
    )
    final_embedding, final_labels, final_loss_mask = LLaVAModel._preprocess_data(
        pp_model,
        image_embeddings=torch.ones(1, 5, hidden_size),
        language_embeddings=torch.ones(3, batch["input_ids"].shape[1], hidden_size),
        input_ids=batch["input_ids"],
        loss_mask=batch["loss_mask"],
        labels=batch["labels"],
        use_inference_kv_cache=False,
        inference_context=None,
        image_token_index=NEMO_IMAGE_TOKEN_ID,
        num_image_tiles=batch["num_image_tiles"],
        is_packed_dynamic_res=True,
    )

    assert final_embedding.shape == (32, 3, hidden_size)
    assert final_labels.shape == final_loss_mask.shape == (3, 32)


def test_nemotron_omni_collate_counts_common_padding_in_model_merge_limit(monkeypatch):
    processor = _DynamicNemotronOmniProcessor()
    processor.rows[0] = [10, 11, *range(40, 51), 2]
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)

    with pytest.raises(
        ValueError,
        match=r"compact width 14 produces model row lengths \[14, 14, 16\].*sequence_length=15",
    ):
        collate.nemotron_omni_collate_fn(
            _heterogeneous_nemotron_examples(),
            processor,
            sequence_length=15,
            pad_to_multiple_of=1,
        )


def test_nemotron_omni_collate_checks_post_vision_merge_length_before_contraction(monkeypatch):
    processor = _ExpandingDynamicNemotronOmniProcessor()
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)

    with pytest.raises(
        ValueError,
        match=r"compact width 5 produces model row lengths \[8\].*sequence_length=6",
    ):
        collate.nemotron_omni_collate_fn(
            [_heterogeneous_nemotron_examples()[1]],
            processor,
            sequence_length=6,
            pad_to_multiple_of=1,
        )


def test_nemotron_omni_collate_checks_temporal_model_expansion_before_truncation(monkeypatch):
    processor = _NemotronOmniProcessor()
    rows = [
        torch.tensor([10, NEMO_IMG_START_TOKEN_ID, NEMO_IMAGE_TOKEN_ID, NEMO_IMG_END_TOKEN_ID, 11]),
        torch.tensor(
            [
                20,
                NEMO_IMG_START_TOKEN_ID,
                NEMO_IMAGE_TOKEN_ID,
                NEMO_IMG_END_TOKEN_ID,
                21,
                NEMO_IMG_START_TOKEN_ID,
                NEMO_IMAGE_TOKEN_ID,
                NEMO_IMG_END_TOKEN_ID,
                22,
            ]
        ),
    ]
    input_ids, attention_mask = nemotron_omni_collate._pad_text_rows(
        rows, pad_token_id=processor.tokenizer.pad_token_id
    )
    examples = [
        {"conversation": [{"role": "user", "content": "one tubelet"}]},
        {"conversation": [{"role": "user", "content": "two tubelets"}]},
    ]
    prepared = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "visual_inputs": GenericVisualInputs(pixel_values=torch.ones(1, 1, 768)),
    }
    monkeypatch.setattr(
        nemotron_omni_collate,
        "_prepare_temporal_rows",
        lambda *args, **kwargs: (prepared, examples, torch.ones(3, dtype=torch.long)),
    )
    monkeypatch.setattr(nemotron_omni_collate, "build_assistant_loss_mask", _zero_assistant_loss_mask)

    with pytest.raises(
        ValueError,
        match=r"compact width 9 produces model row lengths \[264, 519\].*sequence_length=512",
    ):
        collate.nemotron_omni_collate_fn(
            examples,
            processor,
            sequence_length=512,
            use_temporal_video_embedder=True,
            patch_dim=16,
        )


class _NemotronVLProcessor:
    def __init__(self, input_ids: torch.Tensor, num_patches: list[int]):
        self.tokenizer = _NemotronOmniTokenizer()
        self.input_ids = input_ids
        self.num_patches = num_patches

    def apply_chat_template(self, conversations, **kwargs):  # noqa: ARG002
        return {
            "input_ids": self.input_ids.clone(),
            "attention_mask": (self.input_ids != self.tokenizer.pad_token_id).to(dtype=torch.long),
            "num_patches": torch.tensor(self.num_patches, dtype=torch.long),
            "pixel_values": torch.ones(sum(self.num_patches), 3, 16, 16),
        }


def test_nemotron_vl_collate_uses_each_rows_flat_image_tile_counts(monkeypatch):
    vl_img_start_id = 131073
    vl_img_end_id = 131074
    raw_rows = torch.tensor(
        [
            [10, 11, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [20, vl_img_start_id, 92, 92, 92, 92, vl_img_end_id, 21, 2, 0, 0, 0, 0, 0, 0],
            [
                30,
                vl_img_start_id,
                92,
                92,
                92,
                vl_img_end_id,
                32,
                vl_img_start_id,
                92,
                92,
                92,
                92,
                vl_img_end_id,
                31,
                2,
            ],
        ]
    )
    processor = _NemotronVLProcessor(raw_rows, [1, 2, 3])
    monkeypatch.setattr(nemotron_vl_collate, "extract_skipped_token_ids", lambda processor: torch.empty(0))
    monkeypatch.setattr(nemotron_vl_collate, "infer_assistant_mask_boundary_config", lambda processor: None)
    monkeypatch.setattr(nemotron_vl_collate, "build_assistant_loss_mask", _sentinel_assistant_loss_mask)

    batch = collate.nemotron_nano_v2_vl_collate_fn(_heterogeneous_nemotron_examples(), processor)

    assert batch["input_ids"].shape == (3, 13)
    assert batch["input_ids"].tolist() == [
        [10, 11, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [20, vl_img_start_id, 92, vl_img_end_id, 21, 2, 0, 0, 0, 0, 0, 0, 0],
        [30, vl_img_start_id, 92, 92, vl_img_end_id, 32, vl_img_start_id, 92, 92, 92, vl_img_end_id, 31, 2],
    ]
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["loss_mask"].nonzero(as_tuple=False).tolist() == [[0, 0], [1, 3], [2, 10]]
