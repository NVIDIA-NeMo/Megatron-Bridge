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

"""Focused tests for deterministic HF inference verification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.unit


def _load_module():
    script = (
        Path(__file__).resolve().parents[4]
        / "skills"
        / "create-model-verification-card"
        / "scripts"
        / "verify_hf_inference.py"
    )
    spec = importlib.util.spec_from_file_location("test_verify_hf_inference_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Batch(dict):
    def to(self, device):
        self.device = device
        return self


class _Tokenizer:
    pad_token_id = None
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        if kwargs.get("tokenize") is False:
            return "formatted text prompt"
        return _Batch(input_ids=torch.tensor([[1, 2, 3]]), pixel_values=torch.tensor([1]))

    def __call__(self, prompt, *, return_tensors):
        self.prompt = prompt
        assert return_tensors == "pt"
        return _Batch(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "verified image"


class _Processor(_Tokenizer):
    def __init__(self):
        self.tokenizer = _Tokenizer()


class _SeparateImageProcessor(_Processor):
    image_token = "<image>"

    def __call__(self, *, text, images, return_tensors):
        self.direct_text = text
        self.direct_images = images
        assert return_tensors == "pt"
        return _Batch(
            input_ids=torch.tensor([[1, 2, 3]]),
            pixel_values=torch.tensor([1]),
            num_patches=torch.tensor([1]),
            num_tokens=torch.tensor([4]),
            imgs_sizes=torch.tensor([[2, 2]]),
        )


class _Model:
    device = "cpu"

    def __init__(self):
        self.calls = []
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3, 4]])


def test_image_content_uses_processor_native_location_keys():
    module = _load_module()

    assert module._image_content("work/data/example.png") == {
        "type": "image",
        "path": "work/data/example.png",
    }
    assert module._image_content("https://example.test/example.png") == {
        "type": "image",
        "url": "https://example.test/example.png",
    }


def test_loading_info_requires_strict_reload():
    module = _load_module()

    module._validate_loading_info(
        {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }
    )

    with pytest.raises(RuntimeError, match="missing_keys=1, mismatched_keys=1"):
        module._validate_loading_info(
            {
                "missing_keys": ["model.missing"],
                "unexpected_keys": [],
                "mismatched_keys": [("model.wrong_shape", (1,), (2,))],
                "error_msgs": [],
            }
        )


@pytest.mark.parametrize(
    ("device_map", "expected_device_map", "expected_to_calls"),
    [
        (None, None, ["cuda"]),
        ("balanced_low_0", "balanced_low_0", []),
    ],
)
def test_runtime_supports_explicit_multi_gpu_device_map(
    monkeypatch, device_map, expected_device_map, expected_to_calls
):
    module = _load_module()
    processor = _Processor()
    model = _Model()

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(model_path, *, trust_remote_code):
            assert model_path == "exported-model"
            assert trust_remote_code
            return processor

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            assert model_path == "exported-model"
            if expected_device_map is None:
                assert "device_map" not in kwargs
            else:
                assert kwargs["device_map"] == expected_device_map
            return model, {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForImageTextToText=_AutoModel, AutoProcessor=_AutoProcessor),
    )
    args = SimpleNamespace(
        device="cuda",
        device_map=device_map,
        dtype="bfloat16",
        hf_model="exported-model",
        image="image.png",
        trust_remote_code=True,
    )

    _, loaded_model, loaded_processor = module._load_runtime(args)

    assert loaded_model is model
    assert loaded_processor is processor
    assert model.to_calls == expected_to_calls


def test_image_requires_chat_template(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_inference.py",
            "--hf-model",
            "model",
            "--prompt",
            "prompt",
            "--image",
            "image.png",
            "--max-new-tokens",
            "2",
        ],
    )

    with pytest.raises(SystemExit):
        module._parse_args()


def test_separate_image_processing_requires_image(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_inference.py",
            "--hf-model",
            "model",
            "--prompt",
            "prompt",
            "--max-new-tokens",
            "2",
            "--separate-image-processing",
        ],
    )

    with pytest.raises(SystemExit):
        module._parse_args()


def test_separate_image_processing_renders_text_then_passes_pil(monkeypatch):
    module = _load_module()
    processor = _SeparateImageProcessor()
    image = object()
    monkeypatch.setattr(module, "_load_pil_image", lambda _: image)
    args = SimpleNamespace(
        image="work/data/example.png",
        prompt="Describe the image.",
        disable_thinking=True,
        separate_image_processing=True,
    )

    inputs = module._prepare_inputs(processor, args)

    assert inputs["pixel_values"].tolist() == [1]
    assert set(inputs) == {"input_ids", "pixel_values"}
    assert processor.messages == [{"role": "user", "content": "<image>\nDescribe the image."}]
    assert processor.template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert processor.direct_text == ["formatted text prompt"]
    assert processor.direct_images == [image]


def test_multimodal_main_uses_processor_chat_and_allows_early_stopping(monkeypatch):
    module = _load_module()
    processor = _Processor()
    model = _Model()
    monkeypatch.setattr(module, "_load_runtime", lambda args: (torch, model, processor))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_inference.py",
            "--hf-model",
            "model",
            "--prompt",
            "What abnormality is shown?",
            "--image",
            "work/data/medpix/verification.png",
            "--max-new-tokens",
            "2",
            "--chat-template",
            "--disable-thinking",
        ],
    )

    result = module.main()

    assert result == 0
    assert processor.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "work/data/medpix/verification.png"},
                {"type": "text", "text": "What abnormality is shown?"},
            ],
        }
    ]
    assert processor.template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["do_sample"] is False
    assert "min_new_tokens" not in call
    assert call["max_new_tokens"] == 2
    assert call["pad_token_id"] == 0
    assert "pixel_values" in call


def test_text_main_keeps_the_legacy_tokenizer_path(monkeypatch):
    module = _load_module()
    tokenizer = _Tokenizer()
    model = _Model()
    monkeypatch.setattr(module, "_load_runtime", lambda args: (torch, model, tokenizer))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_inference.py",
            "--hf-model",
            "model",
            "--prompt",
            "Describe Paris.",
            "--max-new-tokens",
            "2",
            "--chat-template",
        ],
    )

    result = module.main()

    assert result == 0
    assert tokenizer.prompt == "formatted text prompt"
    assert len(model.calls) == 1
    assert "pixel_values" not in model.calls[0]
