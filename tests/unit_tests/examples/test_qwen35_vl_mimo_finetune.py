# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Smoke + contract tests for the Qwen3.5-VL MegatronMIMO finetune example."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch
from PIL import Image


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "megatron_mimo" / "qwen35_vl" / "finetune_qwen35_vl.py"

# Mirror the shared collate defaults (collate_fn.QWEN_VL_MIN/MAX_PIXELS) so the metadata
# grid math matches the visual path. Passed explicitly to keep the test independent of
# collate_fn import order (the example imports those constants lazily).
_MIN_PIXELS = 200704
_MAX_PIXELS = 1003520


def _load_example_module(name: str):
    """Load the example script as a module and keep it registered for attribute access."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qwen35_vl_mimo_finetune_example_imports():
    """Guard the example's module load against moved/removed symbols and import cycles.

    Executing the module top-to-bottom exercises every top-level import, the dataclass
    definitions, and module-level constants. This catches regressions such as data-API
    renames (e.g. ``vlm_datasets`` -> ``hf_datasets``) and the ``collate_fn`` <-> ``vlm_datasets``
    circular import that only triggers when the example imports ``collate_fn`` first.
    """
    name = "qwen35_vl_mimo_finetune_import_under_test"
    try:
        _load_example_module(name)
    finally:
        sys.modules.pop(name, None)


class _DummyImageProcessor:
    patch_size = 14
    merge_size = 2


class _DummyTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    added_tokens_decoder: dict = {}
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    _MARKER_IDS = {"<|im_start|>assistant\n": [102], "<|im_end|>": [103]}

    def encode(self, text, add_special_tokens=False):
        return self._MARKER_IDS.get(text, [1])

    def __call__(
        self,
        text=None,
        add_special_tokens=False,
        padding=True,
        return_tensors=None,
        return_token_type_ids=False,
        **kwargs,
    ):
        # Single-string call: marker tokenization for the boundary-config builder.
        if isinstance(text, str):
            return {"input_ids": self._MARKER_IDS.get(text, [1])}
        # List call: batch tokenization for the metadata collate.
        rows = len(text) if text is not None else 1
        input_ids = torch.tensor([[1, 2, 3]] * rows)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class _DummyMetadataProcessor:
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
    image_token = "<|image_pad|>"

    def __init__(self):
        self.tokenizer = _DummyTokenizer()
        self.image_processor = _DummyImageProcessor()

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):
        if tokenize:
            # HF generation-mask path consumed by build_assistant_loss_mask.
            return {"input_ids": [1, 2, 3], "assistant_masks": [0, 0, 1]}
        # Text path for the metadata collate: must contain the image placeholder.
        return "<|im_start|>user <|image_pad|><|im_end|><|im_start|>assistant\nhi<|im_end|>"


def test_qwen35_vl_mimo_metadata_collate_builds_batch():
    """Drive the metadata-only collate path (no pixel decode) end-to-end on CPU.

    Unlike the import smoke, this actually calls ``_build_qwen_metadata_batch``, so it
    guards the runtime contract: it exercises ``assistant_mask_boundary_config_from_markers``
    and ``build_assistant_loss_mask(boundary_config=...)`` (catching signature drift such as
    the removed ``require_matches`` kwarg) and locks down the ``image_grid_thw`` math derived
    from image size alone.
    """
    name = "qwen35_vl_mimo_finetune_metadata_under_test"
    try:
        module = _load_example_module(name)
        item = {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (336, 336))},
                        {"type": "text", "text": "What is this?"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "A test image."}]},
            ]
        }

        batch, image_grid_thw = module._build_qwen_metadata_batch(
            [item],
            processor=_DummyMetadataProcessor(),
            spec=module.Qwen35MIMOHFSpec(),
            min_pixels=_MIN_PIXELS,
            max_pixels=_MAX_PIXELS,
        )

        # Grid is computed from image size only; 336x336 -> (t=1, h=32, w=32).
        assert image_grid_thw.dtype == torch.long
        assert image_grid_thw.tolist() == [[1, 32, 32]]

        input_ids = batch["input_ids"]
        assert set(batch) >= {"input_ids", "attention_mask", "labels", "loss_mask", "visual_inputs"}
        assert batch["labels"].shape == input_ids.shape
        assert batch["loss_mask"].shape == input_ids.shape
        assert torch.equal(batch["visual_inputs"].image_grid_thw, image_grid_thw)
    finally:
        sys.modules.pop(name, None)
