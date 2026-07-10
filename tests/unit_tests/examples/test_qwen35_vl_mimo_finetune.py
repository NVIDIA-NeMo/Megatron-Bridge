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
from types import SimpleNamespace

import pytest
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
    renames across ``datasets``, ``sources``, and ``collators``, plus model-collator import cycles.
    """
    name = "qwen35_vl_mimo_finetune_import_under_test"
    try:
        _load_example_module(name)
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    "dataset_name",
    ["cord_v2", "rdr", "medpix"],
)
def test_dataset_names_select_matching_source_presets(dataset_name):
    name = f"qwen35_vl_mimo_dataset_defaults_{dataset_name}"
    try:
        module = _load_example_module(name)
        source = module._build_dataset_source(
            SimpleNamespace(
                dataset_name=dataset_name,
                dataset_path=None,
                dataset_subset=None,
                schema_adapter=None,
            )
        )
        assert source.dataset_name == dataset_name
        assert source.path_or_dataset is None
    finally:
        sys.modules.pop(name, None)


def test_custom_dataset_keeps_explicit_source_and_adapter():
    name = "qwen35_vl_mimo_custom_dataset_source"
    try:
        module = _load_example_module(name)
        source = module._build_dataset_source(
            SimpleNamespace(
                dataset_name=None,
                dataset_path="org/custom",
                dataset_subset="subset",
                schema_adapter="rdr",
            )
        )
        assert source.path_or_dataset == "org/custom"
        assert source.subset == "subset"
        assert source.schema_adapter == "rdr"
    finally:
        sys.modules.pop(name, None)


def test_named_dataset_rejects_custom_source_flags():
    name = "qwen35_vl_mimo_named_dataset_conflict"
    try:
        module = _load_example_module(name)
        with pytest.raises(ValueError, match="owns its path, subset, and schema adapter"):
            module._build_dataset_source(
                SimpleNamespace(
                    dataset_name="cord_v2",
                    dataset_path=None,
                    dataset_subset="other",
                    schema_adapter=None,
                )
            )
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("dataset_name", "dataset_path", "do_validation", "expected_validation"),
    [
        ("cord_v2", None, None, True),
        ("rdr", None, None, False),
        (None, "org/custom", None, False),
        (None, "org/custom", True, True),
    ],
)
def test_dataset_config_enables_only_requested_or_known_validation_splits(
    dataset_name, dataset_path, do_validation, expected_validation
):
    name = f"qwen35_vl_mimo_validation_{dataset_name or 'custom'}"
    try:
        module = _load_example_module(name)
        config = module._build_dataset_config(
            SimpleNamespace(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                dataset_subset=None,
                schema_adapter=None,
                seq_length=128,
                processor_path=None,
                hf_model="org/model",
                num_workers=0,
                dataloader_type="single",
                trust_remote_code=False,
                do_validation=do_validation,
            )
        )
        config.validate()
        assert config.do_validation is expected_validation
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


def test_qwen35_vl_mimo_rejects_truncated_visual_tokens():
    name = "qwen35_vl_mimo_visual_truncation_under_test"
    try:
        module = _load_example_module(name)
        spec = module.Qwen35MIMOHFSpec()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, spec.image_token_id, spec.image_token_id]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "labels": torch.ones(1, 5, dtype=torch.long),
            "loss_mask": torch.ones(1, 5),
        }

        with pytest.raises(ValueError, match="truncates Qwen visual tokens"):
            module._adapt_qwen35_hf_batch(batch, spec, seq_length=4, pad_to_seq_length=True)
    finally:
        sys.modules.pop(name, None)
