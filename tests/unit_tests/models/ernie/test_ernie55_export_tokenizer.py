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

"""CPU tests of the ERNIE 5.5 Hugging Face export's tokenizer class name.

transformers 5 saves a fast tokenizer as ``TokenizersBackend``, which transformers 4 (the reference
implementation's stack) cannot load. ``Ernie55Bridge.postprocess_hf_export_artifacts`` writes back
``PreTrainedTokenizerFast``, an alias of the same class in transformers 5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te  # noqa: E402


allow_conversion_imports_without_te()
pytest.importorskip("megatron.bridge.models.conversion.model_bridge", reason="needs Bridge's conversion stack")

from megatron.bridge.models.ernie.ernie55_bridge import Ernie55Bridge  # noqa: E402


pytestmark = pytest.mark.unit

TEXT = "hello ERNIE <|im_end|>"


def _postprocess(path: Path) -> None:
    """Run the export hook; it reads nothing but ``path``, so the bridge needs no config."""
    Ernie55Bridge.postprocess_hf_export_artifacts(object.__new__(Ernie55Bridge), path)


def _toy_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|im_end|>")


def _write(path: Path, settings: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return path


def _read(path: Path) -> dict:
    return json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))


def test_transformers5_class_name_is_written_back(tmp_path):
    settings = {"backend": "tokenizers", "eos_token": "<|ERNIE_MASK|>", "tokenizer_class": "TokenizersBackend"}
    _postprocess(_write(tmp_path, settings))
    assert _read(tmp_path) == {**settings, "tokenizer_class": "PreTrainedTokenizerFast"}


@pytest.mark.parametrize("tokenizer_class", ["PreTrainedTokenizerFast", "LlamaTokenizerFast", None])
def test_other_class_names_are_left_alone(tmp_path, tokenizer_class):
    settings = {"eos_token": "<|ERNIE_MASK|>"}
    if tokenizer_class is not None:
        settings["tokenizer_class"] = tokenizer_class
    _write(tmp_path, settings)
    before = (tmp_path / "tokenizer_config.json").read_bytes()
    _postprocess(tmp_path)
    assert (tmp_path / "tokenizer_config.json").read_bytes() == before


def test_export_without_a_tokenizer_is_left_alone(tmp_path):
    _postprocess(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_saved_tokenizer_reloads_with_the_same_tokens(tmp_path):
    from transformers import AutoTokenizer

    tokenizer = _toy_tokenizer()
    tokenizer.save_pretrained(tmp_path)
    saved = _read(tmp_path)
    _postprocess(tmp_path)
    fixed = _read(tmp_path)

    assert fixed["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert {k: v for k, v in fixed.items() if k != "tokenizer_class"} == {
        k: v for k, v in saved.items() if k != "tokenizer_class"
    }
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    assert reloaded(TEXT)["input_ids"] == tokenizer(TEXT)["input_ids"]
    assert reloaded.eos_token == tokenizer.eos_token
    assert reloaded.pad_token == tokenizer.pad_token
