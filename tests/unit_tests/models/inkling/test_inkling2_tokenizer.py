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

"""Inkling2MegatronTokenizer's BOS and EOS roles, built from a tokenizer configuration as the recipes build it."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from megatron.bridge.models.inkling.data.tokenizer import (
    INKLING2_BOS_TOKEN,
    INKLING2_EOS_TOKEN,
    Inkling2MegatronTokenizer,
    inkling2_tokenizer_metadata,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


pytestmark = pytest.mark.unit


def _save_tokenizer_without_roles(path, special_tokens: tuple[str, ...]) -> None:
    """Save a word-level tokenizer that holds ``special_tokens`` but assigns them no BOS or EOS role."""
    vocab = {token: index for index, token in enumerate((*special_tokens, "[UNK]", "hello", "world"))}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", additional_special_tokens=list(special_tokens)
    )
    tokenizer.save_pretrained(str(path))


def _build(path, **overrides):
    return build_tokenizer(
        TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(path), **overrides)
    )


def test_roles_are_assigned_without_changing_the_vocabulary(tmp_path):
    _save_tokenizer_without_roles(tmp_path, (INKLING2_EOS_TOKEN, INKLING2_BOS_TOKEN))
    plain = _build(tmp_path)
    tokenizer = _build(tmp_path, metadata_path=inkling2_tokenizer_metadata())

    assert plain.eod is None
    assert isinstance(tokenizer, Inkling2MegatronTokenizer)
    assert (tokenizer.eos_id, tokenizer.eod, tokenizer.bos_id) == (0, 0, 1)
    assert tokenizer.vocab_size == plain.vocab_size
    assert tokenizer.tokenize("hello world") == plain.tokenize("hello world") == [3, 4]


def test_a_vocabulary_without_the_role_tokens_is_rejected(tmp_path):
    _save_tokenizer_without_roles(tmp_path, (INKLING2_EOS_TOKEN,))

    with pytest.raises(ValueError, match="must already be in the tokenizer's vocabulary"):
        _build(tmp_path, metadata_path=inkling2_tokenizer_metadata())


def test_only_a_hugging_face_tokenizer_is_loaded(tmp_path):
    _save_tokenizer_without_roles(tmp_path, (INKLING2_EOS_TOKEN, INKLING2_BOS_TOKEN))

    with pytest.raises(ValueError, match="loads a Hugging Face tokenizer"):
        _build(tmp_path, metadata_path={**inkling2_tokenizer_metadata(), "library": "sentencepiece"})
