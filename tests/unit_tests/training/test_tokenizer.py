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

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer

from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import CustomTikTokenizer, build_tokenizer


class TestTokenizerConfig:
    """Test cases for TokenizerConfig dataclass."""

    def test_tokenizer_config_default_hf_kwargs(self):
        """Test that hf_tokenizer_kwargs defaults to empty dict."""
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="bert-base-uncased",
        )
        assert config.hf_tokenizer_kwargs == {}

    def test_tokenizer_config_with_hf_kwargs(self):
        """Test that hf_tokenizer_kwargs can be set."""
        custom_kwargs = {
            "use_fast": True,
            "trust_remote_code": True,
            "chat_template": "custom_template",
        }
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Llama-2-7b-chat-hf",
            hf_tokenizer_kwargs=custom_kwargs,
        )
        assert config.hf_tokenizer_kwargs == custom_kwargs


class TestBuildTokenizer:
    """Test cases for build_tokenizer function."""

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_hf_tokenizer_with_config_kwargs(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that hf_tokenizer_kwargs from config are passed to HuggingFaceTokenizer."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        custom_kwargs = {
            "use_fast": True,
            "trust_remote_code": False,
            "padding_side": "left",
        }
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            hf_tokenizer_kwargs=custom_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        mock_hf_tokenizer_class.assert_called_once_with("gpt2", **custom_kwargs)
        assert tokenizer == mock_tokenizer_instance

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_hf_tokenizer_kwargs_override(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that passed kwargs override config hf_tokenizer_kwargs."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        config_kwargs = {
            "use_fast": True,
            "trust_remote_code": False,
        }
        passed_kwargs = {
            "use_fast": False,  # This should override
            "padding_side": "right",  # This should be added
        }
        expected_kwargs = {
            "use_fast": False,  # Overridden
            "trust_remote_code": False,  # From config
            "padding_side": "right",  # From passed kwargs
        }

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            hf_tokenizer_kwargs=config_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config, **passed_kwargs)

        # Verify
        mock_hf_tokenizer_class.assert_called_once_with("gpt2", **expected_kwargs)
        assert tokenizer == mock_tokenizer_instance

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_hf_tokenizer_no_config_kwargs(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that HuggingFaceTokenizer works without hf_tokenizer_kwargs."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            # hf_tokenizer_kwargs not set, should default to {}
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        mock_hf_tokenizer_class.assert_called_once_with("gpt2")
        assert tokenizer == mock_tokenizer_instance

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_hf_tokenizer_with_chat_template(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that chat_template can be passed via hf_tokenizer_kwargs."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        custom_kwargs = {
            "chat_template": chat_template,
        }
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Llama-2-7b-chat-hf",
            hf_tokenizer_kwargs=custom_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        mock_hf_tokenizer_class.assert_called_once_with("meta-llama/Llama-2-7b-chat-hf", chat_template=chat_template)
        assert tokenizer == mock_tokenizer_instance

    @patch("megatron.bridge.training.tokenizers.tokenizer._SentencePieceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_non_hf_tokenizer_ignores_hf_kwargs(self, mock_get_rank, mock_sp_tokenizer_class):
        """Test that non-HuggingFace tokenizers don't use hf_tokenizer_kwargs."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_sp_tokenizer_class.return_value = mock_tokenizer_instance

        # Even if hf_tokenizer_kwargs is set, it shouldn't affect SentencePiece tokenizer
        config = TokenizerConfig(
            tokenizer_type="SentencePieceTokenizer",
            tokenizer_model="tokenizer.model",
            hf_tokenizer_kwargs={"use_fast": True},  # Should be ignored
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify - SentencePiece should be called without hf_tokenizer_kwargs
        mock_sp_tokenizer_class.assert_called_once_with("tokenizer.model", vocab_extra_ids=0)
        assert tokenizer == mock_tokenizer_instance


class TestHuggingFaceTokenizerIntegration:
    """Integration tests for HuggingFace tokenizer with mocked transformers."""

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_hf_tokenizer_with_use_fast_integration(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that use_fast parameter flows through correctly in a realistic scenario."""
        # Setup a realistic mock that behaves like a real HF tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_underlying_tokenizer = MagicMock()
        mock_underlying_tokenizer.__class__.__name__ = "GPT2Tokenizer"  # Not "Fast"
        mock_tokenizer_instance._tokenizer = mock_underlying_tokenizer
        mock_tokenizer_instance.vocab_size = 50257
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            hf_tokenizer_kwargs={"use_fast": False},
        )

        tokenizer = build_tokenizer(config)

        # Verify the kwargs were passed
        mock_hf_tokenizer_class.assert_called_once_with("gpt2", use_fast=False)
        assert tokenizer is not None
        assert hasattr(tokenizer, "_tokenizer")
        # Verify it's not a fast tokenizer
        assert "Fast" not in type(tokenizer._tokenizer).__name__

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_hf_tokenizer_backward_compatibility_integration(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test backward compatibility with mocked tokenizer."""
        # Setup a realistic mock
        mock_tokenizer_instance = MagicMock()
        mock_underlying_tokenizer = MagicMock()
        mock_underlying_tokenizer.__class__.__name__ = "GPT2TokenizerFast"
        mock_tokenizer_instance._tokenizer = mock_underlying_tokenizer
        mock_tokenizer_instance.vocab_size = 50257
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            # No hf_tokenizer_kwargs specified
        )

        tokenizer = build_tokenizer(config)

        # Verify no extra kwargs were passed (backward compatible)
        mock_hf_tokenizer_class.assert_called_once_with("gpt2")
        assert tokenizer is not None
        assert hasattr(tokenizer, "vocab_size")


@pytest.fixture()
def mock_transformers(monkeypatch):
    class _MockHFTokenizer:
        def __init__(self, eos_token_id=2, bos_token_id=1, mask_token_id=None):
            self.eos_token_id = eos_token_id
            self.bos_token_id = bos_token_id
            self.mask_token_id = mask_token_id
            self.chat_template = None

        def get_vocab(self):
            return {"</s>": self.eos_token_id}

        def decode(self, ids, **kwargs):
            return ""

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1, 2])

    class _MockAutoTokenizer:
        def from_pretrained(self, *args, **kwargs):
            eos_id = kwargs.pop("_eos_token_id", 2)
            return _MockHFTokenizer(eos_token_id=eos_id)

    mock_module = type("_T", (), {"AutoTokenizer": _MockAutoTokenizer()})
    monkeypatch.setattr("megatron.bridge.training.tokenizers.tokenizer.transformers", mock_module, raising=False)
    return mock_module


@pytest.fixture()
def mock_sentencepiece(monkeypatch):
    class _MockSP:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 10

        def vocab_size(self):
            return 10

        def get_piece_size(self):
            return 10

        def id_to_piece(self, i):
            mapping = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
            return mapping.get(i, f"_{i}")

        def pad_id(self):
            return 0

        def bos_id(self):
            return 1

        def eos_id(self):
            return 2

        def encode_as_ids(self, s):
            return [1, 2]

        def decode_ids(self, ids):
            return ""

        def encode(self, s):
            return [1, 2]

        def decode_ids_as_immutable_proto(self, ids):
            class _Piece:
                def __init__(self, begin):
                    self.begin = begin

            class _Proto:
                def __init__(self):
                    self.pieces = [_Piece(0)]

            return _Proto()

    mock_module = type("_S", (), {"SentencePieceProcessor": _MockSP})
    monkeypatch.setitem(sys.modules, "sentencepiece", mock_module)
    return mock_module


@patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
@patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
def test_hf_tokenizer_eos_property(mock_get_rank, mock_hf_cls):
    inst = MagicMock()
    type(inst).eos = 7
    type(inst).eos_id = 7
    mock_hf_cls.return_value = inst

    cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="dummy-model",
        hf_tokenizer_kwargs={"_eos_token_id": 7},
    )
    tok = build_tokenizer(cfg)
    assert tok.eos == 7
    assert tok.eos_id == 7


@patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
@patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
def test_hf_tokenizer_none_eos_property(mock_get_rank, mock_hf_cls):
    inst = MagicMock()
    type(inst).eos = None
    type(inst).eos_id = None
    mock_hf_cls.return_value = inst

    cfg = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model="dummy")
    tok = build_tokenizer(cfg)
    assert tok.eos is None
    assert tok.eos_id is None


def test_sentencepiece_tokenizer_eos_property(mock_sentencepiece):
    cfg = TokenizerConfig(tokenizer_type="SentencePieceTokenizer", tokenizer_model="sp.model")
    tok = build_tokenizer(cfg)
    assert tok.eos == 2
    assert tok.eos_id == 2


def test_gpt_sentencepiece_tokenizer_eos_property(mock_sentencepiece):
    cfg = TokenizerConfig(tokenizer_type="GPTSentencePieceTokenizer", tokenizer_model="sp.model")
    tok = build_tokenizer(cfg)
    assert tok.eos == 2
    assert tok.eos_id == 2
    assert tok.eod == 2


@patch("megatron.bridge.training.tokenizers.tokenizer._Llama2Tokenizer")
@patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
def test_llama2_tokenizer_eos_property(mock_get_rank, mock_llama2_cls, mock_sentencepiece):
    inst = MagicMock()
    type(inst).eos = 2
    type(inst).eos_id = 2
    type(inst).eod = 2
    mock_llama2_cls.return_value = inst

    cfg = TokenizerConfig(tokenizer_type="Llama2Tokenizer", tokenizer_model="sp.model")
    tok = build_tokenizer(cfg)
    assert tok.eos == 2
    assert tok.eos_id == 2
    assert tok.eod == 2


@patch("megatron.bridge.training.tokenizers.tokenizer._GPT2BPETokenizer")
@patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
def test_gpt2_bpe_tokenizer_eos_property(mock_get_rank, mock_gpt2_cls):
    inst = MagicMock()
    type(inst).eos = 50256
    type(inst).eos_id = 50256
    type(inst).eod = 50256
    mock_gpt2_cls.return_value = inst

    cfg = TokenizerConfig(tokenizer_type="GPT2BPETokenizer", vocab_file="vocab.json", merge_file="merges.txt")
    tok = build_tokenizer(cfg)
    assert tok.eos == 50256
    assert tok.eos_id == 50256
    assert tok.eod == 50256


def test_null_tokenizer_eos_property():
    cfg = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=10)
    tok = build_tokenizer(cfg)
    assert tok.eos == 9
    assert tok.eos_id == 9
    assert tok.eod == 9


@pytest.mark.skipif(not hasattr(CustomTikTokenizer, "__init__"), reason="CustomTikTokenizer not available")
def test_tiktokenizer_eos_property(tmp_path):
    try:
        import tiktoken  # noqa: F401
    except Exception:
        pytest.skip("tiktoken not installed")

    vocab = [
        {"rank": 0, "token_bytes": "AA==", "token_str": "\u0000"},
        {"rank": 1, "token_bytes": "AQ==", "token_str": "\u0001"},
    ]
    p = tmp_path / "tiny_vocab.json"
    p.write_text(json.dumps(vocab), encoding="utf-8")

    # Minimal consistent vocab_size = special_tokens(3) + mergeable_ranks(len(vocab)=2) = 5
    cfg = TokenizerConfig(
        tokenizer_type="TikTokenizer",
        tokenizer_model=str(p),
        vocab_size=5,
        tiktoken_pattern="v2",
        tiktoken_num_special_tokens=3,
        tiktoken_special_tokens=["<unk>", "<s>", "</s>"],
    )
    tok = build_tokenizer(cfg)
    # Validate consistency and mapping
    assert tok.eos_id == tok.eos
    assert tok.vocab["</s>"] == tok.eos


def test_null_multimodal_tokenizer_basic_properties():
    cfg = TokenizerConfig(tokenizer_type="NullMultimodalTokenizer", vocab_size=127)
    tok = build_tokenizer(cfg)

    assert tok.vocab_size == 127
    assert tok.eod == 126
    assert tok.eos == 126


def test_null_multimodal_tokenizer_tokenize_detokenize_offsets():
    cfg = TokenizerConfig(tokenizer_type="NullMultimodalTokenizer", vocab_size=100)
    tok = build_tokenizer(cfg)

    s = "1 23 456"
    ids = tok.tokenize(s)
    assert ids == [1, 23, 456]
    text = tok.detokenize(ids)
    assert text == s

    offs = tok.offsets(ids, s)
    assert offs == [0, 2, 5]


def test_null_multimodal_tokenizer_image_token_default():
    cfg = TokenizerConfig(tokenizer_type="NullMultimodalTokenizer", vocab_size=100)
    tok = build_tokenizer(cfg)

    tokens = "1  " + tok._image_token + "  2"
    ids = tok.convert_tokens_to_ids(tokens)
    assert isinstance(ids, list)
    assert len(ids) == 3
    assert ids[0] == 1 and ids[-1] == 2
    assert ids[1] == tok._image_token_id


def test_null_multimodal_tokenizer_image_token_override():
    cfg = TokenizerConfig(tokenizer_type="NullMultimodalTokenizer", vocab_size=100)
    tok = build_tokenizer(cfg)

    tok._image_token = "<<img>>"
    tok._image_token_id = 77

    tokens = "3  <<img>>  4"
    ids = tok.convert_tokens_to_ids(tokens)
    assert ids == [3, 77, 4]


@pytest.mark.timeout(30)
def test_hf_tokenizer_as_local_path_object(tmp_path):
    # Cover the user case where a user has made a local path object of a WIP tokenizer and wants
    #  to use that in some megatron model at train time.

    # First as a proxy download a tokenizer from HF and save it to a local path. A user would
    #  do this differently by exporting their WIP tokenizer to a local path.

    # 1. Download a common, small tokenizer from the Hub
    # "bert-base-uncased" is a safe choice as it's small and standard.
    model_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Define a local path in the temporary directory
    local_model_path = tmp_path / "my_local_tokenizer"

    # 3. Save the tokenizer to disk
    # This creates tokenizer_config.json, vocab.txt, special_tokens_map.json, etc.
    tokenizer.save_pretrained(str(local_model_path))

    # 4. Load it back using the local path
    # This simulates the user providing a path to a folder instead of a Hub ID
    cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=local_model_path,
        hf_tokenizer_kwargs={"trust_remote_code": True},
    )
    loaded_tokenizer = build_tokenizer(cfg)

    # 5. Verify it functions identically
    test_text = "Unit testing is important."

    original_tokens = tokenizer.encode(test_text)
    reloaded_tokens = loaded_tokenizer.tokenize(test_text)

    assert original_tokens == reloaded_tokens
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

    # verify that the directory actually contains files (sanity check)
    assert (local_model_path / "tokenizer_config.json").exists()
    assert (local_model_path / "tokenizer.json").exists()
