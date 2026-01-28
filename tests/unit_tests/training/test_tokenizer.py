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

from unittest.mock import patch

import pytest
from transformers import AutoTokenizer

from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


class TestTokenizers:
    @pytest.mark.parametrize("vocab_size", [32000])
    def test_build_null_tokenizer(self, vocab_size):
        # Setup
        config = TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=vocab_size,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "null-text"
        assert tokenizer.vocab_size == vocab_size

    @patch("megatron.core.tokenizers.text.libraries.MegatronHFTokenizer")
    @pytest.mark.parametrize("use_fast", [True])
    @pytest.mark.parametrize("include_special_tokens", [False])
    def test_build_megatron_tokenizer(self, mock_hf_tokenizer_class, use_fast, include_special_tokens):
        # Setup
        custom_kwargs = {
            "use_fast": use_fast,
            "include_special_tokens": include_special_tokens,
        }
        config = TokenizerConfig(
            tokenizer_type="GPT2BPETokenizer",
            tokenizer_model="gpt2",
            hf_tokenizer_kwargs=custom_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "megatron"
        assert tokenizer.path == "GPT2BPETokenizer"
        assert tokenizer.additional_args["use_fast"] == use_fast
        assert tokenizer.additional_args["include_special_tokens"] == include_special_tokens

    @patch("megatron.core.tokenizers.text.libraries.HuggingFaceTokenizer")
    @pytest.mark.parametrize("chat_template", ["{% for message in messages %}{{ message.content }}{% endfor %}"])
    def test_build_hf_tokenizer(self, mock_hf_tokenizer_class, chat_template):
        # Setup
        metadata_path = {"library": "huggingface", "chat_template": chat_template}
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Llama-2-7b-chat-hf",
            metadata_path=metadata_path,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "huggingface"
        assert tokenizer.chat_template == chat_template

    @patch("megatron.core.tokenizers.text.libraries.SentencePieceTokenizer")
    @pytest.mark.parametrize("legacy", [True])
    def test_build_sp_tokenizer(self, mock_sp_tokenizer, legacy):
        # Setup
        custom_kwargs = {
            "legacy": legacy,
        }

        config = TokenizerConfig(
            tokenizer_type="Llama2Tokenizer",
            tokenizer_model="sp.model",
            special_tokens=["<TEST_SPECIAL>"],
            sp_tokenizer_kwargs=custom_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "sentencepiece"
        assert tokenizer.additional_args["legacy"] == legacy

    @patch("megatron.core.tokenizers.text.libraries.TikTokenTokenizer")
    @pytest.mark.parametrize("pattern", ["v1"])
    @pytest.mark.parametrize("num_special_tokens", [2000])
    def test_build_tiktoken_tokenizer(self, mock_tiktoken_tokenizer, pattern, num_special_tokens):
        # Setup
        config = TokenizerConfig(
            tokenizer_type="TikTokenizer",
            tokenizer_model="tiktoken.json",
            tiktoken_pattern=pattern,
            tiktoken_num_special_tokens=num_special_tokens,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "tiktoken"
        assert tokenizer.path == "tiktoken.json"
        assert tokenizer.additional_args["pattern"] == pattern
        assert tokenizer.additional_args["num_special_tokens"] == num_special_tokens

    @pytest.mark.timeout(30)
    def test_hf_tokenizer_as_local_path_object(self, tmp_path):
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
            hf_tokenizer_kwargs={
                "trust_remote_code": True,
                "include_special_tokens": True,
            },
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
