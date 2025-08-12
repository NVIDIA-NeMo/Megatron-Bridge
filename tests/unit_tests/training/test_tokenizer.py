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

from megatron.bridge.training.tokenizers.build_tokenizer import build_tokenizer
from megatron.bridge.training.tokenizers.config import TokenizerConfig


class TestTokenizerConfig:
    def test_build_tokenizer_tiktoken(self, ensure_test_data):
        config = TokenizerConfig(
            tokenizer_path=f"{ensure_test_data}/tokenizers/tiktoken/tiktoken.vocab.json",
            num_special_tokens=100,
            pattern="v1",
            vocab_size=32000,
        )

        assert config.pattern == "v1"
        assert config.num_special_tokens == 100

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        assert tokenizer.vocab_size == 32000

    def test_build_tokenizer_multimodal(self, ensure_test_data):
        special_tokens = ["<image_start>", "<image_end>"]
        config = TokenizerConfig(
            tokenizer_path="llava-hf/llava-1.5-7b-hf",
            multimodal_tokenizer=True,
            tokenizer_prompt_format="nvlm-yi-34b",
            special_tokens=special_tokens,
            image_tag_type="nvlm",
        )

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

    def test_build_tokenzier_megatron(self, ensure_test_data):
        additional_special_tokens = [f"<extra_id_{i}>" for i in range(100)]
        config = TokenizerConfig(
            tokenizer_path="BertWordPieceCase",
            metadata_path=dict(library="megatron"),
            additional_special_tokens=additional_special_tokens,
        )

        assert config.additional_special_tokens == additional_special_tokens

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        assert len(tokenizer.additional_special_tokens_ids) == 100

    def test_build_tokenizer_huggingface(self, ensure_test_data):
        special_tokens = {"bos_token": "<TEST_BOS>", "eos_token": "<TEST_EOS>"}
        config = TokenizerConfig(
            tokenizer_path=f"{ensure_test_data}/tokenizers/huggingface",
            metadata_path=dict(library="huggingface"),
            special_tokens=special_tokens,
        )

        assert config.special_tokens == special_tokens

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        assert tokenizer.tokenize("<TEST_BOS><TEST_EOS>") == [128257, 128256]

    def test_build_tokenizer_null(self):
        config = TokenizerConfig(
            metadata_path=dict(library="null"),
            vocab_size=100000,
        )

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        assert tokenizer.vocab_size == 100001

    def test_build_tokenizer_sentencepiece(self, ensure_test_data):
        config = TokenizerConfig(
            tokenizer_path=f"{ensure_test_data}/tokenizers/sentencepiece/tokenizer.model",
            legacy=False,
        )

        assert config.legacy == False

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )
    
    def test_write_metadata(self, ensure_test_data):
        chat_template = "test chat template"
        config = TokenizerConfig(
            tokenizer_path=f"{ensure_test_data}/tokenizers/huggingface",
            write_metadata=True,
            tokenizer_library="huggingface",
            chat_template=chat_template,
            overwrite_metadata=True,
        )

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        assert tokenizer.chat_template == chat_template
