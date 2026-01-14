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

from megatron.core.tokenizers import MegatronTokenizer

from megatron.bridge.training.tokenizers.config import TokenizerConfig


MEGATRON_TOKENIZERS = ["BertWordPieceLowerCase", "BertWordPieceCase", "GPT2BPETokenizer"]

SP_TOKENIZERS = ["SentencePieceTokenizer", "GPTSentencePieceTokenizer", "Llama2Tokenizer"]


def build_new_tokenizer(config: TokenizerConfig) -> MegatronTokenizer:
    """Initialize tokenizer from megatron.core.tokenizers based on the provided configuration.

    Args:
        config (TokenizerConfig): Configuration object specifying the tokenizer
                                            type, paths to vocab/model files, and other
                                            tokenizer-specific settings.

    Returns:
        MegatronTokenizer: An instance of the initialized tokenizer.
    """
    kwargs = {}
    tokenizer_library = None
    tokenizer_path = None
    if config.tokenizer_type in MEGATRON_TOKENIZERS:
        tokenizer_library = "megatron"
        tokenizer_path = config.tokenizer_type
        kwargs["additional_special_tokens"] = config.special_tokens if config.special_tokens else []
        if tokenizer_path == "BertWordPieceCase":
            special_tokens = {}
            special_tokens["additional_special_tokens"] = [f"<extra_id_{i}>" for i in range(100)]
            kwargs = special_tokens
        kwargs["vocab_file"] = config.vocab_file
        kwargs["merges_file"] = config.merge_file
        if config.hf_tokenizer_kwargs:
            kwargs.update(config.hf_tokenizer_kwargs)
    elif config.tokenizer_type in SP_TOKENIZERS:
        tokenizer_library = "sentencepiece"
        tokenizer_path = config.tokenizer_model
        kwargs["chat_template"] = config.chat_template
        kwargs["special_tokens"] = config.special_tokens
        kwargs.update(config.sp_tokenizer_kwargs)
    elif config.tokenizer_type == "TikTokenizer":
        tokenizer_library = "tiktoken"
        tokenizer_path = config.tokenizer_model
        kwargs["chat_template"] = config.chat_template
        if config.tiktoken_pattern:
            kwargs["pattern"] = config.tiktoken_pattern
        if config.vocab_size:
            kwargs["vocab_size"] = config.vocab_size
        kwargs["num_special_tokens"] = config.tiktoken_num_special_tokens
        kwargs["special_tokens"] = config.special_tokens
        kwargs["vocab_size"] = config.vocab_size
    elif config.tokenizer_type == "HuggingFaceTokenizer":
        tokenizer_library = "huggingface"
        tokenizer_path = config.tokenizer_model
        kwargs["chat_template"] = config.chat_template
        kwargs["vocab_file"] = config.vocab_file
        kwargs["merges_file"] = config.merge_file
        kwargs["additional_special_tokens"] = config.special_tokens if config.special_tokens else []
        if config.hf_tokenizer_kwargs:
            kwargs.update(config.hf_tokenizer_kwargs)
    elif config.tokenizer_type == "NullTokenizer":
        tokenizer_library = "null"
        metadata = {"library": tokenizer_library}
        if config.vocab_size:
            kwargs["vocab_size"] = config.vocab_size - 1
        tokenizer = MegatronTokenizer.from_pretrained(metadata_path=metadata, **kwargs)

        return tokenizer

    if config.metadata_path:
        metadata = config.metadata_path
    else:
        metadata = {"library": tokenizer_library}
    tokenizer = MegatronTokenizer.from_pretrained(tokenizer_path=tokenizer_path, metadata_path=metadata, **kwargs)

    return tokenizer
