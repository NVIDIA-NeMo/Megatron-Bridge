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

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TokenizerConfig:
    """Configuration settings for the tokenizer."""

    vocab_size: Optional[int] = None
    """Size of vocab before EOD or padding."""

    vocab_file: Optional[str] = None
    """Path to the vocab file."""

    merge_file: Optional[str] = None
    """Path to the BPE merge file."""

    vocab_extra_ids: int = 0
    """Number of additional vocabulary tokens. They are used for span masking in the T5 model"""

    tokenizer_type: Optional[
        Literal[
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "SentencePieceTokenizer",
            "GPTSentencePieceTokenizer",
            "HuggingFaceTokenizer",
            "Llama2Tokenizer",
            "TikTokenizer",
            "MultimodalTokenizer",
            "NullTokenizer",
        ]
    ] = None
    """What type of tokenizer to use."""

    tokenizer_model: Optional[str] = None
    """Sentencepiece tokenizer model."""

    tiktoken_pattern: Optional[str] = None
    """Which tiktoken pattern to use. Options: [v1, v2]"""

    tiktoken_num_special_tokens: int = 1000
    """Number of special tokens in tiktoken tokenizer"""

    tiktoken_special_tokens: Optional[list[str]] = None
    """List of tiktoken special tokens, needs to have ["<unk>", "<s>", "</s>"]"""

    tokenizer_prompt_format: Optional[str] = None
    special_tokens: Optional[list[str]] = None
    image_tag_type: Optional[str] = None
    padded_vocab_size: Optional[int] = None
