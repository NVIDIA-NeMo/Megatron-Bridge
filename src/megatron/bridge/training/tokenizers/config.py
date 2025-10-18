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

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TokenizerConfig:
    """Configuration settings for the tokenizer."""

    vocab_size: int | None = None
    """Size of vocab before EOD or padding."""

    vocab_file: str | None = None
    """Path to the vocab file."""

    merge_file: str | None = None
    """Path to the BPE merge file."""

    vocab_extra_ids: int = 0
    """Number of additional vocabulary tokens. They are used for span masking in the T5 model"""

    tokenizer_type:Literal[
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
        ] | None = None
    """What type of tokenizer to use."""

    tokenizer_model: str | None = None
    """Sentencepiece tokenizer model."""

    tiktoken_pattern: str | None = None
    """Which tiktoken pattern to use. Options: [v1, v2]"""

    tiktoken_num_special_tokens: int = 1000
    """Number of special tokens in tiktoken tokenizer"""

    tiktoken_special_tokens: list[str] | None = None
    """List of tiktoken special tokens, needs to have ["<unk>", "<s>", "</s>"]"""

    tokenizer_prompt_format: str | None = None
    special_tokens: list[str] | None = None
    image_tag_type: str | None = None

    hf_tokenizer_kwargs: dict[str, Any] | None = field(default_factory=dict)
    """Additional keyword arguments to pass to HuggingFace AutoTokenizer.from_pretrained.

    Common options include:
        - use_fast (bool): Whether to use fast tokenizer implementation
        - trust_remote_code (bool): Whether to trust remote code when loading tokenizer
        - chat_template (str): Custom chat template string for conversation formatting

    Example:
        hf_tokenizer_kwargs = {
            "use_fast": True,
            "trust_remote_code": True,
            "chat_template": "custom_template_string"
        }
    """
