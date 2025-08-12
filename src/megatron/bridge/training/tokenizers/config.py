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
from typing import Literal, Optional, Union, Dict, List

from megatron.core.tokenizers import MegatronTokenizerBase


@dataclass
class TokenizerConfig:
    """Configuration settings for the tokenizer."""

    tokenizer_path: str = None
    """Path to tokenizer model."""

    metadata_path: Optional[Union[str | dict]] = None
    """Tokenizer metadata."""

    multimodal_tokenizer: Optional[bool] = False
    """Whether to use multimodal tokenizer."""

    special_tokens: Optional[Union[Dict[str, str], List[str]]] = None
    """Tokenizer special tokens."""

    chat_template: Optional[str] = None
    """Tokenizer chat template."""

    # Sentencepiece tokenizer arguments
    legacy: Optional[bool] = None
    ignore_extra_whitespaces: Optional[bool] = None
    chat_template: Optional[str] = None
    trim_spm_separator_after_special_token: Optional[bool] = None
    spm_separator: Optional[str] = None

    # Huggingface tokenizer arguments
    use_fast: Optional[bool] = None
    trust_remote_code: Optional[bool] = None
    include_special_tokens: Optional[bool] = None
    vocab_file: Optional[str] = None
    meges_file: Optional[str] = None
    additional_special_tokens: Optional[List[str]] = None

    # Tiktoken tokenizer arguments
    num_special_tokens: Optional[int] = None
    pattern: Optional[int] = None

    # Null tokenizer arguments
    vocab_size: Optional[int] = None

    # Multimodal tokenizer arguments
    tokenizer_prompt_format: Optional[str] = None
    image_tag_type: Optional[str] = None

    # Metadata arguments
    write_metadata: Optional[bool] = False
    """Creates tokenizer metadata file."""

    tokenizer_library: Optional[
        Literal[
            "huggingface",
            "sentencepiece",
            "tiktoken",
            "megatron",
            "null",
            "byte-level",
        ]
    ] = None

    model_type: Optional[str] = None
    """Type of LLM model to be used with tokenizer."""

    tokenizer_class: Optional[MegatronTokenizerBase] = None
    """Pre-defined tokenizer class."""

    overwrite_metadata: Optional[bool] = False
    """If overwrite metadata file."""
