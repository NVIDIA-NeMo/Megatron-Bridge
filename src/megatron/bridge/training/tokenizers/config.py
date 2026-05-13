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
from typing import Any, Optional

from megatron.training.config import TokenizerConfig as MTrainTokenizerConfig

from megatron.bridge.utils.common_utils import warn_rank_0


@dataclass(kw_only=True)
class TokenizerConfig(MTrainTokenizerConfig):
    """Configuration settings for tokenizers."""

    hf_tokenizer_kwargs: dict[str, Any] | None = field(default_factory=dict)
    """Additional keyword arguments to pass to HuggingFace AutoTokenizer.from_pretrained.

    Common options include:
        - use_fast (bool): Whether to use fast tokenizer implementation
        - trust_remote_code (bool): Whether to trust remote code when loading tokenizer
        - include_special_tokens (bool): Whether to include special tokens when converting text to ids

    Example:
        hf_tokenizer_kwargs = {
            "use_fast": True,
            "trust_remote_code": True,
            "include_special_tokens": True
        }
    """

    sp_tokenizer_kwargs: dict[str, Any] | None = field(default_factory=dict)
    """Additional keyword arguments to pass to SentencePiece tokenizer.

    Common options include:
        - legacy (bool): Whether to use legacy format of sentencepiece tokenizer

    Example:
        sp_tokenizer_kwargs = {
            "legacy": True,
        }
    """

    tokenizer_prompt_format: Optional[str] = None
    """Prompt format for the tokenizer."""

    image_tag_type: Optional[str] = None
    """Image tag to apply, if any. For example <img><image></img>."""

    force_system_message: Optional[bool] = False

    def __post_init__(self) -> None:
        """Sync with MCore values"""
        # HuggingFace tokenizer kwargs
        self.tokenizer_hf_no_use_fast = not self.hf_tokenizer_kwargs.get("use_fast", True)
        self.tokenizer_hf_no_include_special_tokens = not self.hf_tokenizer_kwargs.get("include_special_tokens", True)
        self.trust_remote_code = self.hf_tokenizer_kwargs.get("trust_remote_code", False)
        if self.hf_tokenizer_kwargs:
            warn_rank_0(
                "`hf_tokenizer_kwargs` is deprecated and will be removed soon. "
                "Please, use `tokenizer_hf_no_use_fast` / `tokenizer_hf_no_include_special_tokens` / "
                "`trust_remote_code` arguments directly instead."
            )

        # SentencePiece tokenizer kwargs
        self.tokenizer_sentencepiece_legacy = self.sp_tokenizer_kwargs.get("legacy", False)
        if self.sp_tokenizer_kwargs:
            warn_rank_0(
                "`sp_tokenizer_kwargs` is deprecated and will be removed soon. "
                "Please, use `tokenizer_sentencepiece_legacy` (bool) argument directly instead."
            )
