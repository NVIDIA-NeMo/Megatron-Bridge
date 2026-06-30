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

"""Shared HuggingFace tokenizer adapter for MCore text generation.

MCore's ``TextGenerationController`` expects a tokenizer exposing ``eod`` / ``bos`` /
``vocab_size`` attributes and ``tokenize`` / ``detokenize`` methods. This adapter wraps a
HuggingFace tokenizer to that interface and is shared by the text-generation scripts and the
VLM inference controllers (which previously each defined their own near-identical wrapper).

Behavior knobs let call sites preserve their historical semantics exactly:
- ``set_pad_token``: default the pad token to EOS when unset (text generation).
- ``expose_vocab_size``: expose ``len(tokenizer)`` vs ``None`` (VLM historically used ``None``).
- ``force_skip_special_tokens``: when not ``None``, ``detokenize`` ignores the caller-supplied
  ``skip_special_tokens`` and always uses this value. The MCore controller passes
  ``skip_special_tokens`` only when the method accepts it, so VLM (which always kept special
  tokens) sets this to ``False`` to retain its prior behavior.
"""

from __future__ import annotations


class HFTokenizerAdapter:
    """Adapt a HuggingFace tokenizer to the MCore text-generation tokenizer interface."""

    def __init__(
        self,
        tokenizer,
        *,
        set_pad_token: bool = True,
        expose_vocab_size: bool = True,
        force_skip_special_tokens: bool | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._force_skip_special_tokens = force_skip_special_tokens
        if set_pad_token and tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self.eod = tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id
        self.vocab_size = len(tokenizer) if expose_vocab_size else None

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token ids (no special tokens added)."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back to text.

        When ``force_skip_special_tokens`` was set at construction, it overrides the
        caller-supplied ``skip_special_tokens``.
        """
        if self._force_skip_special_tokens is not None:
            skip_special_tokens = self._force_skip_special_tokens
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
