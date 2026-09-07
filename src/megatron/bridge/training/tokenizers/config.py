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
from typing import Optional

from megatron.training.config import TokenizerConfig as MTrainTokenizerConfig


@dataclass(kw_only=True)
class TokenizerConfig(MTrainTokenizerConfig):
    """Configuration settings for tokenizers."""

    make_vocab_size_divisible_by: int = 1
    """Keep MCore tokenizer padding neutral; model providers apply vocab padding."""

    tensor_model_parallel_size: int = 1
    """Tensor parallel size used by MCore tokenizer padded vocab-size calculation."""

    rank: int = 0
    """Distributed rank used by MCore tokenizer helper logging."""

    use_tokenizer_vocab_size: bool = False
    """Use the runtime tokenizer vocabulary size for the model.

    Enable this for from-scratch pretraining, where the tokenizer selected for
    the dataset defines the embedding and output vocabulary. Keep it disabled
    when model or checkpoint compatibility requires an explicitly configured
    model vocabulary size. This policy also applies during checkpoint loading;
    disable it and configure the checkpoint's original model vocabulary when
    resuming a run created with a different vocabulary policy.
    """

    chat_template_path: Optional[str] = None
    """Path to a jinja chat template file, loaded at build time as ``chat_template``. Supports local
    paths and ``msc://`` URLs. Mutually exclusive with ``chat_template``. Useful for supplying a
    template from an external/process caller (e.g. CLI overrides) where inlining the jinja is
    impractical."""

    tokenizer_prompt_format: Optional[str] = None
    """Prompt format for the tokenizer."""

    @property
    def sft_tokenizer_prompt_format(self) -> str | None:
        """Expose the prompt-format name expected by MCore's SFT tokenizer builder."""
        return self.tokenizer_prompt_format

    image_tag_type: Optional[str] = None
    """Image tag to apply, if any. For example <img><image></img>."""

    force_system_message: Optional[bool] = False

    def __post_init__(self) -> None:
        """Sync with MCore values"""
        # Don't pad vocab size since MBridge does it's own padding
        self.pad_vocab_size = False
