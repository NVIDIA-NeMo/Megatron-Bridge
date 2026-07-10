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

"""Offline tokenizer fixture shared by the Gemma conversion tests."""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def save_minimal_gemma_tokenizer(model_dir: Path) -> None:
    """Save a deterministic tokenizer without downloading a gated Gemma model."""
    vocabulary = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_dir)
