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

from transformers import CLIPTokenizer


class TestTokenizersCompat:
    def test_clip_tokenizer_special_tokens(self):
        """CLIP constructs its postprocessor and inserts BOS/EOS with the installed Tokenizers."""
        tokenizer = CLIPTokenizer(
            vocab={"<|startoftext|>": 0, "<|endoftext|>": 1, "h": 2, "i</w>": 3},
            merges=[],
        )
        encoded = tokenizer("hi", return_special_tokens_mask=True)

        assert encoded["input_ids"] == [0, 2, 3, 1]
        assert encoded["special_tokens_mask"] == [1, 0, 0, 1]
        assert tokenizer.decode(encoded["input_ids"], skip_special_tokens=True) == "hi"
