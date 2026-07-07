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

import runpy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


_SCRIPT = Path(__file__).parents[3] / "examples" / "conversion" / "hf_to_megatron_generate_text.py"
_SCRIPT_GLOBALS = runpy.run_path(_SCRIPT)
_decode_completion = _SCRIPT_GLOBALS["_decode_completion"]
_tokenize_prompt = _SCRIPT_GLOBALS["_tokenize_prompt"]


@pytest.mark.unit
def test_tokenize_raw_prompt() -> None:
    tokenizer = MagicMock()
    tokenizer.encode.return_value = torch.tensor([[1, 2]])

    input_ids = _tokenize_prompt(tokenizer, "hello", apply_chat_template=False, thinking_mode="adaptive")

    torch.testing.assert_close(input_ids, torch.tensor([[1, 2]]))
    tokenizer.encode.assert_called_once_with("hello", return_tensors="pt")
    tokenizer.apply_chat_template.assert_not_called()


@pytest.mark.unit
def test_tokenize_chat_prompt() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {"input_ids": torch.tensor([[3, 4]])}

    input_ids = _tokenize_prompt(tokenizer, "hello", apply_chat_template=True, thinking_mode="disabled")

    torch.testing.assert_close(input_ids, torch.tensor([[3, 4]]))
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        thinking_mode="disabled",
    )
    tokenizer.encode.assert_not_called()


@pytest.mark.unit
def test_decode_completion_excludes_prompt_and_special_tokens() -> None:
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "The sky appears blue."

    text = _decode_completion(tokenizer, torch.tensor([[10, 11, 20, 21]]), prompt_length=2)

    assert text == "The sky appears blue."
    tokenizer.decode.assert_called_once_with([20, 21], skip_special_tokens=True)
