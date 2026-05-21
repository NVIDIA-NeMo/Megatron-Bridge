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

"""Unit tests for GSM8K dataset processor."""

import pytest

from megatron.bridge.data.hf_processors.gsm8k import _extract_final_answer, process_gsm8k_example


@pytest.mark.unit
class TestExtractFinalAnswer:
    """Test cases for the _extract_final_answer helper."""

    def test_standard_delimiter(self):
        answer = "Janet starts with 3. 3 + 2 = 5.\n#### 5"
        assert _extract_final_answer(answer) == "5"

    def test_delimiter_no_spaces(self):
        assert _extract_final_answer("reasoning\n####42") == "42"

    def test_delimiter_extra_whitespace(self):
        assert _extract_final_answer("reasoning\n####   18  ") == "18"

    def test_no_delimiter_returns_stripped_text(self):
        assert _extract_final_answer("just a plain answer") == "just a plain answer"

    def test_multiple_delimiters_uses_last(self):
        assert _extract_final_answer("text #### wrong #### 7") == "7"

    def test_negative_number(self):
        assert _extract_final_answer("The temperature dropped.\n#### -3") == "-3"

    def test_decimal_number(self):
        assert _extract_final_answer("Total cost is $12.50.\n#### 12.50") == "12.50"


@pytest.mark.unit
class TestProcessGsm8kExample:
    """Test cases for process_gsm8k_example function."""

    def test_basic_example(self):
        example = {
            "question": "Janet has 3 apples. She buys 2 more. How many does she have?",
            "answer": "Janet starts with 3 and buys 2 more. 3 + 2 = <<3+2=5>>5.\n#### 5",
        }
        result = process_gsm8k_example(example)

        assert result["input"] == ("Question: Janet has 3 apples. She buys 2 more. How many does she have? Answer:")
        assert result["output"] == example["answer"]
        assert result["original_answers"] == ["5"]

    def test_multiline_reasoning(self):
        example = {
            "question": "A store has 10 items at $3 each. What is the total?",
            "answer": ("Each item costs $3.\nThere are 10 items.\n10 * 3 = <<10*3=30>>30.\n#### 30"),
        }
        result = process_gsm8k_example(example)

        assert result["input"] == ("Question: A store has 10 items at $3 each. What is the total? Answer:")
        assert result["output"] == example["answer"]
        assert result["original_answers"] == ["30"]

    def test_output_is_full_chain_of_thought(self):
        """The output field should contain the full answer including reasoning, not just the number."""
        example = {
            "question": "What is 1+1?",
            "answer": "1 + 1 = 2.\n#### 2",
        }
        result = process_gsm8k_example(example)
        assert "1 + 1 = 2" in result["output"]
        assert "#### 2" in result["output"]

    def test_original_answers_is_final_answer_only(self):
        example = {
            "question": "How many?",
            "answer": "Long reasoning chain here.\n#### 42",
        }
        result = process_gsm8k_example(example)
        assert result["original_answers"] == ["42"]

    def test_special_characters_in_question(self):
        example = {
            "question": "If x = 5 & y = 3, what is x + y?",
            "answer": "x + y = 5 + 3 = 8.\n#### 8",
        }
        result = process_gsm8k_example(example)
        assert "x = 5 & y = 3" in result["input"]
        assert result["original_answers"] == ["8"]

    def test_unicode_characters(self):
        example = {
            "question": "Café sells 5 crépes at €2 each. What is the total in €?",
            "answer": "5 \u00d7 2 = 10.\n#### 10",
        }
        result = process_gsm8k_example(example)
        assert "Café" in result["input"]
        assert "crépes" in result["input"]
        assert result["original_answers"] == ["10"]

    def test_with_tokenizer_parameter(self):
        """Processor should accept and ignore the optional tokenizer argument."""
        example = {
            "question": "What is 2+2?",
            "answer": "2+2=4.\n#### 4",
        }

        class MockTokenizer:
            vocab_size = 50000

        result = process_gsm8k_example(example, MockTokenizer())
        assert result["input"] == "Question: What is 2+2? Answer:"
        assert result["original_answers"] == ["4"]

    def test_return_type_structure(self):
        example = {
            "question": "Test?",
            "answer": "Answer.\n#### 1",
        }
        result = process_gsm8k_example(example)

        assert "input" in result
        assert "output" in result
        assert "original_answers" in result

        assert isinstance(result["input"], str)
        assert isinstance(result["output"], str)
        assert isinstance(result["original_answers"], list)
        assert all(isinstance(a, str) for a in result["original_answers"])

    def test_answer_without_delimiter(self):
        """Handle answers that lack the #### delimiter gracefully."""
        example = {
            "question": "What is 1+1?",
            "answer": "The answer is 2",
        }
        result = process_gsm8k_example(example)
        assert result["output"] == "The answer is 2"
        assert result["original_answers"] == ["The answer is 2"]
