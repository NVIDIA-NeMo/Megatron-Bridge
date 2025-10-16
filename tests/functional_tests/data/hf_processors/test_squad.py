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

"""Unit tests for Squad dataset processor."""

from megatron.bridge.data.hf_processors.squad import process_squad_example


class TestProcessSquadExample:
    """Test cases for process_squad_example function."""

    def test_basic_squad_example(self):
        """Test processing a basic Squad example."""
        example = {
            "context": "The Amazon rainforest is a moist broadleaf forest.",
            "question": "What type of forest is the Amazon rainforest?",
            "answers": {
                "text": ["moist broadleaf forest", "broadleaf forest", "moist broadleaf"],
                "answer_start": [25, 31, 25],
            },
        }

        result = process_squad_example(example)

        # Check input formatting
        expected_input = "Context: The Amazon rainforest is a moist broadleaf forest. Question: What type of forest is the Amazon rainforest? Answer:"
        assert result["input"] == expected_input

        # Check output (should use first answer)
        assert result["output"] == "moist broadleaf forest"

        # Check original answers are preserved
        assert result["original_answers"] == ["moist broadleaf forest", "broadleaf forest", "moist broadleaf"]

    def test_single_answer(self):
        """Test processing a Squad example with a single answer."""
        example = {
            "context": "Python is a programming language.",
            "question": "What is Python?",
            "answers": {"text": ["a programming language"], "answer_start": [12]},
        }

        result = process_squad_example(example)

        expected_input = "Context: Python is a programming language. Question: What is Python? Answer:"
        assert result["input"] == expected_input
        assert result["output"] == "a programming language"
        assert result["original_answers"] == ["a programming language"]

    def test_empty_context(self):
        """Test processing a Squad example with empty context."""
        example = {
            "context": "",
            "question": "What is the capital of France?",
            "answers": {"text": ["Paris"], "answer_start": [0]},
        }

        result = process_squad_example(example)

        expected_input = "Context:  Question: What is the capital of France? Answer:"
        assert result["input"] == expected_input
        assert result["output"] == "Paris"
        assert result["original_answers"] == ["Paris"]

    def test_long_context_and_question(self):
        """Test processing a Squad example with long context and question."""
        long_context = "This is a very long context " * 10
        long_question = "This is a very long question " * 5

        example = {
            "context": long_context,
            "question": long_question,
            "answers": {"text": ["answer1", "answer2"], "answer_start": [0, 10]},
        }

        result = process_squad_example(example)

        expected_input = f"Context: {long_context} Question: {long_question} Answer:"
        assert result["input"] == expected_input
        assert result["output"] == "answer1"
        assert result["original_answers"] == ["answer1", "answer2"]

    def test_special_characters(self):
        """Test processing a Squad example with special characters."""
        example = {
            "context": "The symbol & is called an ampersand. It's used in HTML & XML.",
            "question": "What is the & symbol called?",
            "answers": {"text": ["an ampersand", "ampersand"], "answer_start": [18, 21]},
        }

        result = process_squad_example(example)

        expected_input = "Context: The symbol & is called an ampersand. It's used in HTML & XML. Question: What is the & symbol called? Answer:"
        assert result["input"] == expected_input
        assert result["output"] == "an ampersand"
        assert result["original_answers"] == ["an ampersand", "ampersand"]

    def test_unicode_characters(self):
        """Test processing a Squad example with Unicode characters."""
        example = {
            "context": "Café is a French word meaning coffee. It's pronounced like 'ka-FEY'.",
            "question": "What does café mean?",
            "answers": {"text": ["coffee"], "answer_start": [34]},
        }

        result = process_squad_example(example)

        expected_input = "Context: Café is a French word meaning coffee. It's pronounced like 'ka-FEY'. Question: What does café mean? Answer:"
        assert result["input"] == expected_input
        assert result["output"] == "coffee"
        assert result["original_answers"] == ["coffee"]

    def test_with_tokenizer_parameter(self):
        """Test that the function accepts a tokenizer parameter (even though it's not used)."""
        example = {
            "context": "Test context",
            "question": "Test question?",
            "answers": {"text": ["test answer"], "answer_start": [0]},
        }

        # Mock tokenizer (not actually used by the function)
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 50000

        mock_tokenizer = MockTokenizer()
        result = process_squad_example(example, tokenizer=mock_tokenizer)

        expected_input = "Context: Test context Question: Test question? Answer:"
        assert result["input"] == expected_input
        assert result["output"] == "test answer"
        assert result["original_answers"] == ["test answer"]

    def test_multiple_identical_answers(self):
        """Test processing a Squad example with duplicate answers."""
        example = {
            "context": "The same answer appears twice in this context.",
            "question": "What appears twice?",
            "answers": {"text": ["answer", "answer", "answer"], "answer_start": [9, 15, 25]},
        }

        result = process_squad_example(example)

        expected_input = (
            "Context: The same answer appears twice in this context. Question: What appears twice? Answer:"
        )
        assert result["input"] == expected_input
        assert result["output"] == "answer"
        assert result["original_answers"] == ["answer", "answer", "answer"]

    def test_return_type_structure(self):
        """Test that the function returns the correct structure."""
        example = {"context": "Test", "question": "Test?", "answers": {"text": ["test"], "answer_start": [0]}}

        result = process_squad_example(example)

        # Check that all required fields are present
        assert "input" in result
        assert "output" in result
        assert "original_answers" in result

        # Check types
        assert isinstance(result["input"], str)
        assert isinstance(result["output"], str)
        assert isinstance(result["original_answers"], list)
        assert all(isinstance(answer, str) for answer in result["original_answers"])

    def test_answer_order_preservation(self):
        """Test that the order of original answers is preserved."""
        example = {
            "context": "First answer, second answer, third answer.",
            "question": "What are the answers?",
            "answers": {"text": ["first", "second", "third"], "answer_start": [0, 14, 28]},
        }

        result = process_squad_example(example)

        # First answer should be the output
        assert result["output"] == "first"

        # Order should be preserved in original_answers
        assert result["original_answers"] == ["first", "second", "third"]
