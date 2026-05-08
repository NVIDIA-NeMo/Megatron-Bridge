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

"""Unit tests for OpenMathInstruct-2 dataset processor."""

import pytest

from megatron.bridge.data.hf_processors.openmathinstruct2 import process_openmathinstruct2_example


@pytest.mark.unit
class TestProcessOpenmathinstruct2Example:
    """Test cases for process_openmathinstruct2_example function."""

    def test_basic_example(self):
        example = {
            "problem": "What is 2 + 3?",
            "generated_solution": "We add 2 and 3 to get 5.",
            "expected_answer": "5",
        }
        result = process_openmathinstruct2_example(example)

        assert result["input"] == "Problem: What is 2 + 3? Solution:"
        assert result["output"] == "We add 2 and 3 to get 5."
        assert result["original_answers"] == ["5"]

    def test_multiline_solution(self):
        example = {
            "problem": "Solve for x: 2x + 4 = 10",
            "generated_solution": (
                "Step 1: Subtract 4 from both sides: 2x = 6\nStep 2: Divide both sides by 2: x = 3\nTherefore, x = 3."
            ),
            "expected_answer": "3",
        }
        result = process_openmathinstruct2_example(example)

        assert result["input"] == "Problem: Solve for x: 2x + 4 = 10 Solution:"
        assert "Step 1" in result["output"]
        assert "Step 2" in result["output"]
        assert result["original_answers"] == ["3"]

    def test_output_is_full_generated_solution(self):
        """The output field should be the complete generated_solution, not just the answer."""
        solution = "Let's think step by step.\n2+2=4.\nThe answer is 4."
        example = {
            "problem": "What is 2+2?",
            "generated_solution": solution,
            "expected_answer": "4",
        }
        result = process_openmathinstruct2_example(example)
        assert result["output"] == solution

    def test_original_answers_is_expected_answer(self):
        example = {
            "problem": "Find the derivative of x^2.",
            "generated_solution": "d/dx(x^2) = 2x",
            "expected_answer": "2x",
        }
        result = process_openmathinstruct2_example(example)
        assert result["original_answers"] == ["2x"]

    def test_latex_in_problem(self):
        example = {
            "problem": "Compute $\\frac{3}{4} + \\frac{1}{4}$.",
            "generated_solution": "$\\frac{3}{4} + \\frac{1}{4} = \\frac{4}{4} = 1$",
            "expected_answer": "1",
        }
        result = process_openmathinstruct2_example(example)
        assert "\\frac{3}{4}" in result["input"]
        assert result["original_answers"] == ["1"]

    def test_special_characters(self):
        example = {
            "problem": "If a & b are integers where a=5, b=3, find a*b.",
            "generated_solution": "a * b = 5 * 3 = 15.",
            "expected_answer": "15",
        }
        result = process_openmathinstruct2_example(example)
        assert "a & b" in result["input"]
        assert result["original_answers"] == ["15"]

    def test_unicode_characters(self):
        example = {
            "problem": "Trouvez la somme: 5 + 7",
            "generated_solution": "5 + 7 = 12. La réponse est 12.",
            "expected_answer": "12",
        }
        result = process_openmathinstruct2_example(example)
        assert "Trouvez" in result["input"]
        assert "réponse" in result["output"]
        assert result["original_answers"] == ["12"]

    def test_empty_problem(self):
        example = {
            "problem": "",
            "generated_solution": "No problem given.",
            "expected_answer": "N/A",
        }
        result = process_openmathinstruct2_example(example)
        assert result["input"] == "Problem:  Solution:"
        assert result["output"] == "No problem given."
        assert result["original_answers"] == ["N/A"]

    def test_with_tokenizer_parameter(self):
        """Processor should accept and ignore the optional tokenizer argument."""
        example = {
            "problem": "What is 1+1?",
            "generated_solution": "1+1=2.",
            "expected_answer": "2",
        }

        class MockTokenizer:
            vocab_size = 50000

        result = process_openmathinstruct2_example(example, MockTokenizer())
        assert result["input"] == "Problem: What is 1+1? Solution:"
        assert result["original_answers"] == ["2"]

    def test_return_type_structure(self):
        example = {
            "problem": "Test",
            "generated_solution": "Solution.",
            "expected_answer": "1",
        }
        result = process_openmathinstruct2_example(example)

        assert "input" in result
        assert "output" in result
        assert "original_answers" in result

        assert isinstance(result["input"], str)
        assert isinstance(result["output"], str)
        assert isinstance(result["original_answers"], list)
        assert all(isinstance(a, str) for a in result["original_answers"])

    def test_long_problem_and_solution(self):
        long_problem = "Consider the following problem. " * 20
        long_solution = "Step: compute. " * 50
        example = {
            "problem": long_problem,
            "generated_solution": long_solution,
            "expected_answer": "42",
        }
        result = process_openmathinstruct2_example(example)

        assert result["input"] == f"Problem: {long_problem} Solution:"
        assert result["output"] == long_solution
        assert result["original_answers"] == ["42"]


@pytest.mark.unit
class TestStripIntermediateBoxed:
    """Test cases for _strip_intermediate_boxed function."""

    def test_single_boxed(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import _strip_intermediate_boxed

        assert _strip_intermediate_boxed(r"The answer is \boxed{5}.") == "The answer is 5."

    def test_nested_braces(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import _strip_intermediate_boxed

        assert _strip_intermediate_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_multiple_boxed(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import _strip_intermediate_boxed

        result = _strip_intermediate_boxed(r"First \boxed{3}, then \boxed{5}.")
        assert result == "First 3, then 5."

    def test_no_boxed(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import _strip_intermediate_boxed

        text = "No boxed expressions here."
        assert _strip_intermediate_boxed(text) == text

    def test_malformed_boxed(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import _strip_intermediate_boxed

        # Malformed \boxed{ with no closing brace — keep as-is
        text = r"Malformed \boxed{no close"
        assert _strip_intermediate_boxed(text) == text

    def test_empty_string(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import _strip_intermediate_boxed

        assert _strip_intermediate_boxed("") == ""


@pytest.mark.unit
class TestProcessOpenmathinstruct2ThinkingPackedExample:
    """Test cases for process_openmathinstruct2_thinking_packed_example."""

    def test_basic_boxed_answer(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import (
            process_openmathinstruct2_thinking_packed_example,
        )

        example = {
            "problem": "What is 2+3?",
            "generated_solution": r"We compute 2+3=5. So the answer is \boxed{5}",
            "expected_answer": "5",
        }
        result = process_openmathinstruct2_thinking_packed_example(example)

        assert result["messages"][0] == {"role": "user", "content": "What is 2+3?"}
        assistant = result["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "#### 5"
        assert "We compute 2+3=5" in assistant["thinking"]
        assert r"\boxed{5}" not in assistant["thinking"]

    def test_no_boxed_in_solution(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import (
            process_openmathinstruct2_thinking_packed_example,
        )

        example = {
            "problem": "What is 1+1?",
            "generated_solution": "The answer is 2.",
            "expected_answer": "2",
        }
        result = process_openmathinstruct2_thinking_packed_example(example)

        assistant = result["messages"][1]
        assert assistant["thinking"] == "The answer is 2."
        assert assistant["content"] == "#### 2"

    def test_intermediate_boxed_stripped_from_thinking(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import (
            process_openmathinstruct2_thinking_packed_example,
        )

        example = {
            "problem": "Solve it.",
            "generated_solution": r"First step gives \boxed{3}. Final answer is \boxed{7}",
            "expected_answer": "7",
        }
        result = process_openmathinstruct2_thinking_packed_example(example)

        assistant = result["messages"][1]
        # Intermediate \boxed{3} should be stripped to just 3
        assert r"\boxed{3}" not in assistant["thinking"]
        assert "3" in assistant["thinking"]
        assert assistant["content"] == "#### 7"

    def test_nested_boxed_answer(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import (
            process_openmathinstruct2_thinking_packed_example,
        )

        example = {
            "problem": "Simplify.",
            "generated_solution": r"The answer is \boxed{\frac{1}{2}}",
            "expected_answer": r"\frac{1}{2}",
        }
        result = process_openmathinstruct2_thinking_packed_example(example)

        assistant = result["messages"][1]
        assert assistant["content"] == r"#### \frac{1}{2}"

    def test_output_structure(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import (
            process_openmathinstruct2_thinking_packed_example,
        )

        example = {
            "problem": "Problem.",
            "generated_solution": r"Solution \boxed{42}",
            "expected_answer": "42",
        }
        result = process_openmathinstruct2_thinking_packed_example(example)

        assert result["input"] == ""
        assert result["output"] == ""
        assert result["original_answers"] == ["42"]
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_expected_answer_cast_to_string(self):
        from megatron.bridge.data.hf_processors.openmathinstruct2 import (
            process_openmathinstruct2_thinking_packed_example,
        )

        example = {
            "problem": "What is 6*7?",
            "generated_solution": r"6 times 7 is \boxed{42}",
            "expected_answer": 42,  # int, not str
        }
        result = process_openmathinstruct2_thinking_packed_example(example)

        assert result["messages"][1]["content"] == "#### 42"
        assert result["original_answers"] == ["42"]
