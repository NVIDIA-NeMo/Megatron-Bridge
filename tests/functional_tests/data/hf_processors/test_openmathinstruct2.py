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

from megatron.bridge.data.hf_processors.openmathinstruct2 import process_openmathinstruct2_example


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

        result = process_openmathinstruct2_example(example, tokenizer=MockTokenizer())
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
