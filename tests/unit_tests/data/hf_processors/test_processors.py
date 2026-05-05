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

#
# Test purpose:
# - Cover the three HuggingFace processors in
#   `megatron.bridge.data.hf_processors`. All three are pure functions
#   (dict-in / dict-out, no I/O, no tokenizer dependency) but had no
#   unit-test coverage — only functional tests under
#   `tests/functional_tests/test_groups/data/hf_processors/`, which
#   require the GPU container to run.
# - Adding fast unit coverage means recipe / dataset regressions in the
#   processor formatting (input prefix, answer extraction, original
#   answers list) get caught at L0 unit-test time instead of waiting
#   for a functional CI slot.
#

import pytest

from megatron.bridge.data.hf_processors.gsm8k import _extract_final_answer, process_gsm8k_example
from megatron.bridge.data.hf_processors.openmathinstruct2 import process_openmathinstruct2_example
from megatron.bridge.data.hf_processors.squad import process_squad_example


# -----------------------------------------------------------------------------
# Squad
# -----------------------------------------------------------------------------


class TestProcessSquadExample:
    """Tests for `process_squad_example`."""

    def test_basic_example(self):
        """Happy path: full Context/Question/Answer formatting."""
        example = {
            "context": "The Amazon rainforest is a moist broadleaf forest.",
            "question": "What type of forest is the Amazon rainforest?",
            "answers": {
                "text": ["moist broadleaf forest", "broadleaf forest"],
                "answer_start": [25, 31],
            },
        }
        result = process_squad_example(example)

        assert result["input"] == (
            "Context: The Amazon rainforest is a moist broadleaf forest. "
            "Question: What type of forest is the Amazon rainforest? Answer:"
        )
        # Output is the FIRST answer in the answers list.
        assert result["output"] == "moist broadleaf forest"
        # original_answers preserves all answer alternatives for eval.
        assert result["original_answers"] == ["moist broadleaf forest", "broadleaf forest"]

    def test_single_answer(self):
        """A single-answer example still produces a valid output."""
        example = {
            "context": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answers": {"text": ["Paris"], "answer_start": [0]},
        }
        result = process_squad_example(example)

        assert result["output"] == "Paris"
        assert result["original_answers"] == ["Paris"]

    def test_input_prefix_format_is_documented(self):
        """The input string strictly follows `Context: ... Question: ... Answer:`."""
        example = {
            "context": "ctx",
            "question": "q?",
            "answers": {"text": ["a"], "answer_start": [0]},
        }
        result = process_squad_example(example)
        assert result["input"] == "Context: ctx Question: q? Answer:"
        # The trailing "Answer:" is bare — no space, no answer text.
        assert result["input"].endswith("Answer:")

    def test_tokenizer_argument_is_unused(self):
        """The processor accepts a tokenizer arg but does not use it."""
        example = {
            "context": "ctx",
            "question": "q?",
            "answers": {"text": ["a"], "answer_start": [0]},
        }
        # A sentinel tokenizer should produce the same result as None.
        result_none = process_squad_example(example, tokenizer=None)
        result_with_tok = process_squad_example(example, tokenizer=object())
        assert result_none == result_with_tok

    def test_missing_required_field_raises_key_error(self):
        """Missing 'context'/'question'/'answers' surfaces a KeyError clearly."""
        with pytest.raises(KeyError, match="context"):
            process_squad_example(
                {"question": "q?", "answers": {"text": ["a"], "answer_start": [0]}},
            )


# -----------------------------------------------------------------------------
# GSM8K
# -----------------------------------------------------------------------------


class TestExtractFinalAnswer:
    """Tests for `_extract_final_answer` — the GSM8K helper for the `####` delimiter."""

    def test_extracts_value_after_delimiter(self):
        assert _extract_final_answer("Reasoning here.\n#### 42") == "42"

    def test_strips_surrounding_whitespace(self):
        assert _extract_final_answer("Reasoning.\n####   72  ") == "72"

    def test_no_delimiter_returns_full_answer_stripped(self):
        """When the answer has no `####`, the helper returns it stripped."""
        assert _extract_final_answer("  Just an answer.  ") == "Just an answer."

    def test_only_delimiter_with_empty_after(self):
        """`####` followed by whitespace returns the empty string."""
        assert _extract_final_answer("Reasoning.\n####  ") == ""

    def test_multiple_delimiters_uses_last(self):
        """If `####` appears multiple times, the LAST split wins."""
        assert _extract_final_answer("a #### b #### final") == "final"


class TestProcessGsm8kExample:
    """Tests for `process_gsm8k_example`."""

    def test_basic_example(self):
        example = {
            "question": "Janet has 3 apples. She buys 2 more. How many does she have?",
            "answer": "Janet starts with 3 apples and buys 2 more. 3 + 2 = <<3+2=5>>5.\n#### 5",
        }
        result = process_gsm8k_example(example)

        assert result["input"] == ("Question: Janet has 3 apples. She buys 2 more. How many does she have? Answer:")
        # Output is the FULL answer (chain of thought + final answer).
        assert result["output"] == example["answer"]
        # original_answers contains ONLY the extracted final numerical answer.
        assert result["original_answers"] == ["5"]

    def test_answer_without_delimiter_keeps_full_string_in_originals(self):
        """No `####` ⇒ original_answers gets the stripped full answer."""
        example = {
            "question": "What is the meaning of life?",
            "answer": "  The answer is undefined.  ",
        }
        result = process_gsm8k_example(example)

        assert result["original_answers"] == ["The answer is undefined."]

    def test_input_prefix_format_is_documented(self):
        """The input strictly follows `Question: ... Answer:`."""
        example = {"question": "q?", "answer": "a #### 1"}
        result = process_gsm8k_example(example)
        assert result["input"] == "Question: q? Answer:"
        assert result["input"].endswith("Answer:")

    def test_tokenizer_argument_is_unused(self):
        example = {"question": "q?", "answer": "a #### 1"}
        result_none = process_gsm8k_example(example, _tokenizer=None)
        result_with_tok = process_gsm8k_example(example, _tokenizer=object())
        assert result_none == result_with_tok


# -----------------------------------------------------------------------------
# OpenMathInstruct-2
# -----------------------------------------------------------------------------


class TestProcessOpenMathInstruct2Example:
    """Tests for `process_openmathinstruct2_example`."""

    def test_basic_example(self):
        example = {
            "problem": "What is 2 + 3?",
            "generated_solution": "We add 2 and 3 to get 5.",
            "expected_answer": "5",
        }
        result = process_openmathinstruct2_example(example)

        assert result["input"] == "Problem: What is 2 + 3? Solution:"
        # Output is the GENERATED solution (chain of thought).
        assert result["output"] == "We add 2 and 3 to get 5."
        # original_answers wraps expected_answer in a one-element list.
        assert result["original_answers"] == ["5"]

    def test_input_prefix_format_is_documented(self):
        """The input strictly follows `Problem: ... Solution:`."""
        example = {
            "problem": "p?",
            "generated_solution": "s",
            "expected_answer": "a",
        }
        result = process_openmathinstruct2_example(example)
        assert result["input"] == "Problem: p? Solution:"
        assert result["input"].endswith("Solution:")

    def test_expected_answer_is_preserved_verbatim(self):
        """The expected answer (string) is wrapped in a list, not parsed."""
        example = {
            "problem": "p?",
            "generated_solution": "s",
            "expected_answer": "  the answer is 7  ",
        }
        result = process_openmathinstruct2_example(example)
        # No stripping — the processor preserves whatever the dataset gave it.
        assert result["original_answers"] == ["  the answer is 7  "]

    def test_tokenizer_argument_is_unused(self):
        example = {"problem": "p?", "generated_solution": "s", "expected_answer": "a"}
        result_none = process_openmathinstruct2_example(example, _tokenizer=None)
        result_with_tok = process_openmathinstruct2_example(example, _tokenizer=object())
        assert result_none == result_with_tok

    def test_missing_required_field_raises_key_error(self):
        """Missing any of the three required fields surfaces a KeyError."""
        with pytest.raises(KeyError):
            process_openmathinstruct2_example(
                {"problem": "p", "generated_solution": "s"},  # missing expected_answer
            )


# -----------------------------------------------------------------------------
# Cross-processor invariants
# -----------------------------------------------------------------------------


class TestProcessorOutputContract:
    """Every processor returns the same output shape: input/output/original_answers."""

    @pytest.mark.parametrize(
        "processor,example",
        [
            (
                process_squad_example,
                {
                    "context": "ctx",
                    "question": "q?",
                    "answers": {"text": ["a"], "answer_start": [0]},
                },
            ),
            (
                process_gsm8k_example,
                {"question": "q?", "answer": "a #### 1"},
            ),
            (
                process_openmathinstruct2_example,
                {"problem": "p?", "generated_solution": "s", "expected_answer": "a"},
            ),
        ],
        ids=["squad", "gsm8k", "openmathinstruct2"],
    )
    def test_output_has_required_keys_and_types(self, processor, example):
        result = processor(example)
        assert isinstance(result, dict)
        assert "input" in result
        assert "output" in result
        assert "original_answers" in result
        assert isinstance(result["input"], str)
        assert isinstance(result["output"], str)
        assert isinstance(result["original_answers"], list)
        assert all(isinstance(a, str) for a in result["original_answers"])
        assert len(result["original_answers"]) >= 1
