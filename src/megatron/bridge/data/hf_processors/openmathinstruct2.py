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

"""Processing functions for OpenMathInstruct-2 dataset.

Dataset: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2

OpenMathInstruct-2 contains math problems with generated solutions. Each example
has ``problem``, ``generated_solution``, and ``expected_answer`` fields.
"""

from typing import Any, Optional

from megatron.bridge.data.builders.hf_dataset import ProcessExampleOutput
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


def process_openmathinstruct2_example(
    example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
) -> ProcessExampleOutput:
    """Process a single OpenMathInstruct-2 example into the required format.

    Transforms a raw OpenMathInstruct-2 dataset example into the standard format
    expected by the HFDatasetBuilder for fine-tuning.

    Args:
        example: Raw example containing 'problem', 'generated_solution', and 'expected_answer'
        tokenizer: Optional tokenizer (not used in this processor)

    Returns:
        ProcessExampleOutput with formatted input/output and original answers

    Example:
        >>> example = {
        ...     "problem": "What is 2 + 3?",
        ...     "generated_solution": "We add 2 and 3 to get 5.",
        ...     "expected_answer": "5",
        ... }
        >>> result = process_openmathinstruct2_example(example)
        >>> print(result["input"])
        Problem: What is 2 + 3? Solution:
    """
    _input = f"Problem: {example['problem']} Solution:"
    _output = example["generated_solution"]
    expected_answer = example["expected_answer"]

    return ProcessExampleOutput(input=_input, output=_output, original_answers=[expected_answer])
