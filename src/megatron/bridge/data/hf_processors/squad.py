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

"""Processing functions for Squad dataset."""

from typing import Any

from megatron.bridge.data.builders.hf_dataset import ProcessExampleOutput
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


def process_squad_example(
    example: dict[str, Any], tokenizer: MegatronTokenizer | None = None
) -> ProcessExampleOutput:
    """Process a single Squad example into the required format.

    This function transforms a raw Squad dataset example into the standard format
    expected by the HFDatasetBuilder for fine-tuning.

    Args:
        example: Raw Squad example containing 'context', 'question', and 'answers'
        tokenizer: Optional tokenizer (not used in this example)

    Returns:
        ProcessExampleOutput with formatted input/output and original answers

    Example:
        >>> example = {
        ...     "context": "The Amazon rainforest is a moist broadleaf forest.",
        ...     "question": "What type of forest is the Amazon rainforest?",
        ...     "answers": {
        ...         "text": ["moist broadleaf forest", "broadleaf forest"],
        ...         "answer_start": [25, 31]
        ...     }
        ... }
        >>> result = process_squad_example(example)
        >>> print(result["input"])
        Context: The Amazon rainforest is a moist broadleaf forest. Question: What type of forest is the Amazon rainforest? Answer:
    """
    # Format input as: "Context: ... Question: ... Answer:"
    _input = f"Context: {example['context']} Question: {example['question']} Answer:"

    # Use the first answer as the primary output
    _output = example["answers"]["text"][0]

    # Keep all original answers for evaluation purposes
    original_answers = example["answers"]["text"]

    return ProcessExampleOutput(input=_input, output=_output, original_answers=original_answers)
