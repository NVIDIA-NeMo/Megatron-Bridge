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

"""Processing functions for GSM8K (Grade School Math 8K) dataset.

Dataset: https://huggingface.co/datasets/openai/gsm8k

GSM8K contains 8.5K grade school math word problems. Each example has a
``question`` and an ``answer`` field where the answer contains chain-of-thought
reasoning followed by ``####`` and the final numerical answer.
"""

from typing import Any, Optional

from megatron.bridge.data.builders.hf_dataset import ProcessExampleOutput
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


def _extract_final_answer(answer: str) -> str:
    """Extract the final numerical answer after the ``####`` delimiter."""
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def process_gsm8k_example(
    example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
) -> ProcessExampleOutput:
    """Process a single GSM8K example into the required format.

    Transforms a raw GSM8K dataset example into the standard format expected by
    the HFDatasetBuilder for fine-tuning.

    Args:
        example: Raw GSM8K example containing 'question' and 'answer'
        tokenizer: Optional tokenizer (not used in this processor)

    Returns:
        ProcessExampleOutput with formatted input/output and original answers

    Example:
        >>> example = {
        ...     "question": "Janet has 3 apples. She buys 2 more. How many does she have?",
        ...     "answer": "Janet starts with 3 apples and buys 2 more. 3 + 2 = <<3+2=5>>5.\\n#### 5",
        ... }
        >>> result = process_gsm8k_example(example)
        >>> print(result["input"])
        Question: Janet has 3 apples. She buys 2 more. How many does she have? Answer:
    """
    _input = f"Question: {example['question']} Answer:"
    _output = example["answer"]
    final_answer = _extract_final_answer(example["answer"])

    return ProcessExampleOutput(input=_input, output=_output, original_answers=[final_answer])
