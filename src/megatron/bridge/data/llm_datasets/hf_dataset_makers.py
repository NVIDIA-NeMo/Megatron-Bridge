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

"""
Built-in maker functions that transform HuggingFace datasets into
conversation-style examples consumable by LLM tokenizers.
"""

from typing import Any, Dict, List

from datasets import load_dataset


def _single_turn(user_text: str, assistant_text: str) -> Dict[str, Any]:
    return {
        "conversation": [
            {"role": "user", "content": str(user_text)},
            {"role": "assistant", "content": str(assistant_text)},
        ]
    }


def make_openmathinstruct2_dataset(
    path_or_dataset: str = "nvidia/OpenMathInstruct-2",
    split: str = "train",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load NVIDIA OpenMathInstruct-2 and map rows to chat conversation examples.

    Expects dataset columns: `problem`, `generated_solution` or `solution`/`expected_answer`.
    Prioritizes `generated_solution` if present.
    """
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        user_q = example.get("problem") or example.get("question") or "Solve the following problem."
        # Prefer generated_solution; fallback to solution or expected_answer
        asst_a = example.get("generated_solution") or example.get("solution") or example.get("expected_answer") or ""
        return _single_turn(str(user_q), str(asst_a))

    return [format(ex) for ex in dataset]


def make_squad_v2_dataset(
    path_or_dataset: str = "rajpurkar/squad_v2",
    split: str = "train",
    include_unanswerable: bool = False,
    unanswerable_target: str = "",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load SQuAD v2 and map rows to chat conversation examples.

    Each example becomes a single user/assistant turn with the input formatted as
    "Context: <context>\nQuestion: <question>\nAnswer:" and the assistant
    content set to the first available answer text. For unanswerable items in
    SQuAD v2 (empty answers), set `include_unanswerable=True` to keep them with
    `unanswerable_target` as the assistant content; otherwise they are skipped.
    """

    # Allow callers to override dataset path/config but keep a simple default
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example: Dict[str, Any]) -> Dict[str, Any] | None:
        context = str(example.get("context", ""))
        question = str(example.get("question", ""))
        answers = example.get("answers", {})
        texts = answers.get("text") if isinstance(answers, dict) else None

        answer_text: str = ""
        if isinstance(texts, list) and len(texts) > 0 and texts[0] is not None:
            answer_text = str(texts[0])
        else:
            if not include_unanswerable:
                return None
            answer_text = str(unanswerable_target)

        user_q = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return _single_turn(user_q, answer_text)

    formatted = [format(ex) for ex in dataset]
    return [ex for ex in formatted if ex is not None]
