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

from pathlib import Path
from typing import Any, Optional, Union

from datasets import DatasetDict

from nemo_lm.data.builders.hf_dataset import HFDatasetBuilder, ProcessExampleOutput
from nemo_lm.tokenizers.tokenizer import MegatronTokenizer


class SquadDatasetBuilder(HFDatasetBuilder):
    """Builder for the SQuAD (Stanford Question Answering Dataset).

    This builder configures HFDatasetBuilder to download and process the SQuAD
    dataset for tasks like question answering fine-tuning.
    """

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        tokenizer: Optional[MegatronTokenizer] = None,
        dataset_dict: Optional[DatasetDict] = None,
        dataset_subset: Optional[str] = None,
        split: Optional[str] = None,
        seq_length: int = 2048,
        seed: int = 1234,
        memmap_workers: int = 1,
        max_train_samples: Optional[int] = None,
        packed_sequence_specs: Optional[dict[str, Any]] = None,
        download_mode: Optional[str] = None,
        val_proportion: Optional[float] = 0.05, # Proportion of train to use for val if 'validation' split not used/available
        split_val_from_train: bool = True,
        delete_raw: bool = False, # Keep raw HF download by default
        hf_kwargs: Optional[dict[str, Any]] = None,
        dataset_kwargs: Optional[dict[str, Any]] = None,
        hf_filter_lambda: Optional[callable] = None,
        hf_filter_lambda_kwargs: Optional[dict[str, Any]] = None,
        do_validation: bool = True,
        do_test: bool = True,
    ):
        super().__init__(
            dataset_name="squad",
            process_example_fn=_process_squad_example,
            tokenizer=tokenizer,
            dataset_dict=dataset_dict,
            dataset_subset=dataset_subset,
            dataset_root=dataset_root,
            split=split,
            seq_length=seq_length,
            seed=seed,
            memmap_workers=memmap_workers,
            max_train_samples=max_train_samples,
            packed_sequence_specs=packed_sequence_specs,
            download_mode=download_mode,
            val_proportion=val_proportion,
            split_val_from_train=split_val_from_train,
            delete_raw=delete_raw,
            hf_kwargs=hf_kwargs,
            dataset_kwargs=dataset_kwargs,
            hf_filter_lambda=hf_filter_lambda,
            hf_filter_lambda_kwargs=hf_filter_lambda_kwargs,
            do_validation=do_validation,
            do_test=do_test,
        ) 


def _process_squad_example(example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None) -> ProcessExampleOutput:
    """Processes a single SQuAD example into the nemo_lm expected format.

    Args:
        example: A dictionary representing a single example from the SQuAD dataset.
                 Expected keys: "context", "question", "answers" (with "text" subkey).
        tokenizer: Optional tokenizer, not used in this specific processor.

    Returns:
        A ProcessExampleOutput dictionary with "input", "output", and "original_answers".
    """
    context = example.get("context", "")
    question = example.get("question", "")
    answers = example.get("answers", {}).get("text", [])

    processed_input = f"Context: {context} Question: {question} Answer:"
    processed_output = answers[0] if answers else ""

    return {
        "input": processed_input,
        "output": processed_output,
        "original_answers": answers,
    }
