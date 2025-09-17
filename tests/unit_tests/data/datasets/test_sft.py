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

import datetime
import json
import os

import megatron.core.parallel_state as parallel_state
import numpy as np
import pytest
import torch
import torch.distributed as dist

from megatron.bridge.data.datasets.sft import GPTSFTChatDataset, GPTSFTDataset, GPTSFTPackedDataset
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


def get_gpt_sft(ensure_test_data, dataset_type="sft"):
    path = os.path.join(ensure_test_data, "datasets/sft.jsonl")
    line = {"input": "hi how are you?", "output": "I'm fine, thanks."}

    num_samples = 100
    with open(path, "w") as f:
        for i in range(num_samples):
            f.write(json.dumps(line) + "\n")

    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=f"{ensure_test_data}/tokenizers/huggingface",
    )
    tokenizer = build_tokenizer(
        tokenizer_config=tokenizer_config,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
    )

    if dataset_type == "sft":
        dataset = GPTSFTDataset(
            file_path=path,
            tokenizer=tokenizer,
            label_key="output",
            max_num_samples=num_samples,
            prompt_template="{input}\n\n### Response:\n{output}",
            truncation_field="output",
        )
    elif dataset_type == "packed":
        path = os.path.join(ensure_test_data, "datasets/sft.jsonl.idx.npy")
        dataset = GPTSFTPackedDataset(
            file_path=path,
            tokenizer=tokenizer,
            label_key="output",
            prompt_template="{input}\n\n### Response:\n{output}",
            truncation_field="output",
        )
    else:
        dataset = GPTSFTChatDataset(
            file_path=path,
            tokenizer=tokenizer,
            label_key="output",
            prompt_template="{input}\n\n### Response:\n{output}",
            truncation_field="output",
        )

    return dataset, num_samples


class TestDataGPTSFTDataset:
    @pytest.fixture(autouse=True)
    def setup_and_teardown_parallel_state(self):
        """Setup and teardown parallel state for Megatron tests."""

        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            init_process_group_kwargs = {
                "backend": "nccl" if device_count > 0 else "gloo",
                "world_size": 1,
                "rank": 0,
                "timeout": datetime.timedelta(minutes=30),
            }

            dist.init_process_group(**init_process_group_kwargs)

        assert dist.is_initialized(), "Distributed backend not initialized"
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"
        from megatron.bridge.training.initialize import _set_random_seed

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
        )

        yield

        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
                # Clean up environment variables
                for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    def test_build_samples_mapping(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data)
        dataset._build_samples_mapping()

    def test_gpt_sft_dataset(self, ensure_test_data):
        dataset, dataset_length = get_gpt_sft(ensure_test_data)

        assert len(dataset) == dataset_length
        assert type(dataset[11]) is dict
        assert type(dataset[-11]) is dict

    def test_separate_template(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data)
        template_strings, template_strings_keys = dataset._separate_template(["output"])

        assert template_strings == ["output", "\n\n### Response:\n", "{output}"]
        assert template_strings_keys == ["input", "<template>", "output"]

    def test_multiple_truncation(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data)

        template_ids = [
            [101, 102, 103, 104],
            [201, 202, 203],
            [301, 302],
        ]
        template_ids_keys = ["input", "<template>", "output"]
        context_ids, label_ids = dataset._multiple_truncation(template_ids, template_ids_keys)

        assert context_ids == [101, 102, 103, 104, 201, 202, 203]
        assert label_ids == [301, 302]

    def test_utils_func(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data)

        assert dataset._truncation([101, 102, 103, 104], 0) == []
        assert dataset._truncation([101, 102, 103, 104], 2) == [101, 102]

        assert dataset._maybe_cast_to_list([1]) == [1]
        assert dataset._maybe_cast_to_list(np.array([1])) == [1]

        assert dataset._ceil_to_nearest(1, 2) == 2

        assert dataset._collate_item([[1, 2, 3, 4, 5]], 3, 0) == [[1, 2, 3, 4, 5]]

        processed_example = {"input_ids": [0, 1, 2, 11, 54], "answer_start_idx": 3}
        assert dataset._build_loss_mask(processed_example) == [0.0, 0.0, 0.0, 1.0, 1.0]

        dataset._create_attention_mask(3)

    def test_collate_fn(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data)

        batch = [
            {
                "input_ids": [101, 102, 103, 104, 105],
                "context_ids": [101, 102],
                "answer_start_idx": 2,
                "context_length": 2,
                "answer_ids": [104, 105],
                "metadata": {"id": "ex1"},
                "token_count": 5,
            },
            {
                "input_ids": [201, 202, 203, 204],
                "context_ids": [201],
                "answer_start_idx": 1,
                "context_length": 1,
                "answer_ids": [203, 204],
                "metadata": {"id": "ex2"},
                "token_count": 4,
            },
        ]
        dataset.collate_fn(batch)


class TestDataGPTSFTPackedDataset:
    def test_gpt_sft_packed_dataset(self, ensure_test_data):
        dataset, dataset_length = get_gpt_sft(ensure_test_data, dataset_type="packed")

        assert len(dataset) == dataset_length

    def test_collate_fn(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data, dataset_type="packed")

        batch = [
            {
                "input_ids": [101, 102, 103, 104, 105],
                "context_ids": [101, 102],
                "answer_start_idx": 2,
                "context_length": 2,
                "answer_ids": [104, 105],
                "metadata": {"id": "ex1"},
                "seq_boundaries": (0, 3),
                "loss_mask": [0, 0, 0, 1, 1],
                "token_count": 5,
            },
            {
                "input_ids": [201, 202, 203, 204],
                "context_ids": [201],
                "answer_start_idx": 1,
                "context_length": 1,
                "answer_ids": [203, 204],
                "metadata": {"id": "ex2"},
                "seq_boundaries": (0, 2),
                "loss_mask": [0, 0, 1, 1],
                "token_count": 4,
            },
        ]
        dataset.collate_fn(batch)

    def test_utils_func_packed(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data, dataset_type="packed")

        assert dataset._maybe_cast_to_list([11]) == [11]
        assert dataset._maybe_cast_to_list(np.array([11])) == [11]

        processed_example = {
            "input_ids": [101, 102, 103, 104, 105],
            "seq_boundaries": (0, 3),
            "loss_mask": [0, 0, 0, 1, 1],
        }
        dataset._build_loss_mask(processed_example)

        assert dataset._build_samples_mapping() == None
        dataset._load_dataset()


class TestDataGPTSFTChatDataset:
    def test_maybe_validate_prompt_template(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data, dataset_type="chat")

        assert dataset._maybe_validate_prompt_template() == None

    def test_collate_fn(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data, dataset_type="chat")
        batch = [
            {
                "input_ids": np.array([101, 102, 103, 104, 105]),
                "context_ids": np.array([101, 102]),
                "answer_start_idx": 2,
                "context_length": 2,
                "answer_ids": np.array([104, 105]),
                "metadata": {"id": "ex1"},
                "seq_boundaries": (0, 3),
                "mask": np.array([0, 0, 0, 1, 1]),
                "metadata": {},
                "token_count": 5,
            },
            {
                "input_ids": np.array([201, 202, 203, 204]),
                "context_ids": np.array([201]),
                "answer_start_idx": 1,
                "context_length": 1,
                "answer_ids": np.array([203, 204]),
                "metadata": {"id": "ex2"},
                "seq_boundaries": (0, 2),
                "mask": np.array([0, 0, 1, 1]),
                "metadata": {},
                "token_count": 4,
            },
        ]
        dataset.collate_fn(batch)

    def test_build_samples_mapping(self, ensure_test_data):
        dataset, _ = get_gpt_sft(ensure_test_data, dataset_type="chat")
        dataset._build_samples_mapping()
