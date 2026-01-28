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

import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import compile_helpers, get_blend_from_list
from megatron.core.tokenizers import MegatronTokenizer
from megatron.bridge.data.datasets.fim_dataset import GPTFIMDataset
from megatron.bridge.training.config import GPTFIMDatasetConfig
from tests.functional_tests.utils import initialize_distributed


@pytest.mark.parametrize("spm_rate", [0.0, 1.0])
@pytest.mark.parametrize("split_sample", [None, "python"])
def test_fim_gpt_dataset(spm_rate, split_sample):

    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/huggingface",
        metadata_path={"library": "huggingface"},
        additional_special_tokens=["<prefix>", "<middle>", "<suffix>", "<pad>", "<eod>"],
        include_special_tokens=True,
    )
    blend = get_blend_from_list(["/opt/data/datasets/fim/fim_text_document"])
    extra_tokens = {
        "prefix": "<prefix>",
        "middle": "<middle>",
        "suffix": "<suffix>",
        "pad": "<pad>",
        "eod": "<eod>",
    }
    seq_length = 32
    rate = 1.0
    fragment_rate = 1.0
    config = GPTFIMDatasetConfig(
        blend=blend,
        random_seed=1234,
        sequence_length=seq_length,
        split="990,9,1",
        tokenizer=tokenizer,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        fim_extra_tokens=extra_tokens,
        fim_rate=rate,
        fim_spm_rate=spm_rate,
        fim_fragment_rate=fragment_rate,
        fim_split_sample=split_sample,
    )

    datasets = BlendedMegatronDatasetBuilder(
        GPTFIMDataset, [10, 10, 10], lambda: True, config
    ).build()

    prefix_id = tokenizer.tokenize("<prefix>")[1]
    suffix_id = tokenizer.tokenize("<suffix>")[1]
    middle_id = tokenizer.tokenize("<middle>")[1]

    dataset = datasets[0]
    assert dataset.fim_rate == rate
    assert dataset.fim_spm_rate == spm_rate
    assert dataset.fragment_fim_rate == fragment_rate

    tokens = dataset[0]["tokens"].tolist()
    if split_sample:
        split_sample_id = tokenizer.tokenize(split_sample)[1]
        split_sample_index = tokens.index(split_sample_id)
        assert prefix_id == tokens[split_sample_index + 1]
    if spm_rate == 0.0:
        assert prefix_id == tokens[0]
        assert suffix_id in tokens
        assert middle_id in tokens
        assert tokens.index(suffix_id) < tokens.index(middle_id)
    else:
        assert prefix_id == tokens[0]
        assert suffix_id == tokens[1]
        assert middle_id in tokens
