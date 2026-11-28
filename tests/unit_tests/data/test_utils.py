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
from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset

from megatron.bridge.data import utils as data_utils
from megatron.bridge.data.datasets.fim_dataset import GPTFIMDataset
from megatron.bridge.training.config import GPTDatasetConfig, GPTFIMDatasetConfig


pytestmark = pytest.mark.unit


def _finalized_dataset_config(config_class, *, mock, inter_document_masking, **kwargs):
    """Return a finalized dataset config; without a blend it is a mock config."""
    config = config_class(
        random_seed=1234,
        seq_length=8,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        blend=None if mock else (["unused-prefix"], None),
        split=None if mock else "1,1,1",
        **kwargs,
    )
    if not hasattr(config, "inter_document_masking"):
        raise ValueError("Config has no field 'inter_document_masking'")
    config.inter_document_masking = inter_document_masking
    config.finalize()
    assert config.mock is mock
    return config


def _built_dataset_classes(monkeypatch, config):
    """Run the GPT pretraining provider with Core's builder recorded, not run, and return the classes it got."""
    built = []

    class _RecordingBuilder:
        def __init__(self, cls, sizes, is_built_on_rank, config):
            built.append(cls)

        def build(self):
            return None, None, None

    monkeypatch.setattr(data_utils, "BlendedMegatronDatasetBuilder", _RecordingBuilder)
    data_utils.pretrain_train_valid_test_datasets_provider([1, 1, 1], config)
    return built


@pytest.mark.parametrize("fim_data", [True, False], ids=["fim_data_true", "fim_data_false"])
def test_document_masking_is_rejected_where_the_fim_dataset_is_selected(monkeypatch, fim_data):
    """The FIM dataset returns no per-document lengths, so a masked config that selects it is rejected.

    The provider selects the FIM dataset by the presence of ``fim_data``, not its value.
    """
    config = _finalized_dataset_config(
        GPTFIMDatasetConfig, mock=False, inter_document_masking=True, fim_rate=0.5, fim_spm_rate=0.5
    )
    if not hasattr(config, "fim_data"):
        raise ValueError("Config has no field 'fim_data'")
    config.fim_data = fim_data

    with pytest.raises(ValueError, match="inter_document_masking=True is not supported with the FIM dataset"):
        _built_dataset_classes(monkeypatch, config)


@pytest.mark.parametrize(
    "config_class, mock, inter_document_masking, expected",
    [
        (GPTFIMDatasetConfig, True, True, MockGPTDataset),
        (GPTFIMDatasetConfig, False, False, GPTFIMDataset),
        (GPTDatasetConfig, False, True, GPTDataset),
    ],
    ids=["masked_mock_with_fim_marker", "unmasked_fim", "masked_gpt"],
)
def test_dataset_selection_is_unchanged_outside_masked_fim(
    monkeypatch, config_class, mock, inter_document_masking, expected
):
    """A mock config builds MockGPTDataset first, even with the FIM marker; unmasked FIM still builds FIM."""
    kwargs = {"fim_rate": 0.5, "fim_spm_rate": 0.5} if config_class is GPTFIMDatasetConfig else {}
    config = _finalized_dataset_config(
        config_class, mock=mock, inter_document_masking=inter_document_masking, **kwargs
    )

    assert _built_dataset_classes(monkeypatch, config) == [expected]
