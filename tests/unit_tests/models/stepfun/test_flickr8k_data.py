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

from types import SimpleNamespace

import pytest

from megatron.bridge.data import DatasetBuildContext
from megatron.bridge.models.stepfun.data.flickr8k import ImageForInsert, Step37Flickr8kSFTDataProvider
from megatron.bridge.models.stepfun.data.flickr8k.packing import pack
from megatron.bridge.models.stepfun.modelling_step37.image_insert_embedding import (
    ImageForInsert as ModelImageForInsert,
)


def test_flickr8k_package_uses_model_owned_image_contract():
    assert ImageForInsert is ModelImageForInsert


def test_non_truncating_packing_preserves_ranges_and_oversize_policy():
    dropped = pack([4, 3, 9, 2], max_len=8, oversize_policy="drop")
    extended = pack([4, 3, 9, 2], max_len=8, oversize_policy="extend")

    assert dropped.packed_sample_ranges == [(0, 2), (3, 1)]
    assert dropped.num_droped == 1
    assert extended.packed_sample_ranges == [(0, 2), (2, 1), (3, 1)]
    assert extended.num_droped == 0


def test_provider_keeps_singleton_collate_and_no_eval_splits(monkeypatch: pytest.MonkeyPatch):
    class FakePackedDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"index": index}

    provider = Step37Flickr8kSFTDataProvider(tokenizer_path="stepfun/model")
    packed = FakePackedDataset()
    monkeypatch.setattr(provider, "_build_train_packed_dataloader", lambda: packed)

    train, validation, test = provider.build_datasets(
        DatasetBuildContext(train_samples=1, valid_samples=1, test_samples=1)
    )

    assert train is packed
    assert validation is None
    assert test is None
    assert train.collate_fn([{"packed": True}]) == {"packed": True}
    with pytest.raises(AssertionError, match="micro_batch_size=1"):
        train.collate_fn([{}, {}])


def test_recipe_constructs_model_owned_flickr8k_provider(monkeypatch: pytest.MonkeyPatch):
    from megatron.bridge.recipes.stepfun.h100 import step37 as recipe_module

    class FakeAutoBridge:
        @staticmethod
        def from_hf_pretrained(path):
            assert path == "stepfun-ai/Step-3.7-Flash"
            return FakeAutoBridge()

        def to_megatron_provider(self, load_weights=False):
            assert load_weights is False
            return SimpleNamespace()

    monkeypatch.setattr(recipe_module, "AutoBridge", FakeAutoBridge)
    monkeypatch.setattr(recipe_module, "apply_flex_dispatcher_backend", lambda *_, **__: None)

    config = recipe_module.step37_sft_64gpu_h100_bf16_flickr8k_config()

    assert isinstance(config.dataset, Step37Flickr8kSFTDataProvider)
    assert config.dataset.tokenizer_path == "stepfun-ai/Step-3.7-Flash"
    assert config.dataset.seq_length == config.model.seq_length == 2048


def test_forward_step_imports_model_owned_preprocessor():
    from megatron.bridge.models.stepfun import step37_flickr8k_step
    from megatron.bridge.models.stepfun.data.flickr8k.preprocess import preprocess_packed_batch

    assert step37_flickr8k_step.preprocess_packed_batch is preprocess_packed_batch
