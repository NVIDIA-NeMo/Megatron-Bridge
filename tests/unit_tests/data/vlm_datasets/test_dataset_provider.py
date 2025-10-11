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

import torch

from megatron.bridge.training.config import DatasetBuildContext


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    added_tokens_decoder = {}

    def __call__(self, text, add_special_tokens=False):
        # Very small deterministic tokenization
        if isinstance(text, list):
            # Map list of strings to flat ids
            return {"input_ids": [self.__call__(t, add_special_tokens=add_special_tokens)["input_ids"] for t in text]}
        ids = [1, 2, 3][: max(1, min(3, len(str(text))))]
        return {"input_ids": ids}


class _DummyProcessor:
    def __init__(self):
        self.tokenizer = _DummyTokenizer()

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):
        if tokenize:
            # Return minimal dict used by default_collate_fn
            input_ids = torch.tensor([[1, 2, 3]])
            pixel_values = torch.randn(1, 1, 3, 4, 4)
            return {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
        return "dummy"

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
        input_ids = torch.tensor([[1, 2, 3]])
        out = {"input_ids": input_ids}
        if images is not None:
            n = len(images)
            out["pixel_values"] = torch.randn(1, n, 3, 4, 4)
            out["image_grid_thw"] = torch.tensor([[[1, 2, 2]] * n])
        return out


def _example():
    return {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}


def test_vlm_conversation_dataset_basic(monkeypatch):
    # Import locally to ensure monkeypatches apply to module under test
    import megatron.bridge.data.vlm_datasets.collate as collate
    from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset

    # Enable collate dependencies
    monkeypatch.setattr(collate, "HAVE_QWEN_VL_UTILS", True)

    proc = _DummyProcessor()
    ds = VLMConversationDataset(base_examples=[_example()], target_length=3, processor=proc, collate_impl=None)
    assert len(ds) == 3
    # Wraps over base list
    assert ds[0]["conversation"][0]["role"] == "user"

    batch = ds.collate_fn([_example(), _example()])
    assert set(["input_ids", "labels", "loss_mask", "position_ids", "visual_inputs"]).issubset(batch.keys())


def test_hf_provider_builds_splits_and_binds_collate(monkeypatch):
    # Arrange monkeypatches: stub AutoProcessor and maker
    # Stub AutoProcessor.from_pretrained to avoid network
    import transformers

    from megatron.bridge.data.vlm_datasets import hf_provider as dp_mod

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: _DummyProcessor()))

    # Provide a tiny maker registry by monkeypatching _get_maker to return our lambda
    def _fake_get_maker(self):
        return lambda **kwargs: [_example(), _example()]

    monkeypatch.setattr(dp_mod.HFDatasetConversationProvider, "_get_maker", _fake_get_maker)

    provider = dp_mod.HFDatasetConversationProvider(
        sequence_length=16, hf_processor_path="dummy/model", maker_name="rdr"
    )

    ctx = DatasetBuildContext(train_samples=2, valid_samples=1, test_samples=0)
    train_ds, valid_ds, test_ds = provider.build_datasets(ctx)
    assert train_ds is not None and len(train_ds) == 2
    assert valid_ds is not None and len(valid_ds) == 1
    assert test_ds is None

    # Ensure collate_fn is bound and callable
    batch = train_ds.collate_fn([_example()])
    assert isinstance(batch, dict)
