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

import pytest
import torch

from megatron.bridge.training.config import DatasetBuildContext


class _DummyTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 2
    added_tokens_decoder = {}
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    def __call__(self, text, add_special_tokens=False):
        # Very small deterministic tokenization
        if isinstance(text, list):
            # Map list of strings to flat ids
            return {"input_ids": [self.__call__(t, add_special_tokens=add_special_tokens)["input_ids"] for t in text]}
        ids = [1, 2, 3][: max(1, min(3, len(str(text))))]
        return {"input_ids": ids}


class Gemma3Processor:
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    def __init__(self):
        self.tokenizer = _DummyTokenizer()

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):
        if tokenize:
            if kwargs.get("return_assistant_tokens_mask"):
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 0, 0]}
            # Return minimal dict used by gemma3_vl_collate_fn
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


def _packable_collate(examples, processor, *, pack_sequences=False):
    return {"pack_sequences": pack_sequences}


def test_vlm_conversation_dataset_basic():
    from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset

    proc = Gemma3Processor()
    ds = VLMConversationDataset(base_examples=[_example()], target_length=3, processor=proc, collate_impl=None)
    assert len(ds) == 3
    # Wraps over base list
    assert ds[0]["conversation"][0]["role"] == "user"

    batch = ds.collate_fn([_example(), _example()])
    assert set(["input_ids", "labels", "loss_mask", "position_ids", "visual_inputs"]).issubset(batch.keys())


def test_vlm_conversation_dataset_binds_text_chat_collate_for_messages():
    from megatron.bridge.data.vlm_datasets.collate import text_chat_collate_fn
    from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset

    class TextTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
            if tokenize:
                return {"input_ids": [7, 8, 9], "assistant_masks": [0, 1, 1]}
            return "rendered"

        def __call__(self, text, padding=True, truncation=False, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    example = {
        "messages": [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ]
    }
    ds = VLMConversationDataset(
        base_examples=[example],
        target_length=1,
        processor=TextTokenizer(),
        collate_impl=text_chat_collate_fn,
    )

    batch = ds.collate_fn([ds[0]])

    assert batch["tokens"].tolist() == [[7, 8, 9]]
    assert batch["labels"].tolist() == [[8, 9, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0]]


def test_vlm_conversation_dataset_rejects_unknown_processor_without_collate_impl():
    from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset

    class UnknownProcessor:
        pass

    with pytest.raises(ValueError, match="No VLM collate function registered"):
        VLMConversationDataset(
            base_examples=[_example()],
            target_length=1,
            processor=UnknownProcessor(),
            collate_impl=None,
        )


def test_hf_provider_builds_splits_and_binds_collate(monkeypatch):
    # Arrange monkeypatches: stub AutoProcessor and maker
    # Stub AutoProcessor.from_pretrained to avoid network
    import transformers

    from megatron.bridge.data.vlm_datasets import hf_provider as dp_mod

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: Gemma3Processor()))

    # Provide a tiny maker registry by monkeypatching _get_maker to return our lambda
    def _fake_get_maker(self):
        return lambda **kwargs: [_example(), _example()]

    monkeypatch.setattr(dp_mod.HFDatasetConversationProvider, "_get_maker", _fake_get_maker)

    provider = dp_mod.HFDatasetConversationProvider(seq_length=16, hf_processor_path="dummy/model", maker_name="rdr")

    ctx = DatasetBuildContext(train_samples=2, valid_samples=1, test_samples=0)
    train_ds, valid_ds, test_ds = provider.build_datasets(ctx)
    assert train_ds is not None and len(train_ds) == 2
    assert valid_ds is not None and len(valid_ds) == 1
    assert test_ds is None

    # Ensure collate_fn is bound and callable
    batch = train_ds.collate_fn([_example()])
    assert isinstance(batch, dict)


def test_hf_provider_falls_back_to_tokenizer_for_text_chat_collate(monkeypatch):
    import transformers

    from megatron.bridge.data.vlm_datasets import hf_provider as dp_mod
    from megatron.bridge.data.vlm_datasets.collate import text_chat_collate_fn

    class TextTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
            if tokenize:
                return {"input_ids": [3, 4, 5], "assistant_masks": [0, 1, 1]}
            return "rendered"

        def __call__(self, text, padding=True, truncation=False, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(ValueError("no processor"))),
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: TextTokenizer()))

    monkeypatch.setattr(
        dp_mod,
        "make_text_chat_dataset",
        lambda **kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            }
        ],
    )

    provider = dp_mod.HFDatasetConversationProvider(
        seq_length=16,
        hf_processor_path="dummy/text-model",
        maker_name="text_chat",
        collate_impl=text_chat_collate_fn,
    )

    ctx = DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    assert train_ds.collate_fn([train_ds[0]])["tokens"].tolist() == [[3, 4, 5]]


def test_hf_provider_keeps_runtime_packing_out_of_conversation_dataset(monkeypatch):
    import transformers

    from megatron.bridge.data.vlm_datasets import hf_provider as dp_mod

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: Gemma3Processor()))

    def _fake_get_maker(self):
        return lambda **kwargs: [_example(), _example()]

    monkeypatch.setattr(dp_mod.HFDatasetConversationProvider, "_get_maker", _fake_get_maker)

    provider = dp_mod.HFDatasetConversationProvider(
        seq_length=16,
        hf_processor_path="dummy/model",
        maker_name="rdr",
        pack_sequences_in_batch=True,
    )

    ctx = DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None and len(train_ds) == 2


def test_hf_provider_forwards_packing_to_supported_collate(monkeypatch):
    import transformers

    from megatron.bridge.data.vlm_datasets import hf_provider as dp_mod

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: Gemma3Processor()))

    def _fake_get_maker(self):
        return lambda **kwargs: [_example(), _example()]

    monkeypatch.setattr(dp_mod.HFDatasetConversationProvider, "_get_maker", _fake_get_maker)

    provider = dp_mod.HFDatasetConversationProvider(
        seq_length=16,
        hf_processor_path="dummy/model",
        maker_name="rdr",
        collate_impl=_packable_collate,
        pack_sequences_in_batch=True,
    )

    ctx = DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    assert train_ds.collate_fn([_example()])["pack_sequences"] is True
