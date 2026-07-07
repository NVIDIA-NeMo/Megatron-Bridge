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
from megatron.training.config.instantiate_utils import instantiate

from megatron.bridge.training.config import ConfigContainer, DatasetBuildContext, HFConversationDatasetConfig


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


def _packable_collate(
    examples,
    processor,
    *,
    sequence_length=None,
    pad_to_max_length=False,
    pad_to_multiple_of=128,
    enable_in_batch_packing=False,
    in_batch_packing_pad_to_multiple_of=1,
):
    del (
        examples,
        processor,
        sequence_length,
        pad_to_max_length,
        pad_to_multiple_of,
        in_batch_packing_pad_to_multiple_of,
    )
    return {"enable_in_batch_packing": enable_in_batch_packing}


def _legacy_collate(examples, processor):
    del processor
    return {"num_examples": len(examples)}


class _ConversationBuilderHarness:
    """Keep the existing behavioral cases focused on the canonical builder."""

    def __init__(self, **kwargs):
        self.collate_impl = kwargs.pop("collate_impl", None)
        self.config = HFConversationDatasetConfig(**kwargs)

    def _get_maker(self):
        from megatron.bridge.data.builders import hf_conversation_dataset as builder_mod

        return builder_mod.get_hf_dataset_maker(self.config.maker_name)

    def _load_processor_or_tokenizer(self, tokenizer=None):
        from megatron.bridge.data.builders.hf_conversation_dataset import load_hf_conversation_processor

        return load_hf_conversation_processor(self.config, tokenizer)

    def build_datasets(self, context):
        from megatron.bridge.data.builders.hf_conversation_dataset import HFConversationDatasetBuilder

        return HFConversationDatasetBuilder(
            self.config,
            maker=self._get_maker(),
            collate_impl=self.collate_impl,
        ).build(context)


def test_conversation_dataset_basic():
    from megatron.bridge.data.hf_datasets.conversation_dataset import ConversationDataset

    proc = Gemma3Processor()
    ds = ConversationDataset(base_examples=[_example()], target_length=3, processor=proc, collate_impl=None)
    assert len(ds) == 3
    # Wraps over base list
    assert ds[0]["conversation"][0]["role"] == "user"

    batch = ds.collate_fn([_example(), _example()])
    assert set(["input_ids", "labels", "loss_mask", "position_ids", "visual_inputs"]).issubset(batch.keys())


def test_conversation_dataset_binds_text_chat_collate_for_messages():
    from megatron.bridge.data.hf_datasets.conversation_dataset import ConversationDataset
    from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn

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
    ds = ConversationDataset(
        base_examples=[example],
        target_length=1,
        processor=TextTokenizer(),
        collate_impl=text_chat_collate_fn,
    )

    batch = ds.collate_fn([ds[0]])

    assert batch["tokens"].tolist() == [[7, 8, 9]]
    assert batch["labels"].tolist() == [[8, 9, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0]]


def test_conversation_dataset_preserves_legacy_custom_collate_contract():
    from megatron.bridge.data.hf_datasets.conversation_dataset import ConversationDataset

    ds = ConversationDataset(
        base_examples=[_example()],
        target_length=1,
        processor=Gemma3Processor(),
        collate_impl=_legacy_collate,
        sequence_length=16,
        pad_to_max_length=True,
    )

    assert ds.collate_fn([ds[0]]) == {"num_examples": 1}


def test_conversation_dataset_rejects_legacy_custom_collate_when_packing_requested():
    from megatron.bridge.data.hf_datasets.conversation_dataset import ConversationDataset

    with pytest.raises(ValueError, match="does not accept enable_in_batch_packing=True"):
        ConversationDataset(
            base_examples=[_example()],
            target_length=1,
            processor=Gemma3Processor(),
            collate_impl=_legacy_collate,
            enable_in_batch_packing=True,
        )


def test_conversation_dataset_rejects_unknown_processor_without_collate_impl():
    from megatron.bridge.data.hf_datasets.conversation_dataset import ConversationDataset

    class UnknownProcessor:
        pass

    with pytest.raises(ValueError, match="No conversation collate function registered"):
        ConversationDataset(
            base_examples=[_example()],
            target_length=1,
            processor=UnknownProcessor(),
            collate_impl=None,
        )


def test_hf_builder_builds_splits_and_binds_collate(monkeypatch):
    # Arrange monkeypatches: stub AutoProcessor and maker
    # Stub AutoProcessor.from_pretrained to avoid network
    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: Gemma3Processor()))

    # Provide a tiny maker registry by monkeypatching _get_maker to return our lambda
    def _fake_get_maker(self):
        return lambda **kwargs: [_example(), _example()]

    monkeypatch.setattr(_ConversationBuilderHarness, "_get_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(seq_length=16, hf_processor_path="dummy/model", maker_name="rdr")

    ctx = DatasetBuildContext(train_samples=2, valid_samples=1, test_samples=0)
    train_ds, valid_ds, test_ds = provider.build_datasets(ctx)
    assert train_ds is not None and len(train_ds) == 2
    assert valid_ds is not None and len(valid_ds) == 1
    assert test_ds is None

    # Ensure collate_fn is bound and callable
    batch = train_ds.collate_fn([_example()])
    assert isinstance(batch, dict)


def test_hf_builder_defaults_trust_remote_code_false(monkeypatch):
    """Test that the HF conversation builder disables remote code by default."""
    from megatron.bridge.data.builders import hf_conversation_dataset as dp_mod

    seen = {}

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(path, trust_remote_code=None):
            seen["processor"] = (path, trust_remote_code)
            return Gemma3Processor()

    monkeypatch.setattr(dp_mod, "AutoProcessor", _AutoProcessor)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path="Qwen/attacker_processor",
        maker_name="rdr",
    )

    provider._load_processor_or_tokenizer()  # noqa: SLF001

    assert seen["processor"] == ("Qwen/attacker_processor", False)


def test_hf_builder_normalizes_missing_pad_token_to_eos():
    from megatron.bridge.data.builders.hf_conversation_dataset import normalize_hf_conversation_processor

    class _Tokenizer:
        pad_token_id = None
        pad_token = None
        eos_token_id = 2
        eos_token = "</s>"

    tokenizer = _Tokenizer()

    assert normalize_hf_conversation_processor(tokenizer) is tokenizer
    assert tokenizer.pad_token == tokenizer.eos_token


def test_hf_builder_falls_back_to_tokenizer_for_text_chat_collate(monkeypatch, caplog):
    import logging

    import transformers

    from megatron.bridge.data.builders import hf_conversation_dataset as dp_mod
    from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn

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

    def _fake_get_maker(maker_name):
        assert maker_name == "text_chat"
        return lambda **kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            }
        ]

    monkeypatch.setattr(dp_mod, "get_hf_dataset_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path="dummy/text-model",
        maker_name="text_chat",
        collate_impl=text_chat_collate_fn,
    )

    caplog.set_level(logging.DEBUG, logger=dp_mod.__name__)
    ctx = DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    assert train_ds.collate_fn([train_ds[0]])["tokens"].tolist() == [[3, 4, 5]]
    assert "falling back to AutoTokenizer" in caplog.text


def test_hf_builder_uses_context_tokenizer_when_processor_path_is_unset(monkeypatch):
    from megatron.bridge.data.builders import hf_conversation_dataset as dp_mod
    from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn

    class TextTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
            if tokenize:
                return {"input_ids": [6, 7, 8], "assistant_masks": [0, 1, 1]}
            return "rendered"

        def __call__(self, text, padding=True, truncation=False, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[6, 7, 8]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class WrappedTokenizer:
        _tokenizer = TextTokenizer()

    def _fake_get_maker(maker_name):
        assert maker_name == "text_chat"
        return lambda **kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            }
        ]

    monkeypatch.setattr(dp_mod, "get_hf_dataset_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path=None,
        maker_name="text_chat",
        collate_impl=text_chat_collate_fn,
    )

    ctx = DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0, tokenizer=WrappedTokenizer())
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    assert train_ds.collate_fn([train_ds[0]])["tokens"].tolist() == [[6, 7, 8]]


def test_hf_builder_unwraps_megatron_hf_tokenizer_for_text_chat_collate(monkeypatch):
    from megatron.bridge.data.builders import hf_conversation_dataset as dp_mod
    from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn

    class TextTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
            if tokenize:
                return {"input_ids": [6, 7, 8], "assistant_masks": [0, 1, 1]}
            return "rendered"

        def __call__(self, text, padding=True, truncation=False, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[6, 7, 8]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class MegatronHFTokenizerWrapper:
        tokenizer = TextTokenizer()

        def apply_chat_template(self, conversation, chat_template, **kwargs):
            raise AssertionError("provider should unwrap the raw HF tokenizer")

    class MegatronTokenizerTextWrapper:
        _tokenizer = MegatronHFTokenizerWrapper()

    def _fake_get_maker(maker_name):
        assert maker_name == "text_chat"
        return lambda **kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            }
        ]

    monkeypatch.setattr(dp_mod, "get_hf_dataset_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path=None,
        maker_name="text_chat",
        collate_impl=text_chat_collate_fn,
    )

    ctx = DatasetBuildContext(
        train_samples=1,
        valid_samples=0,
        test_samples=0,
        tokenizer=MegatronTokenizerTextWrapper(),
    )
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    assert train_ds.collate_fn([train_ds[0]])["tokens"].tolist() == [[6, 7, 8]]


def test_text_chat_collate_prefers_unwrapped_tokenizer_over_megatron_wrapper():
    from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn

    class TextTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
            if tokenize:
                return {"input_ids": [6, 7, 8], "assistant_masks": [0, 1, 1]}
            return "rendered"

        def __call__(self, text, padding=True, truncation=False, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[6, 7, 8]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class MegatronHFTokenizerWrapper:
        tokenizer = TextTokenizer()

        def apply_chat_template(self, conversation, chat_template, **kwargs):
            raise AssertionError("text_chat_collate_fn should prefer the raw HF tokenizer")

    example = {
        "messages": [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ]
    }

    batch = text_chat_collate_fn([example], MegatronHFTokenizerWrapper())

    assert batch["tokens"].tolist() == [[6, 7, 8]]


def test_hf_builder_enables_in_batch_packing_for_text_chat_collate(monkeypatch):
    from megatron.bridge.data.builders import hf_conversation_dataset as dp_mod
    from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn

    class TextTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
            if tokenize:
                return {"input_ids": [6, 7, 8], "assistant_masks": [0, 1, 1]}
            return "rendered"

        def __call__(self, text, padding=True, truncation=False, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[6, 7, 8]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class WrappedTokenizer:
        _tokenizer = TextTokenizer()

    def _fake_get_maker(maker_name):
        assert maker_name == "text_chat"
        return lambda **kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            }
        ]

    monkeypatch.setattr(dp_mod, "get_hf_dataset_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path=None,
        maker_name="text_chat",
        collate_impl=text_chat_collate_fn,
        enable_in_batch_packing=True,
    )

    ctx = DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0, tokenizer=WrappedTokenizer())
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    batch = train_ds.collate_fn([train_ds[0]])
    assert batch["tokens"].tolist() == [[6, 7, 8]]
    assert batch["attention_mask"] is None
    assert batch["cu_seqlens_q"].tolist() == [0, 3]
    assert batch["cu_seqlens_kv"].tolist() == [0, 3]
    assert batch["max_seqlen_q"].item() == 3
    assert batch["max_seqlen_kv"].item() == 3
    assert "cu_seqlens" not in batch
    assert "cu_seqlens_argmin" not in batch
    assert "cu_seqlens_unpadded" not in batch


def test_hf_builder_auto_selects_text_chat_collate_for_messages(monkeypatch):
    from megatron.bridge.data.builders import hf_conversation_dataset as dp_mod

    class TextTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
            if tokenize:
                return {"input_ids": [6, 7, 8], "assistant_masks": [0, 1, 1]}
            return "rendered"

        def __call__(self, text, padding=True, truncation=False, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[6, 7, 8]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class WrappedTokenizer:
        _tokenizer = TextTokenizer()

    def _fake_get_maker(maker_name):
        assert maker_name == "squad"
        return lambda **kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            }
        ]

    monkeypatch.setattr(dp_mod, "get_hf_dataset_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path=None,
        maker_name="squad",
        enable_in_batch_packing=True,
    )

    ctx = DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0, tokenizer=WrappedTokenizer())
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    batch = train_ds.collate_fn([train_ds[0]])
    assert batch["tokens"].tolist() == [[6, 7, 8]]
    assert batch["attention_mask"] is None
    assert batch["cu_seqlens_q"].tolist() == [0, 3]
    assert batch["cu_seqlens_kv"].tolist() == [0, 3]


def test_hf_builder_forwards_in_batch_packing_padding_multiple(monkeypatch):
    from megatron.bridge.data.builders import hf_conversation_dataset as dp_mod
    from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn

    class TextTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
            if tokenize:
                return {"input_ids": [6, 7, 8], "assistant_masks": [0, 1, 1]}
            return "rendered"

        def __call__(self, text, padding=True, truncation=False, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[6, 7, 8]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class WrappedTokenizer:
        _tokenizer = TextTokenizer()

    def _fake_get_maker(maker_name):
        assert maker_name == "text_chat"
        return lambda **kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            }
        ]

    monkeypatch.setattr(dp_mod, "get_hf_dataset_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path=None,
        maker_name="text_chat",
        collate_impl=text_chat_collate_fn,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    ctx = DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0, tokenizer=WrappedTokenizer())
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    batch = train_ds.collate_fn([train_ds[0]])
    assert batch["tokens"].tolist() == [[6, 7, 8, 0]]
    assert batch["cu_seqlens_q"].tolist() == [0, 3]
    assert batch["cu_seqlens_kv"].tolist() == [0, 3]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4]
    assert batch["cu_seqlens_kv_padded"].tolist() == [0, 4]
    assert "cu_seqlens" not in batch
    assert "cu_seqlens_unpadded" not in batch


def test_hf_builder_keeps_runtime_packing_out_of_conversation_dataset(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: Gemma3Processor()))

    def _fake_get_maker(self):
        return lambda **kwargs: [_example(), _example()]

    monkeypatch.setattr(_ConversationBuilderHarness, "_get_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path="dummy/model",
        maker_name="rdr",
        enable_in_batch_packing=True,
    )

    ctx = DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None and len(train_ds) == 2


def test_hf_builder_forwards_packing_to_supported_collate(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: Gemma3Processor()))

    def _fake_get_maker(self):
        return lambda **kwargs: [_example(), _example()]

    monkeypatch.setattr(_ConversationBuilderHarness, "_get_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path="dummy/model",
        maker_name="rdr",
        collate_impl=_packable_collate,
        enable_in_batch_packing=True,
    )

    ctx = DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    assert train_ds.collate_fn([_example()])["enable_in_batch_packing"] is True


def test_hf_builder_can_defer_in_batch_packing_to_training_step(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: Gemma3Processor()))

    def _fake_get_maker(self):
        return lambda **kwargs: [_example(), _example()]

    monkeypatch.setattr(_ConversationBuilderHarness, "_get_maker", _fake_get_maker)

    provider = _ConversationBuilderHarness(
        seq_length=16,
        hf_processor_path="dummy/model",
        maker_name="rdr",
        collate_impl=_packable_collate,
        enable_in_batch_packing=True,
        defer_in_batch_packing_to_step=True,
    )

    ctx = DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    assert train_ds is not None
    assert train_ds.collate_fn([_example()])["enable_in_batch_packing"] is False


def test_hf_conversation_config_round_trip_is_declarative():
    config = HFConversationDatasetConfig(
        seq_length=128,
        maker_name="text_chat",
        maker_kwargs={"path_or_dataset": "json", "data_files": {"train": "training.jsonl"}},
        do_test=False,
        enable_in_batch_packing=True,
    )

    serialized = ConfigContainer._convert_value_to_dict(config)
    restored = instantiate(serialized)

    assert isinstance(restored, HFConversationDatasetConfig)
    assert restored.maker_kwargs == config.maker_kwargs
    assert "collate_impl" not in serialized
    assert "processor" not in serialized
    assert "tokenizer" not in serialized


def test_hf_conversation_config_rejects_runtime_objects():
    config = HFConversationDatasetConfig(
        seq_length=128,
        maker_name="text_chat",
        maker_kwargs={"transform": lambda row: row},
    )

    with pytest.raises(TypeError, match="declarative values"):
        config.validate()


def test_hf_conversation_legacy_provider_is_deprecated_and_delegates(monkeypatch):
    from megatron.bridge.data.builders import hf_conversation_dataset as builder_mod
    from megatron.bridge.data.hf_datasets.provider import HFConversationDatasetProvider

    with pytest.warns(DeprecationWarning, match="HFConversationDatasetConfig"):
        provider = HFConversationDatasetProvider(
            seq_length=128,
            maker_name="text_chat",
            hf_processor_path="dummy/model",
            maker_kwargs={"path_or_dataset": "json"},
            collate_impl=_packable_collate,
            num_workers=0,
            data_sharding=False,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            trust_remote_code=True,
            skip_getting_attention_mask_from_dataset=False,
            enable_in_batch_packing=True,
            defer_in_batch_packing_to_step=True,
            pad_to_max_length=True,
            pad_to_multiple_of=64,
            in_batch_packing_pad_to_multiple_of=8,
            do_validation=False,
            do_test=False,
        )

    config = provider._to_config()  # noqa: SLF001
    assert isinstance(config, HFConversationDatasetConfig)
    assert config.maker_name == "text_chat"
    assert config.hf_processor_path == "dummy/model"
    assert config.num_workers == 0
    assert config.data_sharding is False
    assert config.pin_memory is False
    assert config.drop_last is False
    assert config.persistent_workers is False
    assert config.trust_remote_code is True
    assert config.skip_getting_attention_mask_from_dataset is False
    assert config.enable_in_batch_packing is True
    assert config.defer_in_batch_packing_to_step is True
    assert config.pad_to_max_length is True
    assert config.pad_to_multiple_of == 64
    assert config.in_batch_packing_pad_to_multiple_of == 8
    assert config.do_validation is False
    assert config.do_test is False

    monkeypatch.setattr(provider, "_get_maker", lambda: lambda **kwargs: [_example()])
    monkeypatch.setattr(builder_mod, "load_hf_conversation_processor", lambda config, tokenizer: Gemma3Processor())
    train_ds, valid_ds, test_ds = provider.build_datasets(
        DatasetBuildContext(train_samples=2, valid_samples=1, test_samples=1)
    )

    assert train_ds is not None and len(train_ds) == 2
    assert (valid_ds, test_ds) == (None, None)
    assert train_ds.collate_fn([train_ds[0]])["enable_in_batch_packing"] is False


def test_hf_conversation_config_resolves_canonical_builder(monkeypatch):
    from megatron.bridge.data import utils as data_utils

    seen = {}

    class _FakeBuilder:
        def __init__(self, config):
            seen["config"] = config

        def build(self, context):
            seen["context"] = context
            return "train", "validation", "test"

    monkeypatch.setattr(data_utils, "HFConversationDatasetBuilder", _FakeBuilder)
    config = HFConversationDatasetConfig(seq_length=128, maker_name="text_chat")
    provider = data_utils.get_dataset_provider(config)
    tokenizer = object()

    result = provider([12, 3, 1], config, tokenizer=tokenizer)

    assert result == ("train", "validation", "test")
    assert seen["config"] is config
    assert seen["context"].train_samples == 12
    assert seen["context"].tokenizer is tokenizer
