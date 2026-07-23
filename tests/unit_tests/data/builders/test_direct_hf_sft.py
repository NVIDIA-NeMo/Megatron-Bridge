# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib

import pytest
from megatron.training.config.instantiate_utils import instantiate

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    DirectHFSFTDatasetBuilder,
    DirectHFSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)
from megatron.bridge.data.builders import direct_hf_sft as builder_module
from megatron.bridge.data.builders.direct_hf_sft import (
    build_direct_hf_sft_split,
    load_direct_hf_sft_processor,
    select_direct_hf_sft_collate,
)
from megatron.bridge.data.sources.hf import resolve_hf_dataset_source
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


pytestmark = pytest.mark.unit


class _Tokenizer:
    added_tokens_decoder = {}
    pad_token_id = 0
    eos_token_id = 3
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    def apply_chat_template(self, conversation, tokenize=True, **kwargs):
        assert tokenize is True
        assert conversation[-1]["role"] == "assistant"
        return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]}


def test_legacy_hf_sft_builder_api_is_removed():
    from megatron.bridge.data import builders

    assert not hasattr(builders, "HFSFTDatasetConfig")
    assert not hasattr(builders, "HFSFTDatasetBuilder")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("megatron.bridge.data.builders.hf_sft_dataset")


@pytest.mark.parametrize("column", ["messages", "conversation", "conversations"])
def test_builder_auto_selects_shared_text_collate_for_all_chat_columns(monkeypatch, column):
    turns = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    row = (
        {column: [{"from": turn["role"], "value": turn["content"]} for turn in turns]}
        if column == "conversations"
        else {column: turns}
    )
    monkeypatch.setattr(builder_module, "load_and_adapt_hf_dataset", lambda source: [row])
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        pad_to_multiple_of=1,
        do_validation=False,
        do_test=False,
    )

    train, validation, test = DirectHFSFTDatasetBuilder(config).build(
        DatasetBuildContext(1, 0, 0, tokenizer=_Tokenizer())
    )

    assert train is not None
    assert train.collate_fn([train[0]])["tokens"].tolist() == [[1, 2, 3]]
    assert (validation, test) == (None, None)


def test_config_validates_source_and_padding():
    config = DirectHFSFTDatasetConfig(
        seq_length=0,
        source=HFDatasetSourceConfig(path_or_dataset=""),
        pad_to_multiple_of=0,
    )

    with pytest.raises(ValueError, match="seq_length"):
        config.validate()


def test_config_allows_disabled_split_sources_for_later_config_overrides():
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        validation_source=HFDatasetSourceConfig(path_or_dataset="org/validation"),
        do_validation=False,
    )

    config.validate()


@pytest.mark.parametrize("media_type", ["image", "video", "audio"])
def test_builder_leaves_multimodal_conversations_to_processor_collators(media_type):
    row = {
        "conversation": [
            {
                "role": "user",
                "content": [{"type": media_type, media_type: object()}, {"type": "text", "text": "describe"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }

    assert select_direct_hf_sft_collate([row]) is None


def test_builder_keeps_text_shaped_nemotron_omni_data_on_model_collator(monkeypatch):
    row = {"conversation": [{"role": "user", "content": "question"}]}
    processor_type = type("NemotronH_Nano_Omni_Reasoning_V3Processor", (_Tokenizer,), {})
    captured = {}
    monkeypatch.setattr(builder_module, "load_direct_hf_sft_examples", lambda source, preprocessing: [row])

    class _Dataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(builder_module, "DirectSFTDataset", _Dataset)
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        do_validation=False,
        do_test=False,
    )

    build_direct_hf_sft_split(config, config.source, 1, processor_type())

    assert captured["collate_impl"] is None


def test_builder_keeps_other_registered_processors_on_generic_text_collator(monkeypatch):
    row = {"conversation": [{"role": "user", "content": "question"}]}
    processor_type = type("Qwen3VLProcessor", (_Tokenizer,), {})
    captured = {}
    monkeypatch.setattr(builder_module, "load_direct_hf_sft_examples", lambda source, preprocessing: [row])

    class _Dataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(builder_module, "DirectSFTDataset", _Dataset)
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        do_validation=False,
        do_test=False,
    )

    build_direct_hf_sft_split(config, config.source, 1, processor_type())

    assert captured["collate_impl"] is not None


def test_builder_rejects_text_prompt_completion_for_nemotron_omni(monkeypatch):
    row = {"question": "question", "answer": "answer"}
    processor_type = type("NemotronH_Nano_Omni_Reasoning_V3Processor", (_Tokenizer,), {})
    preprocessing = PromptCompletionSFTPreprocessingConfig(prompt_column="question", completion_column="answer")
    monkeypatch.setattr(builder_module, "load_direct_hf_sft_examples", lambda source, preprocessing: [row])
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/paired"),
        preprocessing=preprocessing,
        do_validation=False,
        do_test=False,
    )

    with pytest.raises(ValueError, match="requires chat preprocessing"):
        build_direct_hf_sft_split(config, config.source, 1, processor_type())


def test_builder_preserves_canonical_conversation_key_for_multimodal_collators(monkeypatch):
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": object()}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }
    monkeypatch.setattr(builder_module, "load_and_adapt_hf_dataset", lambda source: [row])
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/vlm"),
        do_validation=False,
        do_test=False,
    )

    train, _, _ = DirectHFSFTDatasetBuilder(config, collate_impl=lambda *_args, **_kwargs: {}).build(
        DatasetBuildContext(1, 0, 0, tokenizer=_Tokenizer())
    )

    assert train is not None
    assert "conversation" in train[0]
    assert "messages" not in train[0]


@pytest.mark.parametrize(
    "row",
    [
        {"conversation": [{"role": "user", "content": "describe <image>"}], "image": object()},
        {"conversation": [{"role": "user", "content": "describe <video>"}], "videos": [object()]},
        {"conversation": [{"role": "user", "content": "transcribe"}], "audio": object()},
        {"conversation": [{"role": "user", "content": "transcribe"}], "audio_path": "audio.wav"},
        {"conversation": [{"role": "user", "content": "describe"}], "image_paths": ["image.png"]},
        {"conversation": [{"role": "user", "content": "describe"}], "video_path": "video.mp4"},
    ],
)
def test_builder_preserves_top_level_media_for_processor_collators(row):
    assert select_direct_hf_sft_collate([row]) is None


def test_builder_rejects_unsupported_multimodal_chat_loss_mode():
    row = {
        "conversation": [
            {"role": "user", "content": [{"type": "image", "image": object()}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }

    with pytest.raises(ValueError, match="only assistant chat loss"):
        select_direct_hf_sft_collate([row], ChatSFTPreprocessingConfig(loss_mode="full"))


def test_builder_does_not_classify_mixed_rows_from_first_text_example():
    text_row = {"messages": [{"role": "assistant", "content": "text"}]}
    image_row = {
        "conversation": [
            {"role": "user", "content": [{"type": "image", "image": object()}]},
            {"role": "assistant", "content": [{"type": "text", "text": "image"}]},
        ]
    }

    assert select_direct_hf_sft_collate([text_row, image_row]) is None


def test_builder_loads_all_requested_sources(monkeypatch):
    calls = []
    row = {"messages": [{"role": "assistant", "content": "answer"}]}

    def _load(source):
        source = resolve_hf_dataset_source(source)
        calls.append((source.path_or_dataset, source.split))
        return [row]

    monkeypatch.setattr(builder_module, "load_and_adapt_hf_dataset", _load)
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/train"),
        validation_source=HFDatasetSourceConfig(path_or_dataset="org/validation", split="dev"),
        test_source=HFDatasetSourceConfig(path_or_dataset="org/test", split="evaluation"),
        pad_to_multiple_of=1,
    )

    train, validation, test = DirectHFSFTDatasetBuilder(config).build(
        DatasetBuildContext(3, 2, 1, tokenizer=_Tokenizer())
    )

    assert [len(dataset) for dataset in (train, validation, test)] == [3, 2, 1]
    assert calls == [
        ("org/train", "train"),
        ("org/validation", "dev"),
        ("org/test", "evaluation"),
    ]


def test_builder_rejects_requested_implicit_unsupported_split_before_loading():
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(dataset_name="rdr"),
        do_validation=True,
        do_test=False,
        pad_to_multiple_of=1,
    )

    with pytest.raises(ValueError, match="has no validation split"):
        DirectHFSFTDatasetBuilder(config).build(DatasetBuildContext(3, 1, 0, tokenizer=_Tokenizer()))


def test_builder_forwards_runtime_packing_to_collate(monkeypatch):
    row = {"messages": [{"role": "assistant", "content": "answer"}]}
    monkeypatch.setattr(builder_module, "load_and_adapt_hf_dataset", lambda source: [row])

    def _collate(
        examples,
        processor,
        *,
        sequence_length,
        pad_to_max_length,
        pad_to_multiple_of,
        enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of,
    ):
        del examples, processor
        return {
            "sequence_length": sequence_length,
            "pad_to_max_length": pad_to_max_length,
            "pad_to_multiple_of": pad_to_multiple_of,
            "enable_in_batch_packing": enable_in_batch_packing,
            "in_batch_packing_pad_to_multiple_of": in_batch_packing_pad_to_multiple_of,
        }

    config = DirectHFSFTDatasetConfig(
        seq_length=64,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        do_validation=False,
        do_test=False,
        pad_to_max_length=True,
        pad_to_multiple_of=16,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=8,
    )

    train, _, _ = DirectHFSFTDatasetBuilder(config, collate_impl=_collate).build(
        DatasetBuildContext(1, 0, 0, tokenizer=_Tokenizer())
    )

    assert train is not None
    assert train.collate_fn([train[0]]) == {
        "sequence_length": 64,
        "pad_to_max_length": True,
        "pad_to_multiple_of": 16,
        "enable_in_batch_packing": True,
        "in_batch_packing_pad_to_multiple_of": 8,
    }


def test_processor_loading_disables_untrusted_remote_code(monkeypatch):
    seen = {}

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(path, *, trust_remote_code):
            seen["call"] = (path, trust_remote_code)
            return _Tokenizer()

    monkeypatch.setattr(builder_module, "AutoProcessor", _AutoProcessor)
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        hf_processor_path="org/processor",
    )

    processor = load_direct_hf_sft_processor(config, tokenizer=None)

    assert isinstance(processor, _Tokenizer)
    assert seen["call"] == ("org/processor", False)


def test_processor_loading_forwards_declarative_kwargs(monkeypatch):
    seen = {}

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen["call"] = (path, kwargs)
            return _Tokenizer()

    monkeypatch.setattr(builder_module, "AutoProcessor", _AutoProcessor)
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        hf_processor_path="org/processor",
        hf_processor_kwargs={"revision": "0123456789abcdef"},
    )

    processor = load_direct_hf_sft_processor(config, tokenizer=None)

    assert isinstance(processor, _Tokenizer)
    assert seen["call"] == (
        "org/processor",
        {"trust_remote_code": False, "revision": "0123456789abcdef"},
    )


def test_processor_kwargs_cannot_override_trust_policy():
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        hf_processor_path="org/processor",
        hf_processor_kwargs={"trust_remote_code": True},
    )

    with pytest.raises(ValueError, match="must not override trust_remote_code"):
        config.validate()


def test_processor_loading_falls_back_to_tokenizer(monkeypatch):
    class _AutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("processor unavailable")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _Tokenizer()

    monkeypatch.setattr(builder_module, "AutoProcessor", _AutoProcessor)
    monkeypatch.setattr(builder_module, "AutoTokenizer", _AutoTokenizer)
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        hf_processor_path="org/tokenizer",
    )

    assert isinstance(load_direct_hf_sft_processor(config, tokenizer=None), _Tokenizer)


def test_direct_hf_sft_config_round_trip_is_declarative():
    config = DirectHFSFTDatasetConfig(
        seq_length=128,
        source=HFDatasetSourceConfig(
            path_or_dataset="json",
            load_kwargs={"data_files": {"train": "training.jsonl"}},
        ),
        do_test=False,
        enable_in_batch_packing=True,
    )

    serialized = ConfigContainer._convert_value_to_dict(config)
    restored = instantiate(serialized)

    assert isinstance(restored, DirectHFSFTDatasetConfig)
    assert restored.preprocessing.loss_mode == "assistant"
    assert restored.source.load_kwargs == config.source.load_kwargs
    assert "collate_impl" not in serialized
    assert "processor" not in serialized
    assert "tokenizer" not in serialized


def test_hf_json_data_file_can_be_supplied_by_cli_override():
    config = DirectHFSFTDatasetConfig(
        seq_length=128,
        source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="train",
            load_kwargs={"data_files": {"train": None}},
        ),
        do_validation=False,
        do_test=False,
    )

    process_config_with_overrides(
        config,
        cli_overrides=["source.load_kwargs.data_files.train=/data/training.jsonl"],
    )

    assert config.source.load_kwargs == {"data_files": {"train": "/data/training.jsonl"}}
    config.validate()


def test_prompt_completion_config_round_trip_is_declarative():
    config = DirectHFSFTDatasetConfig(
        seq_length=128,
        source=HFDatasetSourceConfig(path_or_dataset="json"),
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="question",
            completion_column="answer",
            separator="\n",
            loss_mode="full",
        ),
        do_validation=False,
        do_test=False,
    )

    restored = instantiate(ConfigContainer._convert_value_to_dict(config))

    assert isinstance(restored.preprocessing, PromptCompletionSFTPreprocessingConfig)
    assert restored.preprocessing.prompt_column == "question"
    assert restored.preprocessing.loss_mode == "full"


def test_builder_uses_prompt_completion_without_chat_template(monkeypatch):
    row = {"question": "2 + 2 =", "answer": "4"}
    monkeypatch.setattr(builder_module, "load_and_adapt_hf_dataset", lambda source: [row])
    tokenizer = _Tokenizer()
    tokenizer.encode = lambda text, add_special_tokens=False: [ord(character) for character in text]
    tokenizer.apply_chat_template = lambda *args, **kwargs: pytest.fail("prompt-completion must not render chat")
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/paired"),
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="question",
            completion_column="answer",
            add_eos=True,
        ),
        pad_to_multiple_of=1,
        do_validation=False,
        do_test=False,
    )

    train, _, _ = DirectHFSFTDatasetBuilder(config).build(DatasetBuildContext(1, 0, 0, tokenizer=tokenizer))

    assert train is not None
    batch = train.collate_fn([train[0]])
    assert batch["tokens"].tolist()[0][-2:] == [ord("4"), tokenizer.eos_token_id]
    assert batch["loss_mask"].sum().item() == 2


def test_direct_hf_sft_config_resolves_canonical_builder(monkeypatch):
    from megatron.bridge.data import utils as data_utils

    seen = {}

    class _FakeBuilder:
        def __init__(self, config):
            seen["config"] = config

        def build(self, context):
            seen["context"] = context
            return "train", "validation", "test"

    monkeypatch.setattr(builder_module, "DirectHFSFTDatasetBuilder", _FakeBuilder)
    config = DirectHFSFTDatasetConfig(
        seq_length=128,
        source=HFDatasetSourceConfig(path_or_dataset="json"),
    )
    tokenizer = object()

    result = data_utils.get_dataset_provider(config)([12, 3, 1], config, tokenizer=tokenizer)

    assert result == ("train", "validation", "test")
    assert seen["config"] is config
    assert seen["context"].train_samples == 12
    assert seen["context"].tokenizer is tokenizer
