# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data.collators.sft import text_prompt_completion_collate_fn
from megatron.bridge.data.datasets.gpt_sft import GPTSFTDataset
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.sft_processing import (
    ChatSFTPreprocessingConfig,
    PromptCompletionSFTPreprocessingConfig,
    normalize_sft_example,
    tokenize_prompt_completion_example,
)
from megatron.bridge.data.sources.hf_adapters import adapt_hf_dataset


pytestmark = pytest.mark.unit


class _Tokenizer:
    bos_token_id = 101
    eos_token_id = 102
    eos_id = eos_token_id
    pad_token_id = 0
    space_sensitive = True
    added_tokens_decoder = {}

    def __init__(self):
        self.encoded = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.encoded.append(text)
        return [ord(character) for character in text]

    def apply_chat_template(self, *args, **kwargs):
        pytest.fail("prompt-completion preprocessing must not call apply_chat_template")


def test_chat_preprocessing_promotes_canonical_pair():
    row = normalize_sft_example(
        {"prompt": "question", "completion": "answer", "id": 7},
        ChatSFTPreprocessingConfig(),
    )

    assert row == {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "id": 7,
    }


def test_prompt_completion_rejects_structured_conversation():
    with pytest.raises(ValueError, match="structured conversations"):
        normalize_sft_example(
            {"messages": [{"role": "assistant", "content": "answer"}]},
            PromptCompletionSFTPreprocessingConfig(),
        )


def test_prompt_completion_tokenizes_separately_and_masks_prompt():
    tokenizer = _Tokenizer()
    preprocessing = PromptCompletionSFTPreprocessingConfig(
        prompt_column="question",
        completion_column="answer",
        prompt_template="Q: {prompt}",
        separator="\nA: ",
        add_bos=True,
        add_eos=True,
    )

    tokenized = tokenize_prompt_completion_example(
        {"question": "2+2", "answer": "4"},
        tokenizer,
        preprocessing,
    )

    assert tokenizer.encoded == ["Q: 2+2", "\nA: 4"]
    assert tokenized.input_ids.tolist() == [
        tokenizer.bos_token_id,
        *[ord(character) for character in "Q: 2+2\nA: "],
        ord("4"),
        tokenizer.eos_token_id,
    ]
    assert tokenized.loss_mask.tolist() == [
        *([False] * (len("Q: 2+2") + 1)),
        *([True] * (len("\nA: 4") + 1)),
    ]
    assert tokenized.completion_ids.tolist() == [ord(character) for character in "\nA: 4"]


def test_prompt_completion_drops_one_boundary_space_for_non_space_sensitive_tokenizer():
    tokenizer = _Tokenizer()
    tokenizer.space_sensitive = False

    tokenized = tokenize_prompt_completion_example(
        {"prompt": "Q", "completion": "A"},
        tokenizer,
        PromptCompletionSFTPreprocessingConfig(separator=" "),
    )

    assert tokenizer.encoded == ["Q", "A"]
    assert tokenized.input_ids.tolist() == [ord("Q"), ord("A"), tokenizer.eos_token_id]


def test_prompt_completion_full_loss_and_truncation_preserve_special_tokens():
    tokenizer = _Tokenizer()
    preprocessing = PromptCompletionSFTPreprocessingConfig(
        loss_mode="full",
        add_bos=True,
        add_eos=True,
        truncation_method="left",
    )

    tokenized = tokenize_prompt_completion_example(
        {"prompt": "abcd", "completion": "xy"},
        tokenizer,
        preprocessing,
        max_length=5,
    )

    assert tokenized.input_ids.tolist() == [
        tokenizer.bos_token_id,
        ord("d"),
        ord("x"),
        ord("y"),
        tokenizer.eos_token_id,
    ]
    assert tokenized.loss_mask.tolist() == [True] * 5


@pytest.mark.parametrize(
    ("truncation_method", "expected_completion"),
    [("right", "bc"), ("left", "defg")],
)
def test_prompt_completion_preserves_gpt_generation_reserve_truncation(
    truncation_method,
    expected_completion,
):
    tokenizer = _Tokenizer()
    preprocessing = PromptCompletionSFTPreprocessingConfig(truncation_method=truncation_method)

    tokenized = tokenize_prompt_completion_example(
        {"prompt": "a", "completion": "bcdefg"},
        tokenizer,
        preprocessing,
        max_length=5,
        minimum_completion_length=2,
    )

    assert tokenized.prompt_ids.tolist() == []
    assert tokenized.completion_ids.tolist() == [ord(character) for character in expected_completion]
    assert tokenized.input_ids.tolist()[-1] == tokenizer.eos_token_id


def test_squad_row_has_matching_gpt_sft_and_direct_hf_collation():
    tokenizer = _Tokenizer()
    tokenizer.added_tokens_decoder = {ord("A"): "<image>"}
    preprocessing = PromptCompletionSFTPreprocessingConfig(separator=" ")
    example = adapt_hf_dataset(
        [{"context": "C", "question": "Q", "answers": {"text": ["A"]}}],
        adapter_name="squad",
    )[0]
    gpt_dataset = GPTSFTDataset.__new__(GPTSFTDataset)
    gpt_dataset.prompt_completion_config = preprocessing
    gpt_dataset.tokenizer = tokenizer
    gpt_dataset.max_seq_length = 64
    gpt_dataset.virtual_tokens = 0
    gpt_dataset.tokens_to_generate = 0
    gpt_dataset.is_test = False
    gpt_dataset.output_original_text = False
    gpt_dataset.pad_to_max_length = False
    gpt_dataset.pad_seq_length_to_mult = 1
    gpt_dataset.ceil_to_power_2 = False
    gpt_dataset.get_attention_mask_from_fusion = True

    processed = gpt_dataset._process_example(example)
    gpt_batch = gpt_dataset.collate_fn([processed])
    direct_batch = text_prompt_completion_collate_fn(
        [example],
        tokenizer,
        preprocessing=preprocessing,
    )

    assert direct_batch["tokens"][0, :-1].tolist() == processed["input_ids"][:-1]
    shifted_loss_mask = [float(value) for value in processed["loss_mask"][1:]]
    expected_labels = [
        token_id if contributes_to_loss else IGNORE_INDEX
        for token_id, contributes_to_loss in zip(processed["input_ids"][1:], shifted_loss_mask)
    ]
    assert direct_batch["labels"][0, :-1].tolist() == expected_labels
    assert direct_batch["loss_mask"][0, :-1].tolist() == shifted_loss_mask
    assert direct_batch["loss_mask"][0, -4:-1].tolist() == [1.0, 0.0, 1.0]

    pair_count = len(processed["input_ids"]) - 1
    assert gpt_batch["tokens"][0, :pair_count].tolist() == direct_batch["tokens"][0, :pair_count].tolist()
    gpt_trainable = gpt_batch["loss_mask"][0, :pair_count].bool()
    direct_trainable = direct_batch["loss_mask"][0, :pair_count].bool()
    assert gpt_trainable.tolist() == direct_trainable.tolist()
    assert (
        gpt_batch["labels"][0, :pair_count][gpt_trainable].tolist()
        == direct_batch["labels"][0, :pair_count][direct_trainable].tolist()
    )


def test_gpt_prompt_completion_preserves_test_and_runtime_options():
    tokenizer = _Tokenizer()
    preprocessing = PromptCompletionSFTPreprocessingConfig(
        prompt_column="input",
        completion_column="output",
        add_bos=True,
        add_sep=True,
    )
    dataset = GPTSFTDataset.__new__(GPTSFTDataset)
    dataset.prompt_completion_config = preprocessing
    dataset.tokenizer = tokenizer
    dataset.max_seq_length = 8
    dataset.virtual_tokens = 1
    dataset.tokens_to_generate = 2
    dataset.is_test = True
    dataset.output_original_text = True
    dataset.sep_id = 777

    processed = dataset._process_example({"input": " Q "})

    assert processed["context_ids"][:2] == [tokenizer.bos_token_id, tokenizer.eos_token_id]
    assert processed["context_ids"][-1] == dataset.sep_id
    assert processed["answer_ids"] == []
    assert processed["metadata"]["input"] == " Q "
    assert len(processed["input_ids"]) <= dataset.max_seq_length - dataset.tokens_to_generate


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (PromptCompletionSFTPreprocessingConfig(prompt_column="same", completion_column="same"), "different"),
        (PromptCompletionSFTPreprocessingConfig(prompt_template="literal"), "placeholder"),
        (PromptCompletionSFTPreprocessingConfig(prompt_template="{unknown}"), "placeholder"),
    ],
)
def test_prompt_completion_config_validation(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()
