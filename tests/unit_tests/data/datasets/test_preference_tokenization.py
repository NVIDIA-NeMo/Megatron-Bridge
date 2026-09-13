from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import pytest
import torch

from megatron.bridge.data.batch_utils import split_batch_into_microbatches
from megatron.bridge.data.datasets.preference import build_preference_data_loader
from megatron.bridge.data.datasets.preference_pair import PreferencePairDataset
from megatron.bridge.data.datasets.preference_tokenization import build_pair_conversations, tokenize_conversation
from tests.unit_tests.data.preference_fakes import ASSISTANT_HEADER, USER_HEADER, FakeChatTokenizer


class BatchEncodingChatTokenizer(FakeChatTokenizer):
    """transformers>=5 shape: dict-like return, single conversation batched as [[ids]]."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        return {"input_ids": [ids]}


def conversation(prompt="hello there", completion="general kenobi"):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def chat_source(num_pairs):
    return [
        {"chosen": conversation(f"prompt {p}"), "rejected": conversation(f"prompt {p}", f"worse answer {p}")}
        for p in range(num_pairs)
    ]


def explicit_chat_source(num_pairs, with_system=False):
    """The same pairs as ``chat_source`` in TRL's explicit-prompt layout:
    a shared ``messages`` prompt plus completion-only chosen/rejected."""
    return [
        {
            "messages": ([{"role": "system", "content": "be terse"}] if with_system else [])
            + [{"role": "user", "content": f"prompt {p}"}],
            "chosen": [{"role": "assistant", "content": "general kenobi"}],
            "rejected": [{"role": "assistant", "content": f"worse answer {p}"}],
            # Rows may carry extra fields the dataset must ignore.
            "preference_margin": 0.5,
            "pairs_metadata": {"quality_score": {"chosen": 0.0}},
        }
        for p in range(num_pairs)
    ]


def build_chat_preference_records(
    source: Iterable[Mapping[str, Any]],
    tokenizer,
    max_seq_length: int,
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    prompt_key: str | None = None,
    num_pairs: int = 0,
) -> tuple[list[dict], Counter]:
    """Eager twin of PreferencePairDataset: drops bad pairs whole and counts the reasons."""
    records: list[dict] = []
    drops: Counter = Counter()
    for example in source:
        if num_pairs and len(records) >= num_pairs:
            break
        pair = build_pair_conversations(
            example, chosen_key=chosen_key, rejected_key=rejected_key, prompt_key=prompt_key
        )
        if isinstance(pair, str):
            drops[pair] += 1
            continue
        side_c = tokenize_conversation(tokenizer, pair[0], max_seq_length)
        side_r = tokenize_conversation(tokenizer, pair[1], max_seq_length)
        if isinstance(side_c, str) or isinstance(side_r, str):
            drops[side_c if isinstance(side_c, str) else side_r] += 1
            continue
        records.append(
            {
                "pair_id": len(records),
                "chosen_input_ids": side_c[0],
                "chosen_context_len": side_c[1],
                "rejected_input_ids": side_r[0],
                "rejected_context_len": side_r[1],
            }
        )
    return records, drops


def test_tokenize_conversation_returns_ids_with_context_prefix():
    result = tokenize_conversation(FakeChatTokenizer(), conversation(), max_seq_length=100)
    assert not isinstance(result, str), f"unexpected drop: {result}"
    input_ids, context_len = result
    # Context = user header + 2 prompt words; the assistant header belongs to the completion.
    assert context_len == 3
    assert input_ids[:context_len] == [USER_HEADER, 105, 105]
    # Completion = assistant header + two words + appended EOS.
    assert input_ids[context_len:] == [ASSISTANT_HEADER, 107, 106, FakeChatTokenizer.eos_token_id]


def test_context_len_is_a_true_prefix_of_the_single_full_render():
    """The boundary comes from one render, so it cannot disagree with input_ids."""
    multi_turn = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "final answer here"},
    ]
    input_ids, context_len = tokenize_conversation(FakeChatTokenizer(), multi_turn, max_seq_length=100)

    # Only the last assistant turn is completion; earlier assistant turns stay in context.
    assert input_ids[context_len] == ASSISTANT_HEADER
    assert len(input_ids) - context_len == 5  # header + three words + EOS


@pytest.mark.parametrize(
    ("messages", "max_seq_length", "reason"),
    [
        ([{"role": "user", "content": "no reply"}], 100, "no_assistant_completion"),
        (conversation(completion=""), 100, "empty_completion"),
        (conversation(), 3, "over_length"),
    ],
)
def test_tokenize_conversation_drop_reasons(messages, max_seq_length, reason):
    assert tokenize_conversation(FakeChatTokenizer(), messages, max_seq_length) == reason


class ChatMLTokenizer:
    """Character-level stand-in that renders exactly like the Qwen2.5 ChatML template."""

    pad_token_id = 0
    eos_token_id = 1
    extra_generation_newline = False

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("\n" if self.extra_generation_newline else "")
        return [ord(c) for c in text]


class TrailingNewlineChatMLTokenizer(ChatMLTokenizer):
    """Generation prompt ends with a newline the full render does not contain."""

    extra_generation_newline = True


@pytest.mark.parametrize(
    "messages",
    [
        conversation(),
        [{"role": "system", "content": "be terse"}, *conversation()],
        [*conversation("q1", "a1"), *conversation("q2", "a2")],
        conversation("line one\nline two", "reply\nwith\nnewlines"),
    ],
    ids=["single_turn", "with_system", "multi_turn", "multiline_content"],
)
def test_context_len_equals_the_prior_turns_render_length_for_chatml(messages):
    """The boundary sits at the start of the final assistant turn, header included."""
    tokenizer = ChatMLTokenizer()
    input_ids, context_len = tokenize_conversation(tokenizer, messages, max_seq_length=10_000)

    prior_turns = tokenizer.apply_chat_template(messages[:-1])
    assert context_len == len(prior_turns)
    assert input_ids[:context_len] == prior_turns


def test_completion_span_covers_the_whole_final_assistant_turn_plus_eos():
    """NeMo-RL alignment: the scored span is header + content + terminator + EOS."""
    tokenizer = ChatMLTokenizer()
    input_ids, context_len = tokenize_conversation(tokenizer, conversation(), max_seq_length=10_000)
    completion = "".join(map(chr, input_ids[context_len:]))
    assert completion == "<|im_start|>assistant\ngeneral kenobi<|im_end|>\n" + chr(tokenizer.eos_token_id)


class EosTerminatedChatTokenizer(FakeChatTokenizer):
    """Render already ends with EOS, as instruct-tuned chat templates do."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        if not add_generation_prompt:
            ids.append(self.eos_token_id)
        return ids


def test_eos_is_not_appended_when_the_render_already_ends_with_it():
    input_ids, _ = tokenize_conversation(EosTerminatedChatTokenizer(), conversation(), max_seq_length=100)
    assert input_ids[-1] == FakeChatTokenizer.eos_token_id
    assert input_ids[-2] != FakeChatTokenizer.eos_token_id


def test_a_generation_prompt_that_is_not_a_prefix_no_longer_costs_the_pair():
    """Previously dropped as template_prefix_mismatch; the boundary is now derived in-render."""
    input_ids, context_len = tokenize_conversation(
        TrailingNewlineChatMLTokenizer(), conversation(), max_seq_length=10_000
    )
    assert "".join(map(chr, input_ids)).startswith("".join(map(chr, input_ids[:context_len])))
    assert "".join(map(chr, input_ids[context_len:])) == "<|im_start|>assistant\ngeneral kenobi<|im_end|>\n" + chr(
        ChatMLTokenizer.eos_token_id
    )


class CountingChatTokenizer(FakeChatTokenizer):
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        self.calls += 1
        return super().apply_chat_template(messages, tokenize, add_generation_prompt)


def test_dataset_tokenizes_on_fetch_and_stubs_invalid_pairs():
    source = chat_source(3) + [{"chosen": conversation("one prompt"), "rejected": conversation("another prompt")}]
    tokenizer = CountingChatTokenizer()
    dataset = PreferencePairDataset(source, tokenizer, max_seq_length=100)

    assert len(dataset) == 4  # invalid pair kept, not dropped
    assert tokenizer.calls == 0  # nothing tokenized at init

    record = dataset[0]
    assert tokenizer.calls == 4  # 2 sides x (full render + boundary render) — tokenized on fetch
    assert record["loss_multiplier"] == 1.0
    assert record["pair_id"] == 0

    stub = dataset[3]
    assert stub["loss_multiplier"] == 0.0
    assert stub["chosen_context_len"] == 1
    assert len(stub["chosen_input_ids"]) == 2  # two-token stub, valid for the collate


def test_dataset_loss_multiplier_rides_row_aligned_through_the_loader():
    over_length_pair = {"chosen": conversation(completion="x " * 50), "rejected": conversation()}
    source = chat_source(7) + [over_length_pair]
    # BatchEncodingChatTokenizer also covers the transformers>=5 return shape end-to-end.
    dataset = PreferencePairDataset(source, BatchEncodingChatTokenizer(), max_seq_length=20)
    loader = build_preference_data_loader(
        dataset=dataset,
        micro_batch_size=4,
        global_batch_size=8,
        data_parallel_rank=0,
        data_parallel_size=1,
        consumed_samples=0,
        pin_memory=False,
    )
    for global_batch in loader:
        for mb in split_batch_into_microbatches(global_batch, 2):
            assert torch.equal(mb["pair_id"][::2], mb["pair_id"][1::2])
            # Both rows of a stubbed pair are zeroed together.
            assert torch.equal(mb["loss_multiplier"][::2], mb["loss_multiplier"][1::2])
            dead = mb["loss_multiplier"] == 0.0
            assert torch.equal(dead, mb["pair_id"] == 7), "only the over-length pair is dead"


def test_dataset_joins_ref_logprobs_by_source_index():
    # The collate asserts each row's completion-token count matches the artifact's, so the
    # ref entries must carry the counts this tokenization actually produces
    # (completion tokens == len(input_ids) - context_len).
    probe = PreferencePairDataset(chat_source(4), FakeChatTokenizer(), max_seq_length=100)
    ref = {
        p: {
            "ref_chosen_logprob_sum": p + 0.25,
            "ref_chosen_num_tokens": len(probe[p]["chosen_input_ids"]) - probe[p]["chosen_context_len"],
            "ref_rejected_logprob_sum": p + 0.75,
            "ref_rejected_num_tokens": len(probe[p]["rejected_input_ids"]) - probe[p]["rejected_context_len"],
        }
        for p in range(4)
    }
    dataset = PreferencePairDataset(chat_source(4), FakeChatTokenizer(), max_seq_length=100, ref_logprobs=ref)
    assert dataset.require_ref_logprobs
    batch = dataset.collate_fn([dataset[i] for i in range(4)])
    assert torch.equal(batch["ref_logprob_sum"][::2], batch["pair_id"][::2].float() + 0.25)

    incomplete = PreferencePairDataset(
        chat_source(4), FakeChatTokenizer(), max_seq_length=100, ref_logprobs={0: ref[0]}
    )
    with pytest.raises(ValueError, match="no entry for pair_id"):
        incomplete[1]


@pytest.mark.parametrize("tokenizer_cls", [FakeChatTokenizer, ChatMLTokenizer], ids=["fake", "chatml"])
def test_explicit_prompt_rows_tokenize_identically_to_implicit_rows(tokenizer_cls):
    """Load-bearing: the new mode reduces to the already-validated implicit path."""
    implicit, _ = build_chat_preference_records(chat_source(3), tokenizer_cls(), max_seq_length=1000)
    explicit, drops = build_chat_preference_records(
        explicit_chat_source(3), tokenizer_cls(), max_seq_length=1000, prompt_key="messages"
    )
    assert not drops
    assert explicit == implicit


def test_explicit_prompt_carries_multi_message_prompts():
    """A leading system turn belongs to the shared context, not to either completion."""
    records, drops = build_chat_preference_records(
        explicit_chat_source(1, with_system=True), ChatMLTokenizer(), max_seq_length=1000, prompt_key="messages"
    )
    assert not drops
    record = records[0]
    context = "".join(map(chr, record["chosen_input_ids"][: record["chosen_context_len"]]))
    assert context == "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nprompt 0<|im_end|>\n"


def test_explicit_prompt_mode_keeps_pairs_whose_completions_share_no_prefix():
    """The shared prefix is shared by construction, so the mismatch check does not apply."""
    row = {
        "messages": [{"role": "user", "content": "a question"}],
        "chosen": [{"role": "assistant", "content": "yes"}],
        "rejected": [{"role": "assistant", "content": "no"}],
    }
    records, drops = build_chat_preference_records(
        [row], FakeChatTokenizer(), max_seq_length=100, prompt_key="messages"
    )
    assert not drops and len(records) == 1


def test_explicit_prompt_rows_read_without_prompt_key_are_all_dead():
    """Documents the misconfiguration: no crash, just zero training signal."""
    dataset = PreferencePairDataset(explicit_chat_source(3), FakeChatTokenizer(), max_seq_length=100)
    assert [dataset[i]["loss_multiplier"] for i in range(3)] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("completion", "max_seq_length", "reason"),
    [
        ([], 100, "no_assistant_completion"),
        ([{"role": "user", "content": "not a completion"}], 100, "no_assistant_completion"),
        ([{"role": "assistant", "content": ""}], 100, "empty_completion"),
        ([{"role": "assistant", "content": "x " * 50}], 20, "over_length"),
    ],
    ids=["empty_list", "wrong_role", "empty_content", "over_length"],
)
def test_explicit_prompt_malformed_rows_drop_by_reason(completion, max_seq_length, reason):
    row = {"messages": [{"role": "user", "content": "a question"}], "chosen": completion, "rejected": completion}
    _, drops = build_chat_preference_records(
        [row], FakeChatTokenizer(), max_seq_length=max_seq_length, prompt_key="messages"
    )
    assert dict(drops) == {reason: 1}


def test_dataset_stubs_malformed_explicit_rows_without_shrinking():
    source = explicit_chat_source(2) + [
        {"messages": [{"role": "user", "content": "q"}], "chosen": [], "rejected": []},
    ]
    dataset = PreferencePairDataset(source, FakeChatTokenizer(), max_seq_length=100, prompt_key="messages")
    assert len(dataset) == 3
    assert [dataset[i]["loss_multiplier"] for i in range(3)] == [1.0, 1.0, 0.0]
    assert dataset[2]["chosen_input_ids"] == [0, 0]


def standard_source(num_pairs):
    """The same pairs as ``chat_source`` in TRL's *standard* layout: every field a plain
    string, the shape most public preference sets on the hub ship in."""
    return [
        {"prompt": f"prompt {p}", "chosen": "general kenobi", "rejected": f"worse answer {p}"}
        for p in range(num_pairs)
    ]


@pytest.mark.parametrize("tokenizer_cls", [FakeChatTokenizer, ChatMLTokenizer], ids=["fake", "chatml"])
def test_standard_string_rows_tokenize_identically_to_conversational_rows(tokenizer_cls):
    """Load-bearing: strings reduce to the already-validated message-list path."""
    conversational, _ = build_chat_preference_records(chat_source(3), tokenizer_cls(), max_seq_length=1000)
    standard, drops = build_chat_preference_records(
        standard_source(3), tokenizer_cls(), max_seq_length=1000, prompt_key="prompt"
    )
    assert not drops
    assert standard == conversational


def test_string_prompt_becomes_a_user_turn_and_string_completion_an_assistant_turn():
    """Wrapping roles match NeMo-RL's BinaryPreferenceDataset, so the same row reads the same in both."""
    records, drops = build_chat_preference_records(
        standard_source(1), ChatMLTokenizer(), max_seq_length=1000, prompt_key="prompt"
    )
    assert not drops
    record = records[0]
    rendered = "".join(map(chr, record["chosen_input_ids"]))
    assert rendered.startswith("<|im_start|>user\nprompt 0<|im_end|>\n")
    assert rendered.endswith("<|im_start|>assistant\ngeneral kenobi<|im_end|>\n" + chr(ChatMLTokenizer.eos_token_id))


@pytest.mark.parametrize(
    "row",
    [
        {"prompt": "a question", "chosen": [{"role": "assistant", "content": "yes"}], "rejected": "no"},
        {"prompt": [{"role": "user", "content": "a question"}], "chosen": "yes", "rejected": "no"},
    ],
    ids=["string_prompt_list_completions", "list_prompt_string_completions"],
)
def test_string_and_message_list_fields_mix_freely(row):
    records, drops = build_chat_preference_records([row], FakeChatTokenizer(), max_seq_length=100, prompt_key="prompt")
    assert not drops and len(records) == 1


def test_string_completions_without_a_prompt_key_drop_instead_of_crashing():
    """Strings are only meaningful alongside a prompt field; the misconfiguration must
    report a drop reason, not raise out of the tokenizer."""
    _, drops = build_chat_preference_records(
        [{"chosen": "yes", "rejected": "no"}], FakeChatTokenizer(), max_seq_length=100
    )
    assert dict(drops) == {"no_assistant_completion": 1}
