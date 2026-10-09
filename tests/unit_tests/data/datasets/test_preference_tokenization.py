import json

import pytest

from megatron.bridge.data.datasets.preference_pair import PreferencePairDataset
from megatron.bridge.data.datasets.preference_tokenization import tokenize_conversation
from tests.unit_tests.data.preference_fakes import ChatMLTokenizer, ImEndEOSTokenizer


def conversation(prompt, completion):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]


def scored_text(input_ids, loss_mask):
    return "".join(chr(token) for token, scored in zip(input_ids, loss_mask) if scored)


def test_a_shared_history_scores_only_the_final_assistant_turn():
    """Everything before the final assistant turn is context; that turn (header included) through its EOS is
    scored. A template whose turn terminator is the EOS (``<|im_end|>`` on Nemotron 3 / Qwen2.5-Instruct) ends
    the completion there, with no second EOS; otherwise one is appended."""
    messages = [{"role": "system", "content": "be terse"}, *conversation("q1", "a1"), *conversation("q2", "a2")]

    for tokenizer in (ChatMLTokenizer(), ImEndEOSTokenizer()):
        eos = chr(tokenizer.eos_token_id)
        input_ids, loss_mask = tokenize_conversation(tokenizer, messages, max_seq_length=10_000)

        context_len = loss_mask.index(True)
        assert loss_mask == [False] * context_len + [True] * (len(input_ids) - context_len)
        assert input_ids[:context_len] == tokenizer.apply_chat_template(messages[:-1])
        turn_end = eos if isinstance(tokenizer, ImEndEOSTokenizer) else "<|im_end|>\n" + eos
        assert scored_text(input_ids, loss_mask) == "<|im_start|>assistant\na2" + turn_end, type(tokenizer).__name__


def test_a_trajectory_may_end_on_a_tool_call():
    """A final assistant turn with tool calls and no text is a completion, not an ``empty_completion`` stub."""
    call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}
    messages = [{"role": "user", "content": "q"}, call]
    tokenizer = ChatMLTokenizer()

    input_ids, loss_mask = tokenize_conversation(tokenizer, messages, max_seq_length=10_000)

    rendered_call = '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
    assert scored_text(input_ids, loss_mask) == f"<|im_start|>assistant\n{rendered_call}<|im_end|>\n" + chr(
        tokenizer.eos_token_id
    )
    assert tokenize_conversation(tokenizer, [messages[0], {"role": "assistant", "content": " "}], 10_000) == (
        "empty_completion"
    )


def test_json_string_tool_calls_and_tool_schemas():
    """``tool_calls`` stored as a JSON string (HF ``datasets``) and ``function.arguments`` stored as a JSON string
    (OpenAI wire format) decode like the mapping forms, and the tool schemas are rendered into the prompt without
    being scored."""
    tokenizer = ChatMLTokenizer()
    call = {"function": {"name": "f", "arguments": {"x": 1}}}
    wire_call = {"function": {"name": "f", "arguments": json.dumps({"x": 1})}}
    schemas = [{"type": "function", "function": {"name": "f"}}]

    def row(tool_calls, tools):
        rejected = [{"role": "assistant", "content": "no"}]
        return {
            "prompt": "q",
            "chosen": [{"role": "assistant", "content": "", "tool_calls": tool_calls}],
            "rejected": rejected,
            "tools": tools,
        }

    records = [
        PreferencePairDataset([r], tokenizer, max_seq_length=1000, prompt_key="prompt", tools_key="tools")[0]
        for r in (row(json.dumps([wire_call]), json.dumps(schemas)), row([call], schemas))
    ]
    assert records[0] == records[1]

    ids, mask = records[0]["chosen_input_ids"], records[0]["chosen_loss_mask"]
    unscored = "".join(chr(token) for token, scored in zip(ids, mask) if not scored)
    assert unscored.startswith(f"<|im_start|>system\n<tools>{json.dumps(schemas)}</tools><|im_end|>\n")
    assert scored_text(ids, mask).startswith(
        '<|im_start|>assistant\n<tool_call>\n{"name": "f", "arguments": {"x": 1}}'
    )


def test_trajectories_that_split_mid_way_score_every_assistant_turn():
    """Full trajectories that first differ at an earlier assistant turn score all assistant turns (content plus
    end-of-turn), the final one with its header; tool results are not scored. A split at a non-assistant turn is
    a ``context_mismatch`` stub."""
    tokenizer = ChatMLTokenizer()
    prefix = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "q2"},
    ]

    def side(call, result, answer):
        return prefix + [
            {"role": "assistant", "content": call},
            {"role": "tool", "content": result},
            {"role": "assistant", "content": answer},
        ]

    row = {"chosen": side("c1", "t1", "c2"), "rejected": side("r1", "t2", "r2")}
    record = PreferencePairDataset([row], tokenizer, max_seq_length=1000)[0]

    assert record["loss_multiplier"] == 1.0
    eos = chr(tokenizer.eos_token_id)
    for key, (turn, answer) in {"chosen": ("c1", "c2"), "rejected": ("r1", "r2")}.items():
        scored = scored_text(record[f"{key}_input_ids"], record[f"{key}_loss_mask"])
        assert scored == f"a0<|im_end|>\n{turn}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n{eos}", key

    tool_split = {"chosen": side("c1", "t1", "c2"), "rejected": side("c1", "t2", "c2")}
    assert PreferencePairDataset([tool_split], tokenizer, max_seq_length=1000)[0]["loss_multiplier"] == 0.0


def test_dataset_reads_every_layout_alike_stubs_bad_pairs_and_joins_ref_logprobs():
    """The explicit-prompt and plain-string TRL layouts must tokenize exactly like the conversational
    one; a bad pair becomes a zero-loss stub so ``__len__`` stays fixed for the sampler; reference
    logprobs join by source row index and a missing entry fails loudly."""
    tokenizer = ChatMLTokenizer()
    conversational = [
        {"chosen": conversation(f"prompt {p}", "yes"), "rejected": conversation(f"prompt {p}", f"no {p}")}
        for p in range(3)
    ]
    explicit = [
        {
            "messages": [{"role": "user", "content": f"prompt {p}"}],
            "chosen": [{"role": "assistant", "content": "yes"}],
            "rejected": [{"role": "assistant", "content": f"no {p}"}],
            "preference_margin": 0.5,  # rows may carry extra fields the dataset must ignore
        }
        for p in range(3)
    ]
    standard = [{"prompt": f"prompt {p}", "chosen": "yes", "rejected": f"no {p}"} for p in range(3)]
    mismatched_prompts = {"chosen": conversation("one prompt", "yes"), "rejected": conversation("another", "no")}

    dataset = PreferencePairDataset(conversational + [mismatched_prompts], tokenizer, max_seq_length=1000)
    records = [dataset[i] for i in range(3)]
    for source, prompt_key in [(explicit, "messages"), (standard, "prompt")]:
        other = PreferencePairDataset(source, tokenizer, max_seq_length=1000, prompt_key=prompt_key)
        assert [other[i] for i in range(3)] == records, prompt_key

    assert len(dataset) == 4
    assert records[0]["loss_multiplier"] == 1.0
    stub = dataset[3]
    assert stub["loss_multiplier"] == 0.0
    assert stub["chosen_input_ids"] == stub["rejected_input_ids"] == [tokenizer.pad_token_id] * 2

    ref = {
        0: {
            "ref_chosen_logprob_sum": -1.5,
            "ref_chosen_num_tokens": 4,
            "ref_rejected_logprob_sum": -2.5,
            "ref_rejected_num_tokens": 5,
        }
    }
    with_ref = PreferencePairDataset(conversational, tokenizer, max_seq_length=1000, ref_logprobs=ref)
    assert with_ref[0]["ref_chosen_logprob_sum"] == -1.5
    with pytest.raises(ValueError, match="no entry for pair_id"):
        with_ref[1]
