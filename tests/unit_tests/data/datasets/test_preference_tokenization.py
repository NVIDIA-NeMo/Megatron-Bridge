import pytest

from megatron.bridge.data.datasets.preference_pair import PreferencePairDataset
from megatron.bridge.data.datasets.preference_tokenization import tokenize_conversation
from tests.unit_tests.data.preference_fakes import ChatMLTokenizer


def conversation(prompt, completion):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]


def test_context_len_marks_the_start_of_the_final_assistant_turn():
    """Everything before the final assistant turn is context; that turn (header included) plus EOS is scored."""
    messages = [{"role": "system", "content": "be terse"}, *conversation("q1", "a1"), *conversation("q2", "a2")]
    tokenizer = ChatMLTokenizer()

    input_ids, context_len = tokenize_conversation(tokenizer, messages, max_seq_length=10_000)

    assert input_ids[:context_len] == tokenizer.apply_chat_template(messages[:-1])
    assert "".join(map(chr, input_ids[context_len:])) == "<|im_start|>assistant\na2<|im_end|>\n" + chr(
        tokenizer.eos_token_id
    )


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
