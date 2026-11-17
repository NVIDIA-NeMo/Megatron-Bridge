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

"""ERNIE 5.5 text SFT rows: the reference's chat-template encoding and the dataset's text-row path.

``fixtures/ernie55_reference_template_cases.json`` is the reference implementation's own template test data: for each
conversation, the rendered text and the supervised text its encoder must produce, with the byte-level toy tokenizer of
that test. Its special tags are renamed to the released tokenizer's ``ernie:`` names, as in Bridge's template.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te  # noqa: E402


allow_conversion_imports_without_te()
pytest.importorskip("megatron.bridge.models.conversion.model_bridge", reason="needs Bridge's conversion stack")
pytest.importorskip("megatron.bridge.training.utils.train_utils", reason="needs Bridge's training stack")

from megatron.bridge.data.base import DatasetBuildContext  # noqa: E402
from megatron.bridge.models.ernie.ernie55_chat import (  # noqa: E402
    ERNIE55_CHAT_TEMPLATE_PATH,
    encode_ernie55_sft_row,
    ernie55_sft_messages,
)
from megatron.bridge.models.ernie.ernie55_data import Ernie55SFTDatasetConfig, collate_ernie55_rows  # noqa: E402
from megatron.bridge.models.ernie.ernie55_step import ernie55_documents_from_batch  # noqa: E402


CASES = json.loads((Path(__file__).parent / "fixtures" / "ernie55_reference_template_cases.json").read_text())["cases"]
TEMPLATE_SHA256 = "b23eda7a585bb420dea436165ed016aa45bed7db91ea0d6b84f024b6fbc249b3"  # pragma: allowlist secret
REFERENCE_TEMPLATE_SHA256 = (
    "5a58cf518bb2ac0c804e488b9b4647c442b7db84846898ba0000dd95396b36c3"  # pragma: allowlist secret
)
_RELEASED_TAG_NAMES = {
    "<|im_start|>": "<|ernie:im_start|>",
    "<|im_end|>": "<|ernie:im_end|>",
    **{
        f"<{close}{name}>": f"<{close}ernie:{name}>"
        for name in ("system_setting", "global_setting", "tool_list", "tool_output", "think")
        for close in ("", "/")
    },
}


def _toy_tokenizer(tmp_path: Path | None = None):
    """The reference test's byte-level tokenizer: one id per byte, no merges."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|ernie:im_end|>", pad_token="<|ernie:im_end|>"
    )
    if tmp_path is not None:
        tokenizer.save_pretrained(tmp_path)
    return tokenizer


def _supervised_text(tokenizer, sample):
    return tokenizer.decode([label for label in sample["labels"] if label != -100])


def test_template_is_the_reference_file_in_the_released_tag_names():
    template = ERNIE55_CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    assert hashlib.sha256(template.encode()).hexdigest() == TEMPLATE_SHA256
    for plain, released in _RELEASED_TAG_NAMES.items():
        assert plain not in template, plain
        template = template.replace(released, plain)
    assert hashlib.sha256(template.encode()).hexdigest() == REFERENCE_TEMPLATE_SHA256


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_reference_template_cases(case):
    tokenizer = _toy_tokenizer()
    row = {"messages": case["messages"], **({"tools": case["tools"]} if case.get("tools") else {})}
    sample = encode_ernie55_sft_row(tokenizer, row, chat_template_kwargs=case.get("kwargs", {}))
    assert tokenizer.decode(sample["input_ids"]) == case["text"]
    assert len(sample["labels"]) == len(sample["input_ids"])
    assert sample["labels"][0] == -100
    if not case.get("inference"):
        assert _supervised_text(tokenizer, sample) == case["supervised_text"]


def test_string_tool_arguments_are_decoded():
    case = deepcopy(next(case for case in CASES if case["name"] == "tools_xml"))
    call = case["messages"][1]["tool_calls"][0]["function"]
    call["arguments"] = json.dumps(call["arguments"])
    tokenizer = _toy_tokenizer()
    sample = encode_ernie55_sft_row(tokenizer, {"messages": case["messages"], "tools": case["tools"]})
    assert tokenizer.decode(sample["input_ids"]) == case["text"]
    assert _supervised_text(tokenizer, sample) == case["supervised_text"]
    assert isinstance(call["arguments"], str), "the caller's row must not be modified"


def test_prompt_completion_row_is_one_user_and_one_assistant_turn():
    tokenizer = _toy_tokenizer()
    as_text = encode_ernie55_sft_row(tokenizer, {"input": "What is SFT?", "output": "Supervised fine-tuning."})
    as_chat = encode_ernie55_sft_row(
        tokenizer,
        {
            "messages": [
                {"role": "user", "content": "What is SFT?"},
                {"role": "assistant", "content": "Supervised fine-tuning."},
            ]
        },
    )
    assert as_text == as_chat
    assert _supervised_text(tokenizer, as_text).endswith("Supervised fine-tuning.<|ernie:im_end|>")
    custom = encode_ernie55_sft_row(
        tokenizer, {"q": "What is SFT?", "a": "Supervised fine-tuning."}, prompt_column="q", completion_column="a"
    )
    assert custom == as_text


def test_loss_all_supervises_every_token_but_the_first():
    tokenizer = _toy_tokenizer()
    sample = encode_ernie55_sft_row(tokenizer, {"input": "a", "output": "b"}, loss="all")
    assert sample["labels"] == [-100, *sample["input_ids"][1:]]


def test_rows_of_neither_format_are_rejected():
    with pytest.raises(ValueError, match="needs 'messages'"):
        ernie55_sft_messages({"text": "plain pretraining text"})
    with pytest.raises(ValueError, match="empty conversation"):
        ernie55_sft_messages({"messages": [{"role": "assistant", "content": None}]})


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


TEXT_ROWS = [
    {"input": "What is SFT?", "output": "Supervised fine-tuning."},
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},
]


@pytest.mark.parametrize("pack", [False, True])
def test_dataset_encodes_text_rows(tmp_path, pack):
    tokenizer = _toy_tokenizer(tmp_path / "tok")
    pretokenized = encode_ernie55_sft_row(tokenizer, {"input": "Mixed in", "output": "as ids"})
    path = _write_jsonl(tmp_path / "train.jsonl", [*TEXT_ROWS, pretokenized])
    config = Ernie55SFTDatasetConfig(
        seq_length=4096,
        pad_token_id=tokenizer.pad_token_id,
        path=str(path),
        hf_tokenizer=str(tmp_path / "tok"),
        pack=pack,
    )
    config.finalize()
    train, _, _ = config.build_datasets(DatasetBuildContext(train_samples=3, valid_samples=0, test_samples=0))
    expected = [encode_ernie55_sft_row(tokenizer, row) for row in TEXT_ROWS] + [pretokenized]
    wanted = {tuple(sample["input_ids"]): sample["labels"][1:] + [-100] for sample in expected}
    seen = set()
    for index in range(1 if pack else 3):
        batch = collate_ernie55_rows([train[index]], pad_token_id=tokenizer.pad_token_id)
        for ids, labels in ernie55_documents_from_batch(batch):
            key = tuple(ids.tolist())
            assert labels.tolist() == wanted[key]
            seen.add(key)
    assert seen == set(wanted)


def test_dataset_matches_offline_pretokenization(tmp_path):
    tokenizer = _toy_tokenizer(tmp_path / "tok")
    text = _write_jsonl(tmp_path / "text.jsonl", TEXT_ROWS)
    ids = _write_jsonl(
        tmp_path / "ids.jsonl",
        [encode_ernie55_sft_row(tokenizer, row, chat_template_kwargs={"enable_thinking": False}) for row in TEXT_ROWS],
    )
    common = dict(seq_length=4096, pad_token_id=tokenizer.pad_token_id)
    online = Ernie55SFTDatasetConfig(
        **common, path=str(text), hf_tokenizer=str(tmp_path / "tok"), chat_template_kwargs={"enable_thinking": False}
    )
    offline = Ernie55SFTDatasetConfig(**common, path=str(ids))
    context = DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0)
    a, _, _ = online.build_datasets(context)
    b, _, _ = offline.build_datasets(context)
    assert [a[i] for i in range(2)] == [b[i] for i in range(2)]


def test_text_rows_without_a_tokenizer_are_rejected(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", TEXT_ROWS)
    config = Ernie55SFTDatasetConfig(seq_length=64, pad_token_id=0, path=str(path))
    with pytest.raises(ValueError, match="needs dataset.hf_tokenizer"):
        config.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0))


def test_tokenizer_needs_a_path():
    config = Ernie55SFTDatasetConfig(
        seq_length=64, pad_token_id=0, hf_tokenizer="some/tokenizer", mock_document_lengths=(8,)
    )
    with pytest.raises(ValueError, match="set dataset.path too"):
        config.finalize()


def test_token_ids_past_the_vocabulary_are_rejected(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", [{"input_ids": [1, 2, 300], "labels": [1, 2, 300]}])
    config = Ernie55SFTDatasetConfig(seq_length=64, pad_token_id=0, path=str(path), vocab_size=256)
    train, _, _ = config.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0))
    with pytest.raises(ValueError, match="token id 300, but the model's vocab_size is 256"):
        train[0]


def test_tokenizer_path_survives_the_framework_tokenizer_injection():
    """Bridge's setup assigns its built tokenizer to ``cfg.dataset.tokenizer`` whenever that attribute exists
    (training/setup.py). The text-row tokenizer path must live under another name, or it is replaced by that object."""
    config = Ernie55SFTDatasetConfig(seq_length=64, pad_token_id=0, path="train.jsonl", hf_tokenizer="/tok")
    assert not hasattr(config, "tokenizer")
    assert config.hf_tokenizer == "/tok"
