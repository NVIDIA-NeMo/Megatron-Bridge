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

import pickle
from dataclasses import dataclass

import pytest

from megatron.bridge.data.collators.contracts import ModelCollator, PreparedSequenceCollator


pytestmark = pytest.mark.unit


@dataclass
class _Prepared:
    sequence_length: int


def _picklable_collate(examples, processor, **kwargs):
    return {"examples": examples, "processor": processor, **kwargs}


def _picklable_prepare(example, processor, **kwargs):
    del processor
    return _Prepared(sequence_length=len(example["tokens"]) + kwargs.get("extra_tokens", 0))


def _picklable_pack(sequences, **kwargs):
    return {"lengths": [sequence.sequence_length for sequence in sequences], **kwargs}


def test_model_collator_preserves_full_batch_callable_contract():
    seen = []

    def collate(examples, processor, **kwargs):
        seen.append((examples, processor, kwargs))
        return {"batch": examples}

    collator = ModelCollator(collate)
    examples = [{"text": "hello"}]

    assert collator(examples, "processor", sequence_length=128) == {"batch": examples}
    assert collator.collate(examples, "processor", sequence_length=64) == {"batch": examples}
    assert collator.collate_fn is collate
    assert seen == [
        (examples, "processor", {"sequence_length": 128}),
        (examples, "processor", {"sequence_length": 64}),
    ]


def test_prepared_sequence_collator_exposes_prepare_and_pack_contract():
    def collate(examples, processor, **kwargs):  # noqa: ARG001
        return {"batch": examples}

    def prepare_one(example, processor, **kwargs):
        assert processor == "processor"
        return _Prepared(sequence_length=len(example["tokens"]) + kwargs["extra_tokens"])

    def pack_prepared(sequences, **kwargs):
        return {"lengths": [sequence.sequence_length for sequence in sequences], **kwargs}

    collator = PreparedSequenceCollator(
        collate,
        prepare_one=prepare_one,
        pack_prepared=pack_prepared,
    )
    prepared = collator.prepare_one({"tokens": [1, 2]}, "processor", extra_tokens=1)

    assert prepared.sequence_length == 3
    assert collator.aligned_length(prepared, multiple=4) == 4
    assert collator.pack_prepared([prepared], sequence_length=8, pad_to_multiple_of=4) == {
        "lengths": [3],
        "sequence_length": 8,
        "pad_to_multiple_of": 4,
    }


@pytest.mark.parametrize("multiple", [0, -1])
def test_prepared_sequence_collator_rejects_invalid_alignment(multiple):
    collator = PreparedSequenceCollator(
        lambda *_args, **_kwargs: {},
        prepare_one=lambda *_args: _Prepared(1),
        pack_prepared=lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ValueError, match="multiple must be greater than 0"):
        collator.aligned_length(_Prepared(1), multiple=multiple)


@pytest.mark.parametrize(
    "collator",
    [
        ModelCollator(_picklable_collate),
        PreparedSequenceCollator(
            _picklable_collate,
            prepare_one=_picklable_prepare,
            pack_prepared=_picklable_pack,
        ),
    ],
)
def test_model_collator_contracts_support_worker_serialization(collator):
    restored = pickle.loads(pickle.dumps(collator))

    assert restored([{"tokens": [1, 2]}], "processor", sequence_length=8) == {
        "examples": [{"tokens": [1, 2]}],
        "processor": "processor",
        "sequence_length": 8,
    }
    if isinstance(restored, PreparedSequenceCollator):
        prepared = restored.prepare_one({"tokens": [1, 2]}, "processor", extra_tokens=1)
        assert restored.pack_prepared([prepared], sequence_length=8) == {"lengths": [3], "sequence_length": 8}
