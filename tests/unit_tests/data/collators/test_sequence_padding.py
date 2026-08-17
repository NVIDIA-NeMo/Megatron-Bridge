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

import pytest
import torch

from megatron.bridge.data.collators.sequence_padding import (
    pad_or_truncate_sequence_batch,
    use_processor_right_padding,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX


pytestmark = pytest.mark.unit


class _Tokenizer:
    def __init__(self, padding_side="left"):
        self.padding_side = padding_side


class _Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _batch(length=3, batch_size=1):
    return {
        "input_ids": torch.arange(1, length + 1).repeat(batch_size, 1),
        "labels": torch.arange(2, length + 2).repeat(batch_size, 1),
        "loss_mask": torch.ones(batch_size, length),
        "position_ids": torch.arange(length).repeat(batch_size, 1),
        "attention_mask": torch.ones(batch_size, length, dtype=torch.long),
    }


def test_use_processor_right_padding_forces_and_restores_padding_side():
    processor = _Processor(_Tokenizer(padding_side="left"))

    with use_processor_right_padding(processor):
        assert processor.tokenizer.padding_side == "right"

    assert processor.tokenizer.padding_side == "left"


def test_use_processor_right_padding_restores_on_exception():
    processor = _Processor(_Tokenizer(padding_side="left"))

    with pytest.raises(RuntimeError):
        with use_processor_right_padding(processor):
            raise RuntimeError("boom")

    assert processor.tokenizer.padding_side == "left"


def test_use_processor_right_padding_accepts_bare_tokenizer():
    tokenizer = _Tokenizer(padding_side="left")

    with use_processor_right_padding(tokenizer):
        assert tokenizer.padding_side == "right"

    assert tokenizer.padding_side == "left"


def test_use_processor_right_padding_noop_without_padding_side():
    class _NoPadding:
        pass

    processor = _Processor(_NoPadding())
    with use_processor_right_padding(processor):
        pass  # must not raise


def test_pad_to_efficiency_multiple_caps_at_sequence_length():
    batch = _batch(length=3)

    pad_or_truncate_sequence_batch(batch, sequence_length=16, pad_to_multiple_of=4)

    assert batch["input_ids"].shape == (1, 4)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0]]
    assert batch["labels"].tolist() == [[2, 3, 4, IGNORE_INDEX]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 1.0, 0.0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0]]


def test_pad_to_max_length_pads_to_sequence_length():
    batch = _batch(length=3)

    pad_or_truncate_sequence_batch(batch, sequence_length=6, pad_to_max_length=True, pad_to_multiple_of=4)

    assert batch["input_ids"].shape == (1, 6)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 0, 0]]


def test_multiple_shorter_than_sequence_length_wins():
    batch = _batch(length=5)

    pad_or_truncate_sequence_batch(batch, sequence_length=128, pad_to_multiple_of=8)

    assert batch["input_ids"].shape == (1, 8)


def test_custom_pad_token_id():
    batch = _batch(length=3)

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True, pad_token_id=7)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 7]]


def test_truncates_overlong_batch():
    batch = _batch(length=6)

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert batch["labels"].shape == (1, 4)
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3]]


def test_truncation_that_removes_all_supervision_raises():
    batch = _batch(length=6)
    batch["loss_mask"] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="rows \\[0\\]"):
        pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)


def test_truncation_keeping_some_supervision_passes():
    batch = _batch(length=6)
    batch["loss_mask"] = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 1.0]])

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["loss_mask"].tolist() == [[0.0, 1.0, 0.0, 0.0]]


def test_tokens_key_variant_and_mirroring():
    batch = _batch(length=3)
    batch["tokens"] = batch["input_ids"].clone()

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["tokens"].tolist() == [[1, 2, 3, 0]]
    assert torch.equal(batch["tokens"], batch["input_ids"])


def test_tokens_only_batch():
    batch = {"tokens": torch.tensor([[1, 2, 3]])}

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["tokens"].tolist() == [[1, 2, 3, 0]]
    assert "input_ids" not in batch


def test_bool_attention_mask_pads_with_false():
    batch = _batch(length=3)
    batch["attention_mask"] = torch.ones(1, 3, dtype=torch.bool)

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["attention_mask"].dtype == torch.bool
    assert batch["attention_mask"].tolist() == [[True, True, True, False]]


def test_4d_attention_mask_pads_query_and_key_dims():
    batch = _batch(length=3)
    batch["attention_mask"] = torch.ones(1, 1, 3, 3, dtype=torch.bool)

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["attention_mask"].shape == (1, 1, 4, 4)
    assert not batch["attention_mask"][0, 0, 3, 3]


def test_4d_attention_mask_truncates_to_target():
    batch = _batch(length=6)
    batch["attention_mask"] = torch.ones(1, 1, 6, 6, dtype=torch.bool)

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["attention_mask"].shape == (1, 1, 4, 4)


def test_3d_attention_mask_raises():
    batch = _batch(length=3)
    batch["attention_mask"] = torch.ones(1, 3, 3)

    with pytest.raises(ValueError, match="2D or 4D"):
        pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)


def test_position_ids_extension_continues_ramp():
    batch = _batch(length=3)

    pad_or_truncate_sequence_batch(batch, sequence_length=6, pad_to_max_length=True)

    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5]]


def test_sequence_tensor_pad_values_pads_extra_keys():
    batch = _batch(length=3)
    batch["token_type_ids"] = torch.zeros(1, 3, dtype=torch.long)

    pad_or_truncate_sequence_batch(
        batch,
        sequence_length=5,
        pad_to_max_length=True,
        sequence_tensor_pad_values={"token_type_ids": 9},
    )

    assert batch["token_type_ids"].tolist() == [[0, 0, 0, 9, 9]]


def test_sequence_tensor_pad_values_ignores_missing_keys():
    batch = _batch(length=3)

    pad_or_truncate_sequence_batch(
        batch,
        sequence_length=4,
        pad_to_max_length=True,
        sequence_tensor_pad_values={"not_present": 1},
    )

    assert "not_present" not in batch


def test_none_sequence_length_is_noop():
    batch = _batch(length=3)
    before = batch["input_ids"].clone()

    pad_or_truncate_sequence_batch(batch, sequence_length=None)

    assert torch.equal(batch["input_ids"], before)


def test_nonpositive_sequence_length_raises():
    batch = _batch(length=3)

    with pytest.raises(ValueError, match="sequence_length"):
        pad_or_truncate_sequence_batch(batch, sequence_length=0)


def test_missing_token_tensor_raises():
    with pytest.raises(ValueError, match="input_ids"):
        pad_or_truncate_sequence_batch({"labels": torch.tensor([[1]])}, sequence_length=4)


def test_non_2d_tokens_raise():
    with pytest.raises(ValueError, match="2D"):
        pad_or_truncate_sequence_batch({"input_ids": torch.tensor([1, 2, 3])}, sequence_length=4)


def test_multi_row_batch_pads_every_row():
    batch = _batch(length=3, batch_size=2)

    pad_or_truncate_sequence_batch(batch, sequence_length=4, pad_to_max_length=True)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["input_ids"][1].tolist() == [1, 2, 3, 0]
