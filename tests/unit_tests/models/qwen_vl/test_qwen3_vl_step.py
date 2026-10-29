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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.qwen_vl.qwen3_vl_step import (
    _pad_and_pack_qwen3_vl_step,
    forward_step,
    get_batch_from_iterator,
)


pytestmark = pytest.mark.unit


def test_get_batch_from_iterator_rejects_collate_time_packing_metadata():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "visual_inputs": None,
        "cu_seqlens_q": torch.tensor([0, 3], dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="does not support collate-time in-batch packing"):
        get_batch_from_iterator(
            iter([batch]),
            is_first_pp_stage=True,
            is_last_pp_stage=True,
        )


def test_get_batch_from_iterator_allows_deferred_none_packing_metadata(monkeypatch):
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "labels": torch.tensor([[2, 3, -100]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "visual_inputs": None,
        "cu_seqlens_q": None,
        "cu_seqlens": None,
    }
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)

    result = get_batch_from_iterator(
        iter([batch]),
        is_first_pp_stage=True,
        is_last_pp_stage=True,
    )

    assert torch.equal(result["input_ids"], batch["input_ids"])


class _ProcessGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


class _Timer:
    def __call__(self, *args, **kwargs):  # noqa: ARG002
        return self

    def start(self):
        return None

    def stop(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):  # noqa: ARG002
        return False


class _RecordingModel:
    pre_process = True

    def __init__(self, output=None):
        self.output = torch.tensor(0.0) if output is None else output
        self.forward_kwargs = None

    def __call__(self, **kwargs):
        self.forward_kwargs = kwargs
        return self.output


def _pg_collection():
    process_group = _ProcessGroup()
    return type(
        "PGCollection",
        (),
        {"pp": process_group, "tp": process_group, "cp": process_group, "ep": process_group},
    )()


def _model_config(**overrides):
    values = {"mtp_num_layers": 0, "overlap_moe_expert_parallel_comm": False, **overrides}
    return type("ModelConfig", (), values)()


def _state(*, enable_in_batch_packing, seq_length=4):
    return type(
        "State",
        (),
        {
            "timers": _Timer(),
            "straggler_timer": _Timer(),
            "cfg": type(
                "Config",
                (),
                {
                    # A builder-backed model's config is its transformer config, so the step
                    # reads the sequence length from the outer model config.
                    "model": SimpleNamespace(seq_length=seq_length),
                    "dataset": type("Dataset", (), {"enable_in_batch_packing": enable_in_batch_packing})(),
                    "rerun_state_machine": type(
                        "Rerun", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False}
                    )(),
                },
            )(),
        },
    )()


def _patch_step_dependencies(monkeypatch, *, config, batch):
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.get_pg_collection", lambda _: _pg_collection())
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.is_pp_first_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.is_pp_last_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.get_model_config", lambda _: config)
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.get_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step.get_batch_on_this_cp_rank",
        lambda forward_args, **kwargs: forward_args,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step._create_loss_function", lambda loss_mask, *args: object()
    )


def test_forward_step_uses_compacted_loss_mask_from_model(monkeypatch):
    config = _model_config()
    compacted_loss_mask = torch.tensor([[1.0, 0.0]])
    model = _RecordingModel(output=(torch.tensor(0.0), compacted_loss_mask))
    tokens = torch.tensor([[1, 2, 3, 0]])
    labels = torch.tensor([[2, 3, -100, -100]])
    original_loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    position_ids = torch.arange(4).unsqueeze(0)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool)

    _patch_step_dependencies(
        monkeypatch,
        config=config,
        batch=(tokens, labels, original_loss_mask, attention_mask, position_ids, {}, None),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step._pad_and_pack_qwen3_vl_step",
        lambda *args, **kwargs: (*args[:5], object(), kwargs["padding_mask"]),
    )
    captured_loss_masks = []
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step._create_loss_function",
        lambda loss_mask, *args: captured_loss_masks.append(loss_mask) or object(),
    )

    output, _ = forward_step(_state(enable_in_batch_packing=True), iter(()), model)

    assert output.item() == 0.0
    assert captured_loss_masks == [compacted_loss_mask]


def test_get_batch_from_iterator_moves_padding_mask_to_device(monkeypatch):
    padding_mask = torch.tensor([[False, False, True]])
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "visual_inputs": None,
        "padding_mask": padding_mask,
    }
    moved = []
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: moved.append(self) or self)

    result = get_batch_from_iterator(iter([batch]), is_first_pp_stage=False, is_last_pp_stage=False)

    assert result["padding_mask"] is padding_mask
    assert any(tensor is padding_mask for tensor in moved)


def test_pad_and_pack_marks_added_positions_as_padding():
    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])

    outputs = _pad_and_pack_qwen3_vl_step(
        tokens,
        torch.full_like(tokens, -100),
        torch.zeros_like(tokens, dtype=torch.float),
        None,
        torch.arange(4).expand(2, -1),
        _pg_collection(),
        force_to_pad_to_seq_len=True,
        seq_length=6,
        padding_mask=padding_mask,
    )

    assert outputs[0].shape == (2, 6)
    expected = torch.tensor(
        [[False, False, False, True, True, True], [False, False, True, True, True, True]],
    )
    assert torch.equal(outputs[-1], expected)


@pytest.mark.parametrize(("force_to_pad_to_seq_len", "expected_len"), [(True, 8), (False, 16)])
def test_pad_and_pack_max_seqlen_is_a_host_int_without_item(monkeypatch, force_to_pad_to_seq_len, expected_len):
    """max_seqlen is the padded length as a Python int, computed without Tensor.item().

    A device-to-host read such as .item() is not allowed while a full-iteration CUDA graph is captured.
    """

    def _no_item(self):
        raise AssertionError("Tensor.item() called while packing the Qwen3-VL step batch")

    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    monkeypatch.setattr(torch.Tensor, "item", _no_item)

    outputs = _pad_and_pack_qwen3_vl_step(
        tokens,
        torch.full_like(tokens, -100),
        torch.zeros_like(tokens, dtype=torch.float),
        None,
        torch.arange(4).expand(2, -1),
        _pg_collection(),
        use_fp8_padding=True,
        force_to_pad_to_seq_len=force_to_pad_to_seq_len,
        seq_length=8,
    )

    packed_seq_params = outputs[5]
    assert outputs[0].shape == (2, expected_len)
    assert type(packed_seq_params.max_seqlen_q) is int and type(packed_seq_params.max_seqlen_kv) is int
    assert packed_seq_params.max_seqlen_q == packed_seq_params.max_seqlen_kv == expected_len
    assert torch.equal(packed_seq_params.cu_seqlens_q, torch.tensor([0, expected_len, 2 * expected_len]))


@pytest.mark.parametrize("enable_in_batch_packing", [False, True])
def test_forward_step_passes_padding_mask_for_padded_batches_only(monkeypatch, enable_in_batch_packing):
    config = _model_config()
    model = _RecordingModel()
    tokens = torch.tensor([[1, 2, 3, 0]])
    padding_mask = torch.tensor([[False, False, False, True]])
    _patch_step_dependencies(
        monkeypatch,
        config=config,
        batch=(
            tokens,
            torch.tensor([[2, 3, -100, -100]]),
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            None,
            torch.arange(4).unsqueeze(0),
            {},
            padding_mask,
        ),
    )

    forward_step(_state(enable_in_batch_packing=enable_in_batch_packing), iter(()), model)

    if enable_in_batch_packing:
        assert "padding_mask" not in model.forward_kwargs
    else:
        # The step pads the batch to a multiple of 16 for MXFP8 and marks the new positions as padding.
        expected = torch.cat([padding_mask, torch.ones((1, 12), dtype=torch.bool)], dim=1)
        assert torch.equal(model.forward_kwargs["padding_mask"], expected)


def test_forward_step_scatters_padding_mask_on_later_sequence_parallel_stages(monkeypatch):
    config = _model_config(sequence_parallel=True)
    model = _RecordingModel()
    model.pre_process = False
    padding_mask = torch.zeros((2, 16), dtype=torch.bool)
    padding_mask[:, 6:] = True
    _patch_step_dependencies(
        monkeypatch,
        config=config,
        batch=(
            torch.zeros((2, 16), dtype=torch.long),
            torch.full((2, 16), -100),
            torch.zeros((2, 16)),
            None,
            torch.arange(16).expand(2, -1),
            {},
            padding_mask,
        ),
    )
    scattered = []

    def scatter_first_half(tensor, group):
        del group
        scattered.append(tensor)
        return tensor[: tensor.shape[0] // 2]

    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step.tensor_parallel.scatter_to_sequence_parallel_region",
        scatter_first_half,
    )

    forward_step(_state(enable_in_batch_packing=False, seq_length=16), iter(()), model)

    # The mask is scattered sequence-first, like the activations this stage receives.
    assert len(scattered) == 1 and scattered[0].shape == (16, 2)
    assert torch.equal(model.forward_kwargs["padding_mask"], padding_mask[:, :8])
