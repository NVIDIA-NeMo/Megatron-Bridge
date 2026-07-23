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

from megatron.bridge.training import nemotron_omni_step
from megatron.bridge.training.nemotron_omni_step import get_batch, get_batch_from_iterator
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params


def test_batch_moves_only_one_compatible_token_alias_to_cuda(monkeypatch):
    cuda_inputs = []

    def record_cuda(tensor, **kwargs):  # noqa: ARG001
        cuda_inputs.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "cuda", record_cuda)
    tokens = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[0, 1, 2]])
    batch = {"tokens": tokens, "input_ids": tokens, "position_ids": position_ids}

    moved = get_batch_from_iterator(
        iter([batch]),
        is_first_pp_stage=True,
        is_last_pp_stage=False,
    )

    assert moved["tokens"] is tokens
    assert moved["input_ids"] is None
    assert sum(tensor is tokens for tensor in cuda_inputs) == 1


def test_packed_batch_preserves_mamba_sequence_boundaries(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    batch = {
        "input_ids": torch.tensor([[1, 2, 0, 0, 3, 4, 5, 0]]),
        "labels": torch.tensor([[2, -100, -100, -100, 4, 5, -100, -100]]),
        "loss_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
        "attention_mask": None,
        "cu_seqlens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_seqlens_q_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "cu_seqlens_kv_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "max_seqlen_q": torch.tensor(4, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(4, dtype=torch.int32),
        "total_tokens": 8,
    }

    moved = get_batch_from_iterator(
        iter([batch]),
        is_first_pp_stage=True,
        is_last_pp_stage=True,
    )
    metadata = {
        key: value
        for key, value in moved.items()
        if key.startswith("cu_seqlens") or key.startswith("max_seqlen") or key == "total_tokens"
    }
    packed_seq_params = get_packed_seq_params(metadata)

    assert moved["total_tokens"] == 8
    assert packed_seq_params.seq_idx.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]


def _packed_pipeline_batch():
    tokens = torch.tensor([[18, 1, 18, 2]])
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    return {
        "input_ids": tokens,
        "labels": tokens.clone(),
        "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": None,
        "visual_inputs": SimpleNamespace(pixel_values=torch.ones(1, 4, 8)),
        "sound_clips": torch.ones(1, 4, 8),
        "sound_length": torch.tensor([4]),
        "imgs_sizes": torch.tensor([[32, 32], [32, 32]]),
        "num_frames": torch.tensor([1, 1]),
        "num_image_tiles": torch.tensor([1, 1], dtype=torch.int),
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_kv": cu_seqlens,
        "max_seqlen_q": torch.tensor(2, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(2, dtype=torch.int32),
        "total_tokens": 4,
    }


def _pipeline_cfg(*, packed=True, defer_packing=False, temporal_patch_dim=1):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            skip_getting_attention_mask_from_dataset=True,
            enable_in_batch_packing=packed,
            defer_in_batch_packing_to_step=defer_packing,
            enable_offline_packing=False,
        ),
        model=SimpleNamespace(temporal_patch_dim=temporal_patch_dim, image_token_index=18),
    )


def test_forward_rejects_deferred_multimodal_packing():
    state = SimpleNamespace(
        timers=None,
        straggler_timer=None,
        cfg=_pipeline_cfg(defer_packing=True),
    )

    with pytest.raises(ValueError, match="requires collate-time in-batch packing"):
        nemotron_omni_step.forward_step(state, iter(()), object())


def test_middle_pipeline_stage_preserves_only_packed_attention_metadata(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(nemotron_omni_step, "is_pp_first_stage", lambda group: False)
    monkeypatch.setattr(nemotron_omni_step, "is_pp_last_stage", lambda group: False)

    result = get_batch(iter([_packed_pipeline_batch()]), _pipeline_cfg(), pg_collection=SimpleNamespace(pp=object()))

    assert result[2] is None
    assert result[0] is None
    assert result[7]["cu_seqlens_q"].tolist() == [0, 2, 4]
    assert result[7]["total_tokens"] == 4


def test_middle_unpacked_pipeline_stage_does_not_consume_iterator(monkeypatch):
    monkeypatch.setattr(nemotron_omni_step, "is_pp_first_stage", lambda group: False)
    monkeypatch.setattr(nemotron_omni_step, "is_pp_last_stage", lambda group: False)
    data_iterator = iter([_packed_pipeline_batch()])

    result = get_batch(data_iterator, _pipeline_cfg(packed=False), pg_collection=SimpleNamespace(pp=object()))

    assert result == (None,) * 14
    assert next(data_iterator)["input_ids"].tolist() == [[18, 1, 18, 2]]


def test_last_pipeline_stage_keeps_label_expansion_inputs_without_media(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    batch = _packed_pipeline_batch()

    moved = get_batch_from_iterator(
        iter([batch]),
        is_first_pp_stage=False,
        is_last_pp_stage=True,
    )

    assert moved["input_ids"] is batch["input_ids"]
    assert moved["labels"] is batch["labels"]
    assert moved["loss_mask"] is batch["loss_mask"]
    assert moved["num_image_tiles"] is batch["num_image_tiles"]
    assert moved["visual_inputs"] is None
    assert moved["sound_clips"] is None
    assert moved["imgs_sizes"] is None
    assert moved["cu_seqlens_q"] is batch["cu_seqlens_q"]


def test_packed_middle_pipeline_forward_uses_boundaries_without_input_tensors(monkeypatch):
    class _Timer:
        def __call__(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def start(self):
            return None

        def stop(self):
            return None

    class _Model:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor(1.0)

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(nemotron_omni_step, "is_pp_first_stage", lambda group: False)
    monkeypatch.setattr(nemotron_omni_step, "is_pp_last_stage", lambda group: False)
    monkeypatch.setattr(
        nemotron_omni_step,
        "get_pg_collection",
        lambda model: SimpleNamespace(pp=object()),
    )
    model = _Model()
    state = SimpleNamespace(
        timers=_Timer(),
        straggler_timer=_Timer(),
        cfg=SimpleNamespace(
            **vars(_pipeline_cfg()),
            rerun_state_machine=SimpleNamespace(
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            ),
        ),
    )

    output, _ = nemotron_omni_step.forward_step(state, iter([_packed_pipeline_batch()]), model)

    assert output.item() == 1.0
    assert model.kwargs["images"] is None
    assert model.kwargs["input_ids"] is None
    assert model.kwargs["packed_seq_params"].cu_seqlens_q.tolist() == [0, 2, 4]


def test_forward_unwraps_model_output_and_uses_expanded_loss_mask(monkeypatch):
    class _Timer:
        def __call__(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def start(self):
            return None

        def stop(self):
            return None

    losses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expanded_loss_mask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])[:, :2]
    assert not expanded_loss_mask.is_contiguous()

    class _Model:
        def __call__(self, **kwargs):
            del kwargs
            return losses, expanded_loss_mask

    monkeypatch.setattr(
        nemotron_omni_step,
        "get_batch",
        lambda *args, **kwargs: (
            None,
            None,
            torch.tensor([[1, 2]]),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2),
            None,
            torch.arange(2).unsqueeze(0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        nemotron_omni_step,
        "get_pg_collection",
        lambda model: SimpleNamespace(pp=object()),
    )
    state = SimpleNamespace(
        timers=_Timer(),
        straggler_timer=_Timer(),
        cfg=SimpleNamespace(
            dataset=SimpleNamespace(
                enable_in_batch_packing=False,
                defer_in_batch_packing_to_step=False,
            ),
            rerun_state_machine=SimpleNamespace(
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            ),
        ),
    )

    output, loss_function = nemotron_omni_step.forward_step(state, iter(()), _Model())

    assert output is losses
    assert torch.equal(loss_function.args[0], expanded_loss_mask)
    assert loss_function.args[0].is_contiguous()

    monkeypatch.setattr(
        "megatron.bridge.training.losses.get_rerun_state_machine",
        lambda: SimpleNamespace(),
    )
    loss, num_tokens, _ = loss_function(output)
    assert loss.item() == 5.0
    assert num_tokens.item() == 2
