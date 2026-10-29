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

from megatron.bridge.training.llava_step import forward_step, get_batch
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


class _NoCudaTensor(torch.Tensor):
    def cuda(self, non_blocking=False):  # type: ignore[override]
        return self


def _as_nocuda(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.as_subclass(_NoCudaTensor)


def test_get_batch_accepts_current_packed_nemotron_vl_batch(monkeypatch):
    """The LLaVA step must consume the packed schema emitted by its recipe collator."""
    visual_inputs = GenericVisualInputs(pixel_values=_as_nocuda(torch.randn(2, 3, 4, 4)))
    batch = {
        "input_ids": _as_nocuda(torch.tensor([[1, 2, 3, 4]])),
        "position_ids": _as_nocuda(torch.tensor([[0, 1, 0, 1]])),
        "labels": _as_nocuda(torch.tensor([[2, 3, 4, -100]])),
        "loss_mask": _as_nocuda(torch.tensor([[1.0, 1.0, 1.0, 0.0]])),
        "attention_mask": None,
        "cu_seqlens_q": _as_nocuda(torch.tensor([0, 2, 4], dtype=torch.int32)),
        "cu_seqlens_kv": _as_nocuda(torch.tensor([0, 2, 4], dtype=torch.int32)),
        "max_seqlen_q": torch.tensor(2, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(2, dtype=torch.int32),
        "visual_inputs": visual_inputs,
    }
    cfg = SimpleNamespace(dataset=SimpleNamespace(skip_getting_attention_mask_from_dataset=True))
    pg_collection = SimpleNamespace(pp=object(), cp=object())
    monkeypatch.setattr("megatron.bridge.training.llava_step.is_pp_first_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.training.llava_step.is_pp_last_stage", lambda _: True)
    monkeypatch.setattr(
        "megatron.bridge.training.llava_step.get_batch_on_this_cp_rank",
        lambda batch, **_: batch,
    )

    images, num_image_tiles, input_ids, labels, loss_mask, attention_mask, position_ids, packed_metadata = get_batch(
        iter([batch]), cfg, pg_collection=pg_collection
    )

    assert images is visual_inputs.pixel_values
    assert num_image_tiles is None
    assert torch.equal(input_ids, batch["input_ids"])
    assert torch.equal(labels, batch["labels"])
    assert torch.equal(loss_mask, batch["loss_mask"])
    assert attention_mask is None
    assert torch.equal(position_ids, batch["position_ids"])
    assert set(packed_metadata) == {"cu_seqlens_q", "cu_seqlens_kv", "max_seqlen_q", "max_seqlen_kv"}


class _Timer:
    def __call__(self, *args, **kwargs):
        return self

    def start(self):
        return self

    def stop(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _PackedLlavaModel:
    def __init__(self, *, max_sequence_length=1024):
        self.llava_model = SimpleNamespace(
            image_token_index=131072,
            img_seq_len=4,
            _language_max_sequence_length=max_sequence_length,
        )
        self.packed_seq_params = None

    def __call__(self, **kwargs):
        self.packed_seq_params = kwargs["packed_seq_params"]
        return torch.tensor(0.0)


def test_forward_step_expands_packed_boundaries_for_visual_embeddings(monkeypatch):
    """Packed boundaries must describe the post-vision decoder sequence."""
    image_token_id = 131072
    input_ids = _as_nocuda(torch.tensor([[10, image_token_id, 11, 20, 21, image_token_id, 22]]))
    batch = {
        "input_ids": input_ids,
        "position_ids": _as_nocuda(torch.tensor([[0, 1, 2, 0, 1, 2, 3]])),
        "labels": _as_nocuda(torch.tensor([[image_token_id, 11, -100, 21, image_token_id, 22, -100]])),
        "loss_mask": _as_nocuda(torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])),
        "attention_mask": None,
        "cu_seqlens_q": _as_nocuda(torch.tensor([0, 3, 7], dtype=torch.int32)),
        "cu_seqlens_kv": _as_nocuda(torch.tensor([0, 3, 7], dtype=torch.int32)),
        "max_seqlen_q": torch.tensor(4, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(4, dtype=torch.int32),
        "visual_inputs": GenericVisualInputs(pixel_values=_as_nocuda(torch.randn(2, 3, 4, 4))),
    }
    state = SimpleNamespace(
        cfg=SimpleNamespace(
            dataset=SimpleNamespace(skip_getting_attention_mask_from_dataset=True),
            rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=False),
        ),
        timers=_Timer(),
        straggler_timer=_Timer(),
    )
    model = _PackedLlavaModel()
    pg_collection = SimpleNamespace(pp=object(), cp=object())
    monkeypatch.setattr(
        "megatron.bridge.training.llava_step.get_model_config", lambda _: SimpleNamespace(is_hybrid_model=True)
    )
    monkeypatch.setattr("megatron.bridge.training.llava_step.get_pg_collection", lambda _: pg_collection)
    monkeypatch.setattr("megatron.bridge.training.llava_step.is_pp_first_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.training.llava_step.is_pp_last_stage", lambda _: True)
    monkeypatch.setattr(
        "megatron.bridge.training.llava_step.get_batch_on_this_cp_rank",
        lambda batch, **_: batch,
    )

    forward_step(state, iter([batch]), model)

    packed_seq_params = model.packed_seq_params
    assert torch.equal(packed_seq_params.cu_seqlens_q, torch.tensor([0, 6, 13], dtype=torch.int32))
    assert torch.equal(packed_seq_params.cu_seqlens_kv, torch.tensor([0, 6, 13], dtype=torch.int32))
    assert packed_seq_params.cu_seqlens_q.dtype == torch.int32
    assert packed_seq_params.cu_seqlens_kv.dtype == torch.int32
    assert packed_seq_params.max_seqlen_q == 7
    assert packed_seq_params.max_seqlen_kv == 7
    assert torch.equal(
        packed_seq_params.seq_idx,
        torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32),
    )


def test_forward_step_clips_padded_packed_boundaries_after_visual_expansion(monkeypatch):
    """Post-vision logical and padded boundaries must match MCore truncation."""
    image_token_id = 131072
    batch = {
        "input_ids": _as_nocuda(torch.tensor([[10, image_token_id, 11, 0, 20, 21, image_token_id, 22]])),
        "position_ids": _as_nocuda(torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])),
        "labels": _as_nocuda(torch.tensor([[image_token_id, 11, -100, -100, 21, image_token_id, 22, -100]])),
        "loss_mask": _as_nocuda(torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])),
        "attention_mask": None,
        "cu_seqlens_q": _as_nocuda(torch.tensor([0, 3, 7], dtype=torch.int32)),
        "cu_seqlens_kv": _as_nocuda(torch.tensor([0, 3, 7], dtype=torch.int32)),
        "cu_seqlens_q_padded": _as_nocuda(torch.tensor([0, 4, 8], dtype=torch.int32)),
        "cu_seqlens_kv_padded": _as_nocuda(torch.tensor([0, 4, 8], dtype=torch.int32)),
        "max_seqlen_q": torch.tensor(4, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(4, dtype=torch.int32),
        "visual_inputs": GenericVisualInputs(pixel_values=_as_nocuda(torch.randn(2, 3, 4, 4))),
    }
    state = SimpleNamespace(
        cfg=SimpleNamespace(
            dataset=SimpleNamespace(skip_getting_attention_mask_from_dataset=True),
            rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=False),
        ),
        timers=_Timer(),
        straggler_timer=_Timer(),
    )
    model = _PackedLlavaModel(max_sequence_length=12)
    pg_collection = SimpleNamespace(pp=object(), cp=object())
    monkeypatch.setattr(
        "megatron.bridge.training.llava_step.get_model_config", lambda _: SimpleNamespace(is_hybrid_model=True)
    )
    monkeypatch.setattr("megatron.bridge.training.llava_step.get_pg_collection", lambda _: pg_collection)
    monkeypatch.setattr("megatron.bridge.training.llava_step.is_pp_first_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.training.llava_step.is_pp_last_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.training.llava_step.get_batch_on_this_cp_rank", lambda batch, **_: batch)

    forward_step(state, iter([batch]), model)

    packed_seq_params = model.packed_seq_params
    assert torch.equal(packed_seq_params.cu_seqlens_q, torch.tensor([0, 6, 11], dtype=torch.int32))
    assert torch.equal(packed_seq_params.cu_seqlens_kv, torch.tensor([0, 6, 11], dtype=torch.int32))
    assert torch.equal(packed_seq_params.cu_seqlens_q_padded, torch.tensor([0, 7, 12], dtype=torch.int32))
    assert torch.equal(packed_seq_params.cu_seqlens_kv_padded, torch.tensor([0, 7, 12], dtype=torch.int32))
    assert packed_seq_params.max_seqlen_q == 7
    assert packed_seq_params.max_seqlen_kv == 7
    assert torch.equal(
        packed_seq_params.seq_idx,
        torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.int32),
    )


def test_get_batch_rejects_packed_pipeline_or_context_parallelism(monkeypatch):
    """Packed LLaVA batches must fail before unsupported PP/CP slicing."""
    batch = {
        "input_ids": _as_nocuda(torch.tensor([[1, 2, 3, 4]])),
        "position_ids": _as_nocuda(torch.tensor([[0, 1, 0, 1]])),
        "labels": _as_nocuda(torch.tensor([[2, 3, 4, -100]])),
        "loss_mask": _as_nocuda(torch.tensor([[1.0, 1.0, 1.0, 0.0]])),
        "attention_mask": None,
        "cu_seqlens_q": _as_nocuda(torch.tensor([0, 2, 4], dtype=torch.int32)),
        "cu_seqlens_kv": _as_nocuda(torch.tensor([0, 2, 4], dtype=torch.int32)),
        "max_seqlen_q": torch.tensor(2, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(2, dtype=torch.int32),
        "visual_inputs": GenericVisualInputs(pixel_values=_as_nocuda(torch.randn(2, 3, 4, 4))),
    }
    cfg = SimpleNamespace(dataset=SimpleNamespace(skip_getting_attention_mask_from_dataset=True))
    pp_group = object()
    cp_group = object()
    pg_collection = SimpleNamespace(pp=pp_group, cp=cp_group)
    monkeypatch.setattr("megatron.bridge.training.llava_step.is_pp_first_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.training.llava_step.is_pp_last_stage", lambda _: True)
    monkeypatch.setattr(
        "megatron.bridge.training.llava_step.get_pg_size",
        lambda group: 2 if group is pp_group else 1,
    )

    with pytest.raises(ValueError, match="pipeline_model_parallel_size=1 and context_parallel_size=1"):
        get_batch(iter([batch]), cfg, pg_collection=pg_collection)
