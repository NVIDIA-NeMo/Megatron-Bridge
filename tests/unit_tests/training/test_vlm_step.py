# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs
from megatron.bridge.training.vlm_step import (
    forward_step,
    get_batch,
    get_batch_from_iterator,
)


class _Iterator:
    def __init__(self, batch):
        self.batch = batch
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self.batch


class _NoCudaTensor(torch.Tensor):
    def cuda(self, non_blocking=False):  # type: ignore[override]
        return self


def _as_nocuda(tensor):
    return tensor.as_subclass(_NoCudaTensor)


def _make_batch(device="cpu"):
    # Minimal text tensors
    tokens = torch.tensor([[1, 2, 3]], device=device)
    input_ids = tokens.clone()
    position_ids = torch.tensor([[0, 1, 2]], device=device)
    labels = torch.tensor([[2, 3, 4]], device=device)
    loss_mask = torch.ones_like(labels, dtype=torch.float, device=device)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=device)

    # Visual inputs container
    pixel_values = torch.randn(1, 2, 3, 4, 4, device=device)
    image_grid_thw = torch.tensor([[[1, 2, 2], [1, 2, 2]]], device=device)
    vi = GenericVisualInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    batch = {
        "tokens": tokens,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "visual_inputs": vi,
    }
    return batch


def _make_forward_step_state():
    class _Timer:
        def __call__(self, *args, **kwargs):  # noqa: ARG002
            return self

        def start(self):
            return self

        def stop(self):
            return self

    class _StragglerTimer:
        def __call__(self, *args, **kwargs):  # noqa: ARG002
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ARG002
            return False

    cfg = type(
        "Cfg",
        (),
        {
            "model": type("M", (), {"seq_length": 16, "pipeline_model_parallel_size": 1})(),
            "dataset": type("D", (), {"skip_getting_attention_mask_from_dataset": True})(),
            "rerun_state_machine": type("R", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False})(),
        },
    )()
    return type("State", (), {"cfg": cfg, "timers": _Timer(), "straggler_timer": _StragglerTimer()})()


def _patch_forward_step_deps(monkeypatch, model):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr("megatron.bridge.training.vlm_step.get_model_config", lambda _: model.config, raising=True)
    monkeypatch.setattr(
        "megatron.bridge.training.vlm_step.get_pg_collection", lambda _: model.pg_collection, raising=True
    )
    monkeypatch.setattr("megatron.bridge.training.vlm_step.is_pp_first_stage", lambda _: True, raising=True)
    monkeypatch.setattr("megatron.bridge.training.vlm_step.is_pp_last_stage", lambda _: True, raising=True)


def _make_visual_forward_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "loss_mask": torch.ones(1, 4),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "visual_inputs": GenericVisualInputs(
            pixel_values=torch.randn(1, 3, 4, 4),
            image_position_ids=torch.zeros(1, 4, 2, dtype=torch.long),
            mm_token_type_ids=torch.ones(1, 4, dtype=torch.long),
        ),
    }


def test_get_batch_from_iterator_moves_visual_inputs_to_cuda(monkeypatch):
    # Simulate Training on CPU-only env by making .cuda a no-op that returns the same tensor
    class _NoCudaTensor(torch.Tensor):
        def cuda(self, non_blocking=False):  # type: ignore[override]
            return self

    def _as_nocuda(t):
        return t.as_subclass(_NoCudaTensor)

    batch = _make_batch()
    # Replace tensors with _NoCudaTensor so calling .cuda works without a GPU
    for k in ["tokens", "input_ids", "position_ids", "labels", "loss_mask", "attention_mask"]:
        batch[k] = _as_nocuda(batch[k])
    vi = batch["visual_inputs"]
    vi.pixel_values = _as_nocuda(vi.pixel_values)
    vi.image_grid_thw = _as_nocuda(vi.image_grid_thw)

    it = _Iterator(batch)
    out = get_batch_from_iterator(
        it,
        use_mtp=False,
        skip_getting_attention_mask_from_dataset=True,
        is_first_pp_stage=True,
        is_last_pp_stage=True,
    )

    assert "visual_inputs" in out
    out_vi = out["visual_inputs"]
    assert isinstance(out_vi, GenericVisualInputs)
    # Verify fields are preserved
    assert out_vi.pixel_values is not None and out_vi.image_grid_thw is not None


def test_get_batch_from_iterator_projects_visual_payload_on_middle_pp_stage():
    batch = _make_batch()
    for key in ["tokens", "input_ids", "position_ids", "labels", "loss_mask", "attention_mask"]:
        batch[key] = _as_nocuda(batch[key])
    vi = batch["visual_inputs"]
    original_pixel_values = vi.pixel_values
    vi.pixel_values = _as_nocuda(vi.pixel_values)
    vi.image_grid_thw = _as_nocuda(vi.image_grid_thw)

    out = get_batch_from_iterator(
        _Iterator(batch),
        use_mtp=False,
        skip_getting_attention_mask_from_dataset=True,
        is_first_pp_stage=False,
        is_last_pp_stage=False,
    )

    out_vi = out["visual_inputs"]
    assert isinstance(out_vi, GenericVisualInputs)
    assert out_vi.pixel_values is None
    assert out_vi.image_grid_thw is not None
    assert vi.pixel_values is not None
    assert vi.pixel_values.shape == original_pixel_values.shape
    # Token and position IDs stay available to every PP stage.
    assert out["input_ids"] is not None


def test_get_batch_from_iterator_keeps_input_ids_with_multiaxis_position_ids():
    batch = _make_batch()
    batch["position_ids"] = torch.arange(3).view(1, 1, 3).expand(3, 1, -1).clone()
    for key in ["tokens", "input_ids", "position_ids", "labels", "loss_mask", "attention_mask"]:
        batch[key] = _as_nocuda(batch[key])
    vi = batch["visual_inputs"]
    vi.pixel_values = _as_nocuda(vi.pixel_values)
    vi.image_grid_thw = _as_nocuda(vi.image_grid_thw)

    out = get_batch_from_iterator(
        _Iterator(batch),
        use_mtp=False,
        skip_getting_attention_mask_from_dataset=True,
        is_first_pp_stage=False,
        is_last_pp_stage=False,
    )

    assert out["tokens"] is not None
    assert out["input_ids"] is not None
    assert out["position_ids"].shape == (3, 1, 3)
    assert out["visual_inputs"].pixel_values is None
    assert out["visual_inputs"].image_grid_thw is not None


class _MockProcessGroup:
    """Mock process group with rank/size methods for testing."""

    def rank(self):
        return 0

    def size(self):
        return 1


class _MockPGCollection:
    """Mock PG collection for testing."""

    def __init__(self, cp_size=1):
        self.pp = _MockProcessGroup()
        self.tp = _MockProcessGroup()
        self._cp_size = cp_size

    @property
    def cp(self):
        pg = _MockProcessGroup()
        pg.size = lambda: self._cp_size
        return pg


class _ForwardModelBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"mtp_num_layers": 0, "overlap_moe_expert_parallel_comm": False})()
        self.pg_collection = _MockPGCollection()
        self.received_kwargs = None


class _ForwardWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.config = module.config
        self.pg_collection = module.pg_collection

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _Gemma4LikeForwardModel(_ForwardModelBase):
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_mask=None,
        pixel_values=None,
        image_position_ids=None,
    ):
        self.received_kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
        }
        return torch.tensor(0.0)


class _MmTokenTypeForwardModel(_ForwardModelBase):
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_mask=None,
        pixel_values=None,
        image_position_ids=None,
        mm_token_type_ids=None,
    ):
        self.received_kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return torch.tensor(0.0)


class _PackedForwardModel(_ForwardModelBase):
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_mask=None,
        packed_seq_params=None,
    ):
        self.received_kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "packed_seq_params": packed_seq_params,
        }
        return torch.tensor(0.0)


def test_forward_step_filters_unsupported_visual_kwargs(monkeypatch):
    inner_model = _Gemma4LikeForwardModel()
    model = _ForwardWrapper(inner_model)
    _patch_forward_step_deps(monkeypatch, model)

    output, _ = forward_step(_make_forward_step_state(), _Iterator(_make_visual_forward_batch()), model)

    assert output.item() == 0.0
    assert inner_model.received_kwargs is not None
    assert inner_model.received_kwargs["pixel_values"] is not None
    assert inner_model.received_kwargs["image_position_ids"] is not None
    assert "mm_token_type_ids" not in inner_model.received_kwargs


def test_forward_step_preserves_supported_mm_token_type_ids(monkeypatch):
    inner_model = _MmTokenTypeForwardModel()
    model = _ForwardWrapper(inner_model)
    _patch_forward_step_deps(monkeypatch, model)

    output, _ = forward_step(_make_forward_step_state(), _Iterator(_make_visual_forward_batch()), model)

    assert output.item() == 0.0
    assert inner_model.received_kwargs is not None
    assert inner_model.received_kwargs["mm_token_type_ids"] is not None


def test_forward_step_rejects_deferred_in_batch_packing(monkeypatch):
    inner_model = _PackedForwardModel()
    model = _ForwardWrapper(inner_model)
    _patch_forward_step_deps(monkeypatch, model)
    state = _make_forward_step_state()
    state.cfg.dataset.enable_in_batch_packing = True
    state.cfg.dataset.defer_in_batch_packing_to_step = True

    with pytest.raises(ValueError, match="requires collate-time in-batch packing"):
        forward_step(state, _Iterator(_make_visual_forward_batch()), model)


def test_get_batch_consumes_collated_sequence_shape(monkeypatch):
    # Simulate both first and last pipeline stages so tensors are returned
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)

    # Disable context parallel slicing effects
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x, **kwargs: x,
        raising=True,
    )

    # Minimal cfg
    cfg = type("Cfg", (), {})()
    cfg.model = type(
        "M",
        (),
        {
            "seq_length": 32,
            "seq_len_interpolation_factor": 1.0,
            "seq_length_interpolation_factor": 1.0,
            "seq_length_interpolation": None,
            "seq_length_interpolation_power": 1.0,
            "pipeline_model_parallel_size": 1,
        },
    )()  # noqa: E501
    cfg.dataset = type("D", (), {"skip_getting_attention_mask_from_dataset": True})()

    # The collate layer is now responsible for padding/truncation, so get_batch
    # should preserve the incoming sequence length.
    short_tokens = torch.tensor([[1, 2, 3, 4]])
    vi = GenericVisualInputs(pixel_values=torch.randn(1, 1, 3, 4, 4), image_grid_thw=torch.tensor([[[1, 2, 2]]]))
    batch = {
        "input_ids": short_tokens,
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "loss_mask": torch.ones_like(short_tokens, dtype=torch.float),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones_like(short_tokens, dtype=torch.bool),
        "visual_inputs": vi,
    }

    # Iterator
    it = _Iterator(batch)

    tokens, labels, loss_mask, attention_mask, position_ids, *_ = get_batch(
        it, cfg, use_mtp=False, pg_collection=_MockPGCollection()
    )
    assert tokens.shape[1] == 4
    assert labels.shape[1] == 4
    assert loss_mask.shape[1] == 4
    assert position_ids.shape[1] == 4


def test_get_batch_consumes_collated_packed_metadata(monkeypatch):
    """Test get_batch with collate-provided packed-sequence metadata."""
    # Simulate both first and last pipeline stages so tensors are returned
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)

    # Disable context parallel slicing effects
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x, **kwargs: x,
        raising=True,
    )

    cfg = type("Cfg", (), {})()
    cfg.model = type(
        "M",
        (),
        {
            "seq_length": 64,
            "pipeline_model_parallel_size": 1,
        },
    )()
    cfg.dataset = type(
        "D",
        (),
        {
            "skip_getting_attention_mask_from_dataset": True,
            "enable_in_batch_packing": True,
        },
    )()

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[2, 3, -100, 5, 6, 7, 8, -100]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])

    vi = GenericVisualInputs(pixel_values=torch.randn(1, 1, 3, 4, 4), image_grid_thw=torch.tensor([[[1, 2, 2]]]))
    batch = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": None,
        "cu_seqlens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor([0, 3, 8], dtype=torch.int32),
        "max_seqlen_q": torch.tensor(5, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(5, dtype=torch.int32),
        "visual_inputs": vi,
    }

    it = _Iterator(batch)

    (
        out_tokens,
        out_labels,
        out_loss_mask,
        out_attention_mask,
        out_position_ids,
        packed_seq_params,
        visual_inputs,
    ) = get_batch(it, cfg, use_mtp=False, pg_collection=_MockPGCollection())

    assert out_tokens.shape == (1, 8)
    assert out_labels.shape == (1, 8)
    assert out_loss_mask.shape == (1, 8)
    assert out_position_ids.shape == (1, 8)
    assert packed_seq_params["cu_seqlens_q"].tolist() == [0, 3, 8]
    assert packed_seq_params["cu_seqlens_kv"].tolist() == [0, 3, 8]
    assert packed_seq_params["max_seqlen_q"].item() == 5
    assert packed_seq_params["max_seqlen_kv"].item() == 5
    assert "cu_seqlens_argmin" not in packed_seq_params
    assert "cu_seqlens_unpadded" not in packed_seq_params
    assert out_attention_mask is None
    assert torch.equal(out_tokens.cpu(), tokens)
    assert visual_inputs is not None


def test_get_batch_consumes_current_padded_cu_seqlens(monkeypatch):
    """Test get_batch forwards collate-provided current padded cu-seqlens fields."""
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x, **kwargs: x,
        raising=True,
    )

    cfg = type("Cfg", (), {})()
    cfg.model = type("M", (), {"seq_length": 64, "pipeline_model_parallel_size": 1})()
    cfg.dataset = type(
        "D",
        (),
        {
            "skip_getting_attention_mask_from_dataset": True,
            "enable_in_batch_packing": True,
        },
    )()

    tokens = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 7, 8, 0, 0, 0]])
    labels = torch.full_like(tokens, -100)
    loss_mask = torch.zeros_like(tokens, dtype=torch.float)
    position_ids = torch.arange(12).unsqueeze(0)

    batch = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": None,
        "cu_seqlens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seqlens_q_padded": torch.tensor([0, 4, 12], dtype=torch.int32),
        "cu_seqlens_kv_padded": torch.tensor([0, 4, 12], dtype=torch.int32),
        "max_seqlen_q": torch.tensor(8, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(8, dtype=torch.int32),
        "visual_inputs": None,
    }

    it = _Iterator(batch)

    (
        out_tokens,
        out_labels,
        out_loss_mask,
        _,
        out_position_ids,
        packed_seq_params,
        _,
    ) = get_batch(it, cfg, use_mtp=False, pg_collection=_MockPGCollection(cp_size=2))

    assert out_tokens.shape[1] == 12
    assert out_labels.shape[1] == 12
    assert out_loss_mask.shape[1] == 12
    assert out_position_ids.shape[1] == 12
    assert packed_seq_params["cu_seqlens_q"].tolist() == [0, 3, 8]
    assert packed_seq_params["cu_seqlens_kv"].tolist() == [0, 3, 8]
    assert packed_seq_params["cu_seqlens_q_padded"].tolist() == [0, 4, 12]
    assert packed_seq_params["cu_seqlens_kv_padded"].tolist() == [0, 4, 12]
    assert packed_seq_params["max_seqlen_q"].item() == 8
    assert packed_seq_params["max_seqlen_kv"].item() == 8


def test_forward_step_schedule_plan(monkeypatch):
    # Configure pipeline last/first to enable labels & loss_mask path
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)

    # No-op CUDA and CP functions
    monkeypatch.setattr("megatron.core.utils.get_batch_on_this_cp_rank", lambda x, **kwargs: x, raising=True)

    # Create a proper mock process group with rank/size methods
    class _MockProcessGroup:
        def rank(self):
            return 0

        def size(self):
            return 1

    # Create mock pg_collection with proper process groups
    class _MockPGCollection:
        def __init__(self):
            self.pp = _MockProcessGroup()
            self.tp = _MockProcessGroup()
            self.cp = _MockProcessGroup()

    # Dummy model with required interface
    class _Model:
        def __init__(self):
            self.config = type("C", (), {"mtp_num_layers": 0, "overlap_moe_expert_parallel_comm": True})()
            self._pg_collection = _MockPGCollection()

        @property
        def pg_collection(self):
            return self._pg_collection

        def build_schedule_plan(self, tokens, position_ids, attention_mask, labels=None, loss_mask=None):  # noqa: ARG002
            return torch.tensor(1)

        def __call__(self, **kwargs):  # noqa: ARG002
            return torch.tensor(0.0)

    # Return model config
    monkeypatch.setattr("megatron.core.utils.get_model_config", lambda m: m.config, raising=True)

    # Dummy timers/straggler_timer
    class _Timer:
        def __call__(self, *a, **k):  # noqa: ARG002
            return self

        def start(self):
            return self

        def stop(self):
            return self

    class _Strag:
        def __call__(self, *a, **k):  # noqa: ARG002
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ARG002
            return False

    class _State:
        def __init__(self):
            self.cfg = type(
                "Cfg",
                (),
                {
                    "rerun_state_machine": type(
                        "R", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False}
                    )()
                },
            )()  # noqa: E501
            self.timers = _Timer()
            self.straggler_timer = _Strag()

    # Reuse small iterator producing already-sized batch
    vi = GenericVisualInputs(pixel_values=torch.randn(1, 1, 3, 4, 4), image_grid_thw=torch.tensor([[[1, 2, 2]]]))
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "loss_mask": torch.ones(1, 4),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "visual_inputs": vi,
    }
    it = _Iterator(batch)

    # Minimal cfg for get_batch within forward_step
    cfg = type(
        "C2",
        (),
        {
            "model": type("M", (), {"seq_length": 16, "pipeline_model_parallel_size": 1})(),
            "dataset": type("D", (), {"skip_getting_attention_mask_from_dataset": True})(),
            "rerun_state_machine": type("R", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False})(),
        },
    )()  # noqa: E501

    state = _State()
    state.cfg = cfg
    model = _Model()

    # Execute schedule plan path
    plan, loss_fn = forward_step(state, it, model, return_schedule_plan=True)
    assert isinstance(plan, torch.Tensor)
