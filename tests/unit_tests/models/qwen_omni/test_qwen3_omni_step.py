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

import torch

from megatron.bridge.models.qwen_omni.qwen3_omni_step import (
    _normalize_multimodal_inputs,
    forward_step,
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


def _as_nocuda(tensor: torch.Tensor) -> torch.Tensor:
    class _NoCudaTensor(torch.Tensor):
        def cuda(self, non_blocking=False):  # type: ignore[override]
            return self

    return tensor.as_subclass(_NoCudaTensor)


def _make_batch():
    tokens = torch.tensor([[1, 2, 3, 4]])
    return {
        "input_ids": tokens,
        "position_ids": torch.arange(tokens.size(1)).unsqueeze(0),
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "loss_mask": torch.ones(1, 4),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "pixel_values": torch.randn(1, 2, 3, 4, 4),
        "image_grid_thw": torch.tensor([[[1, 2, 2], [1, 2, 2]]]),
        "pixel_values_videos": torch.randn(1, 1, 3, 4, 4),
        "video_grid_thw": torch.tensor([[[1, 2, 2]]]),
        "video_second_per_grid": torch.tensor([[0.5]]),
        "input_features": torch.randn(1, 80, 6),
        "feature_attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long),
        "audio_feature_lengths": torch.tensor([4], dtype=torch.long),
    }


def test_get_batch_from_iterator_moves_omni_tensors_to_cuda():
    batch = _make_batch()
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = _as_nocuda(value)

    out = get_batch_from_iterator(
        _Iterator(batch),
        use_mtp=False,
        skip_getting_attention_mask_from_dataset=False,
        is_first_pp_stage=True,
        is_last_pp_stage=True,
    )

    assert out["pixel_values"] is not None
    assert out["input_features"] is not None
    assert out["video_second_per_grid"] is not None


def test_normalize_multimodal_inputs_flattens_expected_shapes():
    normalized = _normalize_multimodal_inputs(_make_batch())

    assert normalized["pixel_values"].shape == (2, 3, 4, 4)
    assert normalized["image_grid_thw"].shape == (2, 3)
    assert normalized["pixel_values_videos"].shape == (1, 3, 4, 4)
    assert normalized["video_grid_thw"].shape == (1, 3)
    assert normalized["video_second_per_grid"].shape == (1,)
    assert normalized["input_features"].shape == (1, 80, 6)


def test_forward_step_passes_omni_multimodal_args(monkeypatch):
    class _MockProcessGroup:
        def rank(self):
            return 0

        def size(self):
            return 1

    class _MockPGCollection:
        def __init__(self):
            self.pp = _MockProcessGroup()
            self.cp = _MockProcessGroup()

    class _Model:
        def __init__(self):
            self.config = type("Cfg", (), {"mtp_num_layers": 0, "overlap_moe_expert_parallel_comm": True})()
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor(0.0)

        def build_schedule_plan(self, *args, **kwargs):  # noqa: ARG002
            return torch.tensor(1)

    class _Timer:
        def __call__(self, *args, **kwargs):  # noqa: ARG002
            return self

        def start(self):
            return self

        def stop(self):
            return self

    class _Strag:
        def __call__(self, *args, **kwargs):  # noqa: ARG002
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ARG002
            return False

    state = type("State", (), {})()
    state.cfg = type(
        "Cfg",
        (),
        {
            "dataset": type("D", (), {"skip_getting_attention_mask_from_dataset": False})(),
            "model": type("M", (), {"pipeline_model_parallel_size": 1, "seq_length": 16})(),
            "rerun_state_machine": type("R", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False})(),
        },
    )()
    state.timers = _Timer()
    state.straggler_timer = _Strag()

    monkeypatch.setattr(
        "megatron.bridge.models.qwen_omni.qwen3_omni_step.get_pg_collection",
        lambda model: _MockPGCollection(),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_omni.qwen3_omni_step.get_model_config",
        lambda model: model.config,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_omni.qwen3_omni_step.get_batch_on_this_cp_rank",
        lambda batch, cp_group=None: batch,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_omni.qwen3_omni_step.is_pp_first_stage",
        lambda pg: True,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_omni.qwen3_omni_step.is_pp_last_stage",
        lambda pg: True,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_omni.qwen3_omni_step.get_batch",
        lambda data_iterator, cfg, use_mtp, pg_collection: (
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[2, 3, 4, -100]]),
            torch.ones(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            torch.arange(4).unsqueeze(0),
            {
                "pixel_values": torch.randn(2, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
                "input_features": torch.randn(1, 80, 6),
                "feature_attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
                "audio_feature_lengths": torch.tensor([4]),
            },
        ),
    )

    model = _Model()
    output, loss_fn = forward_step(state, iter([{}]), model)

    assert isinstance(output, torch.Tensor)
    assert callable(loss_fn)
    assert model.kwargs is not None
    assert model.kwargs["position_ids"] is None
    assert "pixel_values" in model.kwargs
    assert "input_features" in model.kwargs
    assert "audio_feature_lengths" in model.kwargs
