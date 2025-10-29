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

import torch

from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs
from megatron.bridge.training.vlm_step import forward_step, get_batch, get_batch_from_iterator


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
    vi = Qwen2_5_VLVisualInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

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


def test_get_batch_from_iterator_moves_visual_inputs_to_cuda(monkeypatch):
    # Avoid requiring distributed/parallel initialization in unit test
    monkeypatch.setattr(
        "megatron.core.parallel_state.is_pipeline_last_stage",
        lambda: False,
        raising=True,
    )

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
    out = get_batch_from_iterator(it, use_mtp=False, skip_getting_attention_mask_from_dataset=True)

    assert "visual_inputs" in out
    out_vi = out["visual_inputs"]
    assert isinstance(out_vi, Qwen2_5_VLVisualInputs)
    # Verify fields are preserved
    assert out_vi.pixel_values is not None and out_vi.image_grid_thw is not None


def test_get_batch_padding_paths(monkeypatch):
    # Simulate both first and last pipeline stages so tensors are returned
    monkeypatch.setattr("megatron.core.parallel_state.is_pipeline_first_stage", lambda: True, raising=True)
    monkeypatch.setattr("megatron.core.parallel_state.is_pipeline_last_stage", lambda: True, raising=True)

    # Disable context parallel slicing effects
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x: x,
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

    # Make batch shorter than 128 to trigger ceil-to-128 padding path
    short_tokens = torch.tensor([[1, 2, 3, 4]])
    vi = Qwen2_5_VLVisualInputs(pixel_values=torch.randn(1, 1, 3, 4, 4), image_grid_thw=torch.tensor([[[1, 2, 2]]]))
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
    tokens, labels, loss_mask, attention_mask, position_ids, *_ = get_batch(it, cfg, use_mtp=False)
    # Length padded up to min(seq_cap, ceil_to_128(4)) == 32
    assert tokens.shape[1] == 32
    assert labels.shape[1] == 32
    assert loss_mask.shape[1] == 32
    assert position_ids.shape[1] == 32


def test_forward_step_schedule_plan(monkeypatch):
    # Configure pipeline last/first to enable labels & loss_mask path
    monkeypatch.setattr("megatron.core.parallel_state.is_pipeline_first_stage", lambda: True, raising=True)
    monkeypatch.setattr("megatron.core.parallel_state.is_pipeline_last_stage", lambda: True, raising=True)

    # No-op CUDA and CP functions
    monkeypatch.setattr("megatron.core.utils.get_batch_on_this_cp_rank", lambda x: x, raising=True)

    # Dummy model with required interface
    class _Model:
        def __init__(self):
            self.config = type("C", (), {"mtp_num_layers": 0, "overlap_moe_expert_parallel_comm": True})()

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
    vi = Qwen2_5_VLVisualInputs(pixel_values=torch.randn(1, 1, 3, 4, 4), image_grid_thw=torch.tensor([[[1, 2, 2]]]))
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
