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

"""CPU contracts for the DiffusionGemma Megatron forward step."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.diffusion_gemma import diffusion_gemma_step as step


pytestmark = pytest.mark.unit


class _Timer:
    def start(self):
        return None

    def stop(self):
        return None


class _Model:
    training = True

    def __init__(self, vocab_size=7):
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append({"kwargs": kwargs, "grad_enabled": torch.is_grad_enabled()})
        batch, length = kwargs["decoder_input_ids"].shape
        logits = torch.full((batch, length, self.vocab_size + 2), float(len(self.calls)), requires_grad=True)
        return SimpleNamespace(logits=logits, encoder_loss=torch.tensor(0.25))


def _state():
    timer = _Timer()
    return SimpleNamespace(
        timers=lambda name, log_level=None: timer,
        straggler_timer=nullcontext(),
        train_state=SimpleNamespace(step=4),
        cfg=SimpleNamespace(
            rng=SimpleNamespace(seed=17),
            rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=True),
        ),
    )


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "position_ids": None,
        "pixel_values": torch.ones(2, 3, dtype=torch.float32),
        "image_position_ids": torch.zeros(2, 3, 2, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(2, 4, dtype=torch.long),
        "canvas_ids": torch.tensor([[1, 2], [3, 4]]),
        "canvas_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
    }


def _config(**overrides):
    values = {
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "vocab_size": 0,
        "text_config": SimpleNamespace(vocab_size=7),
        "params_dtype": torch.bfloat16,
        "diffusion_loss_variant": "loo-reweighted-ce",
        "diffusion_weight_clip": 3.0,
        "diffusion_decoder_loss_weight": 0.5,
        "diffusion_encoder_loss_weight": 0.25,
        "diffusion_self_conditioning_prob": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_forward_dependencies(monkeypatch, batch, config):
    monkeypatch.setattr(step, "get_pg_collection", lambda model: SimpleNamespace(pp="pp", dp="dp"))
    monkeypatch.setattr(step, "is_pp_first_stage", lambda group: True)
    monkeypatch.setattr(step, "is_pp_last_stage", lambda group: True)
    monkeypatch.setattr(step, "get_batch_from_iterator", lambda iterator, **kwargs: batch)
    monkeypatch.setattr(step, "get_model_config", lambda model: config)


def test_corruption_rejects_invalid_noise_bounds_and_respects_endpoints():
    x0 = torch.tensor([[3, 4, 5]])
    mask = torch.tensor([[1, 0, 1]])
    with pytest.raises(ValueError, match="diffusion_noise_min"):
        step.corrupt_canvas(x0, mask, vocab_size=10, noise_min=0.8, noise_max=0.2)

    unchanged, t = step.corrupt_canvas(x0, mask, vocab_size=10, noise_min=0.0, noise_max=0.0)
    assert torch.equal(unchanged, x0)
    assert t.tolist() == [0.0]

    generator = torch.Generator().manual_seed(9)
    corrupted, t = step.corrupt_canvas(x0, mask, vocab_size=10_000, noise_min=1.0, noise_max=1.0, generator=generator)
    assert t.tolist() == [1.0]
    assert corrupted[0, 1] == x0[0, 1]
    assert corrupted[0, 0] != x0[0, 0] and corrupted[0, 2] != x0[0, 2]


def test_loss_function_validates_nan_inf_and_spikes_without_encoder_loss(monkeypatch):
    calls = []

    class _RerunStateMachine:
        def validate_result(self, **kwargs):
            calls.append(kwargs)

        def is_unexpectedly_large(self, value, *, threshold, context):
            return threshold == 10.0 and context == "loss"

    monkeypatch.setattr(step, "get_rerun_state_machine", lambda: _RerunStateMachine())
    logits = torch.zeros(2, 2, 5)
    x0 = torch.zeros(2, 2, dtype=torch.long)
    loss, count, metrics = step.diffusion_loss_function(
        (logits, x0, x0, torch.tensor([0.1, 0.2]), torch.ones(2, 2)),
        variant="base-sft",
        vocab_size=5,
        check_for_nan_in_loss=True,
        check_for_spiky_loss=True,
    )

    assert int(count) == 2
    assert set(metrics) == {"lm loss"}
    assert torch.allclose(loss / count, torch.log(torch.tensor(5.0)))
    assert [call["message"] for call in calls] == [
        "found NaN in local DiffusionGemma loss calculation",
        "found Inf in local DiffusionGemma loss calculation",
        "Spiky DiffusionGemma loss",
    ]
    assert calls[0]["fatal"] and calls[1]["fatal"] and not calls[2]["fatal"]
    assert calls[2]["rejection_func"](loss)


def test_batch_loader_requires_single_pipeline_stage_and_required_keys(monkeypatch):
    with pytest.raises(NotImplementedError, match="pipeline_model_parallel_size=1"):
        step.get_batch_from_iterator(iter([]), is_first_pp_stage=False)
    with pytest.raises(KeyError, match="canvas_mask"):
        step.get_batch_from_iterator(
            iter([{"input_ids": torch.ones(1), "attention_mask": torch.ones(1), "canvas_ids": torch.ones(1)}])
        )

    moved = []
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: moved.append(non_blocking) or self + 1)
    batch = step.get_batch_from_iterator(
        iter(
            [
                {
                    "input_ids": torch.tensor([1]),
                    "attention_mask": torch.tensor([1]),
                    "canvas_ids": torch.tensor([2]),
                    "canvas_mask": torch.tensor([1.0]),
                    "position_ids": None,
                    "unused": torch.tensor([99]),
                }
            ]
        )
    )
    assert moved == [True, True, True, True]
    assert batch["input_ids"].tolist() == [2]
    assert batch["position_ids"] is None
    assert "unused" not in batch


def test_microbatch_counter_rejects_unknown_phase():
    with pytest.raises(ValueError, match="mode must be"):
        step.next_microbatch_index("predict", 1)


def test_noise_generator_keys_cuda_rng_by_phase_step_microbatch_and_dp_rank(monkeypatch):
    created = []

    class _Generator:
        def __init__(self, *, device):
            self.device = device
            self.seed = None
            created.append(self)

        def manual_seed(self, seed):
            self.seed = seed
            return self

    step._MICROBATCH_COUNTERS["eval"].update(step=None, index=0)
    monkeypatch.setattr(step.torch, "Generator", _Generator)
    monkeypatch.setattr(step.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(step.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(step.torch.distributed, "get_rank", lambda group: 2 if group == "dp" else -1)
    monkeypatch.setattr(step, "get_pg_collection", lambda model: SimpleNamespace(dp="dp"))
    state = _state()
    model = SimpleNamespace(training=False)

    first = step._noise_generator(state, model)
    second = step._noise_generator(state, model)
    assert [generator.device for generator in created] == ["cuda:3", "cuda:3"]
    assert first.seed == step.noise_seed(seed=17, mode="eval", step=4, microbatch=0, dp_rank=2)
    assert second.seed == step.noise_seed(seed=17, mode="eval", step=4, microbatch=1, dp_rank=2)


def test_forward_step_uses_explicit_noise_detached_self_conditioning_and_encoder_targets(monkeypatch):
    batch = {
        **_batch(),
        "noisy_canvas_ids": torch.tensor([[6, 2], [3, 5]]),
        "diffusion_t": torch.tensor([0.25, 0.75], dtype=torch.float64),
        "self_conditioning_mask": torch.tensor([True, False]),
    }
    config = _config()
    _patch_forward_dependencies(monkeypatch, batch, config)
    monkeypatch.setattr(
        step, "_noise_generator", lambda state, model: pytest.fail("explicit noise should not draw RNG")
    )
    model = _Model()

    output, loss_function = step.forward_step(_state(), iter([]), model)

    assert len(model.calls) == 2
    first, second = model.calls
    assert not first["grad_enabled"]
    assert "self_conditioning_mask" not in first["kwargs"]
    assert "position_ids" not in first["kwargs"]
    assert first["kwargs"]["pixel_values"].dtype == torch.bfloat16
    assert second["grad_enabled"]
    assert torch.equal(second["kwargs"]["self_conditioning_mask"], batch["self_conditioning_mask"])
    assert not second["kwargs"]["self_conditioning_logits"].requires_grad
    assert torch.equal(second["kwargs"]["encoder_target_ids"], batch["canvas_ids"])
    assert torch.equal(second["kwargs"]["encoder_target_mask"], batch["canvas_mask"])
    assert torch.equal(output[2], batch["noisy_canvas_ids"])
    assert output[3].dtype == torch.float32
    assert loss_function.keywords == {
        "variant": "reweighted-loo-ce",
        "vocab_size": 7,
        "diffusion_weight_clip": 3.0,
        "check_for_nan_in_loss": False,
        "check_for_spiky_loss": True,
        "decoder_loss_weight": 0.5,
        "encoder_loss_weight": 0.25,
    }


def test_forward_step_draws_noise_and_self_conditioning_from_one_generator(monkeypatch):
    batch = _batch()
    config = _config(diffusion_self_conditioning_prob=1.0, diffusion_encoder_loss_weight=0.0)
    _patch_forward_dependencies(monkeypatch, batch, config)
    generators = []

    def generator(state, model):
        value = torch.Generator().manual_seed(123)
        generators.append(value)
        return value

    monkeypatch.setattr(step, "_noise_generator", generator)
    model = _Model()
    output, _ = step.forward_step(_state(), iter([]), model)

    assert len(generators) == 1
    assert len(model.calls) == 2
    assert model.calls[1]["kwargs"]["self_conditioning_mask"].tolist() == [True, True]
    assert "encoder_target_ids" not in model.calls[1]["kwargs"]
    assert output[3].shape == (2,)


def test_forward_step_rejects_unsupported_parallel_shapes(monkeypatch):
    batch = _batch()
    _patch_forward_dependencies(monkeypatch, batch, _config(context_parallel_size=2))
    with pytest.raises(NotImplementedError, match="context parallelism"):
        step.forward_step(_state(), iter([]), _Model())

    batch["input_ids"] = torch.ones(2, 3, dtype=torch.long)
    _patch_forward_dependencies(monkeypatch, batch, _config(sequence_parallel=True, tensor_model_parallel_size=2))
    with pytest.raises(ValueError, match="input_ids length divisible by TP=2"):
        step.forward_step(_state(), iter([]), _Model())
