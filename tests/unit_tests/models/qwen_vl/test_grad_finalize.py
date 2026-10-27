"""Ownership, scaling and callback integration for combined Qwen3.5-VL."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.qwen_vl import grad_finalize as gf
from megatron.bridge.training.state import GlobalState


class FakeGDN(torch.nn.Module):
    def __init__(self, sp):
        super().__init__()
        self.out_norm = torch.nn.Linear(2, 2, bias=False)
        self.out_norm.weight.sequence_parallel = sp


class FakeDDP(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.module = raw
        self.config = raw.config
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
        self.sync_calls = []
        self.scale_calls = 0

    def finish_grad_sync(self, force_all_reduce=False):
        self.sync_calls.append(force_all_reduce)
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.main_grad is not None:
                parameter.main_grad.mul_(3)

    def scale_gradients(self, scaling_factor):
        self.scale_calls += 1
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.main_grad is not None:
                parameter.main_grad.mul_(scaling_factor)


def make_model(language_sp, vision_sp, tp_size):
    pg = SimpleNamespace(tp=SimpleNamespace(size=lambda: tp_size))
    config = SimpleNamespace(
        sequence_parallel=language_sp,
        vision_sequence_parallel=vision_sp,
        tensor_model_parallel_size=tp_size,
        gtp_weight_remat_size=1,
        expert_gtp_weight_remat_size=1,
        finalize_model_grads_func=None,
    )
    raw = torch.nn.Module()
    raw.config = config
    raw.vision_model = torch.nn.Linear(2, 2)
    raw.vision_model.config = SimpleNamespace(sequence_parallel=vision_sp)
    raw.vision_model.pg_collection = pg
    raw.vision_model.weight.sequence_parallel = vision_sp
    raw.vision_model.bias.sequence_parallel = vision_sp
    raw.vision_model.bias.average_gradients_across_tp_domain = True
    raw.vision_model.register_parameter("unused", torch.nn.Parameter(torch.ones(2)))
    raw.vision_model.unused.sequence_parallel = vision_sp
    raw.vision_model.register_parameter("frozen", torch.nn.Parameter(torch.ones(2), requires_grad=False))
    raw.vision_model.frozen.sequence_parallel = vision_sp
    raw.language_model = torch.nn.Sequential(FakeGDN(language_sp))
    raw.language_model.pg_collection = pg
    for parameter in raw.parameters():
        parameter.main_grad = torch.ones_like(parameter)
    raw.vision_model.unused.main_grad = None
    return FakeDDP(raw), config, pg


@pytest.mark.unit
@pytest.mark.parametrize("language_sp,vision_sp", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("root_gate", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_reduction_ownership_and_scaling(monkeypatch, language_sp, vision_sp, root_gate, tp_size):
    model, config, pg = make_model(language_sp, vision_sp, tp_size)
    original_buffers = {name: parameter.main_grad for name, parameter in model.named_parameters()}
    monkeypatch.setattr(gf, "DistributedDataParallel", FakeDDP)
    monkeypatch.setattr(gf, "GatedDeltaNet", FakeGDN)
    monkeypatch.setattr(gf, "unwrap_model", lambda wrapped: wrapped.module)
    sums = []

    def sum_tp(grad, group):
        assert group is pg.tp
        sums.append(grad)
        grad.mul_(tp_size)

    monkeypatch.setattr(gf.torch.distributed, "all_reduce", sum_tp)
    core_calls = []

    def core_finalize(chunks, num_tokens, pg_collection, force_all_reduce):
        core_calls.append(chunks)
        assert pg_collection is pg
        for chunk in chunks:
            chunk.finish_grad_sync(force_all_reduce=force_all_reduce)
            # Simulate both the pinned Core gate and a Core version without it.
            for name, parameter in chunk.named_parameters(prefix="check", remove_duplicate=True):
                assert name.startswith("check.")
                if parameter.requires_grad and parameter.main_grad is not None:
                    if getattr(parameter, "average_gradients_across_tp_domain", False):
                        continue  # Averaging identical simulated rank values leaves them unchanged.
                    if getattr(parameter, "sequence_parallel", False) and (not root_gate or language_sp):
                        sum_tp(parameter.main_grad, pg.tp)
            num_tokens.mul_(2)  # Model the finalizer's in-place token count update.
            chunk.scale_gradients(1 / num_tokens)

    monkeypatch.setattr(gf, "finalize_model_grads", core_finalize)
    tokens = torch.tensor(2.0)
    gf.finalize_qwen35vl_model_grads([model], tokens, pg, force_all_reduce=True)
    assert len(core_calls) == 1
    assert model.sync_calls == [True]
    assert model.scale_calls == 1
    assert tokens.item() == 4
    for name, parameter in model.named_parameters():
        assert parameter.main_grad is original_buffers[name]
        if name.endswith("unused"):
            assert parameter.main_grad is None
        elif name.endswith("frozen"):
            torch.testing.assert_close(parameter.main_grad, torch.ones_like(parameter))
        else:
            reduced = ("vision_model.weight" in name and vision_sp) or "out_norm" in name
            expected = 0.75 * (tp_size if reduced else 1)
            torch.testing.assert_close(parameter.main_grad, torch.full_like(parameter, expected))


@pytest.mark.unit
def test_configure_restart_and_custom_callback():
    model, config, _ = make_model(False, True, 2)
    gf.configure_qwen35vl_grad_finalize(config)
    assert config.finalize_model_grads_func is gf.finalize_qwen35vl_model_grads
    state = GlobalState()
    state._cfg = SimpleNamespace(model=config)
    state.reset_for_restart()
    assert config.finalize_model_grads_func is None
    gf.configure_qwen35vl_grad_finalize(config)
    assert config.finalize_model_grads_func is gf.finalize_qwen35vl_model_grads
    existing = partial(gf.finalize_qwen35vl_model_grads, pg_collection="old", force_all_reduce=True)
    config.finalize_model_grads_func = existing
    gf.configure_qwen35vl_grad_finalize(config)
    assert config.finalize_model_grads_func is existing
    config.finalize_model_grads_func = Mock()
    with pytest.raises(ValueError, match="custom finalizers conflict"):
        gf.configure_qwen35vl_grad_finalize(config)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["group", "wrapper", "gtp", "dtensor"])
def test_reject_unsupported_reduction_before_core(monkeypatch, mode):
    model, config, pg = make_model(False, True, 2)
    monkeypatch.setattr(gf, "DistributedDataParallel", FakeDDP if mode != "wrapper" else torch.nn.Linear)
    monkeypatch.setattr(gf, "GatedDeltaNet", FakeGDN)
    monkeypatch.setattr(gf, "unwrap_model", lambda wrapped: wrapped.module)
    core = Mock()
    monkeypatch.setattr(gf, "finalize_model_grads", core)
    if mode == "group":
        model.module.vision_model.pg_collection = SimpleNamespace(tp=object())
    elif mode == "gtp":
        config.gtp_weight_remat_size = 2
    elif mode == "dtensor":
        monkeypatch.setattr(gf, "DTensor", SimpleNamespace)
        model.module.vision_model.weight.main_grad = SimpleNamespace()
    with pytest.raises(ValueError):
        gf.finalize_qwen35vl_model_grads([model], pg_collection=pg)
    core.assert_not_called()
