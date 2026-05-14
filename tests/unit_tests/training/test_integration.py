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

from __future__ import annotations

import sys
import types

import pytest
import torch

from megatron.bridge.training import integration


pytestmark = pytest.mark.unit


class FakeModel:
    def sharded_state_dict(self, **kwargs):
        self.kwargs = kwargs
        return {
            "adapter.weight": torch.tensor([1.0]),
            "base.weight": torch.tensor([2.0]),
            "adapter._extra_state": torch.tensor([3.0]),
        }

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict


class FakePeft:
    def __call__(self, model, *, training: bool):
        self.call = (model, training)
        return model

    def set_params_to_save(self, model) -> None:
        self.saved_model = model

    def adapter_key_filter(self, key: str) -> bool:
        return key.startswith("adapter.")


def test_peft_pre_wrap_hook_runs_loaders_and_sets_params_to_save() -> None:
    base_model = [FakeModel()]
    loaded_model = [FakeModel()]
    peft = FakePeft()
    events = []

    def base_loader(model):
        events.append(("base", model))
        return loaded_model

    def adapter_loader(model):
        events.append(("adapter", model))

    hook = integration.create_peft_hook(
        peft,
        base_checkpoint_loader=base_loader,
        adapter_checkpoint_loader=adapter_loader,
    )

    assert hook(base_model) is loaded_model
    assert peft.call == (loaded_model, True)
    assert peft.saved_model is loaded_model
    assert events == [("base", base_model), ("adapter", loaded_model)]


def test_create_peft_returns_none_when_rank_disabled() -> None:
    assert integration.create_peft({"rank": 0}) is None


def test_create_peft_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported PEFT type"):
        integration.create_peft({"type": "not_lora", "rank": 4})


def test_create_peft_translates_rank_and_ignores_downstream_keys() -> None:
    peft = integration.create_peft(
        {
            "type": "lora",
            "rank": 4,
            "alpha": 8,
            "unknown_downstream_key": True,
        },
        dtype="bf16",
    )

    assert peft.dim == 4
    assert peft.alpha == 8
    assert peft.lora_dtype is torch.bfloat16
    assert not hasattr(peft, "unknown_downstream_key")


def test_create_peft_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported torch dtype"):
        integration.create_peft({"type": "lora", "rank": 4}, dtype="fp8")


def test_load_peft_adapter_checkpoint_filters_and_loads(monkeypatch) -> None:
    model = [FakeModel()]
    peft = FakePeft()
    calls = {}

    def fake_filter(state_dict, peft):
        return {
            "model": {
                key: value
                for key, value in state_dict["model"].items()
                if peft.adapter_key_filter(key) and "_extra_state" not in key
            }
        }

    fake_checkpointing = types.SimpleNamespace(
        _generate_model_state_dict=lambda model, model_sd_kwargs, ckpt_format, pg_collection=None: {
            "model": model[0].sharded_state_dict()
        },
        apply_peft_adapter_filter_to_state_dict=fake_filter,
    )
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.checkpointing", fake_checkpointing)

    def fake_load(sharded_state_dict, checkpoint_path, load_strategy):
        calls["sharded_state_dict"] = sharded_state_dict
        calls["checkpoint_path"] = checkpoint_path
        calls["load_strategy"] = load_strategy
        return {"model": {"adapter.weight": torch.tensor([4.0])}}

    monkeypatch.setattr("megatron.core.dist_checkpointing.load", fake_load)

    integration.load_peft_adapter_checkpoint(
        model,
        "/adapter",
        peft=peft,
        strict=False,
        fully_parallel_load=False,
        load_strategy="strategy",
    )

    assert sorted(calls["sharded_state_dict"]["model"]) == ["adapter.weight"]
    assert calls["checkpoint_path"] == "/adapter"
    assert calls["load_strategy"] == "strategy"
    assert torch.equal(model[0].loaded_state_dict["adapter.weight"], torch.tensor([4.0]))
    assert model[0].loaded_strict is False


def test_load_peft_adapter_checkpoint_errors_for_missing_model_key(monkeypatch) -> None:
    model = [FakeModel()]
    peft = FakePeft()

    fake_checkpointing = types.SimpleNamespace(
        _generate_model_state_dict=lambda model, model_sd_kwargs, ckpt_format, pg_collection=None: {
            "model": model[0].sharded_state_dict()
        },
        apply_peft_adapter_filter_to_state_dict=lambda state_dict, peft: state_dict,
    )
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.checkpointing", fake_checkpointing)
    monkeypatch.setattr(
        "megatron.core.dist_checkpointing.load",
        lambda sharded_state_dict, checkpoint_path, load_strategy: {"optimizer": {}},
    )

    with pytest.raises(KeyError, match="Expected adapter checkpoint"):
        integration.load_peft_adapter_checkpoint(
            model,
            "/adapter",
            peft=peft,
            fully_parallel_load=False,
            load_strategy="strategy",
        )


def test_create_ddp_config_builds_and_finalizes(monkeypatch) -> None:
    class FakeDDPConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.finalized = False

        def finalize(self):
            self.finalized = True

    fake_config = types.SimpleNamespace(DistributedDataParallelConfig=FakeDDPConfig)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.config", fake_config)

    ddp_config = integration.create_ddp_config(
        use_distributed_optimizer=False,
        use_megatron_fsdp=True,
        overrides={"overlap_grad_reduce": False},
    )

    assert ddp_config.kwargs == {
        "use_distributed_optimizer": True,
        "check_for_nan_in_grad": True,
        "use_megatron_fsdp": True,
        "data_parallel_sharding_strategy": "optim_grads_params",
        "overlap_grad_reduce": False,
    }
    assert ddp_config.finalized is True


def test_create_ddp_config_returns_none_when_not_wrapping() -> None:
    assert integration.create_ddp_config(wrap_with_ddp=False) is None
