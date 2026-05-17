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


def test_create_peft_imports_only_selected_peft_type(monkeypatch) -> None:
    real_import_module = integration.import_module

    def guarded_import_module(module_name):
        if module_name in {"megatron.bridge.peft.lora", "megatron.bridge.peft.canonical_lora"}:
            raise AssertionError(f"unexpected import of {module_name}")
        return real_import_module(module_name)

    monkeypatch.setattr(integration, "import_module", guarded_import_module)

    peft = integration.create_peft({"type": "dora", "rank": 4})

    assert peft.__class__.__name__ == "DoRA"
    assert peft.dim == 4


def test_create_peft_reports_lora_import_failure(monkeypatch) -> None:
    real_import_module = integration.import_module

    def failing_import_module(module_name):
        if module_name == "megatron.bridge.peft.lora":
            raise ModuleNotFoundError("No module named 'transformer_engine'")
        return real_import_module(module_name)

    monkeypatch.setattr(integration, "import_module", failing_import_module)

    with pytest.raises(ImportError, match=r"PEFT type 'lora'.*\(megatron\.bridge\.peft\.lora:LoRA\).*\[te\] extra"):
        integration.create_peft({"type": "lora", "rank": 4})


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


def test_load_peft_adapter_checkpoint_errors_for_missing_virtual_model_key(monkeypatch) -> None:
    model = [FakeModel(), FakeModel()]
    peft = FakePeft()

    fake_checkpointing = types.SimpleNamespace(
        _generate_model_state_dict=lambda model, model_sd_kwargs, ckpt_format, pg_collection=None: {
            f"model{index}": model_chunk.sharded_state_dict() for index, model_chunk in enumerate(model)
        },
        apply_peft_adapter_filter_to_state_dict=lambda state_dict, peft: state_dict,
    )
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.checkpointing", fake_checkpointing)
    monkeypatch.setattr(
        "megatron.core.dist_checkpointing.load",
        lambda sharded_state_dict, checkpoint_path, load_strategy: {"model0": {}},
    )

    with pytest.raises(KeyError, match="model1"):
        integration.load_peft_adapter_checkpoint(
            model,
            "/adapter",
            peft=peft,
            fully_parallel_load=False,
            load_strategy="strategy",
        )


def test_fsdp_dtensor_checkpoint_helpers_are_reexported() -> None:
    from megatron.bridge.training import checkpointing

    assert "save_fsdp_dtensor_checkpoint" in integration.__all__
    assert "load_fsdp_dtensor_checkpoint" in integration.__all__
    assert integration.save_fsdp_dtensor_checkpoint is checkpointing.save_fsdp_dtensor_checkpoint
    assert integration.load_fsdp_dtensor_checkpoint is checkpointing.load_fsdp_dtensor_checkpoint


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


def test_linear_for_last_layer_returns_megatron_style_tuple() -> None:
    head = integration.LinearForLastLayer(input_size=2, output_size=1, sequence_parallel=False)
    with torch.no_grad():
        head.weight.fill_(2.0)

    logits, bias = head(torch.ones(3, 2))

    assert torch.equal(logits, torch.full((3, 1), 4.0))
    assert logits.dtype == torch.float32
    assert bias is None


def test_linear_for_last_layer_gathers_sequence_parallel_output(monkeypatch) -> None:
    head = integration.LinearForLastLayer(input_size=2, output_size=1, sequence_parallel=True)
    with torch.no_grad():
        head.weight.fill_(1.0)

    calls = {}

    def fake_gather(tensor, *, tensor_parallel_output_grad):
        calls["tensor"] = tensor
        calls["tensor_parallel_output_grad"] = tensor_parallel_output_grad
        return tensor + 1

    monkeypatch.setattr(integration.tensor_parallel, "gather_from_sequence_parallel_region", fake_gather)

    logits, bias = head(torch.ones(2, 2))

    assert torch.equal(logits, torch.full((2, 1), 3.0))
    assert torch.equal(calls["tensor"], torch.full((2, 1), 2.0))
    assert calls["tensor_parallel_output_grad"] is False
    assert bias is None


def test_create_value_head_hook_replaces_last_virtual_pipeline_chunk(monkeypatch) -> None:
    from megatron.core import parallel_state

    def fake_is_pipeline_last_stage(*, ignore_virtual=False, vp_stage=None) -> bool:
        del ignore_virtual
        return vp_stage == 1

    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", fake_is_pipeline_last_stage)

    model_chunks = [torch.nn.Module(), torch.nn.Module()]
    hook = integration.create_value_head_hook(hidden_size=4, output_size=2, sequence_parallel=True)

    result = hook(model_chunks)

    assert result is model_chunks
    assert not hasattr(model_chunks[0], "output_layer")
    output_layer = model_chunks[1].output_layer
    assert isinstance(output_layer, integration.LinearForLastLayer)
    assert output_layer.in_features == 4
    assert output_layer.out_features == 2
    assert output_layer.sequence_parallel is True


def test_create_value_head_hook_requires_chunk_count_to_match_pipeline_flags(monkeypatch) -> None:
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True)

    hook = integration.create_value_head_hook(hidden_size=4, sequence_parallel=False)

    with pytest.raises(ValueError, match="Model list length"):
        hook([torch.nn.Module(), torch.nn.Module()])


def test_make_value_model_alias_creates_value_head_hook(monkeypatch) -> None:
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True)

    model_chunks = [torch.nn.Module()]
    hook = integration.make_value_model(hidden_size=8, sequence_parallel=False)

    result = hook(model_chunks)

    assert result is model_chunks
    assert isinstance(model_chunks[0].output_layer, integration.LinearForLastLayer)
    assert model_chunks[0].output_layer.in_features == 8


def test_freeze_moe_router_freezes_router_and_shared_expert_gates() -> None:
    router = torch.nn.Linear(2, 2)
    shared_experts = types.SimpleNamespace(
        gate_weight=torch.nn.Parameter(torch.ones(2, 2)),
        gate_bias=torch.nn.Parameter(torch.ones(2)),
    )
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(router=router, shared_experts=shared_experts))
    model = types.SimpleNamespace(decoder=types.SimpleNamespace(layers=[layer]))

    result = integration.freeze_moe_router([model])

    assert result == [model]
    assert router.weight.requires_grad is False
    assert router.bias.requires_grad is False
    assert shared_experts.gate_weight.requires_grad is False
    assert shared_experts.gate_bias.requires_grad is False
