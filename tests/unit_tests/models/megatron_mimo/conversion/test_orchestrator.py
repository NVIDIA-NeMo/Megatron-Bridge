# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from collections import namedtuple

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.megatron_mimo.conversion import (
    MegatronMIMOBridge,
    MIMOComponent,
    component_pg_context,
    export_megatron_mimo_to_hf,
    get_mimo_adapter,
    import_hf_to_megatron_mimo,
    register_mimo_conversion,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


HFWeightTuple = namedtuple("HFWeightTuple", ["param_name", "weight"])


class _FakeMimoModel(nn.Module):
    """A minimal MimoModel-shaped container for orchestrator tests."""

    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(4, 4)
        self.vision_branch = nn.Linear(4, 4)


class _RecordingBridge:
    """Fake source bridge that records calls made by the orchestrator.

    The orchestrator clones the bridge via ``copy.copy`` and overrides the
    instance's ``mapping_registry`` attribute. Class-level method recording
    survives this — every clone shares the same class.
    """

    calls_load: list[dict] = []
    calls_export: list[dict] = []
    next_export_yield: list[tuple[str, torch.Tensor]] = []

    @classmethod
    def reset(cls):
        cls.calls_load = []
        cls.calls_export = []
        cls.next_export_yield = []

    def __init__(self):
        self._registry = MegatronMappingRegistry(
            AutoMapping("language_model.weight", "model.lm_head.weight"),
            AutoMapping("vision_branch.weight", "model.visual.proj.weight"),
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        return self._registry

    def load_weights_hf_to_megatron(self, hf_pretrained, megatron_model):
        """Records (a) the route-local registry it sees, (b) the submodule,
        (c) the submodule's pg_collection at call time."""
        self.calls_load.append(
            {
                "registry_param_names": [m.megatron_param for m in self.mapping_registry().mappings],
                "submodule": megatron_model,
                "submodule_pg_collection_at_call": getattr(megatron_model, "pg_collection", None),
                "hf_pretrained": hf_pretrained,
            }
        )
        return [megatron_model]

    def stream_weights_megatron_to_hf(
        self,
        megatron_model,
        hf_pretrained,
        cpu: bool = True,
        show_progress: bool = True,
    ):
        self.calls_export.append(
            {
                "registry_param_names": [m.megatron_param for m in self.mapping_registry().mappings],
                "submodule": megatron_model,
                "submodule_pg_collection_at_call": getattr(megatron_model, "pg_collection", None),
                "cpu": cpu,
                "show_progress": show_progress,
            }
        )
        for name, tensor in self.next_export_yield:
            yield HFWeightTuple(name, tensor)


def _two_routes() -> list[MIMOComponent]:
    return [
        MIMOComponent(
            name="language",
            source_prefix="language_model.",
            target_module_path="language_model",
        ),
        MIMOComponent(
            name="vision",
            source_prefix="vision_branch.",
            target_module_path="vision_branch",
        ),
    ]


class _PgCollection:
    """Identity stub. Real ProcessGroupCollection is duck-typed.

    Exposes ``tp`` / ``dp`` / ``dp_cp`` / ``pp`` attributes (each unique
    sentinels) so that ``_bridged_parallel_state`` can write them onto
    ``parallel_state`` without crashing during orchestrator unit tests.
    """

    def __init__(self, label):
        self.label = label
        self.tp = object()
        self.dp = object()
        self.dp_cp = object()
        self.pp = object()

    def __repr__(self):
        return f"_PgCollection({self.label!r})"


class TestComponentPgContext:
    def test_restores_on_exception(self):
        module = nn.Linear(4, 4)
        pg = _PgCollection("language")

        with pytest.raises(RuntimeError, match="boom"):
            with component_pg_context(module, pg):
                assert module.pg_collection is pg
                raise RuntimeError("boom")

        assert getattr(module, "pg_collection", None) is None

    def test_existing_matching_pg_passthrough(self):
        module = nn.Linear(4, 4)
        pg = _PgCollection("language")
        module.pg_collection = pg

        with component_pg_context(module, pg):
            assert module.pg_collection is pg

        # Preserved — not deleted because we didn't attach it
        assert module.pg_collection is pg

    def test_existing_pg_is_trusted_not_overwritten(self):
        """Module's attached pg_collection is the source of truth.

        ``MegatronMIMOProvider`` injects per-rank pg_collection into specs at
        construction time, so the constructed submodule already carries the
        correct groups. ``build_infra`` produces fresh ``ProcessGroupCollection``
        instances on each call, so the orchestrator's pg_collections map
        won't compare equal to what's already on the module. The context
        manager therefore trusts the existing attachment and never
        overwrites it.
        """
        module = nn.Linear(4, 4)
        existing = _PgCollection("language")
        attempted = _PgCollection("vision")
        module.pg_collection = existing

        with component_pg_context(module, attempted):
            # Inside the context, the module's existing pg_collection is what
            # downstream bridge code will see.
            assert module.pg_collection is existing

        # Existing is preserved on exit because we never attached the new one.
        assert module.pg_collection is existing


class TestImportHfToMegatronMimo:
    def setup_method(self):
        _RecordingBridge.reset()

    def test_calls_wrapped_bridge_per_active_route(self):
        bridge = _RecordingBridge()
        model = _FakeMimoModel()
        routes = _two_routes()
        pg_lang = _PgCollection("language")
        pg_vision = _PgCollection("vision")
        pg_collections = {"language": pg_lang, "vision": pg_vision}

        returned = import_hf_to_megatron_mimo(
            source_bridge=bridge,
            hf_pretrained="hf-handle",
            mimo_model=model,
            routes=routes,
            pg_collections=pg_collections,
        )
        assert returned is model
        assert len(_RecordingBridge.calls_load) == 2

        first, second = _RecordingBridge.calls_load
        # Route-local registry has prefix stripped
        assert first["registry_param_names"] == ["weight"]
        assert second["registry_param_names"] == ["weight"]
        # Submodule resolved through get_submodule(target_module_path)
        assert first["submodule"] is model.language_model
        assert second["submodule"] is model.vision_branch
        # pg_collection attached during the call
        assert first["submodule_pg_collection_at_call"] is pg_lang
        assert second["submodule_pg_collection_at_call"] is pg_vision
        # hf_pretrained forwarded unchanged
        assert first["hf_pretrained"] == "hf-handle"

    def test_pg_collection_removed_after_each_route(self):
        bridge = _RecordingBridge()
        model = _FakeMimoModel()
        routes = _two_routes()
        pg_collections = {"language": _PgCollection("a"), "vision": _PgCollection("b")}

        import_hf_to_megatron_mimo(
            source_bridge=bridge,
            hf_pretrained="hf",
            mimo_model=model,
            routes=routes,
            pg_collections=pg_collections,
        )

        # After return, neither submodule should carry pg_collection
        assert getattr(model.language_model, "pg_collection", None) is None
        assert getattr(model.vision_branch, "pg_collection", None) is None

    def test_skips_routes_when_rank_does_not_own(self):
        bridge = _RecordingBridge()
        model = _FakeMimoModel()
        routes = _two_routes()
        pg_collections = {"language": _PgCollection("a"), "vision": None}

        import_hf_to_megatron_mimo(
            source_bridge=bridge,
            hf_pretrained="hf",
            mimo_model=model,
            routes=routes,
            pg_collections=pg_collections,
        )

        assert len(_RecordingBridge.calls_load) == 1
        assert _RecordingBridge.calls_load[0]["submodule"] is model.language_model

    def test_source_bridge_unchanged(self):
        bridge = _RecordingBridge()
        original_registry = bridge.mapping_registry()
        model = _FakeMimoModel()
        routes = _two_routes()
        pg_collections = {"language": _PgCollection("a"), "vision": _PgCollection("b")}

        import_hf_to_megatron_mimo(
            source_bridge=bridge,
            hf_pretrained="hf",
            mimo_model=model,
            routes=routes,
            pg_collections=pg_collections,
        )

        # Source bridge's registry still has the unstripped names
        assert bridge.mapping_registry() is original_registry
        names = {m.megatron_param for m in original_registry.mappings}
        assert names == {"language_model.weight", "vision_branch.weight"}


class TestExportMegatronMimoToHf:
    def setup_method(self):
        _RecordingBridge.reset()

    def test_yields_from_each_route_in_order(self):
        bridge = _RecordingBridge()
        model = _FakeMimoModel()
        routes = _two_routes()
        pg_collections = {"language": _PgCollection("a"), "vision": _PgCollection("b")}

        t_lm = torch.ones(2, 2)
        t_vp = torch.zeros(2, 2)
        _RecordingBridge.next_export_yield = [
            ("model.lm_head.weight", t_lm),
            ("model.visual.proj.weight", t_vp),
        ]

        emitted = list(
            export_megatron_mimo_to_hf(
                source_bridge=bridge,
                hf_pretrained="hf",
                mimo_model=model,
                routes=routes,
                pg_collections=pg_collections,
                cpu=False,
                show_progress=False,
            )
        )

        # Each route yields the full mocked list, in route declaration order
        assert len(emitted) == 4  # 2 yields per route × 2 routes
        assert emitted[0].param_name == "model.lm_head.weight"
        assert emitted[2].param_name == "model.lm_head.weight"
        # Verify pg_collection was attached on each call
        assert _RecordingBridge.calls_export[0]["submodule"] is model.language_model
        assert _RecordingBridge.calls_export[0]["submodule_pg_collection_at_call"] is pg_collections["language"]
        assert _RecordingBridge.calls_export[1]["submodule"] is model.vision_branch
        assert _RecordingBridge.calls_export[1]["submodule_pg_collection_at_call"] is pg_collections["vision"]
        # cpu/show_progress forwarded
        assert _RecordingBridge.calls_export[0]["cpu"] is False
        assert _RecordingBridge.calls_export[0]["show_progress"] is False

    def test_skips_routes_when_rank_does_not_own(self):
        bridge = _RecordingBridge()
        model = _FakeMimoModel()
        routes = _two_routes()
        pg_collections = {"language": _PgCollection("a"), "vision": None}

        _RecordingBridge.next_export_yield = [("model.lm_head.weight", torch.ones(2, 2))]

        emitted = list(
            export_megatron_mimo_to_hf(
                source_bridge=bridge,
                hf_pretrained="hf",
                mimo_model=model,
                routes=routes,
                pg_collections=pg_collections,
            )
        )
        # Only the language route ran
        assert len(emitted) == 1
        assert len(_RecordingBridge.calls_export) == 1
        assert _RecordingBridge.calls_export[0]["submodule"] is model.language_model


class _FakeSourceBridgeForMIMOBridge:
    export_weight_dtype = "bf16"


class _FakeProviderForMIMOBridge:
    def __init__(self):
        self.modality_submodules_spec = {"images": object()}


class _FakeInfraForMIMOBridge:
    def __init__(self):
        self.pg_collections = {"language": object(), "images": object()}


def _bridge_parallelism_config() -> MegatronMIMOParallelismConfig:
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1),
        }
    )


def _bridge_routes() -> list[MIMOComponent]:
    return [
        MIMOComponent("language", "language_model.", "language_model"),
        MIMOComponent("images", "vision_branch.", "vision_branch"),
    ]


def _ensure_fake_mimo_bridge_adapter_registered() -> None:
    try:
        get_mimo_adapter(_FakeSourceBridgeForMIMOBridge)
        return
    except KeyError:
        pass

    @register_mimo_conversion(_FakeSourceBridgeForMIMOBridge)
    def _fake_mimo_bridge_adapter(source_bridge, hf_pretrained, parallelism_config):
        return _FakeProviderForMIMOBridge(), _bridge_routes()


def _mimo_bridge() -> MegatronMIMOBridge:
    _ensure_fake_mimo_bridge_adapter_registered()
    return MegatronMIMOBridge(
        PretrainedConfig(),
        parallelism_config=_bridge_parallelism_config(),
        source_bridge=_FakeSourceBridgeForMIMOBridge(),
    )


class TestMegatronMIMOBridge:
    def test_to_megatron_provider_resolves_adapter_and_routes(self):
        bridge = _mimo_bridge()

        provider = bridge.to_megatron_provider()

        assert isinstance(provider, _FakeProviderForMIMOBridge)
        assert [route.name for route in bridge.routes] == ["language", "images"]

    def test_from_bridge_copies_source_state(self):
        _ensure_fake_mimo_bridge_adapter_registered()

        class _StandardBridge:
            hf_pretrained = PretrainedConfig()
            export_weight_dtype = "fp16"
            hf_model_id = "hf-model"
            _model_bridge = _FakeSourceBridgeForMIMOBridge()

        bridge = MegatronMIMOBridge.from_bridge(_StandardBridge(), parallelism_config=_bridge_parallelism_config())

        assert bridge.hf_pretrained is _StandardBridge.hf_pretrained
        assert bridge.export_weight_dtype == "fp16"
        assert bridge.hf_model_id == "hf-model"
        assert isinstance(bridge._model_bridge, _FakeSourceBridgeForMIMOBridge)

    def test_load_hf_weights_delegates_to_route_import(self, monkeypatch):
        import megatron.bridge.models.megatron_mimo.conversion.orchestrator as orchestrator_module

        calls = []

        def fake_import_hf_to_megatron_mimo(**kwargs):
            calls.append(kwargs)
            return kwargs["mimo_model"]

        monkeypatch.setattr(orchestrator_module, "import_hf_to_megatron_mimo", fake_import_hf_to_megatron_mimo)

        bridge = _mimo_bridge()
        bridge.to_megatron_provider()
        bridge._infra = _FakeInfraForMIMOBridge()
        monkeypatch.setattr(bridge, "_resolve_hf_pretrained", lambda hf_path: "hf")

        model = _FakeMimoModel()
        returned = bridge.load_hf_weights(model, allowed_mismatched_params=["ignored.*"])

        assert returned is model
        assert len(calls) == 1
        assert calls[0]["source_bridge"] is bridge._model_bridge
        assert calls[0]["hf_pretrained"] == "hf"
        assert calls[0]["mimo_model"] is model
        assert [route.name for route in calls[0]["routes"]] == ["language", "images"]
        assert calls[0]["pg_collections"] == bridge._infra.pg_collections
        assert calls[0]["allowed_mismatched_params"] == ["ignored.*"]

    def test_import_ckpt_builds_loads_and_saves(self, monkeypatch):
        bridge = _mimo_bridge()
        model = _FakeMimoModel()
        calls = []

        monkeypatch.setattr(
            bridge,
            "to_megatron_model",
            lambda **kwargs: calls.append(("to_megatron_model", kwargs)) or [model],
        )
        monkeypatch.setattr(
            bridge,
            "save_megatron_model",
            lambda model_arg, path_arg, **kwargs: calls.append(("save_megatron_model", model_arg, path_arg, kwargs)),
        )

        bridge.import_ckpt("/ckpt", hf_tokenizer_path="hf", hf_tokenizer_kwargs={"trust_remote_code": True})

        assert calls[0] == (
            "to_megatron_model",
            {"load_weights": True, "wrap_with_ddp": False, "data_parallel_random_init": False},
        )
        assert calls[1] == (
            "save_megatron_model",
            model,
            "/ckpt",
            {"hf_tokenizer_path": "hf", "hf_tokenizer_kwargs": {"trust_remote_code": True}},
        )

    def test_export_ckpt_loads_then_saves_hf(self, monkeypatch):
        bridge = _mimo_bridge()
        model = _FakeMimoModel()
        calls = []

        monkeypatch.setattr(bridge, "load_megatron_model", lambda path: calls.append(("load", path)) or model)
        monkeypatch.setattr(
            bridge,
            "save_hf_pretrained",
            lambda model_arg, path_arg, **kwargs: calls.append(("save_hf", model_arg, path_arg, kwargs)),
        )

        bridge.export_ckpt("/ckpt", "/hf", show_progress=False, strict=False)

        assert calls == [
            ("load", "/ckpt"),
            ("save_hf", model, "/hf", {"show_progress": False, "source_path": None, "strict": False}),
        ]
