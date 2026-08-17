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

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_roundtrip_module(monkeypatch: pytest.MonkeyPatch):
    megatron = types.ModuleType("megatron")
    bridge = types.ModuleType("megatron.bridge")
    bridge.AutoBridge = object
    models = types.ModuleType("megatron.bridge.models")
    decorators = types.ModuleType("megatron.bridge.models.decorators")
    decorators.torchrun_main = lambda function: function
    hf_pretrained = types.ModuleType("megatron.bridge.models.hf_pretrained")
    hf_utils = types.ModuleType("megatron.bridge.models.hf_pretrained.utils")
    hf_utils.is_safe_repo = lambda **_kwargs: False
    utils = types.ModuleType("megatron.bridge.utils")
    slurm_utils = types.ModuleType("megatron.bridge.utils.slurm_utils")
    slurm_utils.resolve_slurm_master_addr = lambda: "node001"
    slurm_utils.resolve_slurm_master_port = lambda: 15456
    modules = {
        "megatron": megatron,
        "megatron.bridge": bridge,
        "megatron.bridge.models": models,
        "megatron.bridge.models.decorators": decorators,
        "megatron.bridge.models.hf_pretrained": hf_pretrained,
        "megatron.bridge.models.hf_pretrained.utils": hf_utils,
        "megatron.bridge.utils": utils,
        "megatron.bridge.utils.slurm_utils": slurm_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    script = REPO_ROOT / "examples/conversion/hf_megatron_roundtrip_multi_gpu.py"
    spec = importlib.util.spec_from_file_location("test_hf_megatron_roundtrip_multi_gpu", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configure_slurm_distributed_environment(monkeypatch: pytest.MonkeyPatch):
    module = _load_roundtrip_module(monkeypatch)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_PROCID", "9")
    monkeypatch.setenv("SLURM_LOCALID", "1")

    module._configure_slurm_distributed_environment()

    assert module.os.environ["RANK"] == "9"
    assert module.os.environ["WORLD_SIZE"] == "16"
    assert module.os.environ["LOCAL_RANK"] == "1"
    assert module.os.environ["MASTER_ADDR"] == "node001"
    assert module.os.environ["MASTER_PORT"] == "15456"


def test_standalone_roundtrip_defaults_to_loose_export(monkeypatch: pytest.MonkeyPatch):
    module = _load_roundtrip_module(monkeypatch)

    args = module._build_parser().parse_args([])

    assert args.strict is False


def test_standalone_roundtrip_strict_export_is_opt_in(monkeypatch: pytest.MonkeyPatch):
    module = _load_roundtrip_module(monkeypatch)

    strict_args = module._build_parser().parse_args(["--strict"])
    legacy_loose_args = module._build_parser().parse_args(["--not-strict"])

    assert strict_args.strict is True
    assert legacy_loose_args.strict is False

    with pytest.raises(SystemExit):
        module._build_parser().parse_args(["--strict", "--not-strict"])


def test_standalone_roundtrip_defaults_to_distributed_export(monkeypatch: pytest.MonkeyPatch):
    module = _load_roundtrip_module(monkeypatch)

    default_args = module._build_parser().parse_args([])
    serial_args = module._build_parser().parse_args(["--no-distributed-save"])

    assert default_args.distributed_save is True
    assert serial_args.distributed_save is False


def test_weight_verification_broadcasts_rank_zero_mismatch(monkeypatch: pytest.MonkeyPatch):
    module = _load_roundtrip_module(monkeypatch)
    mismatch = SimpleNamespace(value=0, item=lambda: mismatch.value)
    tensor_calls = []

    def fake_tensor(value, *, dtype, device):
        tensor_calls.append((value, dtype, device))
        return mismatch

    def fake_broadcast(value, *, src):
        assert value is mismatch
        assert src == 0
        mismatch.value = 1

    monkeypatch.setattr(module.torch, "tensor", fake_tensor)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(module.torch.distributed, "broadcast", fake_broadcast)

    assert module._synchronize_weight_verification(all_match=True) is True
    assert tensor_calls == [(False, torch.int32, 3)]


def test_roundtrip_synchronizes_result_and_saves_before_reporting_summary(
    monkeypatch: pytest.MonkeyPatch,
):
    """Ranks must synchronize and save before rank 0 reports verification."""
    module = _load_roundtrip_module(monkeypatch)
    events = []

    class FakeProvider:
        tensor_model_parallel_size = 1
        pipeline_model_parallel_size = 1
        expert_model_parallel_size = 1
        expert_tensor_parallel_size = 1

        def finalize(self):
            return None

        def initialize_model_parallel(self, seed):
            return None

        def provide_distributed_model(self, wrap_with_ddp):
            return [object()]

    class FakeBridge:
        hf_pretrained = SimpleNamespace(state={"weight": torch.ones(1)})

        def to_megatron_provider(self, load_weights):
            return FakeProvider()

        def export_hf_weights(self, model, show_progress):
            yield "weight", torch.ones(1)

        def save_hf_pretrained(self, model, path, **kwargs):
            events.append(("save", kwargs))

    fake_bridge = FakeBridge()
    monkeypatch.setattr(
        module,
        "AutoBridge",
        SimpleNamespace(from_hf_pretrained=lambda *args, **kwargs: fake_bridge),
    )
    monkeypatch.setattr(module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        module,
        "_synchronize_weight_verification",
        lambda all_match: events.append(("synchronize", all_match)),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_print_verification_results",
        lambda *args, **kwargs: events.append(("report", args)),
    )
    monkeypatch.setenv("WORLD_SIZE", "2")

    module.main(hf_model_id="example/model", distributed_save=True)

    event_names = [event[0] for event in events]
    assert "synchronize" in event_names
    assert event_names.index("synchronize") < event_names.index("save")
    assert event_names.index("save") < event_names.index("report")
    save_kwargs = next(event[1] for event in events if event[0] == "save")
    assert save_kwargs["distributed_save"] is True
