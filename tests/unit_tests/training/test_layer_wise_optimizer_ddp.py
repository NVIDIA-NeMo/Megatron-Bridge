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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import megatron.bridge.models.model_provider as model_provider
import megatron.bridge.training.setup as training_setup
from megatron.bridge.training.config import _validate_and_sync_distributed_optimizer_settings


@pytest.mark.parametrize("optimizer_name, layer_wise", [("muon", True), ("dist_muon", False)])
def test_layer_wise_keeps_optimizer_separate_from_ddp(optimizer_name, layer_wise):
    cfg = SimpleNamespace(
        optimizer=SimpleNamespace(
            optimizer=optimizer_name,
            use_layer_wise_distributed_optimizer=layer_wise,
            use_distributed_optimizer=True,
            overlap_param_gather=True,
        ),
        ddp=SimpleNamespace(use_distributed_optimizer=False, overlap_param_gather=True),
    )

    _validate_and_sync_distributed_optimizer_settings(cfg)

    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.optimizer.use_distributed_optimizer is False
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.optimizer.overlap_param_gather is True


@pytest.mark.parametrize("optimizer_name, layer_wise", [("muon", True), ("dist_muon", False), ("adam", False)])
def test_training_setup_forwards_layer_wise_layout(monkeypatch, optimizer_name, layer_wise):
    provider = SimpleNamespace(finalize=Mock(), provide_distributed_model=Mock(return_value=[]))
    cfg = SimpleNamespace(
        model=provider,
        optimizer=SimpleNamespace(
            optimizer=optimizer_name,
            use_layer_wise_distributed_optimizer=layer_wise,
            overlap_param_gather_with_optimizer_step=False,
        ),
        ddp=object(),
        dist=SimpleNamespace(use_megatron_fsdp=False, use_torch_fsdp2=False),
        rng=SimpleNamespace(data_parallel_random_init=False),
    )
    monkeypatch.setattr(training_setup, "configure_gtp_remat", Mock())
    monkeypatch.setattr(training_setup, "classify_gtp_remat_chains", Mock())

    training_setup._build_distributed_model(cfg, pg_collection=object())

    provider.finalize.assert_called_once_with()
    kwargs = provider.provide_distributed_model.call_args.kwargs
    if layer_wise or optimizer_name.startswith("dist_"):
        assert kwargs["use_layer_wise_distributed_optimizer"] is True
    else:
        assert "use_layer_wise_distributed_optimizer" not in kwargs


def test_model_provider_forwards_layer_wise_flag_to_mcore_ddp(monkeypatch):
    provider = SimpleNamespace(fp16=False, bf16=False, use_cpu_initialization=True, init_model_with_meta_device=False)
    chunk = Mock()
    chunk.parameters.return_value = []
    captured_kwargs = {}

    def ddp_wrap(model, *args, use_layer_wise_distributed_optimizer=False, **kwargs):
        captured_kwargs["use_layer_wise_distributed_optimizer"] = use_layer_wise_distributed_optimizer
        return model

    monkeypatch.setattr(model_provider, "_ddp_wrap", ddp_wrap)
    monkeypatch.setattr(model_provider, "_create_model", Mock(return_value=[chunk]))
    monkeypatch.setattr(model_provider, "_finalize_model_quantization", Mock())
    monkeypatch.setattr(model_provider, "_print_num_params", Mock())
    monkeypatch.setattr(model_provider, "get_model_config", Mock(return_value=provider))
    monkeypatch.setattr(model_provider, "correct_amax_history_if_needed", None)

    result = model_provider.get_model(
        provider,
        ddp_config=object(),
        pg_collection=object(),
        use_layer_wise_distributed_optimizer=True,
    )

    assert result == [chunk]
    assert captured_kwargs["use_layer_wise_distributed_optimizer"] is True


def test_layer_wise_layout_rejects_unsupported_mcore_before_model_allocation(monkeypatch):
    def old_ddp_wrap(model, *args, pg_collection):
        return model

    allocate_model = Mock()
    monkeypatch.setattr(model_provider, "_ddp_wrap", old_ddp_wrap)
    monkeypatch.setattr(model_provider, "_create_model", allocate_model)

    with pytest.raises(RuntimeError, match="does not support layer-wise optimizer layouts"):
        model_provider.get_model(
            SimpleNamespace(),
            ddp_config=object(),
            pg_collection=object(),
            use_layer_wise_distributed_optimizer=True,
        )

    allocate_model.assert_not_called()
