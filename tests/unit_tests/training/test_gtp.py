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

"""Tests for Generalized Tensor Parallelism runtime wiring."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import megatron.core.tensor_parallel as tensor_parallel
import pytest

from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.bridge.training.gtp import (
    _get_checkpoint_weight_topology,
    _get_dataloader_process_group,
    _validate_checkpoint_weight_topology,
    classify_gtp_remat_chains,
    configure_gtp_remat,
    get_data_distribution_group,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("nested", [False, True])
def test_checkpoint_weight_topology_reads_serialized_public_shards(nested):
    config = {
        "tensor_model_parallel_size": 2,
        "expert_tensor_parallel_size": 1,
        "tensor_parallel_num_weight_shards": 4,
        "expert_tensor_parallel_num_weight_shards": 4,
        "gtp_weight_remat_size": 1,
        "expert_gtp_weight_remat_size": 1,
    }
    assert _get_checkpoint_weight_topology({"transformer": config} if nested else config) == (2, 2, 1, 4)


def test_checkpoint_weight_topology_reads_legacy_derived_sizes():
    config = SimpleNamespace(
        tensor_model_parallel_size=2,
        expert_tensor_parallel_size=None,
        gtp_weight_remat_size=4,
        expert_gtp_weight_remat_size=2,
    )
    assert _get_checkpoint_weight_topology(config) == (2, 4, 2, 2)


def test_checkpoint_weight_topology_reads_nested_runtime_config():
    @dataclass
    class ModelConfig:
        transformer: SimpleNamespace

    config = ModelConfig(
        transformer=SimpleNamespace(
            tensor_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            tensor_parallel_num_weight_shards=4,
            expert_tensor_parallel_num_weight_shards=4,
        )
    )
    assert _get_checkpoint_weight_topology(config) == (2, 2, 1, 4)


@pytest.mark.parametrize(
    "saved,requested",
    [
        ((1, 2, 1, 1), (1, 1, 1, 1)),
        ((1, 1, 1, 1), (1, 2, 1, 1)),
        ((1, 1, 1, 2), (1, 1, 1, 1)),
        ((1, 1, 1, 1), (1, 1, 1, 2)),
        ((1, 2, 1, 1), (2, 1, 1, 1)),
        ((1, 2, 1, 2), (1, 2, 2, 1)),
    ],
)
def test_checkpoint_weight_topology_rejects_native_gtp_resharding(saved, requested):
    with pytest.raises(ValueError, match="Resharding a GTP checkpoint"):
        _validate_checkpoint_weight_topology(saved=saved, requested=requested)


@pytest.mark.parametrize(
    "saved,requested",
    [
        ((1, 2, 1, 4), (1, 2, 1, 4)),
        ((1, 1, 1, 1), (2, 1, 4, 1)),
        ((2, 2, 2, 1), (2, 2, 1, 1)),
        ((2, 1, 2, 2), (1, 1, 2, 2)),
    ],
)
def test_checkpoint_weight_topology_preserves_same_gtp_and_plain_resharding(saved, requested):
    _validate_checkpoint_weight_topology(saved=saved, requested=requested)


@pytest.mark.parametrize("dense_gtp_size", [None, 1, 2])
def test_dataloader_process_group_excludes_cp_and_preserves_dense_gtp(dense_gtp_size):
    pg = SimpleNamespace(dp=object(), gtp_remat=None, expt_gtp_remat=MagicMock())
    pg.expt_gtp_remat.size.return_value = 2
    if dense_gtp_size is not None:
        pg.gtp_remat = MagicMock()
        pg.gtp_remat.size.return_value = dense_gtp_size
    with patch("megatron.bridge.training.gtp.parallel_state.get_data_parallel_group") as get_group:
        group = _get_dataloader_process_group(pg)
    if dense_gtp_size == 2:
        get_group.assert_called_once_with(with_gtp_remat=True)
        assert group is get_group.return_value
    else:
        get_group.assert_not_called()
        assert group is pg.dp


def _gtp_config(*, dense_size: int = 2, expert_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        gtp_weight_remat_size=dense_size,
        expert_gtp_weight_remat_size=expert_size,
        fp4=None,
        fp8_recipe=None,
        fp8=None,
        calculate_per_token_loss=True,
        cuda_graph_modules=["attn"],
        moe_shared_expert_overlap=False,
        cuda_graph_impl="none",
    )


def test_transformer_config_derives_gtp_sizes_from_weight_shards():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        tensor_parallel_num_weight_shards=8,
        expert_tensor_parallel_size=2,
        expert_tensor_parallel_num_weight_shards=6,
    )

    config.finalize()

    assert config.gtp_weight_remat_size == 4
    assert config.expert_gtp_weight_remat_size == 3


def test_mla_transformer_config_derives_gtp_sizes_from_weight_shards():
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        tensor_parallel_num_weight_shards=8,
        expert_tensor_parallel_size=2,
        expert_tensor_parallel_num_weight_shards=6,
    )

    config.finalize()

    assert config.gtp_weight_remat_size == 4
    assert config.expert_gtp_weight_remat_size == 3


@pytest.mark.parametrize("num_weight_shards", [1, 3])
def test_transformer_config_rejects_invalid_gtp_weight_shards(num_weight_shards):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        tensor_parallel_num_weight_shards=num_weight_shards,
    )

    with pytest.raises(ValueError, match="tensor_parallel_num_weight_shards"):
        config.finalize()


@pytest.mark.parametrize("fp32_accumulation", [False, True])
@pytest.mark.parametrize("dense_size,expert_size", [(2, 1), (1, 2)])
def test_configure_gtp_remat_forwards_transformer_recipe(monkeypatch, fp32_accumulation, dense_size, expert_size):
    mock_configure = MagicMock()
    monkeypatch.setattr(
        tensor_parallel,
        "gtp_api",
        SimpleNamespace(HAVE_GTP=True, configure_gtp_remat_from_recipe=mock_configure),
        raising=False,
    )
    config = _gtp_config(dense_size=dense_size, expert_size=expert_size)

    configure_gtp_remat(config, reduce_scatter_with_fp32_accumulation=fp32_accumulation)

    mock_configure.assert_called_once_with(
        fp4=False,
        fp8_recipe=None,
        fp8=False,
        calculate_per_token_loss=True,
        reduce_scatter_with_fp32_accumulation=fp32_accumulation,
    )


@pytest.mark.parametrize("nccl_ub", [False, True])
@patch("megatron.core.process_groups_config.resolve_gtp_remat_group")
def test_configure_gtp_remat_registers_dense_user_buffer(mock_resolve_group, monkeypatch, nccl_ub):
    mock_register = MagicMock()
    monkeypatch.setattr(
        tensor_parallel,
        "gtp_api",
        SimpleNamespace(
            HAVE_GTP=True,
            configure_gtp_remat_from_recipe=MagicMock(),
            register_gtp_symm_pool=mock_register,
        ),
        raising=False,
    )
    pg_collection = SimpleNamespace()

    configure_gtp_remat(_gtp_config(), nccl_ub=nccl_ub, pg_collection=pg_collection)

    if nccl_ub:
        mock_resolve_group.assert_called_once_with(pg_collection, is_expert=False)
        mock_register.assert_called_once_with(mock_resolve_group.return_value)
    else:
        mock_resolve_group.assert_not_called()
        mock_register.assert_not_called()


def test_configure_gtp_remat_user_buffer_requires_process_groups(monkeypatch):
    mock_register = MagicMock()
    monkeypatch.setattr(
        tensor_parallel,
        "gtp_api",
        SimpleNamespace(
            HAVE_GTP=True,
            configure_gtp_remat_from_recipe=MagicMock(),
            register_gtp_symm_pool=mock_register,
        ),
        raising=False,
    )

    with pytest.raises(ValueError, match="gtp_remat_nccl_ub requires an initialized process-group collection"):
        configure_gtp_remat(_gtp_config(), nccl_ub=True)

    mock_register.assert_not_called()


def test_configure_gtp_remat_skips_runtime_controls_when_gtp_is_inactive(monkeypatch):
    mock_api = MagicMock()
    monkeypatch.setattr(tensor_parallel, "gtp_api", mock_api, raising=False)

    configure_gtp_remat(
        _gtp_config(dense_size=1, expert_size=1),
        reduce_scatter_with_fp32_accumulation=True,
        nccl_ub=True,
    )

    mock_api.configure_gtp_remat_from_recipe.assert_not_called()
    mock_api.register_gtp_symm_pool.assert_not_called()


def test_classify_gtp_remat_chains_receives_all_model_chunks(monkeypatch):
    mock_classify = MagicMock()
    monkeypatch.setattr(
        tensor_parallel,
        "gtp_api",
        SimpleNamespace(classify_gtp_remat_chains=mock_classify),
        raising=False,
    )
    config = _gtp_config()
    model = [MagicMock(), MagicMock()]

    classify_gtp_remat_chains(model, config)

    mock_classify.assert_called_once_with(
        model,
        cuda_graph_modules=["attn"],
        moe_shared_expert_overlap=False,
        cuda_graph_impl="none",
    )


def test_gtp_off_preserves_existing_data_parallel_groups():
    config = _gtp_config(dense_size=1, expert_size=1)
    pg_collection = SimpleNamespace(dp=object(), dp_cp=object())

    assert get_data_distribution_group(pg_collection, config) is pg_collection.dp
    assert get_data_distribution_group(pg_collection, config, with_context_parallel=True) is pg_collection.dp_cp


@patch("megatron.bridge.training.gtp.parallel_state.get_data_parallel_group")
def test_gtp_uses_full_data_distribution_groups(mock_get_data_parallel_group):
    config = _gtp_config()
    full_dp_group = object()
    full_dp_cp_group = object()
    pg_collection = SimpleNamespace(dp_cp_gtp_remat=full_dp_cp_group)
    mock_get_data_parallel_group.return_value = full_dp_group

    assert get_data_distribution_group(pg_collection, config) is full_dp_group
    assert get_data_distribution_group(pg_collection, config, with_context_parallel=True) is full_dp_cp_group
    mock_get_data_parallel_group.assert_called_once_with(with_gtp_remat=True)


@pytest.mark.parametrize("fp32_accumulation", [False, True])
@pytest.mark.parametrize("nccl_ub", [False, True])
@patch("megatron.bridge.training.setup.classify_gtp_remat_chains")
@patch("megatron.bridge.training.setup.configure_gtp_remat")
def test_distributed_model_build_obeys_gtp_lifecycle(mock_configure, mock_classify, fp32_accumulation, nccl_ub):
    from megatron.bridge.training.config import DistributedInitConfig
    from megatron.bridge.training.setup import _build_distributed_model

    events = []
    model = [MagicMock(), MagicMock()]
    model_config = SimpleNamespace()
    model_config.finalize = MagicMock(side_effect=lambda: events.append("finalize"))
    model_config.provide_distributed_model = MagicMock(side_effect=lambda **_kwargs: events.append("build") or model)
    mock_configure.side_effect = lambda _config, **_kwargs: events.append("configure")
    mock_classify.side_effect = lambda _model, _config: events.append("classify")
    cfg = SimpleNamespace(
        model=model_config,
        ddp=object(),
        optimizer=SimpleNamespace(overlap_param_gather_with_optimizer_step=False),
        dist=DistributedInitConfig(
            use_megatron_fsdp=False,
            use_torch_fsdp2=False,
            gtp_remat_reduce_scatter_with_fp32_accumulation=fp32_accumulation,
            gtp_remat_nccl_ub=nccl_ub,
        ),
        rng=SimpleNamespace(data_parallel_random_init=False),
    )

    pg_collection = MagicMock()
    result = _build_distributed_model(cfg, pg_collection)

    assert result is model
    mock_configure.assert_called_once_with(
        model_config,
        reduce_scatter_with_fp32_accumulation=fp32_accumulation,
        nccl_ub=nccl_ub,
        pg_collection=pg_collection,
    )
    assert events == ["finalize", "configure", "build", "classify"]
