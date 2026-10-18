# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression tests for optimizer selection, DDP layout and checkpoint state."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    DistributedDataParallelConfig,
    OptimizerConfig,
    _validate_and_sync_distributed_optimizer_settings,
)
from megatron.bridge.training.setup import _build_distributed_model
from megatron.bridge.training.utils.optimizer_utils import (
    normalize_optimizer_config,
    uses_distributed_optimizer,
    uses_layer_wise_optimizer,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["muon", "dist_muon", "adaptive_muon", "soap", "lion"])
@pytest.mark.parametrize("source", ["optimizer", "ddp", "layerwise"])
def test_distributed_emerging_optimizer_normalization(name, source):
    cfg = SimpleNamespace(
        optimizer=OptimizerConfig(
            optimizer=name,
            use_distributed_optimizer=source == "optimizer",
            use_layer_wise_distributed_optimizer=source == "layerwise",
        ),
        ddp=DistributedDataParallelConfig(use_distributed_optimizer=source == "ddp"),
    )
    for _ in range(2):
        _validate_and_sync_distributed_optimizer_settings(cfg)
        cfg.optimizer.finalize()
        assert cfg.optimizer.optimizer == name.removeprefix("dist_")
        assert cfg.optimizer.use_layer_wise_distributed_optimizer
        assert not cfg.optimizer.use_distributed_optimizer
        assert not cfg.ddp.use_distributed_optimizer
        assert uses_distributed_optimizer(cfg.optimizer)


@pytest.mark.parametrize("name", ["adam", "sgd", "muon", "lion"])
def test_replicated_optimizer_stays_replicated(name):
    cfg = OptimizerConfig(optimizer=name, use_distributed_optimizer=False)
    normalize_optimizer_config(cfg)
    assert not uses_distributed_optimizer(cfg)


@pytest.mark.parametrize("name", ["adam", "sgd"])
def test_adam_and_sgd_keep_ordinary_distribution(name):
    cfg = SimpleNamespace(
        optimizer=OptimizerConfig(optimizer=name),
        ddp=DistributedDataParallelConfig(use_distributed_optimizer=True),
    )
    _validate_and_sync_distributed_optimizer_settings(cfg)
    assert cfg.optimizer.use_distributed_optimizer
    assert cfg.ddp.use_distributed_optimizer
    assert not uses_layer_wise_optimizer(cfg.optimizer)


@pytest.mark.parametrize("name", ["adam", "sgd", "dist_adam"])
def test_reject_layerwise_adam_or_sgd(name):
    cfg = OptimizerConfig(optimizer=name, use_layer_wise_distributed_optimizer=True)
    with pytest.raises(ValueError, match="Layer-wise distribution requires"):
        cfg.finalize()


@pytest.mark.parametrize("name", ["muonn", "dist_muonn"])
def test_reject_unknown_optimizer_before_distribution_is_changed(name):
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        cfg = OptimizerConfig(optimizer=name, use_distributed_optimizer=True)
        cfg.finalize()


def test_setup_optimizer_normalizes_dist_muon_without_container_validation():
    from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig

    from megatron.bridge.training.config import SchedulerConfig
    from megatron.bridge.training.optim import setup_optimizer

    optimizer_config = MCoreOptimizerConfig(optimizer="dist_muon")
    override_provider = Mock()
    override_provider.build_config_overrides.return_value = None
    model = Mock()

    with (
        patch("megatron.bridge.training.optim.get_model_config", return_value=SimpleNamespace(use_mup=False)),
        patch("megatron.bridge.training.optim.get_megatron_optimizer", return_value=Mock()) as factory,
        patch("megatron.bridge.training.optim.OptimizerParamScheduler"),
    ):
        setup_optimizer(
            optimizer_config,
            SchedulerConfig(),
            model,
            optimizer_config_override_provider=override_provider,
        )

    assert optimizer_config.optimizer == "muon"
    assert optimizer_config.use_layer_wise_distributed_optimizer
    assert not optimizer_config.use_distributed_optimizer
    assert factory.call_args.kwargs["config"] is optimizer_config


def test_reject_layerwise_optimizer_step_overlap():
    cfg = OptimizerConfig(optimizer="dist_muon", overlap_param_gather_with_optimizer_step=True)
    with pytest.raises(ValueError, match="overlap_param_gather_with_optimizer_step"):
        cfg.finalize()


@pytest.mark.parametrize("name,layerwise", [("dist_muon", False), ("muon", True)])
@pytest.mark.parametrize("overlap", [False, True])
def test_layerwise_overlap_configuration(name, layerwise, overlap):
    provider = GPTModelProvider(num_layers=2, hidden_size=128, num_attention_heads=4)
    optimizer = OptimizerConfig(optimizer=name, use_layer_wise_distributed_optimizer=layerwise)
    ddp = DistributedDataParallelConfig(use_distributed_optimizer=False, overlap_param_gather=not overlap)
    comm = CommOverlapConfig(
        tp_comm_overlap=False,
        data_parallel_size=2,
        overlap_param_gather=overlap,
        overlap_grad_reduce=overlap,
    )
    comm.finalize()
    comm.setup(provider, optimizer, ddp)
    assert ddp.overlap_param_gather is overlap
    assert optimizer.overlap_param_gather is overlap
    assert ddp.overlap_grad_reduce is overlap


@pytest.mark.parametrize("builder", [False, True])
@pytest.mark.parametrize("name,layerwise", [("adam", False), ("muon", True), ("dist_muon", False)])
@pytest.mark.parametrize("param_layout", [False, True])
def test_model_setup_forwards_layerwise_layout(builder, name, layerwise, param_layout):
    from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
    from megatron.bridge.models.transformer_config import TransformerConfig
    from megatron.bridge.training.config import DistributedInitConfig

    model = (
        GPTModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4), vocab_size=1024
        )
        if builder
        else Mock()
    )
    model.freeze_base_model_for_mtp = False
    cfg = SimpleNamespace(
        model=model,
        ddp=DistributedDataParallelConfig(),
        optimizer=OptimizerConfig(
            optimizer=name,
            use_layer_wise_distributed_optimizer=layerwise,
            use_layer_wise_param_layout=param_layout,
        ),
        dist=DistributedInitConfig(),
        rng=SimpleNamespace(data_parallel_random_init=False),
    )
    with (
        patch("megatron.bridge.training.setup.configure_gtp_remat"),
        patch("megatron.bridge.training.setup.classify_gtp_remat_chains"),
    ):
        if builder:
            with patch.object(type(model), "get_builder_cls") as factory:
                _build_distributed_model(cfg, Mock())
                kwargs = factory.return_value.return_value.build_distributed_models.call_args.kwargs
        else:
            _build_distributed_model(cfg, Mock())
            kwargs = model.provide_distributed_model.call_args.kwargs
    assert kwargs["use_layer_wise_distributed_optimizer"] is (name != "adam")
    assert kwargs["use_layer_wise_param_layout"] is param_layout


def test_provider_forwards_layerwise_layout_to_get_model():
    provider = GPTModelProvider(num_layers=2, hidden_size=128, num_attention_heads=4)
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("megatron.bridge.models.model_provider.get_model", return_value=[]) as build,
    ):
        provider.provide_distributed_model(
            ddp_config=DistributedDataParallelConfig(),
            pg_collection=Mock(),
            use_layer_wise_distributed_optimizer=True,
        )
    assert build.call_args.kwargs["use_layer_wise_distributed_optimizer"] is True
    assert build.call_args.kwargs["use_layer_wise_param_layout"] is False


@pytest.mark.parametrize("name", ["muon", "dist_muon", "adaptive_muon", "lion", "soap"])
def test_layerwise_layout_defaults_to_compact_after_normalization(name: str) -> None:
    cfg = SimpleNamespace(
        optimizer=OptimizerConfig(optimizer=name, use_distributed_optimizer=True),
        ddp=DistributedDataParallelConfig(use_distributed_optimizer=True),
    )
    _validate_and_sync_distributed_optimizer_settings(cfg)
    cfg.optimizer.finalize()
    assert cfg.optimizer.use_layer_wise_distributed_optimizer
    assert not cfg.optimizer.use_layer_wise_param_layout
    assert not cfg.ddp.use_distributed_optimizer


@pytest.mark.parametrize(
    "name,layerwise,ordinary,expected",
    [
        ("adam", False, True, True),
        ("adam", False, False, False),
        ("dist_muon", False, False, True),
        ("muon", True, False, True),
        ("muon", False, False, False),
    ],
)
def test_checkpoint_metadata_tracks_both_distributed_implementations(name, layerwise, ordinary, expected):
    from megatron.bridge.training.checkpointing import _build_sharded_state_dict_metadata
    from megatron.bridge.training.config import CheckpointConfig

    optimizer = OptimizerConfig(
        optimizer=name, use_layer_wise_distributed_optimizer=layerwise, use_distributed_optimizer=ordinary
    )
    assert uses_distributed_optimizer(optimizer) is expected
    metadata = _build_sharded_state_dict_metadata(uses_distributed_optimizer(optimizer), CheckpointConfig())
    assert ("distrib_optim_sharding_type" in metadata) is expected


@pytest.mark.parametrize("name", ["muon", "dist_muon"])
def test_reject_multiple_layerwise_optimizer_instances(name):
    cfg = SimpleNamespace(
        optimizer=OptimizerConfig(optimizer=name, use_layer_wise_distributed_optimizer=True),
        ddp=DistributedDataParallelConfig(num_distributed_optimizer_instances=2),
    )
    with pytest.raises(ValueError, match="num_distributed_optimizer_instances=1"):
        _validate_and_sync_distributed_optimizer_settings(cfg)


@pytest.mark.parametrize("name", ["muon", "dist_muon"])
@pytest.mark.parametrize("backend", ["torch", "megatron_dist", "megatron_ddp"])
def test_reject_emerging_optimizer_fsdp_before_model_construction(name, backend):
    from megatron.bridge.recipes.gpt.vanilla_gpt import vanilla_gpt_pretrain_config

    cfg = vanilla_gpt_pretrain_config()
    cfg.optimizer.optimizer = name
    cfg.dist.use_torch_fsdp2 = backend == "torch"
    cfg.dist.use_megatron_fsdp = backend == "megatron_dist"
    cfg.ddp.use_megatron_fsdp = backend == "megatron_ddp"
    with pytest.raises(ValueError, match="do not support Megatron-FSDP or Torch-FSDP2"):
        cfg.validate()
