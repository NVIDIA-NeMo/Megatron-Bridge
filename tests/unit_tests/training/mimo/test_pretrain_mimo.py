# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO pretrain entrypoint wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cfg():
    cfg = MagicMock()
    cfg.train = SimpleNamespace(
        rampup_batch_size=None,
        global_batch_size=1,
        micro_batch_size=1,
        decrease_batch_size_if_needed=False,
    )
    cfg.data_parallel_size = 1
    return cfg


def _make_setup_output(module_to_grid_map):
    return SimpleNamespace(
        model=MagicMock(),
        mimo_infra=SimpleNamespace(module_to_grid_map=module_to_grid_map),
        multimodule_communicator=MagicMock(),
        train_data_iterator=iter([]),
        valid_data_iterator=None,
        global_state=MagicMock(),
    )


@patch("megatron.bridge.training.pretrain_mimo.train_mimo")
@patch("megatron.bridge.training.pretrain_mimo.setup_mimo")
@patch("megatron.bridge.training.pretrain_mimo.unwrap_mimo_model")
@patch("megatron.bridge.training.pretrain_mimo.dist")
def test_pretrain_mimo_uses_constructor_wired_config(
    mock_dist, mock_unwrap_mimo_model, mock_setup_mimo, mock_train_mimo
):
    """Pretrain path should use constructor-wired config, not mutate it."""
    from megatron.bridge.training.pretrain_mimo import pretrain_mimo

    cfg = _make_cfg()
    opt_config = MagicMock()
    opt_config.finalize = MagicMock()
    schedulers = {}
    forward_step_func = MagicMock()

    mock_dist.get_rank.return_value = 0

    sentinel_grid_map = {"language": MagicMock()}
    setup_output = _make_setup_output(module_to_grid_map=sentinel_grid_map)
    mock_setup_mimo.return_value = setup_output

    original_grid_map = {"language": MagicMock()}
    unwrapped_model = MagicMock()
    unwrapped_model.mimo_config = SimpleNamespace(
        module_to_grid_map=original_grid_map,
    )
    mock_unwrap_mimo_model.return_value = unwrapped_model

    with (
        patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
        patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
        patch("megatron.core.models.mimo.optimizer.get_mimo_optimizer") as mock_get_optimizer,
    ):
        mock_get_optimizer.return_value = MagicMock()

        pretrain_mimo(
            cfg=cfg,
            mimo_provider=MagicMock(),
            forward_step_func=forward_step_func,
            build_data_iterators_fn=MagicMock(),
            opt_config=opt_config,
            schedulers=schedulers,
            global_state=MagicMock(),
        )

    # No post-construction mutation: keep original references/values.
    assert unwrapped_model.mimo_config.module_to_grid_map is original_grid_map
    mock_train_mimo.assert_called_once()


@patch("megatron.bridge.training.pretrain_mimo.setup_mimo")
@patch("megatron.bridge.training.pretrain_mimo.unwrap_mimo_model")
@patch("megatron.bridge.training.pretrain_mimo.dist")
def test_pretrain_mimo_asserts_when_constructor_fields_missing(mock_dist, mock_unwrap_mimo_model, mock_setup_mimo):
    """Guardrail should fail when constructor-time config wiring is missing."""
    from megatron.bridge.training.pretrain_mimo import pretrain_mimo

    cfg = _make_cfg()
    opt_config = MagicMock()
    opt_config.finalize = MagicMock()

    mock_dist.get_rank.return_value = 0

    # Infra indicates MIMO-parallel path is active.
    mock_setup_mimo.return_value = _make_setup_output(module_to_grid_map={"language": MagicMock()})

    # Missing constructor-wired fields should trigger assertion.
    unwrapped_model = MagicMock()
    unwrapped_model.mimo_config = SimpleNamespace(
        module_to_grid_map=None,
    )
    mock_unwrap_mimo_model.return_value = unwrapped_model

    with (
        patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
        patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
    ):
        with pytest.raises(AssertionError, match="module_to_grid_map must be set"):
            pretrain_mimo(
                cfg=cfg,
                mimo_provider=MagicMock(),
                forward_step_func=MagicMock(),
                build_data_iterators_fn=MagicMock(),
                opt_config=opt_config,
                schedulers={},
                global_state=MagicMock(),
            )
