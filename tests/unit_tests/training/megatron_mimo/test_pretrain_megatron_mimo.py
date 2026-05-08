# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO pretrain and setup wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cfg():
    cfg = MagicMock()
    cfg.train.rampup_batch_size = None
    cfg.train.global_batch_size = 1
    cfg.train.micro_batch_size = 1
    cfg.train.decrease_batch_size_if_needed = False
    cfg.data_parallel_size = 1
    cfg.checkpoint.load = None
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.non_persistent_ckpt_type = None
    cfg.checkpoint.save_rng = False
    return cfg


def _make_setup_output(module_to_grid_map):
    global_state = MagicMock()
    global_state.train_state.step = 0
    mock_checkpoint_manager = MagicMock()
    mock_checkpoint_manager.checkpointing_context = None
    language_pg = MagicMock()
    return SimpleNamespace(
        model=MagicMock(),
        megatron_mimo_infra=SimpleNamespace(
            module_to_grid_map=module_to_grid_map,
            pg_collections={"language": language_pg},
        ),
        multimodule_communicator=MagicMock(),
        multimodule_pg_collection=MagicMock(),
        module_to_grid_tuple=[(MagicMock(), MagicMock())],
        optimizer=MagicMock(),
        schedulers={},
        train_data_iterator=iter([]),
        valid_data_iterator=None,
        global_state=global_state,
        checkpoint_manager=mock_checkpoint_manager,
        active_module_name="language",
        local_pg_collection=language_pg,
    )


def _make_fake_grid(*, rank_offset: int, size: int, tp_rank: int, pp_rank: int) -> MagicMock:
    """Build a mock grid whose ``get_pg`` returns PGs with the given ranks."""
    tp_pg = MagicMock()
    tp_pg.rank.return_value = tp_rank
    pp_pg = MagicMock()
    pp_pg.rank.return_value = pp_rank
    grid = MagicMock()
    grid.rank_offset = rank_offset
    grid.size = size
    grid.get_pg.side_effect = lambda dims: {"tp": tp_pg, "pp": pp_pg}[dims[0]]
    return grid


def test_scheduler_helpers_skip_frozen_module_optimizer_stubs():
    """Frozen modules can create optimizer stubs that must not drive LR scheduling."""
    from megatron.bridge.training.setup_megatron_mimo import (
        _first_scheduler_with_param_groups as setup_first_scheduler,
    )
    from megatron.bridge.training.setup_megatron_mimo import (
        _optimizer_has_params,
    )
    from megatron.bridge.training.train_megatron_mimo import (
        _first_scheduler_with_param_groups as train_first_scheduler,
    )

    stub_optimizer = SimpleNamespace(param_groups=[])
    empty_group_optimizer = SimpleNamespace(param_groups=[{"params": [], "lr": 1e-3}])
    real_optimizer = SimpleNamespace(param_groups=[{"params": [object()], "lr": 1e-3}])
    stub_scheduler = SimpleNamespace(optimizer=stub_optimizer)
    empty_group_scheduler = SimpleNamespace(optimizer=empty_group_optimizer)
    real_scheduler = SimpleNamespace(optimizer=real_optimizer)
    schedulers = {"language": stub_scheduler, "audio": empty_group_scheduler, "images": real_scheduler}

    assert not _optimizer_has_params(stub_optimizer)
    assert not _optimizer_has_params(empty_group_optimizer)
    assert _optimizer_has_params(real_optimizer)
    assert setup_first_scheduler(schedulers) is real_scheduler
    assert train_first_scheduler(schedulers) is real_scheduler


def test_setup_skips_multimodule_pipeline_communicator_for_colocated_models():
    """Colocated schedules communicate inside MimoModel, so setup must not build MCore cross-module P2P."""
    from megatron.core.models.mimo.config.role import ModuleLayout

    from megatron.bridge.training.setup_megatron_mimo import _needs_multimodule_pipeline_communicator

    colocated = SimpleNamespace(role=SimpleNamespace(mode=ModuleLayout.COLOCATED))
    non_colocated = SimpleNamespace(role=SimpleNamespace(mode=ModuleLayout.NON_COLOCATED))

    assert not _needs_multimodule_pipeline_communicator(colocated)
    assert _needs_multimodule_pipeline_communicator(non_colocated)
    assert _needs_multimodule_pipeline_communicator(SimpleNamespace())


def test_learning_rate_for_logging_handles_ranks_without_local_scheduler(monkeypatch):
    """Frozen-only ranks should log the globally active learning rate."""
    from megatron.bridge.training import train_megatron_mimo

    monkeypatch.setattr(train_megatron_mimo.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_megatron_mimo.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_megatron_mimo.torch.cuda, "is_available", lambda: False)

    def fake_all_reduce(tensor, op):
        assert op == train_megatron_mimo.dist.ReduceOp.MAX
        tensor[0] = 1e-3

    monkeypatch.setattr(train_megatron_mimo.dist, "all_reduce", fake_all_reduce)

    assert train_megatron_mimo._learning_rate_for_logging({}) == pytest.approx(1e-3)


def test_seed_per_module_rng_tracker_non_colocated_single_module():
    """Non-colocated rank serving one module → seed called once with that
    module's tp_rank, snapshot dict has one entry."""
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import seed_per_module_rng_tracker

    current_rank = 4  # first rank of vision encoder
    grid = _make_fake_grid(rank_offset=4, size=4, tp_rank=0, pp_rank=0)

    megatron_mimo_infra = SimpleNamespace(
        module_to_grid_map={"vision": grid},
        cuda_rng_states_per_module={},
    )

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed") as mock_seed:
        with patch(
            "megatron.core.tensor_parallel.get_cuda_rng_tracker",
            return_value=MagicMock(get_states=MagicMock(return_value={"sentinel": "vision"})),
        ):
            import torch

            with patch.object(torch.cuda, "device_count", return_value=1):
                seed_per_module_rng_tracker(42, megatron_mimo_infra, current_rank=current_rank)

            # pp_rank=0, so seed stays 42. tp_rank=0 passed explicitly.
            mock_seed.assert_called_once_with(42, tp_rank=0, ep_rank=0, etp_rank=0)
            # Snapshot stashed on infra under the active module's name.
            assert "vision" in megatron_mimo_infra.cuda_rng_states_per_module


def test_seed_per_module_rng_tracker_offsets_by_pp_rank():
    """PP rank > 0 → CUDA tracker seed offset by 100 * pp_rank.

    CPU global seed (``_seed_python_numpy_torch``) stays at the base seed
    regardless of PP rank — see
    ``test_get_global_seed_independent_of_lm_pp_rank``.
    """
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import seed_per_module_rng_tracker

    current_rank = 2
    grid = _make_fake_grid(rank_offset=0, size=4, tp_rank=1, pp_rank=1)

    megatron_mimo_infra = SimpleNamespace(
        module_to_grid_map={"language": grid},
        cuda_rng_states_per_module={},
    )

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed") as mock_seed:
        with patch(
            "megatron.core.tensor_parallel.get_cuda_rng_tracker",
            return_value=MagicMock(get_states=MagicMock(return_value={})),
        ):
            import torch

            with patch.object(torch.cuda, "device_count", return_value=1):
                seed_per_module_rng_tracker(42, megatron_mimo_infra, current_rank=current_rank)

            # CUDA tracker seed = 42 + 100 * 1 = 142, tp_rank=1.
            mock_seed.assert_called_once_with(142, tp_rank=1, ep_rank=0, etp_rank=0)


def test_get_global_seed_independent_of_lm_pp_rank():
    """CPU global seed must not vary with LM PP rank.

    Regression test for the colocated language-PP=2 iter-1 mismatch: when
    ``_get_global_seed_and_module`` returned ``seed + 100 * lm_pp_rank``,
    ``torch.manual_seed`` differed across LM PP stages and the LLaVA vision
    projector (built with ``use_cpu_initialization=True``) randomly
    initialized to different values on vision-DP siblings at LM PP stage 0
    vs stage 1.
    """
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import _get_global_seed_and_module

    base_seed = 42
    for lm_pp_rank in (0, 1, 3):
        grid = _make_fake_grid(rank_offset=0, size=4, tp_rank=0, pp_rank=lm_pp_rank)
        global_seed, module_name = _get_global_seed_and_module(base_seed, {"language": grid})
        assert global_seed == base_seed, (
            f"CPU global seed must equal base seed regardless of lm_pp_rank; "
            f"got {global_seed} at lm_pp_rank={lm_pp_rank}."
        )
        assert module_name == "language"


def test_get_global_seed_falls_back_when_language_absent():
    """Encoder-only ranks (no language module on this rank) still get the
    base seed; the diagnostic anchor falls back to the first active module."""
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import _get_global_seed_and_module

    grid = _make_fake_grid(rank_offset=0, size=4, tp_rank=0, pp_rank=2)
    global_seed, module_name = _get_global_seed_and_module(42, {"images": grid})
    assert global_seed == 42
    assert module_name == "images"

    # No active modules at all → still returns the base seed.
    seed_for_unattached, module_name_for_unattached = _get_global_seed_and_module(42, {})
    assert seed_for_unattached == 42
    assert module_name_for_unattached is None


def test_seed_per_module_rng_tracker_colocated_calls_seed_per_module():
    """Colocated heterogeneous TP rank in BOTH modules → one seed call per
    module with that module's tp_rank, snapshot dict has both entries.

    This is the load-bearing test for asymmetric TP: the pre-task helper
    iterated module_to_grid_map and broke out on the first match (single
    seed call). The new helper must call seeding once per active module so
    each module's TP-region RNG is correct.
    """
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import seed_per_module_rng_tracker

    current_rank = 1
    # Asymmetric TP: vision tp_rank=0, language tp_rank=1.
    vision_grid = _make_fake_grid(rank_offset=0, size=4, tp_rank=0, pp_rank=0)
    language_grid = _make_fake_grid(rank_offset=0, size=4, tp_rank=1, pp_rank=0)

    megatron_mimo_infra = SimpleNamespace(
        module_to_grid_map={"vision": vision_grid, "language": language_grid},
        cuda_rng_states_per_module={},
    )
    # Per-module tracker snapshots — return distinguishable sentinels per call.
    states_per_call = [
        {"tracker_state": "vision_state"},
        {"tracker_state": "language_state"},
    ]
    fake_tracker = MagicMock()
    fake_tracker.get_states.side_effect = states_per_call

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed") as mock_seed:
        with patch(
            "megatron.core.tensor_parallel.get_cuda_rng_tracker",
            return_value=fake_tracker,
        ):
            import torch

            with patch.object(torch.cuda, "device_count", return_value=1):
                seed_per_module_rng_tracker(42, megatron_mimo_infra, current_rank=current_rank)

    # Two seed calls — one per active module — with each module's tp_rank.
    assert mock_seed.call_count == 2
    seed_calls = [tuple(call.args) + tuple(sorted(call.kwargs.items())) for call in mock_seed.call_args_list]
    assert (42, ("ep_rank", 0), ("etp_rank", 0), ("tp_rank", 0)) in seed_calls
    assert (42, ("ep_rank", 0), ("etp_rank", 0), ("tp_rank", 1)) in seed_calls

    # Snapshot dict has both modules' tracker states.
    assert set(megatron_mimo_infra.cuda_rng_states_per_module.keys()) == {"vision", "language"}
    assert megatron_mimo_infra.cuda_rng_states_per_module["vision"] == {"tracker_state": "vision_state"}
    assert megatron_mimo_infra.cuda_rng_states_per_module["language"] == {"tracker_state": "language_state"}


def test_seed_per_module_rng_tracker_global_seed_independent_of_pp_rank():
    """CPU global seed (Python random, numpy, torch.manual_seed) must be the
    base seed on every rank regardless of any active module's PP rank.

    Regression test for the colocated language-PP=2 iter-1 mismatch: the old
    behavior folded ``100 * pp_rank`` into the CPU seed, which made
    CPU-initialized parameters (e.g. the LLaVA vision projector built with
    ``use_cpu_initialization=True``) randomly initialize to different values
    on vision-DP siblings on different LM PP stages. Per-PP-stage
    divergence intentionally lives in the *CUDA RNG tracker*, not in the CPU
    seed.
    """
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import seed_per_module_rng_tracker

    # Encoder-only rank with the unique active module at pp_rank=2 — the old
    # buggy formula would have produced seed=242 here.
    current_rank = 0
    vision_grid = _make_fake_grid(rank_offset=0, size=4, tp_rank=0, pp_rank=2)
    language_grid = _make_fake_grid(rank_offset=4, size=4, tp_rank=0, pp_rank=0)

    megatron_mimo_infra = SimpleNamespace(
        module_to_grid_map={"vision": vision_grid, "language": language_grid},
        cuda_rng_states_per_module={},
    )
    import random as random_module

    import numpy as np_module

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed"):
        with patch(
            "megatron.core.tensor_parallel.get_cuda_rng_tracker",
            return_value=MagicMock(get_states=MagicMock(return_value={})),
        ):
            with patch.object(random_module, "seed") as mock_random_seed:
                with patch.object(np_module.random, "seed") as mock_np_seed:
                    with patch("torch.manual_seed") as mock_torch_seed:
                        import torch

                        with patch.object(torch.cuda, "device_count", return_value=1):
                            seed_per_module_rng_tracker(42, megatron_mimo_infra, current_rank=current_rank)

    mock_random_seed.assert_called_once_with(42)
    mock_np_seed.assert_called_once_with(42)
    mock_torch_seed.assert_called_once_with(42)


def test_seed_per_module_rng_tracker_global_seed_unchanged_with_language_active():
    """Colocated rank with language active at pp_rank=1 — CPU global seed is
    still the base seed; language's pp_rank only drives CUDA tracker seeding,
    not CPU init.
    """
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import seed_per_module_rng_tracker

    current_rank = 0
    vision_grid = _make_fake_grid(rank_offset=0, size=4, tp_rank=0, pp_rank=3)
    language_grid = _make_fake_grid(rank_offset=0, size=4, tp_rank=0, pp_rank=1)

    megatron_mimo_infra = SimpleNamespace(
        module_to_grid_map={"vision": vision_grid, "language": language_grid},
        cuda_rng_states_per_module={},
    )
    import random as random_module

    import numpy as np_module

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed"):
        with patch(
            "megatron.core.tensor_parallel.get_cuda_rng_tracker",
            return_value=MagicMock(get_states=MagicMock(return_value={})),
        ):
            with patch.object(random_module, "seed") as mock_random_seed:
                with patch.object(np_module.random, "seed") as mock_np_seed:
                    with patch("torch.manual_seed") as mock_torch_seed:
                        import torch

                        with patch.object(torch.cuda, "device_count", return_value=1):
                            seed_per_module_rng_tracker(42, megatron_mimo_infra, current_rank=current_rank)

    mock_random_seed.assert_called_once_with(42)
    mock_np_seed.assert_called_once_with(42)
    mock_torch_seed.assert_called_once_with(42)


def test_module_rng_scope_swaps_tracker_state_on_entry_and_saves_on_exit():
    """``module_rng_scope`` swaps the singleton tracker via ``set_states`` on
    entry and writes back via ``get_states`` on exit, so RNG advances inside
    the scope persist into the per-module snapshot dict."""
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
        MegatronMIMOInfra,
        module_rng_scope,
    )

    initial_state = {"tracker": "vision_initial"}
    advanced_state = {"tracker": "vision_advanced"}
    infra = MegatronMIMOInfra(
        module_to_grid_map={},
        topology={},
        pg_collections={},
        participating_modules=[],
        cuda_rng_states_per_module={"vision": initial_state, "language": {"tracker": "lang_state"}},
    )

    fake_tracker = MagicMock()
    # Each get_states() call returns the "current" state. We track what the
    # tracker has been set_states-d to so the test can verify the order.
    fake_tracker.get_states.return_value = advanced_state

    with patch("megatron.core.tensor_parallel.get_cuda_rng_tracker", return_value=fake_tracker):
        with module_rng_scope("vision", infra):
            # set_states called with vision's saved state on entry.
            fake_tracker.set_states.assert_called_once_with(initial_state)

        # get_states called on exit; result stashed under vision's slot.
        fake_tracker.get_states.assert_called_once()
        assert infra.cuda_rng_states_per_module["vision"] == advanced_state
        # Other modules' slots are untouched.
        assert infra.cuda_rng_states_per_module["language"] == {"tracker": "lang_state"}


def test_module_rng_scope_falls_through_when_module_not_in_snapshots():
    """Missing snapshot → no tracker swap (nullcontext semantics).

    Happens when a rank doesn't participate in the requested module (factories
    are sometimes bound for every module name) or before seeding has run.
    Important to fail open here so legacy paths and not-yet-seeded states
    don't crash with a KeyError.
    """
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
        MegatronMIMOInfra,
        module_rng_scope,
    )

    infra = MegatronMIMOInfra(
        module_to_grid_map={},
        topology={},
        pg_collections={},
        participating_modules=[],
        cuda_rng_states_per_module={},  # empty — no snapshots populated yet
    )

    fake_tracker = MagicMock()

    with patch("megatron.core.tensor_parallel.get_cuda_rng_tracker", return_value=fake_tracker):
        with module_rng_scope("vision", infra):
            pass

    # No tracker mutation when there's no snapshot to load.
    fake_tracker.set_states.assert_not_called()
    fake_tracker.get_states.assert_not_called()


def test_module_rng_scope_can_be_re_entered():
    """Same scope entered multiple times must keep working — the factory used
    by ``MimoModel._scope`` produces a fresh context manager per entry, but
    each entry needs to load this module's *current* snapshot, not a stale
    one. Specifically: state advanced inside an earlier scope must be the
    starting state of the next entry (advances persist via get_states on exit).
    """
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
        MegatronMIMOInfra,
        module_rng_scope,
    )

    infra = MegatronMIMOInfra(
        module_to_grid_map={},
        topology={},
        pg_collections={},
        participating_modules=[],
        cuda_rng_states_per_module={"vision": {"step": 0}},
    )

    fake_tracker = MagicMock()
    fake_tracker.get_states.side_effect = [{"step": 1}, {"step": 2}]

    with patch("megatron.core.tensor_parallel.get_cuda_rng_tracker", return_value=fake_tracker):
        with module_rng_scope("vision", infra):
            pass
        first_snapshot = dict(infra.cuda_rng_states_per_module["vision"])
        with module_rng_scope("vision", infra):
            pass
        second_snapshot = dict(infra.cuda_rng_states_per_module["vision"])

    # First exit wrote {step: 1}; second entry loaded that and saved {step: 2}.
    assert first_snapshot == {"step": 1}
    assert second_snapshot == {"step": 2}
    # set_states called twice — second time with the post-first-exit state.
    set_calls = [call.args[0] for call in fake_tracker.set_states.call_args_list]
    assert set_calls == [{"step": 0}, {"step": 1}]


def test_get_rng_state_namespaces_key_with_module_name():
    """get_rng_state should namespace ShardedObject key when module_name is set.

    Unit test: mocks ``torch.cuda.get_rng_state`` and the Megatron CUDA RNG
    tracker so the test runs without a GPU (``get_rng_state`` otherwise calls
    these unconditionally at ``checkpointing.py:425-426``).
    """
    from megatron.bridge.training.checkpointing import get_rng_state

    pg = MagicMock()
    pg.pp.rank.return_value = 0
    pg.pp.size.return_value = 1
    pg.tp.rank.return_value = 0
    pg.tp.size.return_value = 2
    pg.dp_cp.rank.return_value = 0
    pg.dp_cp.size.return_value = 1
    pg.ep = None  # no EP

    with (
        patch("torch.cuda.get_rng_state", return_value=b"dummy_cuda_rng_state"),
        patch("megatron.bridge.training.checkpointing.tensor_parallel.get_cuda_rng_tracker") as mock_tracker,
    ):
        mock_tracker.return_value.get_states.return_value = {}

        # Without module_name: key is "rng_state"
        result = get_rng_state(False, "torch_dist", pg_collection=pg)
        assert result.key == "rng_state"

        # With module_name: key is namespaced
        result = get_rng_state(False, "torch_dist", pg_collection=pg, module_name="language")
        assert result.key == "rng_state.language"

        result = get_rng_state(False, "torch_dist", pg_collection=pg, module_name="vision")
        assert result.key == "rng_state.vision"


@patch("megatron.bridge.training.pretrain_megatron_mimo._finish_train")
@patch("megatron.bridge.training.pretrain_megatron_mimo.train_megatron_mimo")
@patch("megatron.bridge.training.pretrain_megatron_mimo.setup_megatron_mimo")
@patch("megatron.bridge.training.pretrain_megatron_mimo.dist")
@patch("megatron.bridge.training.pretrain_megatron_mimo.megatron_mimo_runtime_config_update")
@patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None)
def test_pretrain_megatron_mimo_calls_setup_and_train(
    mock_runtime_update, mock_dist, mock_setup_megatron_mimo, mock_train_megatron_mimo, mock_finish
):
    """pretrain_megatron_mimo should call setup_megatron_mimo then train_megatron_mimo."""
    from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo

    cfg = _make_cfg()

    mock_dist.get_rank.return_value = 0
    mock_dist.is_initialized.return_value = True
    setup_output = _make_setup_output(module_to_grid_map={"language": MagicMock()})
    mock_setup_megatron_mimo.return_value = setup_output

    pretrain_megatron_mimo(
        cfg=cfg,
        forward_step_func=MagicMock(),
        build_data_iterators_fn=MagicMock(return_value=(iter([]), None)),
        global_state=MagicMock(),
    )

    mock_setup_megatron_mimo.assert_called_once()
    mock_train_megatron_mimo.assert_called_once()
    mock_finish.assert_called_once()


@patch("megatron.bridge.training.pretrain_megatron_mimo._finish_train")
@patch("megatron.bridge.training.pretrain_megatron_mimo.train_megatron_mimo")
@patch("megatron.bridge.training.pretrain_megatron_mimo.setup_megatron_mimo")
@patch("megatron.bridge.training.pretrain_megatron_mimo.dist")
@patch("megatron.bridge.training.pretrain_megatron_mimo.megatron_mimo_runtime_config_update")
@patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None)
def test_pretrain_megatron_mimo_threads_active_module_from_setup(
    mock_runtime_update, mock_dist, mock_setup_megatron_mimo, mock_train_megatron_mimo, mock_finish
):
    """pretrain_megatron_mimo must pass active_module_name and local_pg_collection
    from setup_output into train_megatron_mimo. Prevents reintroduction of the
    rank-dependent inline recompute that fails in colocated mode."""
    from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo

    cfg = _make_cfg()
    mock_dist.get_rank.return_value = 0
    mock_dist.is_initialized.return_value = True
    setup_output = _make_setup_output(module_to_grid_map={"language": MagicMock()})
    mock_setup_megatron_mimo.return_value = setup_output

    pretrain_megatron_mimo(
        cfg=cfg,
        forward_step_func=MagicMock(),
        build_data_iterators_fn=MagicMock(return_value=(iter([]), None)),
        global_state=MagicMock(),
    )

    _, kwargs = mock_train_megatron_mimo.call_args
    assert kwargs["active_module_name"] == setup_output.active_module_name
    assert kwargs["local_pg_collection"] is setup_output.local_pg_collection


def test_finish_train_calls_cleanup():
    """_finish_train should finalize async saves, shut down NVRx/FT, and flush loggers."""
    from megatron.bridge.training.train import _finish_train

    global_state = MagicMock()
    checkpoint_manager = MagicMock()

    with (
        patch("megatron.bridge.training.train.safe_shutdown_nvrx_straggler_manager") as m_nvrx,
        patch("megatron.bridge.training.train.fault_tolerance") as m_ft,
        patch("megatron.bridge.training.train.destroy_global_state") as m_destroy,
    ):
        _finish_train(global_state, checkpoint_manager)

    # Async saves finalized
    checkpoint_manager.finalize_async_saves.assert_called_once_with(
        state=global_state,
        blocking=True,
        terminate=True,
    )

    # NVRx shutdown
    m_nvrx.assert_called_once_with(global_state.nvrx_straggler_manager)

    # Fault tolerance lifecycle — mirror the exact contract at train.py:1445-1448.
    m_ft.on_checkpointing_start.assert_called_once_with(global_state)
    m_ft.on_checkpointing_end.assert_called_once_with(global_state=global_state, is_async_finalization=True)
    m_ft.shutdown.assert_called_once_with(global_state)

    # Logger flush (MagicMock is truthy)
    global_state.wandb_logger.finish.assert_called_once()
    global_state._comet_logger.end.assert_called_once()

    # GlobalState destroyed
    m_destroy.assert_called_once()


@patch("megatron.bridge.training.setup_megatron_mimo.unwrap_megatron_mimo_model")
@patch("megatron.bridge.training.setup_megatron_mimo.get_model_config")
@patch("megatron.bridge.training.setup_megatron_mimo.dist")
def test_setup_megatron_mimo_asserts_when_constructor_fields_missing(
    mock_dist, mock_get_model_config, mock_unwrap_megatron_mimo_model
):
    """setup_megatron_mimo guardrail should fail when module_to_grid_map is missing at construction."""
    from megatron.bridge.training.setup_megatron_mimo import setup_megatron_mimo

    cfg = _make_cfg()
    mock_dist.get_rank.return_value = 0
    mock_dist.get_world_size.return_value = 8

    # Model with missing module_to_grid_map
    unwrapped_model = MagicMock()
    unwrapped_model.mimo_config = SimpleNamespace(module_to_grid_map=None)
    mock_unwrap_megatron_mimo_model.return_value = unwrapped_model

    mock_model_config = MagicMock()
    mock_model_config.pipeline_dtype = None
    mock_model_config.bf16 = True
    mock_get_model_config.return_value = mock_model_config

    # Set cfg.model to a provider that returns infra with an active grid map
    mock_infra = MagicMock()
    mock_infra.module_to_grid_map = {"language": MagicMock()}
    mock_infra.topology = {"language": []}
    mock_infra.module_output_ndim = {"language": 3}
    cfg.model.build_infra.return_value = mock_infra
    cfg.model.provide_distributed_model.return_value = [MagicMock()]

    with (
        patch("megatron.bridge.training.setup_megatron_mimo.validate_no_stub_ranks"),
        patch("megatron.bridge.training.setup_megatron_mimo.build_pg_collection_for_schedule"),
        patch("megatron.bridge.training.setup_megatron_mimo.get_module_to_grid_tuple"),
        patch("megatron.bridge.training.setup_megatron_mimo.MultiModulePipelineCommunicator"),
        patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
        patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
    ):
        mock_state = MagicMock()
        mock_state.cfg = cfg
        with pytest.raises(AssertionError, match="module_to_grid_map must be set"):
            setup_megatron_mimo(state=mock_state)


# ---------------------------------------------------------------------------
# Tests: train_step_megatron_mimo schedule dispatch
# ---------------------------------------------------------------------------


def _make_mimo_model_stub(mode, lm_has_pp: bool):
    """Build a minimal MimoModel-like stub for dispatch tests."""
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

    stub = MagicMock()
    stub.role.mode = mode
    stub.lm_has_pp = lm_has_pp
    stub.role.has_language_module = True
    stub.role.is_last_stage.return_value = False
    # Make MIMO_LANGUAGE_MODULE_KEY resolvable for post-schedule loss code paths
    _ = MIMO_LANGUAGE_MODULE_KEY  # suppress unused warning
    return stub


def _make_minimal_train_step_kwargs():
    """Minimal kwargs for train_step_megatron_mimo; values only need to reach the dispatch."""
    language_pg = MagicMock()
    return dict(
        forward_step_func=MagicMock(),
        data_iterator=iter([]),
        model=MagicMock(),
        optimizer=MagicMock(),
        schedulers={},
        global_state=MagicMock(),
        multimodule_communicator=MagicMock(),
        multimodule_pg_collection=MagicMock(),
        infra=SimpleNamespace(
            module_to_grid_map={"language": MagicMock(), "images": MagicMock()},
            pg_collections={"language": language_pg, "images": MagicMock()},
        ),
        module_to_grid_tuple=[],
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
    )


def test_train_step_colocated_llm_pp_gt_one_dispatches_to_three_phase_adapter():
    """Colocated + LLM PP>1 must use the Bridge three-phase adapter."""
    from megatron.core.models.mimo.config.role import ModuleLayout

    from megatron.bridge.training.train_megatron_mimo import train_step_megatron_mimo

    stub = _make_mimo_model_stub(mode=ModuleLayout.COLOCATED, lm_has_pp=True)
    kwargs = _make_minimal_train_step_kwargs()
    language_pg = kwargs["infra"].pg_collections["language"]
    model_config = MagicMock()
    language_p2p_communicator = MagicMock()
    sentinel = RuntimeError("DISPATCH_REACHED_THREE_PHASE")
    with (
        patch("megatron.bridge.training.train_megatron_mimo.unwrap_megatron_mimo_model", return_value=stub),
        patch("megatron.bridge.training.train_megatron_mimo.get_model_config", return_value=model_config),
        patch(
            "megatron.bridge.training.train_megatron_mimo.P2PCommunicator",
            return_value=language_p2p_communicator,
        ) as mock_p2p,
        patch(
            "megatron.bridge.training.train_megatron_mimo.forward_backward_colocated_mimo_with_pp",
            side_effect=sentinel,
        ) as mock_adapter,
        patch("megatron.bridge.training.train_megatron_mimo.zero_grad_buffer_for_multimodule"),
    ):
        with pytest.raises(RuntimeError, match="DISPATCH_REACHED_THREE_PHASE"):
            train_step_megatron_mimo(**kwargs)
    mock_p2p.assert_called_once_with(pp_group=language_pg.pp, config=model_config)
    call_kwargs = mock_adapter.call_args.kwargs
    assert call_kwargs["model"] is kwargs["model"]
    assert call_kwargs["data_iterator"] is kwargs["data_iterator"]
    assert call_kwargs["infra"] is kwargs["infra"]
    assert call_kwargs["encoder_module_name"] == "images"
    assert call_kwargs["num_microbatches"] == kwargs["num_microbatches"]
    assert call_kwargs["seq_length"] == kwargs["seq_length"]
    assert call_kwargs["micro_batch_size"] == kwargs["micro_batch_size"]
    assert call_kwargs["forward_only"] is False
    assert call_kwargs["p2p_communicator"] is language_p2p_communicator


@pytest.mark.parametrize(
    "mode_name, lm_has_pp, expected_schedule",
    [
        # colocated + LLM PP=1: no-pipelining schedule with language pg_collection.
        # Cross-module flow happens inside MimoModel._forward_all_modules via
        # ColocatedBridgeCommunicator — the schedule does no P2P.
        ("COLOCATED", False, "forward_backward_no_pipelining"),
        # non-colocated + LLM PP=1: pipelining-without-interleaving with the
        # MultiModulePipelineCommunicator — encoder and language live on disjoint
        # ranks so cross-module P2P at the schedule level is required.
        ("NON_COLOCATED", False, "forward_backward_pipelining_without_interleaving"),
        # non-colocated + LLM PP>1: same as above (MultiModulePipelineCommunicator
        # handles both cross-module and intra-LLM-PP P2P).
        ("NON_COLOCATED", True, "forward_backward_pipelining_without_interleaving"),
    ],
    ids=["colocated_pp1", "non_colocated_pp1", "non_colocated_pp_gt1"],
)
def test_train_step_non_three_phase_cases_dispatch_to_correct_schedule(mode_name, lm_has_pp, expected_schedule):
    """Schedule dispatch picks no-pipelining for colocated-PP=1 and pipelining-without-interleaving otherwise."""
    from megatron.core.models.mimo.config.role import ModuleLayout

    from megatron.bridge.training.train_megatron_mimo import train_step_megatron_mimo

    stub = _make_mimo_model_stub(mode=getattr(ModuleLayout, mode_name), lm_has_pp=lm_has_pp)
    # Use a sentinel exception thrown from the mocked schedule as the "got past the
    # dispatch" marker. This avoids exercising downstream code paths (loss
    # aggregation, torch.distributed calls) that aren't the subject under test.
    sentinel = RuntimeError(f"DISPATCH_REACHED_{expected_schedule}")
    with (
        patch("megatron.bridge.training.train_megatron_mimo.unwrap_megatron_mimo_model", return_value=stub),
        patch(
            f"megatron.bridge.training.train_megatron_mimo.{expected_schedule}",
            side_effect=sentinel,
        ) as mock_fb,
        patch("megatron.bridge.training.train_megatron_mimo.zero_grad_buffer_for_multimodule"),
    ):
        with pytest.raises(RuntimeError, match=f"DISPATCH_REACHED_{expected_schedule}"):
            train_step_megatron_mimo(**_make_minimal_train_step_kwargs())
        assert mock_fb.called, f"{expected_schedule} not called for mode={mode_name}, lm_has_pp={lm_has_pp}"


# ---------------------------------------------------------------------------
# Tests: schedule contract — colocated must stay language-PG-scoped, never
# pulls in the MultiModulePipelineCommunicator (would self-send/deadlock).
# ---------------------------------------------------------------------------


def test_colocated_pp1_passes_language_pg_collection_to_no_pipelining():
    """Colocated PP=1 must pass the language module's pg_collection — not multimodule — to the schedule."""
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout

    from megatron.bridge.training.train_megatron_mimo import train_step_megatron_mimo

    stub = _make_mimo_model_stub(mode=ModuleLayout.COLOCATED, lm_has_pp=False)
    kwargs = _make_minimal_train_step_kwargs()
    language_pg = kwargs["infra"].pg_collections[MIMO_LANGUAGE_MODULE_KEY]
    multimodule_pg = kwargs["multimodule_pg_collection"]
    multimodule_comm = kwargs["multimodule_communicator"]

    sentinel = RuntimeError("STOP_AFTER_DISPATCH")
    with (
        patch("megatron.bridge.training.train_megatron_mimo.unwrap_megatron_mimo_model", return_value=stub),
        patch(
            "megatron.bridge.training.train_megatron_mimo.forward_backward_no_pipelining",
            side_effect=sentinel,
        ) as mock_fb,
        patch("megatron.bridge.training.train_megatron_mimo.zero_grad_buffer_for_multimodule"),
    ):
        with pytest.raises(RuntimeError, match="STOP_AFTER_DISPATCH"):
            train_step_megatron_mimo(**kwargs)

    call_kwargs = mock_fb.call_args.kwargs
    assert call_kwargs["pg_collection"] is language_pg, (
        "colocated PP=1 must dispatch with the language pg_collection from infra.pg_collections, "
        "not the MultiModuleProcessGroupCollection (would route cross-module P2P that doesn't exist)."
    )
    assert call_kwargs["pg_collection"] is not multimodule_pg
    # forward_backward_no_pipelining doesn't take a p2p_communicator. Verify the
    # MultiModulePipelineCommunicator never reaches the schedule — it would
    # self-send on degenerate single-rank groups in colocated.
    assert "p2p_communicator" not in call_kwargs
    for value in call_kwargs.values():
        assert value is not multimodule_comm


def test_non_colocated_passes_multimodule_communicator_and_pg_collection():
    """Non-colocated must pass the multimodule communicator + pg_collection — cross-module P2P is required."""
    from megatron.core.models.mimo.config.role import ModuleLayout

    from megatron.bridge.training.train_megatron_mimo import train_step_megatron_mimo

    stub = _make_mimo_model_stub(mode=ModuleLayout.NON_COLOCATED, lm_has_pp=False)
    kwargs = _make_minimal_train_step_kwargs()
    multimodule_pg = kwargs["multimodule_pg_collection"]
    multimodule_comm = kwargs["multimodule_communicator"]

    sentinel = RuntimeError("STOP_AFTER_DISPATCH")
    with (
        patch("megatron.bridge.training.train_megatron_mimo.unwrap_megatron_mimo_model", return_value=stub),
        patch(
            "megatron.bridge.training.train_megatron_mimo.forward_backward_pipelining_without_interleaving",
            side_effect=sentinel,
        ) as mock_fb,
        patch("megatron.bridge.training.train_megatron_mimo.zero_grad_buffer_for_multimodule"),
    ):
        with pytest.raises(RuntimeError, match="STOP_AFTER_DISPATCH"):
            train_step_megatron_mimo(**kwargs)

    call_kwargs = mock_fb.call_args.kwargs
    assert call_kwargs["pg_collection"] is multimodule_pg
    assert call_kwargs["p2p_communicator"] is multimodule_comm
