import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


# The adapter imports MCore lazily; test its CPU-safe contracts without loading
# Bridge's model registry or the optional GPU kernel packages.
_SOURCE = Path(__file__).parents[3] / "src/megatron/bridge/training/dsa_metrics.py"
_SPEC = importlib.util.spec_from_file_location("bridge_dsa_metrics_under_test", _SOURCE)
dsa_metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dsa_metrics)


pytestmark = pytest.mark.unit


@pytest.fixture
def model_config():
    return SimpleNamespace(
        dsa_indexer_loss_coeff=0.5,
        num_layers=4,
        mtp_num_layers=1,
        mtp_use_repeated_layer=False,
        hybrid_layer_pattern="CECE/WE",
        is_hybrid_model=False,
        csa_compress_ratios=[4, 0, 128, 0, 1, 0],
        context_parallel_size=16,
        dynamic_context_parallel=False,
        cuda_graph_impl="local",
    )


@pytest.fixture
def core_logging(monkeypatch):
    helper = SimpleNamespace(
        tracker={},
        clean_loss_in_tracker=Mock(),
        track_indexer_metrics=Mock(),
    )
    module = SimpleNamespace(
        DSAIndexerLossLoggingHelper=helper,
        initialize_dsa_metric_tracker=Mock(),
        resolve_dsa_metric_pg_collection=Mock(side_effect=lambda groups: (True, groups)),
    )
    monkeypatch.setattr(dsa_metrics, "import_module", Mock(return_value=module))
    return module


def _track(config, groups):
    losses = {}
    dsa_metrics.track_dsa_metrics(
        config,
        iteration=7,
        num_microbatches=64,
        writer=None,
        wandb_writer=None,
        total_loss_dict=losses,
        pg_collection=groups,
    )
    return losses


@pytest.mark.parametrize("coefficient", [None, 0.0, -1.0])
def test_disabled_metrics_do_not_import_optional_mcore_api(monkeypatch, model_config, coefficient):
    model_config.dsa_indexer_loss_coeff = coefficient
    load = Mock(side_effect=AssertionError("disabled metrics must not import DSA"))
    monkeypatch.setattr(dsa_metrics, "import_module", load)

    dsa_metrics.initialize_dsa_metrics([], model_config, None)
    dsa_metrics.reset_dsa_metrics(model_config)
    _track(model_config, None)

    load.assert_not_called()


def test_private_pin_without_fixed_storage_api_is_unchanged(monkeypatch, model_config):
    monkeypatch.setattr(
        dsa_metrics,
        "import_module",
        Mock(side_effect=ModuleNotFoundError(name=dsa_metrics._DSA_LOGGING_MODULE)),
    )

    dsa_metrics.initialize_dsa_metrics([], model_config, None)
    dsa_metrics.reset_dsa_metrics(model_config)
    assert _track(model_config, None) == {}


def test_module_without_initializer_is_unchanged(monkeypatch, model_config):
    monkeypatch.setattr(dsa_metrics, "import_module", Mock(return_value=SimpleNamespace()))
    assert dsa_metrics._get_dsa_logging(model_config) is None


def test_optional_import_does_not_hide_missing_mcore_dependencies(monkeypatch, model_config):
    monkeypatch.setattr(
        dsa_metrics,
        "import_module",
        Mock(side_effect=ModuleNotFoundError("missing dependency", name="transformer_engine")),
    )
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        dsa_metrics.initialize_dsa_metrics([], model_config, None)


def test_new_model_lifecycle_discards_old_storage_before_initializing(core_logging, model_config):
    model = [object(), object()]
    groups = object()
    tracker = core_logging.DSAIndexerLossLoggingHelper.tracker
    tracker.update(values=object(), agreed_size=123, reduce_group=object())
    snapshots = []
    core_logging.initialize_dsa_metric_tracker.side_effect = lambda *args: snapshots.append(dict(tracker))

    dsa_metrics.initialize_dsa_metrics(model, model_config, groups)

    assert snapshots == [{}]
    core_logging.initialize_dsa_metric_tracker.assert_called_once_with(model, groups)


def test_reset_preserves_captured_storage_and_reduction_groups(core_logging, model_config):
    storage = object()
    group = object()
    tracker = core_logging.DSAIndexerLossLoggingHelper.tracker
    tracker.update(values=storage, agreed_size=5, reduce_group=group)

    dsa_metrics.reset_dsa_metrics(model_config)

    core_logging.DSAIndexerLossLoggingHelper.clean_loss_in_tracker.assert_called_once_with(preserve_groups=True)
    assert tracker["values"] is storage
    assert tracker["reduce_group"] is group
    assert tracker["agreed_size"] == 5


def test_non_language_model_rank_skips_initialization_and_collectives(core_logging, model_config):
    core_logging.resolve_dsa_metric_pg_collection.side_effect = None
    core_logging.resolve_dsa_metric_pg_collection.return_value = (False, None)
    dsa_metrics.initialize_dsa_metrics([], model_config, object())
    _track(model_config, object())
    core_logging.initialize_dsa_metric_tracker.assert_not_called()
    core_logging.DSAIndexerLossLoggingHelper.track_indexer_metrics.assert_not_called()


@pytest.mark.parametrize("dynamic_cp", [False, True])
@pytest.mark.parametrize("graph_impl", ["none", "local"])
def test_all_ranks_reduce_with_lm_groups_and_nominal_microbatch_count(
    core_logging, model_config, dynamic_cp, graph_impl
):
    model_config.dynamic_context_parallel = dynamic_cp
    model_config.cuda_graph_impl = graph_impl
    lm_groups = SimpleNamespace(pp=object(), dp=object(), dp_cp=object())
    module_groups = object()
    core_logging.resolve_dsa_metric_pg_collection.side_effect = None
    core_logging.resolve_dsa_metric_pg_collection.return_value = (True, lm_groups)

    losses = _track(model_config, module_groups)

    core_logging.resolve_dsa_metric_pg_collection.assert_called_once_with(module_groups)
    core_logging.DSAIndexerLossLoggingHelper.track_indexer_metrics.assert_called_once_with(
        loss_scale=1 / 64,
        iteration=7,
        writer=None,
        wandb_writer=None,
        total_loss_dict=losses,
        num_layers=5,
        num_indexer_layers=1,
        preserve_groups=graph_impl != "none",
        dynamic_cp_parent_group=lm_groups.dp_cp if dynamic_cp else None,
        configured_cp_size=16,
        pp_group=lm_groups.pp,
        dp_group=lm_groups.dp,
    )


def test_legacy_static_groups_are_passed_as_none(core_logging, model_config):
    _track(model_config, None)
    kwargs = core_logging.DSAIndexerLossLoggingHelper.track_indexer_metrics.call_args.kwargs
    assert kwargs["pp_group"] is None
    assert kwargs["dp_group"] is None
    assert kwargs["dynamic_cp_parent_group"] is None


@pytest.mark.parametrize("repeated, expected", [(False, 3), (True, 4)])
def test_csa_normalizer_counts_repeated_mtp_executions(model_config, repeated, expected):
    model_config.hybrid_layer_pattern = None
    model_config.num_layers = 2
    model_config.mtp_num_layers = 3
    model_config.mtp_use_repeated_layer = repeated
    model_config.csa_compress_ratios = [4, 128, 4, 128, 4]
    assert dsa_metrics._num_indexer_executions(model_config) == expected


def test_hybrid_csa_counts_physical_pattern_including_mtp_even_if_flag_is_false(model_config):
    # Hybrid MTP contains multiple symbols, so num_layers + depth is not a
    # sufficient slice of the physical CSA ratio list. Only C (ratio 4) indexes.
    model_config.csa_compress_ratios = [4, 0, 128, 0, 1, 4]
    assert model_config.is_hybrid_model is False
    assert dsa_metrics._num_indexer_executions(model_config) == 2


def test_hybrid_dsa_normalizer_honors_pattern_and_shared_topk(monkeypatch, model_config):
    import sys

    dsa_module = SimpleNamespace(
        is_dsa_skip_topk_layer=lambda number, offset, frequency: max(number - max(offset, 1), 0) % frequency != 0
    )
    pattern_module = SimpleNamespace(
        Symbols=SimpleNamespace(PIPE="|", DS_ATTENTION="D"),
        parse_hybrid_pattern=Mock(
            return_value=SimpleNamespace(main_pattern="D|ED", mtp_pattern="DD", mtp_num_depths=3)
        ),
    )
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.experimental_attention_variant.dsa", dsa_module)
    monkeypatch.setitem(sys.modules, "megatron.core.models.hybrid.hybrid_layer_allocation", pattern_module)
    model_config.num_layers = 3
    model_config.mtp_num_layers = 3
    model_config.csa_compress_ratios = None
    model_config.hybrid_layer_pattern = "D|ED/DD/DD/DD"
    model_config.dsa_indexer_skip_topk_offset = 1
    model_config.dsa_indexer_topk_freq = 2

    assert dsa_metrics._num_indexer_executions(model_config) == 5
    pattern_module.parse_hybrid_pattern.assert_called_once_with(model_config.hybrid_layer_pattern)
