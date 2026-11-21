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
    return SimpleNamespace(dsa_indexer_loss_coeff=0.5)


@pytest.fixture
def core_logging(monkeypatch):
    helper = SimpleNamespace(
        tracker={},
        clean_loss_in_tracker=Mock(),
    )
    module = SimpleNamespace(
        DSAIndexerLossLoggingHelper=helper,
        initialize_dsa_metric_tracker=Mock(),
    )
    monkeypatch.setattr(dsa_metrics, "import_module", Mock(return_value=module))
    return module


@pytest.mark.parametrize("coefficient", [None, 0.0, -1.0])
def test_disabled_metrics_do_not_import_optional_mcore_api(monkeypatch, model_config, coefficient):
    model_config.dsa_indexer_loss_coeff = coefficient
    load = Mock(side_effect=AssertionError("disabled metrics must not import DSA"))
    monkeypatch.setattr(dsa_metrics, "import_module", load)

    dsa_metrics.reset_dsa_metrics(model_config)

    load.assert_not_called()


def test_private_pin_without_fixed_storage_api_is_unchanged(monkeypatch, model_config):
    monkeypatch.setattr(
        dsa_metrics,
        "import_module",
        Mock(side_effect=ModuleNotFoundError(name=dsa_metrics._DSA_LOGGING_MODULE)),
    )

    dsa_metrics.reset_dsa_metrics(model_config)


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
        dsa_metrics.reset_dsa_metrics(model_config)


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
    core_logging.initialize_dsa_metric_tracker.assert_not_called()
