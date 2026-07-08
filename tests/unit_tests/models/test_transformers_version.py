import importlib.machinery
from unittest.mock import Mock

import pytest
from packaging.version import Version

from megatron.bridge.models.conversion import model_bridge, transformers_version
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, get_model_bridge
from megatron.bridge.models.conversion.transformers_version import (
    TransformersVersionError,
    get_transformers_version,
    is_transformers_min_version,
    require_transformers_version,
)


pytestmark = pytest.mark.unit


class _TargetModel:
    pass


def test_get_transformers_version_parses_pep440_local_version(monkeypatch):
    monkeypatch.setattr(transformers_version.importlib.metadata, "version", lambda _: "5.3.0rc1+bridge.1")

    assert get_transformers_version() == Version("5.3.0rc1+bridge.1")


@pytest.mark.parametrize(
    ("installed", "required", "expected"),
    [
        ("5.3.0", "5.3.0", True),
        ("5.3.0+vendor.1", "5.3.0", True),
        ("5.3.0rc1", "5.3.0", False),
        ("5.4.0.dev1", "5.3.0", True),
    ],
)
def test_is_transformers_min_version_uses_pep440(monkeypatch, installed, required, expected):
    monkeypatch.setattr(transformers_version, "get_transformers_version", lambda: Version(installed))

    assert is_transformers_min_version(required) is expected


def test_is_transformers_min_version_rejects_invalid_requirement(monkeypatch):
    monkeypatch.setattr(transformers_version, "get_transformers_version", lambda: Version("5.3.0"))

    with pytest.raises(ValueError, match="Invalid minimum Transformers version"):
        is_transformers_min_version("not-a-version")


def test_require_transformers_version_accepts_dotted_symbol(monkeypatch):
    monkeypatch.setattr(transformers_version, "get_transformers_version", lambda: Version("5.3.0"))

    require_transformers_version(
        "test model",
        "5.3.0",
        symbols=("transformers.configuration_utils.PretrainedConfig",),
    )


def test_require_transformers_version_reports_missing_symbol(monkeypatch):
    monkeypatch.setattr(transformers_version, "get_transformers_version", lambda: Version("5.3.0"))

    with pytest.raises(TransformersVersionError) as error_info:
        require_transformers_version(
            "FutureModel",
            "5.2.0",
            symbols=("transformers.FutureModelForCausalLM",),
            action="construct the provider",
        )

    error = error_info.value
    assert error.model_name == "FutureModel"
    assert error.installed_version == Version("5.3.0")
    assert error.required_version == Version("5.2.0")
    assert error.missing_symbols == ("transformers.FutureModelForCausalLM",)
    assert "construct the provider" in str(error)
    assert "Install or upgrade" in str(error)


def test_symbol_check_does_not_hide_unrelated_import_failure(monkeypatch):
    monkeypatch.setattr(transformers_version, "get_transformers_version", lambda: Version("5.3.0"))

    def fake_find_spec(name):
        if name == "transformers.models.future.configuration_future":
            return importlib.machinery.ModuleSpec(name, loader=None)
        return None

    def fake_import_module(name):
        assert name == "transformers.models.future.configuration_future"
        raise ModuleNotFoundError("No module named 'optional_dependency'", name="optional_dependency")

    monkeypatch.setattr(transformers_version.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(transformers_version.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="optional_dependency"):
        require_transformers_version(
            "FutureModel",
            "5.3.0",
            symbols=("transformers.models.future.configuration_future.FutureConfig",),
        )


def test_symbol_probe_does_not_hide_unrelated_parent_import_failure(monkeypatch):
    monkeypatch.setattr(transformers_version, "get_transformers_version", lambda: Version("5.3.0"))

    def fake_find_spec(name):
        raise ModuleNotFoundError("No module named 'optional_dependency'", name="optional_dependency")

    monkeypatch.setattr(transformers_version.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(ModuleNotFoundError, match="optional_dependency"):
        require_transformers_version(
            "FutureModel",
            "5.3.0",
            symbols=("transformers.models.future.configuration_future.FutureConfig",),
        )


def test_register_bridge_persists_metadata_for_string_source(monkeypatch):
    require_mock = Mock()
    monkeypatch.setattr(model_bridge, "require_transformers_version", require_mock)

    @MegatronModelBridge.register_bridge(
        source="UnitTestFutureForCausalLM",
        target=_TargetModel,
        model_type="unit_test_future",
        min_transformers_version="99.0.0",
        required_transformers_symbols=("transformers.UnitTestFutureForCausalLM",),
    )
    class _StringSourceBridge(MegatronModelBridge):
        def mapping_registry(self):
            raise NotImplementedError

    assert _StringSourceBridge.SOURCE_NAME == "UnitTestFutureForCausalLM"
    assert _StringSourceBridge.MODEL_TYPE == "unit_test_future"
    assert _StringSourceBridge.MIN_TRANSFORMERS_VERSION == "99.0.0"
    assert _StringSourceBridge.REQUIRED_TRANSFORMERS_SYMBOLS == ("transformers.UnitTestFutureForCausalLM",)
    require_mock.assert_not_called()


def test_register_bridge_persists_metadata_for_class_source():
    class _ClassSource:
        pass

    @MegatronModelBridge.register_bridge(source=_ClassSource, target=_TargetModel)
    class _ClassSourceBridge(MegatronModelBridge):
        def mapping_registry(self):
            raise NotImplementedError

    assert _ClassSourceBridge.SOURCE_NAME == "_ClassSource"
    assert _ClassSourceBridge.MIN_TRANSFORMERS_VERSION is None
    assert _ClassSourceBridge.REQUIRED_TRANSFORMERS_SYMBOLS == ()


def test_guard_is_lazy_until_bridge_selection(monkeypatch):
    compatibility_error = TransformersVersionError(
        "UnitTestLazyForCausalLM",
        Version("5.3.0"),
        Version("99.0.0"),
    )
    require_mock = Mock(side_effect=compatibility_error)
    monkeypatch.setattr(model_bridge, "require_transformers_version", require_mock)

    @MegatronModelBridge.register_bridge(
        source="UnitTestLazyForCausalLM",
        target=_TargetModel,
        model_type="unit_test_lazy",
        min_transformers_version="99.0.0",
    )
    class _LazyBridge(MegatronModelBridge):
        initialized = False

        def __init__(self):
            type(self).initialized = True

        def mapping_registry(self):
            raise NotImplementedError

    require_mock.assert_not_called()
    assert _LazyBridge.initialized is False

    with pytest.raises(TransformersVersionError):
        get_model_bridge("UnitTestLazyForCausalLM")

    assert _LazyBridge.initialized is False
    require_mock.assert_called_once_with(
        "UnitTestLazyForCausalLM",
        "99.0.0",
        symbols=(),
        action="use this model bridge",
    )


def test_required_symbols_need_minimum_version():
    with pytest.raises(ValueError, match="requires min_transformers_version"):

        @MegatronModelBridge.register_bridge(
            source="UnitTestInvalidPolicyForCausalLM",
            target=_TargetModel,
            required_transformers_symbols=("transformers.UnitTestInvalidPolicyForCausalLM",),
        )
        class _InvalidPolicyBridge(MegatronModelBridge):
            def mapping_registry(self):
                raise NotImplementedError
