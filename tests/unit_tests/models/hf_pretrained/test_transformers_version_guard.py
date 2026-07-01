from types import SimpleNamespace
from unittest.mock import patch

import pytest
from packaging.version import Version

from megatron.bridge.models.conversion import transformers_version
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.transformers_version import TransformersVersionError
from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry


pytestmark = pytest.mark.unit


class _ConfigGuardTarget:
    pass


@MegatronModelBridge.register_bridge(
    source="UnitTestConfigGuardForCausalLM",
    target=_ConfigGuardTarget,
    model_type="unit_test_config_guard",
    min_transformers_version="99.0.0",
)
class _ConfigGuardBridge(MegatronModelBridge):
    def mapping_registry(self):
        raise NotImplementedError


def test_auto_bridge_validates_guard_before_provider_or_model_loading(monkeypatch):
    monkeypatch.setattr(transformers_version, "get_transformers_version", lambda: Version("5.3.0"))
    config = SimpleNamespace(
        architectures=["UnitTestConfigGuardForCausalLM"],
        model_type="unit_test_config_guard",
    )

    with pytest.raises(TransformersVersionError, match="UnitTestConfigGuardForCausalLM"):
        AutoBridge.from_hf_config(config)


def test_safe_config_loader_wraps_known_architecture_version_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(transformers_version, "get_transformers_version", lambda: Version("5.3.0"))
    monkeypatch.setenv("MEGATRON_CONFIG_LOCK_DIR", str(tmp_path))
    hf_error = ValueError(
        "The checkpoint you are trying to load has model type `unit_test_config_guard` "
        "but Transformers does not recognize this architecture."
    )

    with patch(
        "megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig.from_pretrained",
        side_effect=hf_error,
    ):
        with pytest.raises(TransformersVersionError, match="Transformers>=99.0.0"):
            safe_load_config_with_retry("unused/model", max_retries=0)


def test_safe_config_loader_does_not_wrap_unknown_architecture(monkeypatch, tmp_path):
    monkeypatch.setenv("MEGATRON_CONFIG_LOCK_DIR", str(tmp_path))
    hf_error = ValueError(
        "The checkpoint you are trying to load has model type `not_registered` "
        "but Transformers does not recognize this architecture."
    )

    with patch(
        "megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig.from_pretrained",
        side_effect=hf_error,
    ):
        with pytest.raises(ValueError) as error_info:
            safe_load_config_with_retry("unused/model", max_retries=0)

    assert not isinstance(error_info.value, TransformersVersionError)
    assert "not_registered" in str(error_info.value)


def test_safe_config_loader_does_not_wrap_network_or_auth_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("MEGATRON_CONFIG_LOCK_DIR", str(tmp_path))
    hf_error = OSError("401 Client Error: unauthorized")

    with patch(
        "megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig.from_pretrained",
        side_effect=hf_error,
    ):
        with pytest.raises(ValueError) as error_info:
            safe_load_config_with_retry("private/model", max_retries=0)

    assert not isinstance(error_info.value, TransformersVersionError)
    assert "Ensure the path is valid" in str(error_info.value)
