"""Offline target-policy checks; unsafe operations are replaced with mocks."""

import copy
import functools
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from megatron.bridge.utils.instantiate_utils import (
    InstantiationException,
    InstantiationMode,
    _resolve_target,
    instantiate,
    target_allowlist,
)


pytestmark = pytest.mark.unit
DOWNLOAD_TARGETS = ("torch.hub.load_state_dict_from_url", "torch.utils.model_zoo.load_url")
RESOLVER = "megatron.bridge.utils.instantiate_utils._mcore_resolve_target"


@pytest.mark.parametrize("target", DOWNLOAD_TARGETS)
@pytest.mark.parametrize("mode", list(InstantiationMode))
@pytest.mark.parametrize("form", ["call", "partial", "reference", "nested", "list"])
def test_rejects_checkpoint_download_before_resolution(target: str, mode: InstantiationMode, form: str) -> None:
    operation = Mock(return_value={})
    node: dict[str, object] = {"_target_": target}
    if form == "partial":
        node["_partial_"] = True
    elif form == "reference":
        node["_call_"] = False
    config: object = node
    if form == "nested":
        config = {"tokenizer": node}
    elif form == "list":
        config = [node]

    with patch(RESOLVER, return_value=operation) as resolve:
        with pytest.raises(InstantiationException, match="bypass target validation"):
            instantiate(config, mode=mode)
    resolve.assert_not_called()
    operation.assert_not_called()


@pytest.mark.parametrize("mode", list(InstantiationMode))
@pytest.mark.parametrize("check_callable", [True, False])
def test_rejects_downloader_alias_after_resolution(mode: InstantiationMode, check_callable: bool) -> None:
    operation = Mock(return_value={})
    operation.__module__ = "torch.hub"
    operation.__qualname__ = "load_state_dict_from_url"
    config: dict[str, object] = {"_target_": "torch.some_alias"}
    if not check_callable:
        config["_call_"] = False

    with patch(RESOLVER, return_value=operation):
        with pytest.raises(InstantiationException, match="resolves to the unsafe target"):
            instantiate(config, mode=mode)
    operation.assert_not_called()


@dataclass
class SmallModel:
    scale: int
    bias: int = 0

    def forward(self, value: int) -> int:
        return self.scale * value + self.bias


ModelAlias = SmallModel


@pytest.fixture
def registered_model() -> Iterator[None]:
    prefix = __name__ + "."
    already_allowed = prefix in target_allowlist.allowed_prefixes
    if not already_allowed:
        target_allowlist.add_prefix(prefix)
    yield
    if not already_allowed:
        target_allowlist.remove_prefix(prefix)


@pytest.mark.parametrize("name", ["SmallModel", "ModelAlias"])
@pytest.mark.parametrize("mode", list(InstantiationMode))
def test_valid_model_yaml_roundtrip_and_overrides(registered_model: None, name: str, mode: InstantiationMode) -> None:
    config = {"_target_": f"{__name__}.{name}", "scale": 3, "bias": 2}
    original = copy.deepcopy(config)
    encoded = OmegaConf.to_yaml(OmegaConf.create(config))
    result = instantiate(OmegaConf.create(encoded), mode=mode)
    assert type(result) is SmallModel
    assert asdict(result) == {"scale": 3, "bias": 2}
    assert result.forward(4) == 14
    assert config == original
    assert OmegaConf.to_yaml(OmegaConf.create(config)) == encoded
    assert instantiate(config, mode=mode, bias=5).forward(4) == 17


def test_valid_model_partial_reference_nested_and_errors(registered_model: None) -> None:
    config = {"_target_": f"{__name__}.SmallModel", "scale": 3}
    partial = instantiate({**config, "_partial_": True})
    assert isinstance(partial, functools.partial)
    assert partial(bias=2).forward(4) == 14
    assert instantiate({"_target_": config["_target_"], "_call_": False}) is SmallModel
    nested = instantiate({"models": [config, {**config, "bias": 2}]})
    assert [model.forward(4) for model in nested["models"]] == [12, 14]
    with pytest.raises(InstantiationException, match="Error in call to target"):
        instantiate({"_target_": config["_target_"]})
    with pytest.raises(InstantiationException, match="Unexpected config keys"):
        instantiate({**config, "unused": 1}, mode=InstantiationMode.STRICT)
    assert instantiate({**config, "unused": 1}, mode=InstantiationMode.LENIENT).forward(4) == 12


def test_valid_alias_and_trusted_callable_resolution() -> None:
    operation = Mock(return_value="unchanged")
    operation.__module__ = "torch.hub"
    operation.__qualname__ = "safe_config_factory"
    with patch(RESOLVER, return_value=operation):
        assert instantiate({"_target_": "torch.some_alias"}) == "unchanged"
    operation.assert_called_once_with()
    with patch(RESOLVER, return_value=SmallModel) as resolve:
        assert _resolve_target(SmallModel, "model") is SmallModel
    resolve.assert_called_once_with(SmallModel, "model", True)
