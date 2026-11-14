"""Offline regression coverage for checkpoint downloader targets."""

from unittest.mock import Mock, patch

import pytest

from megatron.bridge.utils.instantiate_utils import InstantiationException, instantiate


@pytest.mark.unit
@pytest.mark.parametrize(
    "target,error,resolver_calls",
    [
        ("torch.hub.load_state_dict_from_url", "bypass target validation", 0),
        ("torch.utils.model_zoo.load_url", "bypass target validation", 0),
        ("torch.some_alias", "resolves to the unsafe target", 1),
    ],
)
def test_rejects_checkpoint_downloader(target: str, error: str, resolver_calls: int) -> None:
    operation = Mock(return_value={})
    operation.__module__ = "torch.hub"
    operation.__qualname__ = "load_state_dict_from_url"

    with patch("megatron.bridge.utils.instantiate_utils._mcore_resolve_target", return_value=operation) as resolve:
        with pytest.raises(InstantiationException, match=error):
            instantiate({"_target_": target})

    assert resolve.call_count == resolver_calls
    operation.assert_not_called()
