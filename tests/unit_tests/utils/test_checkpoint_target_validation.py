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
