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

"""Fresh-process AutoBridge registration contract check."""

from __future__ import annotations

import json
import sys
from typing import cast

from transformers import PretrainedConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import model_bridge


def main() -> None:
    """Validate the expected registration manifest in a fresh interpreter."""
    expected = cast(dict[str, str], json.loads(sys.argv[1]))
    string_registrations = set(cast(list[str], json.loads(sys.argv[2])))
    deprecated_registrations = set(cast(list[str], json.loads(sys.argv[3])))
    # Subset rather than equality: adding a new model must not require editing the manifest.
    # The contract worth guarding is that every architecture already listed stays registered,
    # keeps its key kind, and resolves to the same bridge class.
    supported = set(AutoBridge.list_supported_models()) - deprecated_registrations
    missing = sorted(set(expected) - supported)
    assert not missing, f"expected architectures are no longer registered: {missing!r}"

    key_kinds = {
        key if isinstance(key, str) else key.__name__: isinstance(key, str)
        for key in model_bridge.get_model_bridge._exact_types
        if (key if isinstance(key, str) else key.__name__) not in deprecated_registrations
    }
    mismatched_key_kinds = {
        architecture: key_kinds.get(architecture)
        for architecture in expected
        if key_kinds.get(architecture) != (architecture in string_registrations)
    }
    assert not mismatched_key_kinds, f"registration key-kind mismatch: {mismatched_key_kinds!r}"

    for architecture, expected_bridge_class in expected.items():
        config = PretrainedConfig(architectures=[architecture])
        if architecture in string_registrations:
            config.update({"auto_map": {"AutoModelForCausalLM": f"modeling_test.{architecture}"}})
        assert AutoBridge.supports(config), f"AutoBridge rejected {architecture}"

        bridge = AutoBridge.from_hf_config(config)
        selected_bridge = bridge._model_bridge
        actual_bridge_class = f"{type(selected_bridge).__module__}.{type(selected_bridge).__name__}"

        assert actual_bridge_class == expected_bridge_class, (
            f"{architecture} selected {actual_bridge_class}, expected {expected_bridge_class}"
        )
        assert selected_bridge.hf_config is config


if __name__ == "__main__":
    main()
