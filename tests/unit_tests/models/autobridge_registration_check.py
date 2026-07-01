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

from packaging.version import Version
from transformers import PretrainedConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import model_bridge
from megatron.bridge.models.conversion.transformers_version import TransformersVersionError


def main() -> None:
    """Validate the expected registration manifest in a fresh interpreter."""
    expected = cast(dict[str, str], json.loads(sys.argv[1]))
    string_registrations = set(cast(list[str], json.loads(sys.argv[2])))
    supported = AutoBridge.list_supported_models()
    assert supported == sorted(expected), f"registration manifest mismatch: {supported!r}"

    key_kinds = {
        key if isinstance(key, str) else key.__name__: isinstance(key, str)
        for key in model_bridge.get_model_bridge._exact_types
    }
    expected_key_kinds = {architecture: architecture in string_registrations for architecture in expected}
    assert key_kinds == expected_key_kinds, f"registration key-kind mismatch: {key_kinds!r}"

    for architecture, expected_bridge_class in expected.items():
        config = PretrainedConfig(architectures=[architecture])
        if architecture in string_registrations:
            config.update({"auto_map": {"AutoModelForCausalLM": f"modeling_test.{architecture}"}})
        assert AutoBridge.supports(config), f"AutoBridge rejected {architecture}"

        try:
            bridge = AutoBridge.from_hf_config(config)
        except TransformersVersionError as error:
            bridge_class = model_bridge.get_registered_bridge_class(source_name=architecture)
            assert bridge_class is not None, f"missing guarded bridge class index for {architecture}"
            min_version = bridge_class.MIN_TRANSFORMERS_VERSION
            assert min_version is not None, f"{architecture} raised without compatibility metadata"
            assert error.model_name == architecture
            assert error.required_version == Version(min_version)
            if error.installed_version >= error.required_version:
                assert error.missing_symbols == bridge_class.REQUIRED_TRANSFORMERS_SYMBOLS
            continue

        selected_bridge = bridge._model_bridge
        actual_bridge_class = f"{type(selected_bridge).__module__}.{type(selected_bridge).__name__}"

        assert actual_bridge_class == expected_bridge_class, (
            f"{architecture} selected {actual_bridge_class}, expected {expected_bridge_class}"
        )
        assert selected_bridge.hf_config is config


if __name__ == "__main__":
    main()
