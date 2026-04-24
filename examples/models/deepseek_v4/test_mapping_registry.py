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

"""Smoke-test the DSV4 `mapping_registry()` inside `mbridge`.

Builds a tiny `SimpleNamespace` config matching DSV4-Flash, hands it to the
bridge, and prints how many mapping entries are produced plus a few samples.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from megatron.bridge.models.deepseek_v4 import DeepSeekV4Bridge


def fake_small_config() -> SimpleNamespace:
    """A minimal config with only the fields mapping_registry/provider_bridge read."""
    return SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        n_routed_experts=8,
        num_hash_layers=1,
        # mapping_registry only reads the layer / expert counts; provider_bridge
        # reads the rest when called (we don't call it here).
    )


def main() -> int:
    bridge = DeepSeekV4Bridge()
    bridge.hf_config = fake_small_config()

    registry = bridge.mapping_registry()
    # MegatronMappingRegistry exposes the internal list via a private attr in
    # most builds; fall back to whatever iterable it provides.
    entries = getattr(registry, "_mappings", None) or list(registry)
    print(f"MAPPING_ENTRIES: {len(entries)}")
    for m in entries[:5]:
        mg = getattr(m, "megatron_param", m)
        hf = getattr(m, "hf_param", "")
        print(f"  {type(m).__name__}: {mg!s:70s} <- {hf!s}")
    print(f"  ... and {max(0, len(entries) - 5)} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
