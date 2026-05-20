# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""MegatronMIMO HF<->Megatron conversion framework.

Public surface for the generic MIMO conversion framework. Per-model-family
adapters register themselves via :func:`register_mimo_conversion` and
declare a route table built from :class:`MIMOComponent` entries.
"""

# Side-effect import: registers per-family adapters via the
# ``@register_mimo_conversion`` decorator so ``get_mimo_adapter`` can dispatch
# on a source bridge class without callers needing to import each adapter.
from megatron.bridge.models.megatron_mimo.conversion import adapters  # noqa: F401
from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import (
    load_megatron_mimo_model,
    save_megatron_mimo_model,
)
from megatron.bridge.models.megatron_mimo.conversion.orchestrator import (
    MegatronMIMOBridge,
    MIMOComponent,
    MIMOConversionTask,
    build_route_local_registry,
    component_pg_context,
    export_megatron_mimo_to_hf,
    get_mimo_adapter,
    import_hf_to_megatron_mimo,
    list_mimo_adapters,
    make_route_local_bridge,
    register_mimo_conversion,
    save_hf_pretrained_mimo,
    validate_route_table,
)


__all__ = [
    "MIMOComponent",
    "MIMOConversionTask",
    "MegatronMIMOBridge",
    "build_route_local_registry",
    "component_pg_context",
    "export_megatron_mimo_to_hf",
    "get_mimo_adapter",
    "import_hf_to_megatron_mimo",
    "list_mimo_adapters",
    "load_megatron_mimo_model",
    "make_route_local_bridge",
    "register_mimo_conversion",
    "save_hf_pretrained_mimo",
    "save_megatron_mimo_model",
    "validate_route_table",
]
