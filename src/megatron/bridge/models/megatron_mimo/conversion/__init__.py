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
routes are derived from standard bridge/provider metadata when possible; custom
provider construction can use :func:`register_mimo_conversion_spec`.
"""

from megatron.bridge.models.megatron_mimo.conversion.orchestrator import (
    MegatronMIMOBridge,
    MIMOComponent,
    MIMOConversionTask,
    build_route_local_registry,
    component_pg_context,
    export_megatron_mimo_to_hf,
    get_mimo_conversion_spec,
    import_hf_to_megatron_mimo,
    make_route_local_bridge,
    register_mimo_conversion_spec,
    save_hf_pretrained_mimo,
    supports_mimo_conversion,
    validate_route_table,
)


def __getattr__(name: str):
    if name in {"load_megatron_mimo_model", "save_megatron_mimo_model"}:
        from megatron.bridge.models.megatron_mimo.conversion import mimo_model_io

        return getattr(mimo_model_io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MIMOComponent",
    "MIMOConversionTask",
    "MegatronMIMOBridge",
    "build_route_local_registry",
    "component_pg_context",
    "export_megatron_mimo_to_hf",
    "get_mimo_conversion_spec",
    "import_hf_to_megatron_mimo",
    "load_megatron_mimo_model",
    "make_route_local_bridge",
    "register_mimo_conversion_spec",
    "save_hf_pretrained_mimo",
    "save_megatron_mimo_model",
    "supports_mimo_conversion",
    "validate_route_table",
]
