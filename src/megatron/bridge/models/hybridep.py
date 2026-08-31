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

"""Runtime layout selection for HybridEP uneven-input padding."""

import inspect
from functools import partial

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import unwrap_model


HYBRIDEP_PADDING_FIELDS = (
    "moe_hybridep_pad_uneven_dispatch_inputs",
    "moe_hybridep_pad_variable_tokens",
)

_HYBRIDEP_PADDING_TARGETS = "_mbridge_hybridep_padding_targets"


def _apply_hybridep_padding_layout(
    targets: tuple[tuple[object, tuple[str, ...]], ...],
    packed_seq_params: object | None,
) -> None:
    """Set automatic HybridEP padding targets for the current tensor layout."""
    enable_padding = packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd"
    for config, padding_fields in targets:
        for padding_field in padding_fields:
            if getattr(config, padding_field) != enable_padding:
                setattr(config, padding_field, enable_padding)


def _hybridep_padding_pre_hook(
    targets: tuple[tuple[object, tuple[str, ...]], ...],
    packed_seq_params_position: int | None,
    _module: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    """Select automatic HybridEP padding before a regular model forward."""
    packed_seq_params = kwargs.get("packed_seq_params")
    if packed_seq_params is None and packed_seq_params_position is not None and len(args) > packed_seq_params_position:
        packed_seq_params = args[packed_seq_params_position]
    _apply_hybridep_padding_layout(targets, packed_seq_params)


def _packed_seq_params_position(model: torch.nn.Module) -> int | None:
    """Return the positional index of ``packed_seq_params`` for a bound forward."""
    try:
        parameters = tuple(inspect.signature(model.forward).parameters.values())
    except (TypeError, ValueError):
        return None

    positional_index = 0
    for parameter in parameters:
        if parameter.name == "packed_seq_params":
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                return positional_index
            return None
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional_index += 1
    return None


def register_hybridep_thd_padding(model: torch.nn.Module, config: TransformerConfig) -> None:
    """Register THD-only automatic HybridEP padding on an eager runtime model.

    An explicitly enabled setting is not registered as an automatic target and
    therefore remains enabled for every layout. CUDA-graph models retain their
    configured value because the HybridEP padding path performs a host sync.
    """
    if (
        getattr(config, "moe_token_dispatcher_type", None) != "flex"
        or getattr(config, "moe_flex_dispatcher_backend", None) != "hybridep"
        or getattr(config, "cuda_graph_impl", "none") != "none"
        or _HYBRIDEP_PADDING_TARGETS in vars(model)
    ):
        return

    root_padding_fields = tuple(field_name for field_name in HYBRIDEP_PADDING_FIELDS if hasattr(config, field_name))
    if not root_padding_fields:
        raise AttributeError("Megatron Core TransformerConfig does not expose a HybridEP uneven-input padding field")
    if any(getattr(config, padding_field) for padding_field in root_padding_fields):
        return

    targets_by_id: dict[int, tuple[object, tuple[str, ...]]] = {}
    for module in model.modules():
        module_config = getattr(module, "config", None)
        if (
            module_config is None
            or getattr(module_config, "moe_token_dispatcher_type", None) != "flex"
            or getattr(module_config, "moe_flex_dispatcher_backend", None) != "hybridep"
        ):
            continue
        padding_fields = tuple(
            field_name for field_name in HYBRIDEP_PADDING_FIELDS if hasattr(module_config, field_name)
        )
        if padding_fields and not any(getattr(module_config, padding_field) for padding_field in padding_fields):
            targets_by_id.setdefault(id(module_config), (module_config, padding_fields))

    targets = tuple(targets_by_id.values())
    setattr(model, _HYBRIDEP_PADDING_TARGETS, targets)
    model.register_forward_pre_hook(
        partial(_hybridep_padding_pre_hook, targets, _packed_seq_params_position(model)),
        with_kwargs=True,
    )


def set_hybridep_padding_for_layout(
    model: torch.nn.Module,
    packed_seq_params: object | None,
    *,
    config: TransformerConfig,
) -> None:
    """Select HybridEP padding before a runtime path that bypasses model forward.

    Resolve the standard MCore wrappers once so this per-microbatch path stays
    constant-time and does not traverse every model module.
    """
    if (
        getattr(config, "moe_token_dispatcher_type", None) != "flex"
        or getattr(config, "moe_flex_dispatcher_backend", None) != "hybridep"
        or not isinstance(model, torch.nn.Module)
    ):
        return

    runtime_model = unwrap_model(model)
    targets = vars(runtime_model).get(_HYBRIDEP_PADDING_TARGETS)
    if targets is not None:
        _apply_hybridep_padding_layout(targets, packed_seq_params)
