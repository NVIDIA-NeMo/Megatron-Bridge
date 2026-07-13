# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Patch for https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/instantiate/_instantiate2.py
# until https://github.com/facebookresearch/hydra/issues/2140 is resolved

from typing import Any, Callable

from megatron.training.config import instantiate_utils as _mcore_instantiate_utils
from megatron.training.config.instantiate_utils import (
    InstantiationException,
    InstantiationMode,  # noqa: F401  (re-exported for tests / external callers)
    _call_target,  # noqa: F401  (re-exported for tests / external callers)
    _convert_node,  # noqa: F401  (re-exported for tests / external callers)
    _convert_target_to_string,  # noqa: F401  (re-exported for tests / external callers)
    _extract_pos_args,  # noqa: F401  (re-exported for tests / external callers)
    _filter_kwargs_for_target,  # noqa: F401  (re-exported for tests / external callers)
    _is_target,  # noqa: F401  (re-exported for tests / external callers)
    _Keys,  # noqa: F401  (re-exported for tests / external callers)
    _locate,  # noqa: F401  (re-exported for tests / external callers)
    _prepare_input_dict_or_list,  # noqa: F401  (re-exported for tests / external callers)
    instantiate,  # noqa: F401  (re-exported for tests / external callers)
    instantiate_node,  # noqa: F401  (re-exported for tests / external callers)
    target_allowlist,
)
from megatron.training.config.instantiate_utils import (
    _resolve_target as _mcore_resolve_target,
)


_ALLOWED_TARGET_PREFIXES: set[str] = {
    "megatron.",
    "torch.",
    "nvidia.",
    "transformers.",
    "numpy.",
    "nemo.",
}

_TARGET_ALLOWLIST_MUTATORS: tuple[str, ...] = (
    "add_prefix",
    "remove_prefix",
    "add_exact",
    "remove_exact",
    "disable",
    "enable",
)

_ALLOWED_PRIVATE_TARGETS: set[str] = {
    # PyTorch exposes torch.nn.functional.gelu as this C-extension symbol, and
    # the YAML function representer serializes it with its underlying module.
    "torch._C._nn.gelu",
}

_DISALLOWED_TARGETS: set[str] = {
    "megatron.bridge.models.conversion.auto_bridge.AutoBridge.from_hf_pretrained",
    "megatron.bridge.models.hf_pretrained.safe_config_loader.safe_load_config_with_retry",
    "megatron.bridge.utils.import_utils.safe_import",
    "megatron.bridge.utils.import_utils.safe_import_from",
    "megatron.bridge.utils.instantiate_utils.register_allowed_target_prefix",
    "numpy.ctypeslib.load_library",
    "numpy.load",
    "torch.classes.load_library",
    "torch.ctypes.CDLL",
    "torch.ctypes.OleDLL",
    "torch.ctypes.PyDLL",
    "torch.ctypes.WinDLL",
    "torch.ctypes.cdll.LoadLibrary",
    "torch.ctypes.oledll.LoadLibrary",
    "torch.ctypes.pydll.LoadLibrary",
    "torch.ctypes.windll.LoadLibrary",
    "torch.hub.load",
    "torch.load",
    "torch.ops.load_library",
    "torch.utils.cpp_extension.load",
    "torch.utils.cpp_extension.load_inline",
    "transformers.AutoConfig.from_pretrained",
    "transformers.AutoModel.from_pretrained",
    "transformers.AutoModelForCausalLM.from_pretrained",
    "transformers.AutoProcessor.from_pretrained",
    "transformers.AutoTokenizer.from_pretrained",
    "transformers.models.auto.configuration_auto.AutoConfig.from_pretrained",
    "transformers.models.auto.modeling_auto.AutoModel.from_pretrained",
    "transformers.models.auto.modeling_auto.AutoModelForCausalLM.from_pretrained",
    "transformers.models.auto.processing_auto.AutoProcessor.from_pretrained",
    "transformers.models.auto.tokenization_auto.AutoTokenizer.from_pretrained",
    "transformers.utils.import_utils.direct_transformers_import",
    *{f"megatron.bridge.utils.instantiate_utils.target_allowlist.{method}" for method in _TARGET_ALLOWLIST_MUTATORS},
    *{
        f"megatron.training.config.instantiate_utils.target_allowlist.{method}"
        for method in _TARGET_ALLOWLIST_MUTATORS
    },
}

_DISALLOWED_CALLABLE_FIELD_NAMES: set[str] = {
    "collate_impl",
    "hf_filter_lambda",
    "preprocess_fn",
    "process_example_fn",
}


# Mirror Bridge's allowlist into the MLM `target_allowlist` singleton, which is
# the source of truth consulted by `_validate_target_prefix` below. MLM's
# default prefixes are narrower (megatron.training./megatron.core./torch./
# transformers./signal.) and would otherwise reject e.g. `megatron.bridge.*`,
# `nvidia.*`, `numpy.*`, `nemo.*`.
def _as_module_prefix(prefix: str) -> str:
    """Ensure prefix ends with '.' so allowlist matches at module boundaries."""
    return prefix if prefix.endswith(".") else prefix + "."


def _seed_allowlist() -> None:
    for prefix in _ALLOWED_TARGET_PREFIXES:
        target_allowlist.add_prefix(_as_module_prefix(prefix))


_seed_allowlist()


def register_allowed_target_prefix(prefix: str) -> None:
    """Register an additional allowed module prefix for _target_ instantiation.

    This allows extending the default allowlist for use cases that require
    instantiating classes from other packages.
    """
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"Prefix must be a non-empty string, got {prefix!r}")
    _ALLOWED_TARGET_PREFIXES.add(prefix)
    # MLM's `target_allowlist` is the source of truth for `_validate_target_prefix`
    # and requires the trailing dot to match at module boundaries.
    target_allowlist.add_prefix(_as_module_prefix(prefix))


def _validate_target_prefix(*, target: str, full_key: str | int) -> None:
    """Validate that a _target_ string is permitted by Bridge hardening rules."""
    field_name = full_key.rsplit(".", 1)[-1] if isinstance(full_key, str) else ""
    if field_name in _DISALLOWED_CALLABLE_FIELD_NAMES:
        raise InstantiationException(
            f"Instantiation of '{target}' is not allowed for callable config field '{full_key}'. "
            "Use a registered symbolic option or pass a Python callable from trusted application code."
        )
    if target in _DISALLOWED_TARGETS:
        raise InstantiationException(
            f"Instantiation of '{target}' is not allowed because it can bypass target validation."
            + (f"\nfull_key: {full_key}" if full_key else "")
        )
    private_segments = [segment for segment in target.split(".") if segment.startswith("_")]
    if private_segments and target not in _ALLOWED_PRIVATE_TARGETS:
        raise InstantiationException(
            f"Instantiation of '{target}' is not allowed because private target path segments are not supported: "
            f"{private_segments}." + (f"\nfull_key: {full_key}" if full_key else "")
        )
    if not target_allowlist.is_allowed(target):
        raise InstantiationException(
            f"Instantiation of '{target}' is not allowed because it is not in the allowlist. "
            f"The target must start with one of the allowed prefixes: "
            f"{sorted(target_allowlist.allowed_prefixes)}. "
            f"Use register_allowed_target_prefix() to add additional allowed prefixes."
            + (f"\nfull_key: {full_key}" if full_key else "")
        )


def _resolve_target(
    target: str | type | Callable[..., Any],
    full_key: str | int,
    check_callable: bool = True,
) -> type | Callable[..., Any] | object:
    """Resolve target string, type, or callable after Bridge validation."""
    if isinstance(target, str):
        _validate_target_prefix(target=target, full_key=full_key)
    return _mcore_resolve_target(target, full_key, check_callable)


_mcore_instantiate_utils._resolve_target = _resolve_target
