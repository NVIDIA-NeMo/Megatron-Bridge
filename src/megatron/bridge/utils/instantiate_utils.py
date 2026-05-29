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
    _resolve_target,  # noqa: F401  (re-exported for tests / external callers)
    instantiate,  # noqa: F401  (re-exported for tests / external callers)
    instantiate_node,  # noqa: F401  (re-exported for tests / external callers)
    target_allowlist,
)


_ALLOWED_TARGET_PREFIXES: set[str] = {
    "megatron.",
    "torch.",
    "nvidia.",
    "transformers.",
    "numpy.",
    "nemo.",
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


def _validate_target_prefix(*, target: str, full_key: str) -> None:
    """Validate that a _target_ string is permitted by the allowlist."""
    if not target_allowlist.is_allowed(target):
        raise InstantiationException(
            f"Instantiation of '{target}' is not allowed. "
            f"The target must start with one of the allowed prefixes: "
            f"{sorted(target_allowlist.allowed_prefixes)}. "
            f"Use register_allowed_target_prefix() to add additional allowed prefixes."
            + (f"\nfull_key: {full_key}" if full_key else "")
        )
