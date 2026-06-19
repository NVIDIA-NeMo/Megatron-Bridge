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

from typing import Any, NamedTuple

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
    target_allowlist,
)
from megatron.training.config.instantiate_utils import (
    instantiate as _mlm_instantiate,
)
from megatron.training.config.instantiate_utils import (
    instantiate_node as _mlm_instantiate_node,
)
from omegaconf import OmegaConf
from omegaconf._utils import is_structured_config


_ALLOWED_TARGET_PREFIXES: set[str] = {
    "megatron.",
    "torch.",
    "nvidia.",
    "transformers.",
    "numpy.",
    "nemo.",
}


class _TargetAllowlistSnapshot(NamedTuple):
    """Immutable view of the target allowlist used for one config preflight."""

    enabled: bool
    allowed_prefixes: tuple[str, ...]
    allowed_exact: frozenset[str]


_BLOCKED_TARGETS: frozenset[str] = frozenset(
    {
        "megatron.bridge.utils.instantiate_utils.register_allowed_target_prefix",
    }
)
_BLOCKED_TARGET_PREFIXES: tuple[str, ...] = (
    "megatron.bridge.utils.instantiate_utils.target_allowlist.",
    "megatron.training.config.instantiate_utils.target_allowlist.",
)


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


def _snapshot_allowlist() -> _TargetAllowlistSnapshot:
    """Capture the allowlist state before any target in a config can run."""
    return _TargetAllowlistSnapshot(
        enabled=target_allowlist.enabled,
        allowed_prefixes=target_allowlist.allowed_prefixes,
        allowed_exact=target_allowlist.allowed_exact,
    )


def _is_allowed_by_snapshot(target: str, snapshot: _TargetAllowlistSnapshot) -> bool:
    """Check a target against the pre-instantiation allowlist snapshot."""
    if not snapshot.enabled:
        return True
    if target in snapshot.allowed_exact:
        return True
    return any(target.startswith(prefix) for prefix in snapshot.allowed_prefixes)


def _target_is_blocked(target: str) -> bool:
    """Reject config targets that can mutate the allowlist during traversal."""
    return target in _BLOCKED_TARGETS or any(target.startswith(prefix) for prefix in _BLOCKED_TARGET_PREFIXES)


def _raise_disallowed_target(target: str, full_key: str, snapshot: _TargetAllowlistSnapshot) -> None:
    """Raise an allowlist error using the stable snapshot for diagnostics."""
    raise InstantiationException(
        f"Target '{target}' is not in the allowlist for _target_ instantiation.\n"
        f"Allowed module prefixes: {', '.join(snapshot.allowed_prefixes)}\n"
        f"Allowed exact targets: {', '.join(sorted(snapshot.allowed_exact))}"
        + (f"\nfull_key: {full_key}" if full_key else "")
    )


def _validate_target_for_instantiate(target: Any, full_key: str, snapshot: _TargetAllowlistSnapshot) -> None:
    """Validate one target before import or recursive instantiation can occur."""
    target = _convert_target_to_string(target)
    if not isinstance(target, str):
        return
    if _target_is_blocked(target):
        raise InstantiationException(
            f"Target '{target}' is not allowed in config instantiation because it can modify the target allowlist."
            + (f"\nfull_key: {full_key}" if full_key else "")
        )
    if not _is_allowed_by_snapshot(target, snapshot):
        _raise_disallowed_target(target, full_key, snapshot)


def _preflight_targets(node: Any, snapshot: _TargetAllowlistSnapshot, full_key: str = "") -> None:
    """Validate every target in a config tree against the same allowlist snapshot."""
    if node is None:
        return

    if isinstance(node, (dict, list)):
        node = _prepare_input_dict_or_list(node)
        node = OmegaConf.structured(node, flags={"allow_objects": True})
    elif is_structured_config(node):
        node = OmegaConf.structured(node, flags={"allow_objects": True})

    if OmegaConf.is_dict(node):
        if _Keys.TARGET in node:
            _validate_target_for_instantiate(node.get(_Keys.TARGET), full_key, snapshot)
        for key in node.keys():
            if key in (_Keys.TARGET, _Keys.PARTIAL, _Keys.CALL, _Keys.NAME):
                continue
            child_key = str(key) if not full_key else f"{full_key}.{key}"
            _preflight_targets(node[key], snapshot, child_key)
    elif OmegaConf.is_list(node):
        for idx, value in enumerate(node._iter_ex(resolve=True)):
            child_key = f"{full_key}[{idx}]" if full_key else f"[{idx}]"
            _preflight_targets(value, snapshot, child_key)


def instantiate(
    config: Any,
    *args: Any,
    mode: InstantiationMode = InstantiationMode.LENIENT,
    **kwargs: Any,
) -> Any:
    """Instantiate a config after validating all targets against a stable allowlist."""
    snapshot = _snapshot_allowlist()
    _preflight_targets(config, snapshot)
    _preflight_targets(kwargs, snapshot)
    return _mlm_instantiate(config, *args, mode=mode, **kwargs)


def instantiate_node(
    node: Any,
    *args: Any,
    partial: bool = False,
    mode: InstantiationMode = InstantiationMode.LENIENT,
) -> Any:
    """Instantiate an OmegaConf node after validating all targets against a stable allowlist."""
    snapshot = _snapshot_allowlist()
    _preflight_targets(node, snapshot)
    return _mlm_instantiate_node(node, *args, partial=partial, mode=mode)


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
