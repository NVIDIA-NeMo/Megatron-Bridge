# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO checkpoint extension helpers."""

from __future__ import annotations

from typing import Any

from megatron.core import tensor_parallel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


def _convert_rng_tracker_states(
    rng_tracker_states: dict[str, Any],
    *,
    graph_safe_rng: bool,
) -> dict[str, Any]:
    """Convert serialized CUDA RNG tracker states to the live tracker format."""
    return {
        k: tensor_parallel.convert_cuda_rng_state(v, to_graphable=graph_safe_rng)
        for k, v in rng_tracker_states.items()
    }


def _convert_module_rng_tracker_states(
    module_rng_tracker_states: dict[str, dict[str, Any]],
    *,
    graph_safe_rng: bool,
) -> dict[str, dict[str, Any]]:
    """Convert module-keyed CUDA RNG tracker states to the live tracker format."""
    return {
        module_name: _convert_rng_tracker_states(states, graph_safe_rng=graph_safe_rng)
        for module_name, states in module_rng_tracker_states.items()
    }


def _select_module_rng_tracker_state(
    module_rng_tracker_states: dict[str, dict[str, Any]],
    *,
    module_name: str | None,
) -> dict[str, Any]:
    """Pick a module tracker state to install into MCore's singleton tracker."""
    if module_name is not None and module_name in module_rng_tracker_states:
        return module_rng_tracker_states[module_name]
    if MIMO_LANGUAGE_MODULE_KEY in module_rng_tracker_states:
        return module_rng_tracker_states[MIMO_LANGUAGE_MODULE_KEY]
    return next(iter(module_rng_tracker_states.values()))


def _uses_per_module_rng_checkpoint_mode(megatron_mimo_infra: Any | None) -> bool:
    """Whether checkpointing should treat MIMO RNG state as module-scoped."""
    if megatron_mimo_infra is None:
        return False
    rng_mode = getattr(megatron_mimo_infra, "rng_mode", None)
    return getattr(rng_mode, "value", rng_mode) == "per_module"


def _get_module_rng_tracker_states_for_checkpoint(
    megatron_mimo_infra: Any | None,
) -> dict[str, dict[str, Any]] | None:
    """Return module RNG snapshots when a MIMO setup is in per-module RNG mode."""
    if not _uses_per_module_rng_checkpoint_mode(megatron_mimo_infra):
        return None
    module_rng_tracker_states = getattr(megatron_mimo_infra, "cuda_rng_states_per_module", None)
    if not module_rng_tracker_states:
        return None
    return module_rng_tracker_states


def _restore_module_rng_tracker_states_for_checkpoint(
    module_rng_tracker_states: dict[str, dict[str, Any]],
    *,
    megatron_mimo_infra: Any | None,
    module_name: str | None,
    graph_safe_rng: bool,
    cuda_rng_tracker: Any,
) -> dict[str, dict[str, Any]]:
    """Restore module-keyed CUDA RNG tracker states into the active MIMO infra."""
    module_rng_tracker_states = _convert_module_rng_tracker_states(
        module_rng_tracker_states,
        graph_safe_rng=graph_safe_rng,
    )
    if megatron_mimo_infra is not None:
        megatron_mimo_infra.cuda_rng_states_per_module.clear()
        megatron_mimo_infra.cuda_rng_states_per_module.update(module_rng_tracker_states)
    cuda_rng_tracker.set_states(_select_module_rng_tracker_state(module_rng_tracker_states, module_name=module_name))
    return module_rng_tracker_states


def _get_module_rng_layout_fingerprint_for_checkpoint(
    megatron_mimo_infra: Any | None,
) -> dict[str, Any] | None:
    """Return the per-module RNG layout fingerprint for checkpoint validation."""
    if not _uses_per_module_rng_checkpoint_mode(megatron_mimo_infra):
        return None
    module_to_grid_map = getattr(megatron_mimo_infra, "module_to_grid_map", None)
    if not module_to_grid_map:
        return None
    module_names = sorted(module_to_grid_map)
    return {
        "rng_mode": "per_module",
        "module_names": module_names,
        "module_tp_pp_dp": {
            name: (
                module_to_grid_map[name].get_pg(["tp"]).size(),
                module_to_grid_map[name].get_pg(["pp"]).size(),
                module_to_grid_map[name].get_pg(["dp"]).size(),
            )
            for name in module_names
        },
        "module_rank_offsets": {name: module_to_grid_map[name].rank_offset for name in module_names},
    }


def _validate_module_rng_layout_fingerprint(
    saved_fingerprint: dict[str, Any] | None,
    current_fingerprint: dict[str, Any] | None,
) -> None:
    """Refuse per-module RNG restore when checkpoint and current layouts differ."""
    if current_fingerprint is None:
        return
    if saved_fingerprint != current_fingerprint:
        raise RuntimeError(
            "MegatronMIMO per-module RNG checkpoint layout mismatch. "
            f"Checkpoint fingerprint: {saved_fingerprint}; current fingerprint: {current_fingerprint}. "
            "Set load_rng=False to intentionally ignore checkpoint RNG state."
        )


def _get_module_rng_layout_mismatch_message(
    saved_fingerprint: dict[str, Any] | None,
    current_fingerprint: dict[str, Any] | None,
) -> str | None:
    """Return a warning message when per-module RNG layout restore should be skipped."""
    if current_fingerprint is None or saved_fingerprint == current_fingerprint:
        return None
    return (
        "MegatronMIMO per-module RNG checkpoint layout mismatch. "
        f"Checkpoint fingerprint: {saved_fingerprint}; current fingerprint: {current_fingerprint}. "
        "RNG state will be ignored."
    )


class MegatronMIMORngCheckpointContext:
    """MegatronMIMO implementation of Bridge's generic RNG checkpoint extension."""

    def __init__(self, infra: Any, module_name: str) -> None:
        self._infra = infra
        self._module_name = module_name

    def shard_key_suffix(self) -> str | None:
        return self._module_name

    def collect_extra_rng_state(self) -> object | None:
        return _get_module_rng_tracker_states_for_checkpoint(self._infra)

    def collect_layout_fingerprint(self) -> dict[str, object] | None:
        return _get_module_rng_layout_fingerprint_for_checkpoint(self._infra)

    def layout_mismatch_message(self, saved_fingerprint: dict[str, object] | None) -> str | None:
        return _get_module_rng_layout_mismatch_message(
            saved_fingerprint,
            self.collect_layout_fingerprint(),
        )

    def restore_extra_rng_state(
        self,
        saved_state: object | None,
        saved_fingerprint: dict[str, object] | None,
        *,
        graph_safe_rng: bool,
    ) -> None:
        current_fingerprint = self.collect_layout_fingerprint()
        _validate_module_rng_layout_fingerprint(saved_fingerprint, current_fingerprint)
        if current_fingerprint is None:
            return
        if not saved_state:
            raise RuntimeError(
                "MegatronMIMO per-module RNG checkpoint is missing extra_rng_state. "
                "Set load_rng=False to intentionally ignore checkpoint RNG state."
            )
        _restore_module_rng_tracker_states_for_checkpoint(
            saved_state,  # type: ignore[arg-type]
            megatron_mimo_infra=self._infra,
            module_name=self._module_name,
            graph_safe_rng=graph_safe_rng,
            cuda_rng_tracker=tensor_parallel.get_cuda_rng_tracker(),
        )
