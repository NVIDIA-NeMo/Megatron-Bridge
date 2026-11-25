# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contiguous physical token owners at the TP1 BF16 training head."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy

from .context_parallel import ContiguousBackwardOwners


# The standard MCore head callback receives hidden and labels. The forward scopes its
# token-layout metadata so nested forwards and concurrent Python contexts cannot replace it.
_HEAD_PACK = ContextVar("ernie55_reference_head_pack", default=None)


@contextmanager
def reference_head_pack(packed, group):
    """Expose this model call's physical row layout to both output heads."""
    token = _HEAD_PACK.set((packed, group))
    try:
        yield
    finally:
        _HEAD_PACK.reset(token)


def reference_head_owners(config):
    """Reuse the microbatch's THD route for narrow head activations and token vectors."""
    current = _HEAD_PACK.get()
    if current is None:
        raise ValueError("Reference head weight gradients at CP > 1 require an ERNIE model forward context")
    packed, group = current
    if group is None or group.size() != config.context_parallel_size:
        raise ValueError("Reference head weight gradients require the configured static CP group")
    if packed is not None and getattr(packed, "local_cp_size", None) is not None:
        raise ValueError("Reference head weight gradients do not support dynamic packed CP groups")
    layout = getattr(packed, "cp_partition_mode", None) or config.cp_partition_mode
    if layout == "contiguous":
        return None
    if layout != "zigzag":
        raise ValueError(f"Unrecognized ERNIE head CP layout: {layout!r}")
    # CpPartitionModeConverter records the destination layout on its metadata. Keep the
    # decoder's metadata in its own layout while sharing the already built routing plan.
    return ContiguousBackwardOwners.for_pack(copy(packed), group, config)
