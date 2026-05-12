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

"""Runtime CP-group rebinding for eval-time context parallelism.

This module provides helpers to switch a model's cached process-group references
from the training CP layout to a different eval CP layout, run evaluation, then
restore the training layout. It works only with the decentralized PG path
(use_decentralized_pg=True) and requires no changes to Megatron-Core.

Typical usage::

    with eval_cp_context(model, eval_pgs, train_pgs):
        evaluate_and_print_results(..., pg_collection=eval_pgs)
"""

from contextlib import contextmanager
from typing import Iterator, Union

import torch
from megatron.core.process_groups_config import ProcessGroupCollection


# Maps module attribute name → ProcessGroupCollection field name.
# Single source of truth: update this dict when MCore adds a new CP-bearing attribute.
_GROUP_ATTRS: dict[str, str] = {
    "cp_group": "cp",
    "tp_cp_group": "tp_cp",
    "tp_dp_cp_group": "tp_dp_cp",
    "dp_cp_group": "dp_cp",
}


def install_pg_collection(
    model: Union[list, "torch.nn.Module"],
    target: ProcessGroupCollection,
) -> None:
    """Rebind all CP-affected process groups on every module of *model*.

    Walks every sub-module across all virtual-PP chunks and rebinds:
    - ``pg_collection`` (used by TransformerLayer, DotProductAttention, …)
    - Named CP-bearing group attributes (cp_group, tp_cp_group, …)
    - TEDotProductAttention internal CP comm state via ``set_context_parallel_group``

    TP/PP/EP groups are never touched because those do not change between train
    and eval.

    Args:
        model: Single model chunk or a list of virtual-PP chunks.
        target: The ProcessGroupCollection to install.
    """
    try:
        from megatron.core.extensions.transformer_engine import TEDotProductAttention as _TEDPA

        _te_dpa_cls: type | None = _TEDPA
    except (ImportError, AttributeError):
        _te_dpa_cls = None

    try:
        from megatron.core.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
            RotaryEmbedding,
        )

        _rotary_cls: tuple[type, ...] = (RotaryEmbedding, MultimodalRotaryEmbedding)
    except (ImportError, AttributeError):
        _rotary_cls = ()

    cp_group = target.cp
    cp_size = cp_group.size()
    cp_ranks = torch.distributed.get_process_group_ranks(cp_group)

    # Mutate the (shared) TransformerConfig.context_parallel_size so that
    # RotaryEmbedding.get_rotary_seq_len computes rotary_seq_len against the
    # *live* CP degree. This is the single runtime read of config.cp on the
    # standard GPT eval path; init-time reads (TEDotProductAttention,
    # TransformerLayer, etc.) and CUDA-graph-only reads are not affected.
    # Since install_pg_collection is called both on enter (eval_pgs) and exit
    # (train_pgs) by eval_cp_context, the original value is restored implicitly.
    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        cfg = getattr(chunk, "config", None)
        if cfg is not None and hasattr(cfg, "context_parallel_size"):
            cfg.context_parallel_size = cp_size

    for module in _iter_all_modules(model):
        if hasattr(module, "pg_collection"):
            module.pg_collection = target

        for attr, pg_key in _GROUP_ATTRS.items():
            if hasattr(module, attr):
                setattr(module, attr, getattr(target, pg_key, None))

        # RotaryEmbedding caches forward(max_seq_len, offset, packed_seq, cp_group)
        # via @lru_cache. The cp_group has just been swapped via _GROUP_ATTRS,
        # but cached entries keyed on the previous cp_group would still be
        # returned for the same rotary_seq_len. Clear so the next call
        # recomputes against the current self.cp_group.
        if _rotary_cls and isinstance(module, _rotary_cls):
            if hasattr(module, "forward") and hasattr(module.forward, "cache_clear"):
                module.forward.cache_clear()

        if _te_dpa_cls is not None and isinstance(module, _te_dpa_cls):
            # Lazily create the class-level CP stream if needed (created by TE the
            # first time a CP > 1 model is built; may be None for CP=1 models).
            if _te_dpa_cls.cp_stream is None:
                _te_dpa_cls.cp_stream = torch.cuda.Stream()
            module.set_context_parallel_group(
                cp_group,
                cp_ranks,
                _te_dpa_cls.cp_stream,
                module.cp_comm_type,
            )


@contextmanager
def eval_cp_context(
    model: Union[list, "torch.nn.Module"],
    eval_pgs: ProcessGroupCollection,
    train_pgs: ProcessGroupCollection,
) -> Iterator[None]:
    """Context manager: install *eval_pgs* for the duration of the block.

    On entry, rebinds all CP-affected module attributes to *eval_pgs*.
    On exit (including exceptions), restores *train_pgs*.

    Args:
        model: Single model chunk or list of virtual-PP chunks.
        eval_pgs: ProcessGroupCollection for eval (different CP degree).
        train_pgs: ProcessGroupCollection for training (restored on exit).

    Example::

        with eval_cp_context(model, eval_pgs, train_pgs):
            evaluate_and_print_results(..., pg_collection=eval_pgs)
    """
    install_pg_collection(model, eval_pgs)
    try:
        yield
    finally:
        install_pg_collection(model, train_pgs)


def _iter_all_modules(model: Union[list, "torch.nn.Module"]) -> Iterator["torch.nn.Module"]:
    """Yield every nn.Module across all virtual-PP chunks."""
    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        yield from chunk.modules()
