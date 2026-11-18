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

"""The reference implementation's Muon recipe as parameter attributes Megatron-Core's Muon reads.

The reference's Muon (``muon_qkv_update_mode="split_head"``, ``muon_ffn_split``) orthogonalizes some
weights block by block: every row block below goes through Newton-Schulz alone, with Muon's scale of
its own shape. :func:`tag_ernie55_muon_params` describes the blocks (:func:`ernie55_muon_split_sizes`)
to Megatron-Core as a per-parameter :class:`~megatron.core.transformer.attention.QKVLayout` with one
group whose per-head slices are the blocks (``is_mla=False``), on

* H layers: ``linear_q_up_proj`` per head, ``linear_o_group_proj`` and the output gate per output group;
* D layers: ``linear_q_up_proj`` per head and, inside a head, the no-RoPE and RoPE rows apart;
  ``linear_kv_down_proj`` as the latent and the RoPE key; ``k_b_proj``, ``v_b_proj`` and the output
  gate per head; the indexer's ``linear_wq_b`` per indexer head;
* every gated fc1 (dense MLP, routed and shared experts) as its gate and up halves.

Megatron-Core's emerging-optimizer setup tags every 2-D weight that carries a ``qkv_layout`` (``is_qkv``
and its split shapes). ``muon_split_qkv`` (Megatron-Core's default, which the ERNIE 5.5 recipe keeps) is
what turns the blocks on: with it ``TensorParallelMuon.orthogonalize`` orthogonalizes each block
separately. With ``muon_split_qkv=False`` Muon silently orthogonalizes these weights whole, dropping the
reference's blocks. Since the layout has a single group, ``muon_split_qkv_per_head`` does not change which
rows form a block; it only picks Megatron-Core's per-head implementation. The recipe leaves it off: for
blocks of one size that implementation runs one batched Newton-Schulz and reshapes its result with
``view_as`` (``TensorParallelMuon._orthogonalize_split_qkv``), which fails when the blocks have more rows
than columns, because Newton-Schulz then returns a transposed, non-contiguous tensor. ``k_b_proj`` and the
dense fc1 halves have such blocks. Without it Megatron-Core orthogonalizes the blocks one at a time.
These layouts replace Megatron-Core's own on the H and D query
up projections, which are MLA layouts (``is_mla=True``) that Muon splits only under
``muon_split_mla_qkv``; the recipe leaves that off, so no other weight is split.

The compressor's APE (``muon_exclude_patterns`` ``"ape"``) is tagged ``use_muon=False``, which sends it
to the scalar optimizer; the router matrices (``"mlp.gate"``) follow ``moe_router_skip_muon``, which
the ERNIE 5.5 bridge turns on.

The block sizes describe a weight's local rows, so they are set only when no weight is sharded along
dim 0 (tensor and expert tensor parallel size 1); otherwise Muon orthogonalizes whole matrices, with
a warning.

Per-parameter QKV layouts exist only in Megatron-Core dev: with a Megatron-Core that has no
``QKVLayout``, :func:`tag_ernie55_muon_params` still excludes the APE but leaves every weight unsplit,
with one warning per process.
"""

from __future__ import annotations

import logging
import re

import torch


try:
    from megatron.core.transformer.attention import QKVLayout
except ImportError:  # Megatron-Core main has no per-parameter QKV layouts (Megatron-Core dev only).
    QKVLayout = None

from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import Ernie55HCASelfAttention
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_attention import Ernie55LatentMQASelfAttention
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core import Ernie55LatentMQACoreAttention


logger = logging.getLogger(__name__)

__all__ = ["ernie55_muon_layout", "ernie55_muon_split_sizes", "tag_ernie55_muon_params"]

_GATED_FC1 = re.compile(r"(^|\.)linear_fc1\.weight\d*$")
_warned_no_qkv_layout = False


def ernie55_muon_split_sizes(module: torch.nn.Module) -> dict[str, list[int]]:
    """The Muon row blocks of the weights ``module`` owns directly, by name relative to ``module``."""
    if isinstance(module, Ernie55HCASelfAttention):
        heads, groups = module.num_attention_heads_per_partition, module.o_local_groups
        q_rows = module.linear_q_up_proj.weight.shape[0]
        group_rows = module.linear_o_group_proj.shape[0] // groups
        return {
            "linear_q_up_proj.weight": [q_rows // heads] * heads,
            "linear_o_group_proj": [group_rows] * groups,
            "linear_output_gate.weight": [module.linear_output_gate.weight.shape[0] // groups] * groups,
        }
    if isinstance(module, Ernie55LatentMQASelfAttention):
        config = module.config
        heads = module.num_attention_heads_per_partition
        return {
            "linear_q_up_proj.weight": [config.qk_head_dim, config.qk_pos_emb_head_dim] * heads,
            "linear_kv_down_proj.weight": [config.kv_lora_rank, config.qk_pos_emb_head_dim],
            "k_b_proj": [config.kv_lora_rank] * heads,
            "v_b_proj": [config.v_head_dim] * heads,
        }
    if isinstance(module, Ernie55LatentMQACoreAttention):
        sizes = {"linear_output_gate.weight": [module.v_head_dim] * module.num_heads}
        indexer = module.indexer
        if indexer is not None and getattr(indexer, "linear_wq_b", None) is not None:
            sizes["indexer.linear_wq_b.weight"] = [indexer.index_head_dim] * indexer.index_n_heads
        return sizes
    return {}


def ernie55_muon_layout(sizes: list[int]) -> QKVLayout:
    """The layout that has Megatron-Core's split-QKV Muon orthogonalize every row block alone.

    One group whose projections and per-head slices are both ``sizes``, so Megatron-Core's split is
    ``sizes`` with or without ``muon_split_qkv_per_head`` (``per_head_split_shapes * num_groups`` or
    ``projection_split_shapes`` over one group); only ``muon_split_qkv`` turns it on. ``is_mla=False``,
    so the split does not depend on ``muon_split_mla_qkv``.
    """
    if QKVLayout is None:
        raise RuntimeError(
            "ERNIE 5.5 Muon blocks need Megatron-Core's per-parameter QKVLayout "
            "(megatron.core.transformer.attention.QKVLayout), which this Megatron-Core build lacks; "
            "use Megatron-Core dev."
        )
    blocks = tuple(int(size) for size in sizes)
    return QKVLayout(num_groups=1, projection_split_shapes=blocks, per_head_split_shapes=blocks, is_mla=False)


def _has_glu_interleave(config, name: str) -> bool:
    """Whether the fc1 ``name`` keeps Megatron-Core's block-interleaved ``[g, u, g, u, ...]`` rows."""
    if ".experts." in name and ".shared_experts." not in name:
        return config.moe_mlp_glu_interleave_size is not None
    if ".shared_experts." in name:
        return bool(config.use_grouped_gemm_for_shared_expert) and (
            config.moe_shared_expert_glu_interleave_size is not None
        )
    return False


def tag_ernie55_muon_params(model: torch.nn.Module) -> None:
    """Set the reference's Muon recipe on ``model``'s parameters (see the module docstring).

    Shared SD-MTP parameters are the decoder's Parameters, so they are tagged once.
    """
    global _warned_no_qkv_layout
    config = model.config
    for module in model.modules():
        compressor = getattr(module, "compressor", None)
        ape = getattr(compressor, "ape", None)
        if isinstance(ape, torch.nn.Parameter):
            ape.use_muon = False
    if QKVLayout is None:
        if not _warned_no_qkv_layout:
            _warned_no_qkv_layout = True
            logger.warning(
                "ERNIE 5.5: this Megatron-Core build has no per-parameter QKV layouts, so the reference's "
                "per-head and gate/up Muon blocks are not applied and Muon orthogonalizes those weights "
                "whole (requires Megatron-Core dev)."
            )
        return
    if config.tensor_model_parallel_size > 1 or (config.expert_tensor_parallel_size or 1) > 1:
        logger.warning(
            "ERNIE 5.5: the reference's Muon row blocks assume unsharded weights; with tensor parallelism "
            "Muon orthogonalizes whole matrices."
        )
        return
    for prefix, module in model.named_modules():
        for name, sizes in ernie55_muon_split_sizes(module).items():
            param = module.get_parameter(name)
            if sum(sizes) != param.shape[0]:
                raise RuntimeError(
                    f"ERNIE 5.5 Muon split of {prefix}.{name}: blocks {sizes} do not cover its "
                    f"{tuple(param.shape)} rows."
                )
            param.qkv_layout = ernie55_muon_layout(sizes)
    if not config.gated_linear_unit:
        return
    skipped = []
    for name, param in model.named_parameters():
        if not _GATED_FC1.search(name):
            continue
        if _has_glu_interleave(config, name):
            skipped.append(name)
            continue
        param.qkv_layout = ernie55_muon_layout([param.shape[0] // 2] * 2)
    if skipped:
        logger.warning(
            f"ERNIE 5.5: {len(skipped)} fc1 weights keep GLU-interleaved rows (e.g. {skipped[0]}); Muon "
            "orthogonalizes them whole instead of as gate and up halves."
        )
