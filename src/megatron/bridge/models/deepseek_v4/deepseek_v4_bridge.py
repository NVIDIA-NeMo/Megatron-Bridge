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

"""DeepSeek-V4-Flash weight mapping.

The HF checkpoint ships in inference format (keys like `layers.0.attn.wq_a.weight`).
On the Megatron side we target:

* HC:  block-level `hc_attn` / `hc_ffn` matching `mcore.HyperConnectionModule`:
       `mapping_proj.weight [n²+2n, n·C]` + `bias [n²+2n]` + three scalar
       `alpha_pre / alpha_post / alpha_res [1]`.
       Head-level `hc_head_*` stays raw (the inference head uses an n-row
       variant that mcore's HC module does not cover).
* CSA: `self_attention.*` — Compressed Sparse Attention block (low-rank Q,
       single KV, grouped HCA O, optional Compressor + Indexer).
* HCA: `self_attention.o_head_grouped.linear_o_{down,up}_proj.*`.
* MoE: `mlp.*`.
* MTP: `mtp.layers.N.*` with `mtp_model_layer.*` for the inner block.

Some mappings are 1-to-1 renames. A few are structural:

* HF `hc_*_scale [3]` fans out to 3 separate scalar params on the mcore side.
* HF `hc_*_fn`  → `mapping_proj.weight`;  HF `hc_*_base` → `bias` — same shape.

Export (`megatron → hf`) is the inverse: the three alphas are stacked into a
length-3 tensor. Roundtrip preserves identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterator, List

import torch

from megatron.bridge.models.deepseek_v4.modeling_deepseek_v4.reference import DeepSeekV4ModelArgs


# ============================================================================
# Rename tables (1-to-1 static keys within each sub-tree)
# ============================================================================

# Attention sub-tree — HF `attn.*` → Megatron `self_attention.*`.
_ATTN_RENAMES: Dict[str, str] = {
    "attn.attn_sink": "self_attention.attn_sink",
    "attn.wq_a.weight": "self_attention.linear_q_down_proj.weight",
    "attn.wq_a.scale": "self_attention.linear_q_down_proj.scale",
    "attn.wq_b.weight": "self_attention.linear_q_up_proj.weight",
    "attn.wq_b.scale": "self_attention.linear_q_up_proj.scale",
    "attn.q_norm.weight": "self_attention.q_layernorm.weight",
    "attn.wkv.weight": "self_attention.linear_kv_proj.weight",
    "attn.wkv.scale": "self_attention.linear_kv_proj.scale",
    "attn.kv_norm.weight": "self_attention.kv_layernorm.weight",
    # HCA: grouped low-rank O-projection.
    "attn.wo_a.weight": "self_attention.o_head_grouped.linear_o_down_proj.weight",
    "attn.wo_a.scale": "self_attention.o_head_grouped.linear_o_down_proj.scale",
    "attn.wo_b.weight": "self_attention.o_head_grouped.linear_o_up_proj.weight",
    "attn.wo_b.scale": "self_attention.o_head_grouped.linear_o_up_proj.scale",
    # CSA optional — Compressor (compress_ratio != 0)
    "attn.compressor.ape": "self_attention.compressor.ape",
    "attn.compressor.wkv.weight": "self_attention.compressor.wkv.weight",
    "attn.compressor.wgate.weight": "self_attention.compressor.wgate.weight",
    "attn.compressor.norm.weight": "self_attention.compressor.norm.weight",
    # CSA optional — Indexer (compress_ratio == 4), extends mcore DSA indexer.
    "attn.indexer.wq_b.weight": "self_attention.indexer.linear_q_up_proj.weight",
    "attn.indexer.wq_b.scale": "self_attention.indexer.linear_q_up_proj.scale",
    "attn.indexer.weights_proj.weight": "self_attention.indexer.weights_proj.weight",
    "attn.indexer.compressor.ape": "self_attention.indexer.compressor.ape",
    "attn.indexer.compressor.wkv.weight": "self_attention.indexer.compressor.wkv.weight",
    "attn.indexer.compressor.wgate.weight": "self_attention.indexer.compressor.wgate.weight",
    "attn.indexer.compressor.norm.weight": "self_attention.indexer.compressor.norm.weight",
}

_MOE_FIXED_RENAMES: Dict[str, str] = {
    "ffn.gate.weight": "mlp.router.weight",
    "ffn.gate.bias": "mlp.router.bias",
    "ffn.gate.tid2eid": "mlp.router.tid2eid",
    "ffn.shared_experts.w1.weight": "mlp.shared_experts.linear_fc1_gate.weight",
    "ffn.shared_experts.w1.scale": "mlp.shared_experts.linear_fc1_gate.scale",
    "ffn.shared_experts.w3.weight": "mlp.shared_experts.linear_fc1_up.weight",
    "ffn.shared_experts.w3.scale": "mlp.shared_experts.linear_fc1_up.scale",
    "ffn.shared_experts.w2.weight": "mlp.shared_experts.linear_fc2.weight",
    "ffn.shared_experts.w2.scale": "mlp.shared_experts.linear_fc2.scale",
}

_EXPERT_W_TO_MEGATRON = {"w1": "linear_fc1_gate", "w2": "linear_fc2", "w3": "linear_fc1_up"}
_EXPERT_RE = re.compile(r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")

# Per-block HC: the fn / base rename to mapping_proj.weight / bias. The `scale`
# fan-out is handled separately (not a 1-to-1 rename).
_HC_BLOCK_SIMPLE_RENAMES: Dict[str, str] = {
    "hc_attn_fn": "hc_attn.mapping_proj.weight",
    "hc_attn_base": "hc_attn.bias",
    "hc_ffn_fn": "hc_ffn.mapping_proj.weight",
    "hc_ffn_base": "hc_ffn.bias",
}

# HF `hc_*_scale [3]` -> ordered list of 3 mcore scalar params.
_HC_SCALE_FANOUT: Dict[str, List[str]] = {
    "hc_attn_scale": ["hc_attn.alpha_pre", "hc_attn.alpha_post", "hc_attn.alpha_res"],
    "hc_ffn_scale": ["hc_ffn.alpha_pre", "hc_ffn.alpha_post", "hc_ffn.alpha_res"],
}

_LAYER_FIXED_RENAMES: Dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "pre_mlp_layernorm.weight",
    **_HC_BLOCK_SIMPLE_RENAMES,
}

_MTP_FIXED_RENAMES: Dict[str, str] = {
    "e_proj.weight": "e_proj.weight",
    "e_proj.scale": "e_proj.scale",
    "h_proj.weight": "h_proj.weight",
    "h_proj.scale": "h_proj.scale",
    "enorm.weight": "enorm.weight",
    "hnorm.weight": "hnorm.weight",
    "norm.weight": "final_layernorm.weight",
    "hc_head_fn": "hc_head_fn",
    "hc_head_base": "hc_head_base",
    "hc_head_scale": "hc_head_scale",
}

_ROOT_RENAMES: Dict[str, str] = {
    "embed.weight": "embedding.word_embeddings.weight",
    "norm.weight": "decoder.final_layernorm.weight",
    "head.weight": "output_layer.weight",
    "hc_head_fn": "decoder.hc_head_fn",
    "hc_head_base": "decoder.hc_head_base",
    "hc_head_scale": "decoder.hc_head_scale",
}


_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_MTP_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")


# ============================================================================
# Public API — import / export a state_dict.
#
# The mapping is not a pure rename: HC scale fans 1→3. We therefore operate on
# whole state_dicts (not single keys) so the structural ops live in one place.
# ============================================================================


@dataclass(frozen=True)
class TranslationEntry:
    """One row of the HF ↔ Megatron name translation table.

    `kind` is `"rename"` for 1-to-1 mappings and `"hc_scale_split"` when the
    same HF key fans out to multiple Megatron params (the 3 alphas).
    """

    hf: str
    megatron: str
    kind: str = "rename"  # "rename" | "hc_scale_split" | "hc_scale_join"


def import_hf_state_dict(hf_sd: Dict[str, torch.Tensor], args: DeepSeekV4ModelArgs) -> Dict[str, torch.Tensor]:
    """Translate an HF-format state_dict into the Megatron-mirror layout."""
    out: Dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        _translate_and_place(hf_key, tensor, out)
    _sanity_check_count(hf_sd, out, args, direction="hf->mg")
    return out


def export_hf_state_dict(mg_sd: Dict[str, torch.Tensor], args: DeepSeekV4ModelArgs) -> Dict[str, torch.Tensor]:
    """Inverse — produce an HF-format state_dict from a Megatron-mirror one."""
    out: Dict[str, torch.Tensor] = {}
    # Pre-build a set of mcore keys we'll need to JOIN back into hc_*_scale.
    for hf_key in _enumerate_hf_keys(args):
        _export_one(hf_key, mg_sd, out)
    _sanity_check_count(mg_sd, out, args, direction="mg->hf")
    return out


# --- implementation details -------------------------------------------------


def _translate_and_place(hf_key: str, tensor: torch.Tensor, out: Dict[str, torch.Tensor]) -> None:
    if hf_key in _ROOT_RENAMES:
        out[_ROOT_RENAMES[hf_key]] = tensor
        return

    m = _LAYER_RE.match(hf_key)
    if m is not None:
        layer_idx, suffix = m.group(1), m.group(2)
        prefix = f"decoder.layers.{layer_idx}"
        for mg_suffix, piece in _translate_block_key(suffix, tensor, is_mtp=False):
            out[f"{prefix}.{mg_suffix}"] = piece
        return

    m = _MTP_RE.match(hf_key)
    if m is not None:
        mtp_idx, suffix = m.group(1), m.group(2)
        prefix = f"mtp.layers.{mtp_idx}"
        for mg_suffix, piece in _translate_block_key(suffix, tensor, is_mtp=True):
            out[f"{prefix}.{mg_suffix}"] = piece
        return

    raise KeyError(f"Unrecognized DeepSeek-V4 HF key: {hf_key!r}")


def _translate_block_key(suffix: str, tensor: torch.Tensor, *, is_mtp: bool) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (megatron-suffix, tensor) pairs for a single HF block-relative key.

    Fanout: `hc_*_scale [3]` → three `(mcore_name, scalar)` pairs.
    """
    if is_mtp and suffix in _MTP_FIXED_RENAMES:
        yield _MTP_FIXED_RENAMES[suffix], tensor
        return

    for mg_suffix, piece in _translate_inner(suffix, tensor):
        yield (f"mtp_model_layer.{mg_suffix}" if is_mtp else mg_suffix, piece)


def _translate_inner(suffix: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    if suffix in _LAYER_FIXED_RENAMES:
        yield _LAYER_FIXED_RENAMES[suffix], tensor
        return
    if suffix in _ATTN_RENAMES:
        yield _ATTN_RENAMES[suffix], tensor
        return
    if suffix in _MOE_FIXED_RENAMES:
        yield _MOE_FIXED_RENAMES[suffix], tensor
        return
    if suffix in _HC_SCALE_FANOUT:
        names = _HC_SCALE_FANOUT[suffix]
        # Split tensor [3] into three scalar params of shape [1].
        if tensor.shape != (3,):
            raise ValueError(f"Expected HF {suffix} to be shape (3,), got {tuple(tensor.shape)}")
        for i, mg_name in enumerate(names):
            yield mg_name, tensor[i : i + 1].clone()
        return
    m = _EXPERT_RE.match(suffix)
    if m is not None:
        idx, w, t = m.group(1), m.group(2), m.group(3)
        yield f"mlp.experts.{idx}.{_EXPERT_W_TO_MEGATRON[w]}.{t}", tensor
        return
    raise KeyError(f"Unrecognized DeepSeek-V4 block-relative key: {suffix!r}")


def _export_one(hf_key: str, mg_sd: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> None:
    # Find the *set* of mcore keys that map to this HF key (1, or 3 for scale).
    mg_names = _hf_key_to_megatron_keys(hf_key)
    if len(mg_names) == 1:
        out[hf_key] = mg_sd[mg_names[0]]
    else:
        # hc_*_scale: concatenate three [1] tensors into [3].
        pieces = [mg_sd[name] for name in mg_names]
        out[hf_key] = torch.cat(pieces, dim=0)


def _hf_key_to_megatron_keys(hf_key: str) -> List[str]:
    if hf_key in _ROOT_RENAMES:
        return [_ROOT_RENAMES[hf_key]]
    m = _LAYER_RE.match(hf_key)
    if m is not None:
        layer_idx, suffix = m.group(1), m.group(2)
        prefix = f"decoder.layers.{layer_idx}"
        return [f"{prefix}.{s}" for s in _inner_mcore_suffixes(suffix, is_mtp=False)]
    m = _MTP_RE.match(hf_key)
    if m is not None:
        mtp_idx, suffix = m.group(1), m.group(2)
        prefix = f"mtp.layers.{mtp_idx}"
        return [f"{prefix}.{s}" for s in _inner_mcore_suffixes(suffix, is_mtp=True)]
    raise KeyError(f"Unrecognized DeepSeek-V4 HF key: {hf_key!r}")


def _inner_mcore_suffixes(suffix: str, *, is_mtp: bool) -> List[str]:
    if is_mtp and suffix in _MTP_FIXED_RENAMES:
        return [_MTP_FIXED_RENAMES[suffix]]
    base = _inner_mcore_block_suffixes(suffix)
    return [f"mtp_model_layer.{s}" if is_mtp else s for s in base]


def _inner_mcore_block_suffixes(suffix: str) -> List[str]:
    if suffix in _LAYER_FIXED_RENAMES:
        return [_LAYER_FIXED_RENAMES[suffix]]
    if suffix in _ATTN_RENAMES:
        return [_ATTN_RENAMES[suffix]]
    if suffix in _MOE_FIXED_RENAMES:
        return [_MOE_FIXED_RENAMES[suffix]]
    if suffix in _HC_SCALE_FANOUT:
        return list(_HC_SCALE_FANOUT[suffix])
    m = _EXPERT_RE.match(suffix)
    if m is not None:
        idx, w, t = m.group(1), m.group(2), m.group(3)
        return [f"mlp.experts.{idx}.{_EXPERT_W_TO_MEGATRON[w]}.{t}"]
    raise KeyError(f"Unrecognized DeepSeek-V4 block-relative key: {suffix!r}")


# ============================================================================
# Enumeration (used by tests; cheap to call)
# ============================================================================


def _enumerate_hf_keys(args: DeepSeekV4ModelArgs) -> Iterator[str]:
    yield "embed.weight"
    yield "norm.weight"
    yield "head.weight"
    yield "hc_head_fn"
    yield "hc_head_base"
    yield "hc_head_scale"
    for layer_idx in range(args.n_layers):
        for key in _enumerate_block_keys(layer_idx, args):
            yield f"layers.{layer_idx}.{key}"
    for i in range(args.n_mtp_layers):
        layer_id = args.n_layers + i
        for key in _enumerate_block_keys(layer_id, args):
            yield f"mtp.{i}.{key}"
        yield from (f"mtp.{i}.{k}" for k in _MTP_FIXED_RENAMES)


def _enumerate_block_keys(layer_id: int, args: DeepSeekV4ModelArgs) -> Iterator[str]:
    yield "attn_norm.weight"
    yield "ffn_norm.weight"
    yield "hc_attn_fn"
    yield "hc_attn_base"
    yield "hc_attn_scale"
    yield "hc_ffn_fn"
    yield "hc_ffn_base"
    yield "hc_ffn_scale"
    yield "attn.attn_sink"
    yield "attn.wq_a.weight"
    yield "attn.wq_a.scale"
    yield "attn.wq_b.weight"
    yield "attn.wq_b.scale"
    yield "attn.q_norm.weight"
    yield "attn.wkv.weight"
    yield "attn.wkv.scale"
    yield "attn.kv_norm.weight"
    yield "attn.wo_a.weight"
    yield "attn.wo_a.scale"
    yield "attn.wo_b.weight"
    yield "attn.wo_b.scale"
    compress_ratio = args.compress_ratios[layer_id]
    if compress_ratio:
        yield "attn.compressor.ape"
        yield "attn.compressor.wkv.weight"
        yield "attn.compressor.wgate.weight"
        yield "attn.compressor.norm.weight"
        if compress_ratio == 4:
            yield "attn.indexer.wq_b.weight"
            yield "attn.indexer.wq_b.scale"
            yield "attn.indexer.weights_proj.weight"
            yield "attn.indexer.compressor.ape"
            yield "attn.indexer.compressor.wkv.weight"
            yield "attn.indexer.compressor.wgate.weight"
            yield "attn.indexer.compressor.norm.weight"
    yield "ffn.gate.weight"
    if layer_id < args.n_hash_layers:
        yield "ffn.gate.tid2eid"
    else:
        yield "ffn.gate.bias"
    for i in range(args.n_routed_experts):
        for w in ("w1", "w2", "w3"):
            yield f"ffn.experts.{i}.{w}.weight"
            yield f"ffn.experts.{i}.{w}.scale"
    for w in ("w1", "w2", "w3"):
        yield f"ffn.shared_experts.{w}.weight"
        yield f"ffn.shared_experts.{w}.scale"


def _sanity_check_count(
    inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    args: DeepSeekV4ModelArgs,
    *,
    direction: str,
) -> None:
    # On HF->Mg, 3 HF scale params per HC-block expand to 3+3=6 mcore alphas,
    # plus 2 for fn/base renames (unchanged count for those). Net: +2 per HC
    # group per block. Track it precisely.
    expected_mg = _expected_megatron_key_count(args)
    expected_hf = sum(1 for _ in _enumerate_hf_keys(args))
    if direction == "hf->mg":
        if len(outputs) != expected_mg:
            raise AssertionError(f"hf->mg produced {len(outputs)} keys, expected {expected_mg}")
    else:
        if len(outputs) != expected_hf:
            raise AssertionError(f"mg->hf produced {len(outputs)} keys, expected {expected_hf}")


def _expected_megatron_key_count(args: DeepSeekV4ModelArgs) -> int:
    n_hf = sum(1 for _ in _enumerate_hf_keys(args))
    # Each block has 2 HC scales (attn, ffn) that fan 1→3, so +2 keys per scale.
    blocks = args.n_layers + args.n_mtp_layers
    hc_scale_fanout_extra = 2 * 2 * blocks  # 2 scales * (3-1 extra) = 4 extra per block
    return n_hf + hc_scale_fanout_extra


# ============================================================================
# Backward-compat helpers (used by earlier verification scripts)
# ============================================================================


def translate_hf_to_megatron(hf_key: str) -> str:
    """Single-key rename — returns the *first* mcore key for 1→N fanouts."""
    return _hf_key_to_megatron_keys(hf_key)[0]


def build_translation_table(args: DeepSeekV4ModelArgs):
    """Flat list of HF-key / megatron-key pairs (multi-entry for fanouts)."""
    entries: List[TranslationEntry] = []
    for hf_key in _enumerate_hf_keys(args):
        mg_keys = _hf_key_to_megatron_keys(hf_key)
        if len(mg_keys) == 1:
            entries.append(TranslationEntry(hf=hf_key, megatron=mg_keys[0]))
        else:
            for mg in mg_keys:
                entries.append(TranslationEntry(hf=hf_key, megatron=mg, kind="hc_scale_split"))
    return entries
