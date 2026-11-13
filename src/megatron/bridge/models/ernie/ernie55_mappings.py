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

"""Hugging Face <-> Megatron weight mappings for ERNIE 5.5 (``Ernie5_5_MoeForCausalLM``).

:func:`ernie55_mapping_registry` returns the :class:`MegatronMappingRegistry` a bridge's
``mapping_registry()`` returns; it covers every parameter and persistent buffer of
:class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` and every tensor of a
Hugging Face ERNIE 5.5 checkpoint. Export writes every tensor in its ERNIE 5.5 dtype, so a checkpoint that
stores the keep-in-fp32 tensors in F32 round-trips bitwise; one that stores them in bf16, as the released
checkpoints do, comes back with those tensors in F32 (values unchanged).
:class:`Ernie55WeightConversionMixin` gives a
:class:`~megatron.bridge.models.conversion.model_bridge.MegatronModelBridge` subclass this registry and a
strict import: the checkpoint must hold exactly the tensors the registry reads
(:func:`check_ernie55_checkpoint_keys`), where Bridge alone would ignore a tensor no mapping reads.

Layout. Hugging Face layer ``i`` (``model.layers.{i}``) is two hybrid layers:

* ``decoder.layers.{2i}``, its attention: ``H`` when ``compress_ratios[i] == 128`` (Megatron-Core's DSv4
  attention at compress ratio 128) or ``D`` when ``compress_ratios[i] == -2`` (the latent-MQA layer);
* ``decoder.layers.{2i+1}``, its MLP: a dense MLP for ``i < first_k_dense_replace``, MoE otherwise.

MTP depth ``k`` is Hugging Face layer ``num_hidden_layers + k``. ERNIE 5.5's MTP shares its body with
the decoder's last two hybrid layers (``mtp_shared_last_layer``; SD-MTP): the body's Parameters are the
decoder's, so ``named_parameters()`` and this registry see them once, under their decoder names, and
the checkpoint needs no body tensors (the released checkpoints carry a copy, which the import skips:
:func:`ernie55_mtp_body_copy_keys`). The MTP router's ``expert_bias`` is a buffer, owned by the MTP
layer (Hugging Face ``block_sparse_moe.e_score_correction_bias``). A non-shared MTP has no checkpoint
to validate its key layout against and is rejected.

Every hybrid layer is wrapped in a hyper-connection module, so its weights live under
``.inner_layer.`` and its mHC weights under ``.hyper_connection.`` (Hugging Face
``self_attention_hyper_connection`` / ``mlp_hyper_connection``).

Tensors that are not plain renames:

* ``mapping_proj.weight`` of every hyper-connection module and every ``hc_head_fn`` are stored
  transposed by Hugging Face (``[n C, n (n + 2)]`` and ``[n C, n]``).
* D layers: ``kv_b_proj`` (``[H (nope + v), C]``, per head ``W_UK`` rows then ``W_UV`` rows) is split
  into ``k_b_proj = W_UK^T`` per head (``[H C, nope]``) and ``v_b_proj = W_UV`` per head (``[H v, C]``),
  the two matrices the latent-MQA layer keeps (:func:`split_kv_b_proj`), and joined again on export
  (:func:`join_kv_b_proj`).
* D layers: Megatron-Core's absorbed MLA rotates adjacent RoPE channel pairs. A checkpoint with
  ``mqa_latent_rope_adjacent_pairing: true`` uses the same convention and maps one to one; without it
  the reference implementation rotates channel ``i`` with ``i + R/2``, so the ``R`` RoPE rows of
  ``q_b_proj`` (per head) and ``kv_a_proj_with_mqa`` are permuted, ``megatron[2i] = hf[i]`` and
  ``megatron[2i + 1] = hf[R/2 + i]`` (:func:`rope_row_permutation`). The indexer has no such switch.
* Gated MLPs (dense, shared expert, routed experts): ``linear_fc1`` is ``[gate; up]`` from
  ``gate_proj``/``up_proj`` (MoE: ``w1``/``w3``), and ``w2``/``down_proj`` is ``linear_fc2``. Grouped
  routed experts with Megatron-Core's ``moe_mlp_glu_interleave_size``, and a grouped shared expert
  (``use_grouped_gemm_for_shared_expert``) with ``moe_shared_expert_glu_interleave_size``, keep their
  fc1 rows block-interleaved instead (``[g_0:k, u_0:k, g_k:2k, ...]``); the conversion interleaves them
  on import and restores ``[gate; up]`` on export. ``SequentialMLP`` experts and dense MLPs never do.

Dtypes. In a Hugging Face ERNIE 5.5 checkpoint the router (both gate matrices, the per-expert scale and
``e_score_correction_bias``), every hyper-connection tensor, the ``hc_head_*`` contractions, the H-layer
``attn_sink`` and compressor ``ape``, the indexer ``k_norm``, the VHA factors and the latent-MoE
projections are F32 by the keep-in-fp32 contract, and everything else is in the checkpoint dtype
(``dtype`` / ``torch_dtype``, bf16). The released bf16 checkpoints store the contract tensors in bf16
too, and import widens them exactly (:func:`_check_hf_dtype`). Import rejects any other dtype mismatch,
such as a checkpoint-dtype tensor stored in F32, instead of changing dtype silently. Megatron keeps the
contract tensors in fp32, and loading one into a narrower parameter is an error; the other tensors load
into parameters of the model's dtype (an fp32 checkpoint into a bf16 model rounds them, as for any
model). Export casts every tensor to its checkpoint dtype. Bridge keeps the D-layer ``attn_sink`` in
fp32 while the checkpoint stores ``softmax_offset`` in the checkpoint dtype (bf16): imports widen it
exactly, and exports round it back.

Supported envelope (anything else raises): ``compress_ratios`` entries 128 and -2 only,
``hybrid_mla_attention="mqa_dsa"`` for -2 layers, ``hca_rope_type`` ``"rope"`` or ``"yarn"``, and SD-MTP. Bridge reads
``n_routed_experts``, ``first_k_dense_replace`` and ``n_shared_experts`` only; a config whose other
spellings of them (which the reference implementation also accepts) would give a different value is
rejected. ERNIE 5.5's attention runs at tensor-parallel size 1, so non-expert tensors move whole; routed
experts follow expert parallelism (Bridge renumbers local experts to global ones) and expert tensor
parallelism. Exporting from Megatron-FSDP or GTP-sharded D-layer ``k_b_proj``/``v_b_proj`` is not
supported.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn
from torch.distributed._tensor import DTensor

from megatron.bridge.models.conversion.gtp import _is_gtp_param
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    GatedMLPMapping,
    LocalHFParamSpec,
    MegatronParamMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.ernie.configuration_ernie55 import (
    ERNIE55_ARCHITECTURE,
    check_ernie55_reference_key_spellings,
)


__all__ = [
    "ERNIE55_HF_ARCHITECTURE",
    "HF_HCA_COMPRESS_RATIO",
    "HF_LATENT_MQA_COMPRESS_RATIO",
    "Ernie55DLayerDims",
    "Ernie55HFLayout",
    "Ernie55WeightConversionMixin",
    "check_ernie55_checkpoint_keys",
    "ernie55_hf_param_dtypes",
    "ernie55_mapping_registry",
    "ernie55_mtp_body_copy_keys",
    "ernie55_param_mappings",
    "join_kv_b_proj",
    "rope_row_permutation",
    "split_kv_b_proj",
]

ERNIE55_HF_ARCHITECTURE = ERNIE55_ARCHITECTURE
# Hugging Face ``compress_ratios`` codes of the two ERNIE 5.5 attention layers.
HF_HCA_COMPRESS_RATIO = 128
HF_LATENT_MQA_COMPRESS_RATIO = -2
_ATTENTION_SYMBOLS = {HF_HCA_COMPRESS_RATIO: "H", HF_LATENT_MQA_COMPRESS_RATIO: "D"}
# Hybrid-pattern symbols of the MLP half of a layer.
_DENSE_SYMBOL, _MOE_SYMBOL, _MTP_SEPARATOR = "-", "E", "/"
# Hybrid layer of an MTP depth's body that holds its MoE (the body repeats the decoder's last "DE"/"HE").
_MTP_BODY_MOE_INDEX = 1

_FP32 = torch.float32


def _hf_get(hf_config: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a Hugging Face config object or a plain dict."""
    if isinstance(hf_config, Mapping):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


def _hf_checkpoint_dtype(hf_config: Any) -> torch.dtype:
    """The checkpoint dtype of a Hugging Face config (``dtype`` or ``torch_dtype``; bf16 when unset)."""
    for key in ("dtype", "torch_dtype"):
        value = _hf_get(hf_config, key)
        if value is None:
            continue
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            dtype = getattr(torch, value.removeprefix("torch."), None)
            if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
                return dtype
        raise ValueError(f"ERNIE 5.5 conversion: unsupported Hugging Face {key}={value!r}.")
    return torch.bfloat16


@dataclasses.dataclass(frozen=True)
class Ernie55DLayerDims:
    """Geometry of the D-layer (latent MQA) K/V up-projection and RoPE (Hugging Face ``hybrid_mla_*``)."""

    num_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


@dataclasses.dataclass(frozen=True)
class Ernie55HFLayout:
    """What a Hugging Face ERNIE 5.5 config fixes about the checkpoint's tensors.

    Attributes:
        attention: ``"H"`` or ``"D"`` for every Hugging Face layer, MTP depths included
            (``num_hidden_layers + num_mtp_layers`` entries, from ``compress_ratios``).
        num_hidden_layers: Decoder layers.
        num_mtp_layers: MTP depths (``num_nextn_predict_layers``).
        first_k_dense: Leading decoder layers with a dense MLP (``first_k_dense_replace``); the others
            and every MTP depth are MoE.
        num_experts: Routed experts per MoE layer (``n_routed_experts``).
        shared_expert: Whether MoE layers have a shared expert (``n_shared_experts > 0``).
        moe_latent: Latent MoE (``moe_latent_size``): fp32 ``fc1/fc2_latent_proj``.
        moe_latent_norm: The latent norm (``latent_moe_use_norm``, Megatron-Core's ``fc2_norm``).
        d_dims: D-layer geometry, or None without D layers.
        rope_adjacent_pairing: ``mqa_latent_rope_adjacent_pairing`` (``rope_interleave`` when absent): D-layer
            RoPE rows are already in Megatron-Core's adjacent-pair order.
        dtype: Checkpoint dtype of every tensor that is not F32 by contract.
    """

    attention: tuple[str, ...]
    num_hidden_layers: int
    num_mtp_layers: int
    first_k_dense: int
    num_experts: int
    shared_expert: bool
    moe_latent: bool
    moe_latent_norm: bool
    d_dims: Ernie55DLayerDims | None
    rope_adjacent_pairing: bool
    dtype: torch.dtype

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "Ernie55HFLayout":
        """Parse and validate a Hugging Face ERNIE 5.5 config (a ``PretrainedConfig`` or a dict).

        Raises:
            ValueError: If there is no config, or it is outside the supported envelope (see the module
                docstring).
            NotImplementedError: For MTP depths that do not share the decoder's last layer.
        """

        def fail(message: str) -> None:
            raise ValueError(f"ERNIE 5.5 conversion: {message}")

        if hf_config is None:
            fail(
                "the Hugging Face config is not set. A bridge's mapping_registry() reads self.hf_config, which "
                "Bridge sets when it builds the conversion tasks of an import or export."
            )
        check_ernie55_reference_key_spellings(hf_config)
        num_hidden_layers = int(_hf_get(hf_config, "num_hidden_layers"))
        num_mtp_layers = int(_hf_get(hf_config, "num_nextn_predict_layers", 0) or 0)
        num_layers = num_hidden_layers + num_mtp_layers
        ratios = _hf_get(hf_config, "compress_ratios")
        if ratios is None or len(ratios) < num_layers:
            fail(
                f"compress_ratios must have an entry for every decoder layer and MTP depth ({num_layers}), "
                f"got {ratios!r}."
            )
        ratios = [int(ratio) for ratio in ratios[:num_layers]]
        unsupported = sorted({ratio for ratio in ratios if ratio not in _ATTENTION_SYMBOLS})
        if unsupported:
            fail(
                f"compress_ratios {unsupported} are not ERNIE 5.5 layers; only {HF_HCA_COMPRESS_RATIO} (H, heavily "
                f"compressed attention) and {HF_LATENT_MQA_COMPRESS_RATIO} (D, latent MQA) are supported."
            )
        attention = tuple(_ATTENTION_SYMBOLS[ratio] for ratio in ratios)
        if _hf_get(hf_config, "mla_dense_layers"):
            fail("mla_dense_layers is not supported; every compress_ratios -2 layer is latent MQA.")
        hca_rope_type = _hf_get(hf_config, "hca_rope_type", "rope") or "rope"
        if "H" in attention and hca_rope_type not in ("rope", "yarn"):
            fail(f"H layers use plain RoPE or YaRN; hca_rope_type={hca_rope_type!r} is not supported.")

        first_k_dense = int(_hf_get(hf_config, "first_k_dense_replace", 0) or 0)
        if not 0 <= first_k_dense <= num_hidden_layers:
            fail(f"first_k_dense_replace={first_k_dense} is outside [0, num_hidden_layers={num_hidden_layers}].")
        num_experts = int(_hf_get(hf_config, "n_routed_experts", 0) or 0)
        has_moe = first_k_dense < num_hidden_layers or num_mtp_layers > 0
        if has_moe and num_experts <= 0:
            fail("MoE layers need n_routed_experts > 0.")
        moe_latent = bool(_hf_get(hf_config, "moe_latent_size", None))

        if num_mtp_layers:
            if not _hf_get(hf_config, "mtp_shared_last_layer", False):
                raise NotImplementedError(
                    "ERNIE 5.5 conversion: only MTP depths that share the decoder's last layer "
                    "(mtp_shared_last_layer) are supported; there is no checkpoint to validate a separate MTP "
                    "body's key layout against."
                )
            if first_k_dense == num_hidden_layers:
                fail("an MTP depth shares the decoder's last layer, which must be MoE like the MTP body.")
            for depth in range(num_mtp_layers):
                if attention[num_hidden_layers + depth] != attention[num_hidden_layers - 1]:
                    fail(
                        f"MTP depth {depth} ({attention[num_hidden_layers + depth]}) must have the attention type of "
                        f"the decoder's last layer ({attention[num_hidden_layers - 1]}), whose parameters it shares."
                    )

        d_dims = None
        if "D" in attention:
            mla_attention = _hf_get(hf_config, "hybrid_mla_attention", "mha")
            if mla_attention != "mqa_dsa":
                fail(f"-2 layers are latent MQA: hybrid_mla_attention must be 'mqa_dsa', got {mla_attention!r}.")
            d_dims = Ernie55DLayerDims(
                num_heads=int(_hf_get(hf_config, "hybrid_mla_num_attention_heads")),
                kv_lora_rank=int(_hf_get(hf_config, "hybrid_mla_kv_lora_rank")),
                qk_nope_head_dim=int(_hf_get(hf_config, "hybrid_mla_qk_nope_head_dim")),
                qk_rope_head_dim=int(_hf_get(hf_config, "hybrid_mla_qk_rope_head_dim")),
                v_head_dim=int(_hf_get(hf_config, "hybrid_mla_v_head_dim")),
            )
            if d_dims.qk_rope_head_dim % 2:
                fail(f"hybrid_mla_qk_rope_head_dim must be even, got {d_dims.qk_rope_head_dim}.")

        return cls(
            attention=attention,
            num_hidden_layers=num_hidden_layers,
            num_mtp_layers=num_mtp_layers,
            first_k_dense=first_k_dense,
            num_experts=num_experts,
            shared_expert=int(_hf_get(hf_config, "n_shared_experts", 0) or 0) > 0,
            moe_latent=moe_latent,
            moe_latent_norm=moe_latent and bool(_hf_get(hf_config, "latent_moe_use_norm", False)),
            d_dims=d_dims,
            rope_adjacent_pairing=bool(
                _hf_get(hf_config, "mqa_latent_rope_adjacent_pairing")
                if _hf_get(hf_config, "mqa_latent_rope_adjacent_pairing") is not None
                else _hf_get(hf_config, "rope_interleave", False)
            ),
            dtype=_hf_checkpoint_dtype(hf_config),
        )

    def is_moe(self, hf_layer: int) -> bool:
        """Whether Hugging Face layer ``hf_layer`` (an MTP depth included) has an MoE MLP."""
        return hf_layer >= self.first_k_dense

    @property
    def hybrid_layer_pattern(self) -> str:
        """Megatron-Core hybrid pattern of the model, MTP depths after ``/`` (e.g. ``"H-DEHEDE/DE"``)."""

        def layer(hf_layer: int) -> str:
            return self.attention[hf_layer] + (_MOE_SYMBOL if self.is_moe(hf_layer) else _DENSE_SYMBOL)

        main = "".join(layer(i) for i in range(self.num_hidden_layers))
        mtp = [layer(self.num_hidden_layers + depth) for depth in range(self.num_mtp_layers)]
        return _MTP_SEPARATOR.join([main, *mtp])


# ---------------------------------------------------------------------------------------------
# Tensor transforms
# ---------------------------------------------------------------------------------------------


def split_kv_b_proj(kv_b_proj: torch.Tensor, dims: Ernie55DLayerDims) -> tuple[torch.Tensor, torch.Tensor]:
    """``(k_b_proj, v_b_proj)`` of a D layer from Hugging Face ``kv_b_proj``.

    ``kv_b_proj`` is ``[H (nope + v), C]``; per head its first ``nope`` rows are ``W_UK`` and the next
    ``v`` rows ``W_UV``. ``k_b_proj`` is ``W_UK^T`` per head, ``[H C, nope]``, and ``v_b_proj`` is
    ``W_UV`` per head, ``[H v, C]``.

    Raises:
        ValueError: If ``kv_b_proj`` does not have the shape ``dims`` gives.
    """
    heads, latent, nope, value = dims.num_heads, dims.kv_lora_rank, dims.qk_nope_head_dim, dims.v_head_dim
    expected = (heads * (nope + value), latent)
    if tuple(kv_b_proj.shape) != expected:
        raise ValueError(f"ERNIE 5.5 kv_b_proj must be {expected} for {dims}, got {tuple(kv_b_proj.shape)}.")
    per_head = kv_b_proj.reshape(heads, nope + value, latent)
    k_b_proj = per_head[:, :nope].transpose(1, 2).reshape(heads * latent, nope).contiguous()
    v_b_proj = per_head[:, nope:].reshape(heads * value, latent).contiguous()
    return k_b_proj, v_b_proj


def join_kv_b_proj(k_b_proj: torch.Tensor, v_b_proj: torch.Tensor, dims: Ernie55DLayerDims) -> torch.Tensor:
    """Hugging Face ``kv_b_proj`` from a D layer's ``k_b_proj`` and ``v_b_proj`` (inverse of :func:`split_kv_b_proj`).

    Raises:
        ValueError: If the tensors do not have the shapes ``dims`` gives.
    """
    heads, latent, nope, value = dims.num_heads, dims.kv_lora_rank, dims.qk_nope_head_dim, dims.v_head_dim
    if tuple(k_b_proj.shape) != (heads * latent, nope) or tuple(v_b_proj.shape) != (heads * value, latent):
        raise ValueError(
            f"ERNIE 5.5 k_b_proj/v_b_proj must be {(heads * latent, nope)}/{(heads * value, latent)} for {dims}, "
            f"got {tuple(k_b_proj.shape)}/{tuple(v_b_proj.shape)}."
        )
    w_uk = k_b_proj.reshape(heads, latent, nope).transpose(1, 2)
    w_uv = v_b_proj.reshape(heads, value, latent)
    return torch.cat([w_uk, w_uv], dim=1).reshape(heads * (nope + value), latent).contiguous()


def rope_row_permutation(rope_dim: int) -> torch.Tensor:
    """``perm`` with ``megatron_rows = hf_rows[perm]`` for D-layer RoPE rows without adjacent pairing.

    ``perm[2i] = i`` and ``perm[2i + 1] = rope_dim / 2 + i``: Megatron-Core's de-interleave
    ``cat(x[0::2], x[1::2])`` then restores the Hugging Face order before the half rotation.
    """
    if rope_dim % 2:
        raise ValueError(f"rope_dim must be even, got {rope_dim}.")
    half = rope_dim // 2
    perm = torch.empty(rope_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half) + half
    return perm


def _permute_rope_rows(weight: torch.Tensor, *, block: int, offset: int, perm: torch.Tensor) -> torch.Tensor:
    """Rows ``offset .. offset + len(perm)`` of every ``block``-row block reordered as ``rows[perm]``."""
    if weight.shape[0] % block:
        raise ValueError(f"ERNIE 5.5 RoPE rows: {weight.shape[0]} rows are not a multiple of the block {block}.")
    blocks = weight.reshape(weight.shape[0] // block, block, *weight.shape[1:]).clone()
    window = blocks[:, offset : offset + perm.numel()]
    blocks[:, offset : offset + perm.numel()] = window[:, perm.to(weight.device)]
    return blocks.reshape(weight.shape)


def _glu_interleave(weight: torch.Tensor, size: int) -> torch.Tensor:
    """``[gate; up]`` rows to Megatron-Core's block-interleaved ``[g_0:k, u_0:k, g_k:2k, u_k:2k, ...]``.

    Same layout as the GLU interleave of Bridge's checkpoint loader (``_interleave_glu_tensor``) and
    the one ``TEGroupedMLP._remove_glu_interleaving`` undoes on the fc1 output.
    """
    rows = weight.shape[0]
    if rows % (2 * size):
        raise ValueError(f"ERNIE 5.5 GLU interleave: {rows} fc1 rows are not a multiple of 2 x {size}.")
    return weight.reshape(2, rows // (2 * size), size, *weight.shape[1:]).transpose(0, 1).reshape(weight.shape)


def _glu_deinterleave(weight: torch.Tensor, size: int) -> torch.Tensor:
    """Inverse of :func:`_glu_interleave`."""
    rows = weight.shape[0]
    if rows % (2 * size):
        raise ValueError(f"ERNIE 5.5 GLU interleave: {rows} fc1 rows are not a multiple of 2 x {size}.")
    return weight.reshape(rows // (2 * size), 2, size, *weight.shape[1:]).transpose(0, 1).reshape(weight.shape)


# ---------------------------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------------------------


def _to_hf_dtype(converted: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {name: (tensor.to(dtype) if tensor.is_floating_point() else tensor) for name, tensor in converted.items()}


def _check_hf_dtype(hf_param: str, hf_weight: torch.Tensor, hf_dtype: torch.dtype) -> None:
    """A checkpoint tensor must have its ERNIE 5.5 dtype (the export dtype ``hf_dtype``), or be a 16-bit copy
    of an F32 one.

    The released bf16 checkpoints store the keep-in-fp32 tensors in bf16 too; the import widens them, exactly,
    into their fp32 parameters. A tensor of the checkpoint dtype stored in F32 would be rounded instead, which
    is rejected.

    Raises:
        TypeError: If ``hf_weight`` is neither ``hf_dtype`` nor a 16-bit float for an F32 ``hf_dtype``.
    """
    if hf_dtype == torch.float32 and hf_weight.dtype in (torch.bfloat16, torch.float16):
        return
    if hf_weight.dtype != hf_dtype:
        raise TypeError(
            f"ERNIE 5.5 conversion: {hf_param} is {hf_weight.dtype} in the checkpoint, but ERNIE 5.5 stores it in "
            f"{hf_dtype}: F32 (or 16-bit) for the keep-in-fp32 tensors, the config's dtype for the others."
        )


class _Ernie55TensorMapping(ReplicatedMapping):
    """One Hugging Face tensor <-> one non-expert Megatron parameter or buffer, moved whole.

    ERNIE 5.5's attention runs at tensor-parallel size 1 (Megatron-Core's DSv4 attention asserts it), so
    every non-expert tensor is replicated across the tensor-parallel group. ``hf_dtype`` is the tensor's
    ERNIE 5.5 dtype: export casts to it, and import accepts it (or a 16-bit copy of an F32 tensor, see
    :func:`_check_hf_dtype`). ``keep_fp32`` marks a tensor of the keep-in-fp32 contract (F32 on export),
    which must load into an fp32 Megatron tensor.
    ``transpose`` maps a 2-D tensor that Hugging Face stores transposed.
    """

    # Subclasses whose Megatron tensor is not stored as the Hugging Face tensor is (permuted, split).
    _changes_layout = False

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        *,
        hf_dtype: torch.dtype,
        keep_fp32: bool = False,
        transpose: bool = False,
    ):
        if keep_fp32 and hf_dtype != _FP32:
            raise ValueError(f"ERNIE 5.5 mapping {hf_param}: a keep-in-fp32 tensor is F32, not {hf_dtype}.")
        super().__init__(megatron_param, hf_param)
        self.hf_dtype = hf_dtype
        self.keep_fp32 = keep_fp32
        self.transpose = transpose

    def local_hf_param_specs(self, global_param_name: str | None = None) -> tuple[LocalHFParamSpec, ...]:
        """Canonical local Hugging Face views; a transposed, permuted or split tensor has none."""
        if self.transpose or self._changes_layout:
            return ()
        return super().local_hf_param_specs(global_param_name)

    def resolve(self, captures: tuple[str, ...]) -> "MegatronParamMapping":
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(
            megatron_param, hf_param, hf_dtype=self.hf_dtype, keep_fp32=self.keep_fp32, transpose=self.transpose
        )

    def _check_target(self, megatron_module: nn.Module) -> None:
        """A keep-in-fp32 tensor must land in an fp32 Megatron tensor: a narrower one would round it."""
        if not self.keep_fp32:
            return
        target = getattr(megatron_module, self.megatron_param.rpartition(".")[2])
        if target.dtype != _FP32:
            raise TypeError(
                f"ERNIE 5.5 conversion: {self.megatron_param} is {target.dtype}, but {self.hf_param} is an F32 "
                "checkpoint tensor that ERNIE 5.5 keeps in fp32; loading it would round it."
            )

    def _to_megatron_layout(self, hf_weights: torch.Tensor) -> torch.Tensor:
        if not self.transpose:
            return hf_weights
        if hf_weights.ndim != 2:
            raise ValueError(f"ERNIE 5.5 conversion: {self.hf_param} must be 2-D, got {tuple(hf_weights.shape)}.")
        return hf_weights.t().contiguous()

    def _to_hf_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.t().contiguous() if self.transpose else tensor

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        _check_hf_dtype(self.hf_param, hf_weights, self.hf_dtype)
        self._check_target(megatron_module)
        return super().hf_to_megatron(self._to_megatron_layout(hf_weights), megatron_module)

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None
    ) -> dict[str, torch.Tensor]:
        converted = super().megatron_to_hf(megatron_weights, megatron_module)
        return _to_hf_dtype({name: self._to_hf_layout(tensor) for name, tensor in converted.items()}, self.hf_dtype)


class _Ernie55RopeRowsMapping(_Ernie55TensorMapping):
    """A D-layer projection whose Hugging Face RoPE rows are in half-rotation order (no adjacent pairing).

    The ``rope_dim`` rows at ``offset`` of every ``block``-row block are permuted with
    :func:`rope_row_permutation` on import and back on export: ``q_b_proj`` has one block per head
    (``nope + rope`` rows), ``kv_a_proj_with_mqa`` one block (``C + rope`` rows).
    """

    _changes_layout = True

    def __init__(
        self, megatron_param: str, hf_param: str, *, hf_dtype: torch.dtype, block: int, offset: int, rope_dim: int
    ):
        super().__init__(megatron_param, hf_param, hf_dtype=hf_dtype)
        self.block, self.offset, self.rope_dim = block, offset, rope_dim
        self._perm = rope_row_permutation(rope_dim)
        self._inverse = torch.argsort(self._perm)

    def resolve(self, captures: tuple[str, ...]) -> "MegatronParamMapping":
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(
            megatron_param,
            hf_param,
            hf_dtype=self.hf_dtype,
            block=self.block,
            offset=self.offset,
            rope_dim=self.rope_dim,
        )

    def _to_megatron_layout(self, hf_weights: torch.Tensor) -> torch.Tensor:
        return _permute_rope_rows(hf_weights, block=self.block, offset=self.offset, perm=self._perm)

    def _to_hf_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        return _permute_rope_rows(tensor, block=self.block, offset=self.offset, perm=self._inverse)


class _Ernie55KVUpMapping(_Ernie55TensorMapping):
    """Hugging Face ``kv_b_proj`` <-> a D layer's ``k_b_proj`` (``part="k"``) or ``v_b_proj`` (``part="v"``).

    Both Megatron tensors read the one checkpoint tensor (:func:`split_kv_b_proj`). On export the ``k``
    mapping emits the joined ``kv_b_proj`` from both tensors of its module (:func:`join_kv_b_proj`),
    broadcasting them from the pipeline stage that owns the layer, and the ``v`` mapping emits nothing.
    An export that reports Megatron source names (``with_megatron_names``) therefore attributes
    ``kv_b_proj`` to ``k_b_proj`` alone.

    Bridge prepares the ``k_b_proj`` it passes in (gathering a Megatron-FSDP DTensor or a GTP shard), but
    the mapping reads ``v_b_proj`` from the module itself, so the export rejects a sharded ``v_b_proj``
    (Megatron-FSDP or GTP) instead of joining a shard. The ERNIE 5.5 setups use neither; with SD-MTP,
    Megatron-FSDP is rejected anyway (``validate_ernie55_mtp_sharing``).
    """

    _changes_layout = True

    def __init__(
        self, megatron_param: str, hf_param: str, *, hf_dtype: torch.dtype, part: str, dims: Ernie55DLayerDims
    ):
        if part not in ("k", "v") or not megatron_param.endswith(f"{part}_b_proj"):
            raise ValueError(f"ERNIE 5.5 K/V up-projection mapping: bad part {part!r} for {megatron_param}.")
        super().__init__(megatron_param, hf_param, hf_dtype=hf_dtype)
        self.part, self.dims = part, dims

    def resolve(self, captures: tuple[str, ...]) -> "MegatronParamMapping":
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param, hf_dtype=self.hf_dtype, part=self.part, dims=self.dims)

    def _to_megatron_layout(self, hf_weights: torch.Tensor) -> torch.Tensor:
        k_b_proj, v_b_proj = split_kv_b_proj(hf_weights, self.dims)
        return k_b_proj if self.part == "k" else v_b_proj

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None
    ) -> dict[str, torch.Tensor]:
        if self.part == "v":
            return {}
        local_v = None
        if megatron_module is not None:
            v_param = megatron_module.v_b_proj
            if isinstance(v_param, DTensor) or _is_gtp_param(v_param):
                raise NotImplementedError(
                    f"ERNIE 5.5 conversion: cannot export {self.hf_param} from a sharded v_b_proj (Megatron-FSDP or "
                    "GTP); export from a model without them."
                )
            local_v = v_param.detach()
        k_b_proj = self.broadcast_from_pp_rank(megatron_weights, cache_key=f"{self.hf_param}:k_b_proj")
        v_b_proj = self.broadcast_from_pp_rank(local_v, cache_key=f"{self.hf_param}:v_b_proj")
        if k_b_proj is None:
            return {}
        joined = join_kv_b_proj(self.maybe_dequantize(k_b_proj), self.maybe_dequantize(v_b_proj), self.dims)
        return _to_hf_dtype({str(self.hf_param): joined}, self.hf_dtype)


def _glu_interleave_size(config: Any, glu_interleave: str | None) -> int | None:
    """Megatron-Core's GLU block-interleave size of an fc1, or None when its rows are ``[gate; up]``.

    Only the grouped routed experts (``TEGroupedMLP``, ``moe_mlp_glu_interleave_size``) and the grouped
    shared expert (``use_grouped_gemm_for_shared_expert`` with ``moe_shared_expert_glu_interleave_size``)
    read interleaved rows; ``SequentialMLP`` experts and dense MLPs always use ``[gate; up]``.
    """
    if glu_interleave == "routed":
        return getattr(config, "moe_mlp_glu_interleave_size", None)
    if glu_interleave == "shared" and getattr(config, "use_grouped_gemm_for_shared_expert", False):
        return getattr(config, "moe_shared_expert_glu_interleave_size", None)
    return None


class _Ernie55GatedMLPMapping(GatedMLPMapping):
    """``linear_fc1 = [gate; up]`` of a dense MLP, the shared expert or a routed expert, in the HF dtype.

    ``glu_interleave`` (``"routed"`` or ``"shared"``) names the fc1 whose rows Megatron-Core may keep
    block-interleaved (:func:`_glu_interleave_size`); when the model config enables that, the local fc1
    rows are interleaved on import and restored on export, as Bridge's checkpoint loader does. Tensor-
    and expert-parallel distribution is :class:`GatedMLPMapping`'s. Import requires ``gate`` and ``up``
    to be ``hf_dtype``.
    """

    def __init__(
        self, megatron_param: str, gate: str, up: str, *, hf_dtype: torch.dtype, glu_interleave: str | None = None
    ):
        if glu_interleave not in (None, "routed", "shared"):
            raise ValueError(f"ERNIE 5.5 gated MLP mapping: unknown glu_interleave {glu_interleave!r}.")
        super().__init__(megatron_param, gate, up)
        self.hf_dtype = hf_dtype
        self.glu_interleave = glu_interleave

    def resolve(self, captures: tuple[str, ...]) -> "MegatronParamMapping":
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(
            megatron_param,
            hf_param["gate"],
            hf_param["up"],
            hf_dtype=self.hf_dtype,
            glu_interleave=self.glu_interleave,
        )

    def _interleave_size(self, megatron_module: nn.Module | None) -> int | None:
        if self.glu_interleave is None or megatron_module is None:
            return None
        return _glu_interleave_size(self._get_config(megatron_module), self.glu_interleave)

    def hf_to_megatron(self, hf_weights: dict[str, torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        for part, hf_weight in hf_weights.items():
            _check_hf_dtype(self.hf_param[part], hf_weight, self.hf_dtype)
        local = super().hf_to_megatron(hf_weights, megatron_module)
        size = self._interleave_size(megatron_module)
        return _glu_interleave(local, size) if size else local

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None
    ) -> dict[str, torch.Tensor]:
        size = self._interleave_size(megatron_module)
        if size and megatron_weights is not None:
            megatron_weights = _glu_deinterleave(self.maybe_dequantize(megatron_weights), size)
        return _to_hf_dtype(super().megatron_to_hf(megatron_weights, megatron_module), self.hf_dtype)

    def local_hf_param_specs(self, global_param_name: str | None = None) -> tuple[LocalHFParamSpec, ...]:
        """Canonical local gate/up views; none for an fc1 that Megatron-Core may keep GLU-interleaved.

        Whether those rows are interleaved depends on the model config, which this method does not see,
        so a ``glu_interleave`` mapping reports no views and Bridge converts the tensor normally.
        """
        if self.glu_interleave is not None:
            return ()
        return super().local_hf_param_specs(global_param_name)


class _Ernie55ExpertFc2Mapping(RowParallelMapping):
    """A routed expert's ``linear_fc2`` (row parallel over expert tensor parallelism), in the HF dtype.

    Import requires the checkpoint tensor to be ``hf_dtype``.
    """

    def __init__(self, megatron_param: str, hf_param: str, *, hf_dtype: torch.dtype):
        super().__init__(megatron_param, hf_param)
        self.hf_dtype = hf_dtype

    def resolve(self, captures: tuple[str, ...]) -> "MegatronParamMapping":
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param, hf_dtype=self.hf_dtype)

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        _check_hf_dtype(self.hf_param, hf_weights, self.hf_dtype)
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None
    ) -> dict[str, torch.Tensor]:
        return _to_hf_dtype(super().megatron_to_hf(megatron_weights, megatron_module), self.hf_dtype)


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------


def ernie55_param_mappings(layout: Ernie55HFLayout) -> list[MegatronParamMapping]:
    """Every ERNIE 5.5 mapping for ``layout``, in model order (see the module docstring).

    Args:
        layout: The parsed Hugging Face config (:meth:`Ernie55HFLayout.from_hf_config`).

    Returns:
        The mappings, ready for :class:`MegatronMappingRegistry`.
    """
    model_dtype = layout.dtype

    def tensor(megatron_param: str, hf_param: str, *, fp32: bool = False, transpose: bool = False):
        """A plain mapping; ``fp32`` marks a keep-in-fp32 tensor (F32 on export, fp32 in Megatron)."""
        return _Ernie55TensorMapping(
            megatron_param, hf_param, hf_dtype=_FP32 if fp32 else model_dtype, keep_fp32=fp32, transpose=transpose
        )

    def hc_head(megatron_prefix: str, hf_prefix: str) -> list:
        return [
            tensor(f"{megatron_prefix}.hc_head_fn", f"{hf_prefix}.hc_head_fn", fp32=True, transpose=True),
            tensor(f"{megatron_prefix}.hc_head_base", f"{hf_prefix}.hc_head_base", fp32=True),
            tensor(f"{megatron_prefix}.hc_head_scale", f"{hf_prefix}.hc_head_scale", fp32=True),
        ]

    def hyper_connection(megatron_layer: str, hf_prefix: str) -> list:
        hc = f"{megatron_layer}.hyper_connection"
        return [
            tensor(f"{hc}.mapping_proj.weight", f"{hf_prefix}.mapping_proj.weight", fp32=True, transpose=True),
            *(
                tensor(f"{hc}.{name}", f"{hf_prefix}.{name}", fp32=True)
                for name in ("bias", "alpha_pre", "alpha_post", "alpha_res")
            ),
        ]

    def h_attention(megatron_layer: str, hf_attn: str) -> list:
        attention = f"{megatron_layer}.inner_layer.self_attention"
        core = f"{attention}.core_attention"
        verbatim = [
            "linear_q_down_proj.weight",
            "q_layernorm.weight",
            "linear_q_up_proj.weight",
            "linear_kv_proj.weight",
            "kv_layernorm.weight",
            "linear_o_group_proj",
            "core_attention.compressor.linear_wkv.weight",
            "core_attention.compressor.linear_wgate.weight",
            "core_attention.compressor.norm.weight",
        ]
        return [
            *(tensor(f"{attention}.{name}", f"{hf_attn}.{name}") for name in verbatim),
            tensor(f"{core}.compressor.ape", f"{hf_attn}.core_attention.compressor.ape", fp32=True),
            tensor(f"{core}.attn_sink", f"{hf_attn}.core_attention.attn_sink", fp32=True),
            tensor(f"{core}.vha_postmix_u", f"{hf_attn}.vha_postmix_U", fp32=True),
            tensor(f"{core}.vha_postmix_v", f"{hf_attn}.vha_postmix_V", fp32=True),
            tensor(f"{attention}.linear_output_gate.weight", f"{hf_attn}.gate_proj.weight"),
            tensor(f"{attention}.linear_proj.weight", f"{hf_attn}.o_proj.weight"),
        ]

    def d_attention(megatron_layer: str, hf_attn: str) -> list:
        dims = layout.d_dims
        attention = f"{megatron_layer}.inner_layer.self_attention"
        core = f"{attention}.core_attention"
        indexer = f"{core}.indexer"
        rope = dims.qk_rope_head_dim
        if layout.rope_adjacent_pairing:
            q_up = tensor(f"{attention}.linear_q_up_proj.weight", f"{hf_attn}.q_b_proj.weight")
            kv_down = tensor(f"{attention}.linear_kv_down_proj.weight", f"{hf_attn}.kv_a_proj_with_mqa.weight")
        else:
            q_up = _Ernie55RopeRowsMapping(
                f"{attention}.linear_q_up_proj.weight",
                f"{hf_attn}.q_b_proj.weight",
                hf_dtype=model_dtype,
                block=dims.qk_nope_head_dim + rope,
                offset=dims.qk_nope_head_dim,
                rope_dim=rope,
            )
            kv_down = _Ernie55RopeRowsMapping(
                f"{attention}.linear_kv_down_proj.weight",
                f"{hf_attn}.kv_a_proj_with_mqa.weight",
                hf_dtype=model_dtype,
                block=dims.kv_lora_rank + rope,
                offset=dims.kv_lora_rank,
                rope_dim=rope,
            )
        kv_b_proj = f"{hf_attn}.kv_b_proj.weight"
        return [
            tensor(f"{attention}.linear_q_down_proj.weight", f"{hf_attn}.q_a_proj.weight"),
            tensor(f"{attention}.q_layernorm.weight", f"{hf_attn}.q_a_layernorm.weight"),
            q_up,
            kv_down,
            tensor(f"{attention}.kv_layernorm.weight", f"{hf_attn}.kv_a_layernorm.weight"),
            _Ernie55KVUpMapping(f"{attention}.k_b_proj", kv_b_proj, hf_dtype=model_dtype, part="k", dims=dims),
            _Ernie55KVUpMapping(f"{attention}.v_b_proj", kv_b_proj, hf_dtype=model_dtype, part="v", dims=dims),
            tensor(f"{attention}.linear_proj.weight", f"{hf_attn}.o_proj.weight"),
            tensor(f"{core}.linear_output_gate.weight", f"{hf_attn}.gate_proj.weight"),
            tensor(f"{core}.vha_postmix_u", f"{hf_attn}.vha_postmix_U", fp32=True),
            tensor(f"{core}.vha_postmix_v", f"{hf_attn}.vha_postmix_V", fp32=True),
            # fp32 in Bridge, bf16 (the checkpoint dtype) in the checkpoint.
            tensor(f"{core}.attn_sink", f"{hf_attn}.core_attention.softmax_offset"),
            tensor(f"{indexer}.linear_wq_b.weight", f"{hf_attn}.core_attention.indexer.wq_b.weight"),
            tensor(f"{indexer}.linear_wk.weight", f"{hf_attn}.core_attention.indexer.wk.weight"),
            tensor(f"{indexer}.k_norm.weight", f"{hf_attn}.core_attention.indexer.k_norm.weight", fp32=True),
            tensor(f"{indexer}.k_norm.bias", f"{hf_attn}.core_attention.indexer.k_norm.bias", fp32=True),
            tensor(f"{indexer}.linear_weights_proj.weight", f"{hf_attn}.core_attention.indexer.weights_proj.weight"),
        ]

    def dense_mlp(megatron_layer: str, hf_mlp: str) -> list:
        mlp = f"{megatron_layer}.inner_layer.mlp"
        return [
            _Ernie55GatedMLPMapping(
                f"{mlp}.linear_fc1.weight",
                f"{hf_mlp}.gate_proj.weight",
                f"{hf_mlp}.up_proj.weight",
                hf_dtype=model_dtype,
            ),
            tensor(f"{mlp}.linear_fc2.weight", f"{hf_mlp}.down_proj.weight"),
        ]

    def router_bias(megatron_mlp: str, hf_moe: str):
        return tensor(f"{megatron_mlp}.router.expert_bias", f"{hf_moe}.e_score_correction_bias", fp32=True)

    def moe(megatron_layer: str, hf_moe: str) -> list:
        mlp = f"{megatron_layer}.inner_layer.mlp"
        experts = f"{hf_moe}.experts.*"
        out = [
            tensor(f"{mlp}.router.weight", f"{hf_moe}.gate.weight", fp32=True),
            tensor(f"{mlp}.router.weight_1", f"{hf_moe}.gate.weight_1", fp32=True),
            tensor(
                f"{mlp}.router.routed_scaling_factor_param", f"{hf_moe}.gate.routed_scaling_factor_param", fp32=True
            ),
            router_bias(mlp, hf_moe),
        ]
        # Grouped (TEGroupedMLP) and sequential (SequentialMLP) routed experts; only the grouped ones may
        # keep their fc1 rows GLU-interleaved.
        for fc1, fc2, glu_interleave in (
            (f"{mlp}.experts.linear_fc1.weight*", f"{mlp}.experts.linear_fc2.weight*", "routed"),
            (
                f"{mlp}.experts.local_experts.*.linear_fc1.weight",
                f"{mlp}.experts.local_experts.*.linear_fc2.weight",
                None,
            ),
        ):
            out += [
                _Ernie55GatedMLPMapping(
                    fc1,
                    f"{experts}.w1.weight",
                    f"{experts}.w3.weight",
                    hf_dtype=model_dtype,
                    glu_interleave=glu_interleave,
                ),
                _Ernie55ExpertFc2Mapping(fc2, f"{experts}.w2.weight", hf_dtype=model_dtype),
            ]
        if layout.shared_expert:
            out += [
                _Ernie55GatedMLPMapping(
                    f"{mlp}.shared_experts.linear_fc1.weight",
                    f"{hf_moe}.shared_experts.w1.weight",
                    f"{hf_moe}.shared_experts.w3.weight",
                    hf_dtype=model_dtype,
                    glu_interleave="shared",
                ),
                tensor(f"{mlp}.shared_experts.linear_fc2.weight", f"{hf_moe}.shared_experts.w2.weight"),
            ]
        if layout.moe_latent:
            out += [
                tensor(f"{mlp}.fc1_latent_proj.weight", f"{hf_moe}.fc1_latent_proj.weight", fp32=True),
                tensor(f"{mlp}.fc2_latent_proj.weight", f"{hf_moe}.fc2_latent_proj.weight", fp32=True),
            ]
            if layout.moe_latent_norm:
                out.append(tensor(f"{mlp}.fc2_norm.weight", f"{hf_moe}.latent_norm.weight"))
        return out

    mappings: list[MegatronParamMapping] = [
        tensor("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        tensor("decoder.final_norm.weight", "model.norm.weight"),
        *hc_head("decoder", "model.mhc_contract"),
        tensor("output_layer.weight", "lm_head.weight"),
        tensor("output_layer.multimax_ranges", "model.lm_head.multimax_ranges"),
        tensor("output_layer.multimax_ts", "model.lm_head.multimax_ts"),
    ]
    for hf_layer in range(layout.num_hidden_layers):
        hf = f"model.layers.{hf_layer}"
        attention_layer, mlp_layer = f"decoder.layers.{2 * hf_layer}", f"decoder.layers.{2 * hf_layer + 1}"
        mappings += hyper_connection(attention_layer, f"{hf}.self_attention_hyper_connection")
        mappings.append(
            tensor(f"{attention_layer}.inner_layer.input_layernorm.weight", f"{hf}.input_layernorm.weight")
        )
        if layout.attention[hf_layer] == "H":
            mappings += h_attention(attention_layer, f"{hf}.self_attn")
        else:
            mappings += d_attention(attention_layer, f"{hf}.self_attn")
        mappings += hyper_connection(mlp_layer, f"{hf}.mlp_hyper_connection")
        mappings.append(
            tensor(f"{mlp_layer}.inner_layer.pre_mlp_layernorm.weight", f"{hf}.post_attention_layernorm.weight")
        )
        if layout.is_moe(hf_layer):
            mappings += moe(mlp_layer, f"{hf}.block_sparse_moe")
        else:
            mappings += dense_mlp(mlp_layer, f"{hf}.mlp")
    for depth in range(layout.num_mtp_layers):
        hf = f"model.layers.{layout.num_hidden_layers + depth}"
        mtp = f"mtp.layers.{depth}"
        body_mlp = f"{mtp}.mtp_model_layer.layers.{_MTP_BODY_MOE_INDEX}.inner_layer.mlp"
        mappings += [
            tensor(f"{mtp}.enorm.weight", f"{hf}.enorm.weight"),
            tensor(f"{mtp}.hnorm.weight", f"{hf}.hnorm.weight"),
            tensor(f"{mtp}.e_proj.weight", f"{hf}.e_proj.weight"),
            tensor(f"{mtp}.h_proj.weight", f"{hf}.h_proj.weight"),
            *hc_head(mtp, hf),
            tensor(f"{mtp}.final_layernorm.weight", f"{hf}.shared_head.norm.weight"),
            # The body's Parameters alias the decoder's last layers; its router buffers are its own.
            router_bias(body_mlp, f"{hf}.block_sparse_moe"),
        ]
    return mappings


def ernie55_mapping_registry(hf_config: Any) -> MegatronMappingRegistry:
    """The ERNIE 5.5 mapping registry for a Hugging Face config (a ``PretrainedConfig`` or a dict).

    :class:`Ernie55WeightConversionMixin` makes it a bridge's ``mapping_registry()``.

    Raises:
        ValueError, NotImplementedError: If the config is missing or outside the supported envelope
            (:meth:`Ernie55HFLayout.from_hf_config`).
    """
    return MegatronMappingRegistry(*ernie55_param_mappings(Ernie55HFLayout.from_hf_config(hf_config)))


def ernie55_hf_param_dtypes(hf_config: Any) -> dict[str, torch.dtype]:
    """Every tensor name of a Hugging Face ERNIE 5.5 checkpoint, with its ERNIE 5.5 (export) dtype.

    Routed-expert wildcards are expanded over ``n_routed_experts``. This is the key set the registry
    reads on import and writes on export; export writes these dtypes, and import also accepts a 16-bit
    copy of an F32 tensor.
    """
    layout = Ernie55HFLayout.from_hf_config(hf_config)
    dtypes: dict[str, torch.dtype] = {}
    for mapping in ernie55_param_mappings(layout):
        names = [mapping.hf_param] if isinstance(mapping.hf_param, str) else list(mapping.hf_param.values())
        for name in names:
            expanded = (
                [name.replace("*", str(expert)) for expert in range(layout.num_experts)] if "*" in name else [name]
            )
            for hf_name in expanded:
                dtypes[hf_name] = mapping.hf_dtype
    return dict(sorted(dtypes.items()))


def ernie55_mtp_body_copy_keys(hf_config: Any, hf_keys: Iterable[str] | None = None) -> set[str]:
    """Names of the MTP body copies a checkpoint may hold besides :func:`ernie55_hf_param_dtypes`.

    Every MTP depth shares the decoder's last layer (``mtp_shared_last_layer``), so the registry reads that body
    once, under the last layer's names. A checkpoint that also stores it under the MTP layer, as the released ones
    do, holds a copy of those tensors; these are the copies' names. The import skips them and the export leaves
    them out of the expected source shards
    (:meth:`Ernie55WeightConversionMixin.hf_export_ignored_source_keys`). The released checkpoints name the copied
    hyper-connections ``transformer_layer.{self_attention,mlp}_hyper_connection.*``.

    Args:
        hf_config: The checkpoint's Hugging Face config (a ``PretrainedConfig`` or a dict).
        hf_keys: :func:`ernie55_hf_param_dtypes` of ``hf_config``, when the caller already has it.
    """
    layout = Ernie55HFLayout.from_hf_config(hf_config)
    expected = set(ernie55_hf_param_dtypes(hf_config) if hf_keys is None else hf_keys)
    last = f"model.layers.{layout.num_hidden_layers - 1}."
    bodies = [name[len(last) :] for name in expected if name.startswith(last)]
    bodies += [f"transformer_layer.{body}" for body in bodies if "_hyper_connection." in body]
    copies = {
        f"model.layers.{layout.num_hidden_layers + depth}.{body}"
        for depth in range(layout.num_mtp_layers)
        for body in bodies
    }
    return copies - expected


def check_ernie55_checkpoint_keys(hf_config: Any, hf_keys: Iterable[str]) -> None:
    """Check that a checkpoint holds exactly the tensors the ERNIE 5.5 registry reads for its config.

    Bridge's import fails on a mapped tensor that the checkpoint lacks, but ignores a checkpoint tensor
    that no mapping reads, which would drop weights silently: for example the shared experts of a config
    without ``n_shared_experts``. The one exception is :func:`ernie55_mtp_body_copy_keys`: the released
    checkpoints also store the shared MTP body under every MTP layer, a copy of the decoder's last layer that
    the import does not read.

    Args:
        hf_config: The checkpoint's Hugging Face config (a ``PretrainedConfig`` or a dict).
        hf_keys: The checkpoint's tensor names.

    Raises:
        ValueError: If the config is outside the supported envelope, or tensors are missing or unexpected
            (the first ten of each are named).
        NotImplementedError: For MTP depths that do not share the decoder's last layer.
    """
    expected = set(ernie55_hf_param_dtypes(hf_config))
    keys = set(hf_keys)
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected - ernie55_mtp_body_copy_keys(hf_config, expected))
    if missing or unexpected:
        raise ValueError(
            "ERNIE 5.5 conversion: the checkpoint does not hold exactly the tensors its config describes: "
            f"{len(missing)} missing {missing[:10]}, {len(unexpected)} that no mapping reads {unexpected[:10]}."
        )


def _check_checkpoint_of(hf_pretrained: Any) -> None:
    """:func:`check_ernie55_checkpoint_keys` on the checkpoint behind ``hf_pretrained``, if there is one."""
    source = getattr(getattr(hf_pretrained, "state", None), "source", None)
    if source is not None:
        check_ernie55_checkpoint_keys(hf_pretrained.config, source.get_all_keys())


class Ernie55WeightConversionMixin:
    """ERNIE 5.5 weight conversion for a ``MegatronModelBridge`` subclass, listed before it among the bases.

    * :meth:`mapping_registry` is :func:`ernie55_mapping_registry` of the bridge's Hugging Face config.
    * Loading a checkpoint first checks that it holds exactly the tensors the registry reads
      (:func:`check_ernie55_checkpoint_keys`). A weight source with no checkpoint behind it (an external
      state dict, a config-only wrapper) is not checked.

    Example::

        @MegatronModelBridge.register_bridge(source=ERNIE55_HF_ARCHITECTURE, target=Ernie55HybridModel)
        class Ernie55Bridge(Ernie55WeightConversionMixin, MegatronModelBridge):
            ...
    """

    def mapping_registry(self) -> MegatronMappingRegistry:
        """The ERNIE 5.5 mapping registry of ``self.hf_config``.

        Raises:
            ValueError, NotImplementedError: If ``hf_config`` is unset or outside the supported envelope.
        """
        return ernie55_mapping_registry(getattr(self, "hf_config", None))

    def load_weights_hf_to_megatron(self, hf_pretrained: Any, megatron_model: Any, *args: Any, **kwargs: Any) -> Any:
        """Check the checkpoint's tensor names, then load it as ``MegatronModelBridge`` does.

        Raises:
            ValueError: If the checkpoint does not hold exactly the tensors its config describes.
        """
        _check_checkpoint_of(hf_pretrained)
        return super().load_weights_hf_to_megatron(hf_pretrained, megatron_model, *args, **kwargs)

    def stream_weights_hf_to_megatron(
        self, hf_pretrained: Any, megatron_model: Any, *args: Any, **kwargs: Any
    ) -> Iterable[Any]:
        """Check the checkpoint's tensor names (unless an external ``hf_state_dict`` is given), then stream it.

        Raises:
            ValueError: If the checkpoint does not hold exactly the tensors its config describes.
        """
        if kwargs.get("hf_state_dict") is None:
            _check_checkpoint_of(hf_pretrained)
        return super().stream_weights_hf_to_megatron(hf_pretrained, megatron_model, *args, **kwargs)

    def hf_export_ignored_source_keys(self, hf_pretrained: Any, source_keys: Iterable[str]) -> list[str]:
        """The checkpoint's copies of the shared MTP body, which export writes once under the decoder's names.

        Export yields exactly the tensors the registry maps. A checkpoint that also stores the shared MTP
        body under its MTP layers (:func:`ernie55_mtp_body_copy_keys`; the released checkpoints do) would
        otherwise leave the shards holding those copies incomplete, and a strict save would refuse to write
        them together with every other tensor they hold. A checkpoint written by this export holds no copies,
        and the result is empty.

        Args:
            hf_pretrained: The source checkpoint wrapper; its ``config`` selects the layout.
            source_keys: The source checkpoint's tensor names.
        """
        hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
        return sorted(ernie55_mtp_body_copy_keys(hf_config) & set(source_keys))
