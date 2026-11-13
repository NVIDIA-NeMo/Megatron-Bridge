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

"""ERNIE 5.5 H layer (``compress_ratio == 128``) on Megatron-Core's DSv4 hybrid attention.

An H layer is Megatron-Core's ``DSv4HybridSelfAttention`` with ``CompressedSparseAttention`` and its
``Compressor`` at ratio 128 (no indexer), plus four ERNIE changes. For a query row at document
position ``p``, with ``x`` the layer input and ``c_q = RMSNorm_q(x W_qa^T)`` the q latent:

1. Q/KV (Megatron-Core): ``q = rms(c_q W_qb^T)`` per head (weightless, over all ``v_head_dim``
   channels), ``kv = RMSNorm_kv(x W_kv^T)``, one latent head; RoPE on the last
   ``qk_pos_emb_head_dim`` channels of ``q`` and ``kv``, rotating adjacent channel pairs.
2. Compressor (Megatron-Core): every complete block ``j`` of 128 tokens of a document is pooled,
   ``m_j = sum_i softmax_i(x_i W_gate^T + ape_i) * (x_i W_wkv^T)`` in fp32 and rounded once, then
   RMS-normalized and rotated at position ``128 j``. The ``L mod 128`` tail tokens of a document, and
   documents shorter than 128, have no block.
3. Core (Megatron-Core): each head attends to the window ``[max(0, p - W + 1), p]`` of its document
   and to every complete block ``j < floor((p + 1) / 128)``, with a learnable per-head sink in the
   softmax denominator only and scale ``v_head_dim^-1/2``; key = value = the latent row. The
   output is rotated back (inverse RoPE) at ``p``.
4. VHA (ERNIE, this module's core): ``o'_k = sum_h (I + V U^T)[k, h] o_h`` over the heads.
5. Grouped ``wo_a`` (Megatron-Core): ``z_g = o_g W_oa,g^T`` for ``output_projection_groups`` groups
   of consecutive heads.
6. Output gate (ERNIE, the ``_post_o_group_proj`` hook): ``z <- z * sigmoid(c_q W_g^T)``.
7. ``linear_proj`` (Megatron-Core): ``y = z W_o^T``.

ERNIE pieces, and where they live:

* :class:`Ernie55HCACoreAttention` subclasses ``CompressedSparseAttention``. It owns the VHA factors
  ``vha_postmix_u`` / ``vha_postmix_v`` and applies the head mix out of place to the core output, so
  the sparse-attention ``Function`` keeps its saved output intact. It also draws the all-zero
  attention sink from ``N(0, 0.1)`` at construction when ``perform_initialization`` is set. The
  reference implementation draws it unconditionally; checkpoint loading overwrites it either way.
* :class:`Ernie55HCASelfAttention` subclasses ``DSv4HybridSelfAttention``. It overrides only the
  ``_post_o_group_proj(attn_out, q_compressed)`` hook, where it applies the q-latent sigmoid gate
  ``linear_output_gate``, and with ``ernie_hca_rope_type="rope"`` it replaces the YaRN rotary embedding
  Megatron-Core builds for compressed layers with plain RoPE (see below). ``forward`` is inherited: the core already
  applies the inverse RoPE on Megatron-Core dev, so porting the reference forward would rotate
  twice without any error.

The hook is ``DSv4HybridAttention._post_o_group_proj``, present in Megatron-Core dev from
f78809f6d on. Without it the constructor raises.

RoPE. Megatron-Core builds ``YarnRotaryEmbedding`` for every compressed layer, with base
``csa_compress_rotary_base`` (Hugging Face ``compress_rope_theta``). ``ernie_hca_rope_type`` (Hugging
Face ``hca_rope_type``, ``"rope"`` when the key is missing) selects the H layers' RoPE, as in the
reference implementation:

* ``"rope"``: plain RoPE. ``YarnRotaryEmbedding`` with ``rotary_scaling_factor = 1`` equals plain RoPE
  only up to rounding: in the ramp band its blend ``inv * (1 - m) + inv * m`` can move ``inv_freq`` by
  one fp32 ulp, and the bf16 cos/sin table then differs in a handful of entries. The constructor
  therefore swaps in an exact plain ``RotaryEmbedding`` and rebinds every holder of the YaRN object
  (the attention, the core on dev, the compressor), so the stack's YaRN settings do not reach H layers.
* ``"yarn"``: Megatron-Core's ``YarnRotaryEmbedding`` from the stack's ``rotary_scaling_factor``,
  ``original_max_position_embeddings``, ``beta_fast`` and ``beta_slow``. DSv4 attention rotates with
  ``mscale=1.0`` (no concentration factor), as the reference implementation does. The constructor only
  checks that every holder shares the one YaRN object.

At context_parallel_size > 1 the rows' RoPE (q, kv and the inverse on the output) goes through
Megatron-Core's ``cp_utils.apply_thd_cp_local_rope_unfused``; with ``ernie_fused_ops`` it runs the
reference implementation's Triton kernel on CUDA
(:mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch`).

Expected numerical differences from the reference implementation (rounding only):

* q/kv/compressor norms are Transformer Engine ``RMSNorm`` (fp32 statistics, one rounding); the
  reference casts the input to the weight dtype first.
* The gate projection is a Transformer Engine GEMM rather than ``torch.matmul``.
* On dev, the fused path builds the sparse index lists with ``thd_layout_kernels``, which orders
  the FlashMLA accumulation differently from the reference implementation's pin.

The fused path (``dsa_kernel_backend="cudnn"``) runs FlashMLA's sparse forward and the cuDNN DSA
backward, as the reference implementation does even with ``dsa_kernel_backend="none"``. The
compressor always pools eagerly, which is the reference recipe. On dev, the switch that selects the
fused attention also enables cudnn-frontend's fused ``Compressor`` kernel (``cudnn.csa.compressor``,
which cudnn-frontend 1.29.1 ships) on SM100+ GPUs, so the constructor turns that kernel off for H
layers.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant.csa import CompressedSparseAttention
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridAttention,
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55.context_parallel import validate_ernie55_context_parallel_config
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_attention import Ernie55LatentMQASelfAttention
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core import _param_device
from megatron.bridge.models.ernie.modeling_ernie55.ops import sigmoid_output_gate, vha_postmix


__all__ = [
    "ERNIE55_ATTN_SINK_INIT_STD",
    "ERNIE55_HCA_COMPRESS_RATIO",
    "ERNIE55_HCA_ROPE_TYPES",
    "Ernie55HCACoreAttention",
    "Ernie55HCASelfAttention",
    "Ernie55HCASelfAttentionSubmodules",
    "check_ernie55_attention_layers",
    "rotary_embedding_references",
    "validate_ernie55_hca_config",
]

ERNIE55_HCA_COMPRESS_RATIO = 128
"""Compression ratio of every ERNIE 5.5 H layer (Hugging Face ``compress_ratios`` entries == 128)."""

ERNIE55_HCA_ROPE_TYPES = ("rope", "yarn")
"""Supported H-layer RoPE types (``ernie_hca_rope_type``, Hugging Face ``hca_rope_type``): plain RoPE and
Megatron-Core's YaRN."""

ERNIE55_ATTN_SINK_INIT_STD = 0.1
"""Standard deviation of the H-layer attention sink drawn at initialization (the reference implementation's
``N(0, 0.1)`` seed)."""

_SUPPORTED_DSA_KERNEL_BACKENDS = ("cudnn", "none")


def validate_ernie55_hca_config(config: TransformerConfig) -> None:
    """Raise ``ValueError`` if ``config`` is outside the supported ERNIE 5.5 H-layer envelope.

    Supported: Megatron-Core's DSv4 hybrid attention variant with TP = 1, BF16 (no FP8/FP4, no
    Kitchen quantization), no CUDA graphs and no delayed weight gradients, RMSNorm q/kv/compressor
    norms, no linear biases, a q latent (``q_lora_rank``, the gate input), a positive
    ``ernie_vha_rank``, RoPE on every positional channel (``rotary_percent == 1``) and
    ``dsa_kernel_backend`` ``"cudnn"`` (with a fused ``attention_backend``) or ``"none"``.

    Context parallelism (CP > 1) is Megatron-Core's: DSv4 runs it in the contiguous partition on
    packed THD input (it exchanges the CP-boundary rows the window and the compressor need and
    gathers the compressed keys). The ERNIE additions are per row (VHA, gate) or per layer (plain
    RoPE or YaRN, see :meth:`Ernie55HCASelfAttention._use_plain_rope`). The CP envelope is shared with the D
    layer
    (:func:`~megatron.bridge.models.ernie.modeling_ernie55.context_parallel.validate_ernie55_context_parallel_config`):
    a Megatron-Core build with the CP partition converter (dev; main is not validated and raises)
    and static CP. Megatron-Core's DSv4 CP path builds its compressor layout with CuTe DSL
    kernels, so H layers at CP > 1 run on GPUs only.

    Args:
        config: Stack-level or per-layer transformer config.

    Raises:
        RuntimeError: At CP > 1 if Megatron-Core has no CP partition converter.
        ValueError: On the first violated requirement.
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"ERNIE 5.5 H layer: {message}")

    require(
        config.experimental_attention_variant == "dsv4_hybrid",
        f"experimental_attention_variant must be 'dsv4_hybrid', got {config.experimental_attention_variant!r}.",
    )
    require(getattr(config, "multi_latent_attention", False), "multi_latent_attention must be True.")
    require(config.tensor_model_parallel_size == 1, "tensor_model_parallel_size must be 1.")
    validate_ernie55_context_parallel_config(config, layer="H")
    require(config.bf16 and not config.fp16, "only BF16 is supported (bf16=True, fp16=False).")
    require(not config.fp8 and not config.fp4, "FP8 and FP4 are not supported.")
    require(
        config.quant_recipe is None and not config.use_kitchen and not config.use_kitchen_attention,
        "quant_recipe and Kitchen quantization are not supported.",
    )
    require(
        getattr(config, "cuda_graph_impl", "none") in (None, "none")
        and not getattr(config, "enable_cuda_graph", False),
        "CUDA graphs are not supported.",
    )
    require(not config.delay_wgrad_compute, "delay_wgrad_compute is not supported.")
    require(not config.add_bias_linear, "add_bias_linear must be False.")
    require(config.normalization == "RMSNorm", "normalization must be 'RMSNorm'.")
    require(config.q_lora_rank is not None, "q_lora_rank must be set (the output gate reads the q latent).")
    require(config.rotary_percent == 1.0, f"rotary_percent must be 1.0, got {config.rotary_percent}.")
    rope_type = getattr(config, "ernie_hca_rope_type", "rope")
    require(
        rope_type in ERNIE55_HCA_ROPE_TYPES,
        f"ernie_hca_rope_type must be one of {ERNIE55_HCA_ROPE_TYPES}, got {rope_type!r}.",
    )
    vha_rank = getattr(config, "ernie_vha_rank", None)
    require(
        isinstance(vha_rank, int) and not isinstance(vha_rank, bool) and vha_rank > 0,
        f"ernie_vha_rank must be a positive integer, got {vha_rank!r}.",
    )
    backend = config.dsa_kernel_backend
    require(
        backend in _SUPPORTED_DSA_KERNEL_BACKENDS,
        f"dsa_kernel_backend must be one of {_SUPPORTED_DSA_KERNEL_BACKENDS}, got {backend!r} "
        "(None means the config was not finalized).",
    )
    attention_backend = config.attention_backend
    require(
        backend != "cudnn" or (attention_backend != AttnBackend.unfused and attention_backend != "unfused"),
        "dsa_kernel_backend='cudnn' needs a fused attention_backend; attention_backend=unfused would silently "
        "run the PyTorch reference path. Use dsa_kernel_backend='none' to select the reference path.",
    )
    require(config.csa_window_size >= 1, f"csa_window_size must be at least 1, got {config.csa_window_size}.")


def rotary_embedding_references(root: torch.nn.Module) -> list[tuple[str, RotaryEmbedding]]:
    """Return every rotary embedding reachable from ``root`` with its attribute path.

    Covers registered submodules and plain attributes of every submodule, including the items of
    a list, tuple, set or dict attribute and the arguments of a ``functools.partial`` attribute.
    Shared objects are reported once per path.

    Args:
        root: Module to search.

    Returns:
        ``(path, embedding)`` pairs; ``YarnRotaryEmbedding`` is a ``RotaryEmbedding`` too.
    """
    found = []
    for name, module in root.named_modules(remove_duplicate=False):
        if isinstance(module, RotaryEmbedding):
            found.append((name, module))
        for attr, value in vars(module).items():
            if attr in ("_modules", "_parameters", "_buffers"):
                continue
            path = f"{name}.{attr}" if name else attr
            if isinstance(value, functools.partial):
                items = [*value.args, *value.keywords.values()]
            elif isinstance(value, dict):
                items = list(value.values())
            elif isinstance(value, (list, tuple, set, frozenset)):
                items = list(value)
            else:
                items = [value]
            found.extend((path, item) for item in items if isinstance(item, RotaryEmbedding))
    return found


def _seed_zero_attention_sink_(attn_sink: Tensor) -> None:
    """Draw an all-zero sink from ``N(0, ERNIE55_ATTN_SINK_INIT_STD)`` in place (the reference's init)."""
    if attn_sink.is_meta:
        return
    with torch.no_grad():
        if bool((attn_sink == 0).all()):
            attn_sink.normal_(mean=0.0, std=ERNIE55_ATTN_SINK_INIT_STD)


class Ernie55HCACoreAttention(CompressedSparseAttention):
    """``CompressedSparseAttention`` at ratio 128 followed by the ERNIE VHA head mix.

    Construction and the attention itself are Megatron-Core's (window plus every complete compressed
    block, fp32 sink, fused or reference kernels by ``dsa_kernel_backend``). This subclass adds:

    * ``vha_postmix_u`` / ``vha_postmix_v``: ``[heads, ernie_vha_rank]`` fp32 parameters, marked
      ``keep_in_fp32``. ``U`` takes ``init_method`` when ``perform_initialization`` is set; ``V`` starts
      at zero, so the mix starts as the identity.
    * ``attn_sink`` seeding: the all-zero sink is drawn from ``N(0, 0.1)`` at construction when
      ``perform_initialization`` is set (never in ``forward``).
    * ``forward`` returns ``vha_postmix(core_output)``, computed out of place
      (``ernie_vha_bf16_mix`` selects the bf16 or the fp32 contract).

    The constructor forwards every argument to ``CompressedSparseAttention`` unchanged. Megatron-Core's
    ``Attention.__init__`` first builds a throwaway core without ``compress_ratio`` (a window-only layer) before
    ``DSv4HybridAttention`` builds the real one. The H-layer geometry is therefore checked by
    :class:`Ernie55HCASelfAttention`, not here, and only a ratio-128 core draws ``U`` and the sink:
    the throwaway core allocates the VHA factors but draws no random numbers, so each H layer
    consumes the initialization random streams once.

    Raises:
        ValueError: If ``ernie_vha_rank`` is not a positive integer.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        vha_rank = getattr(config, "ernie_vha_rank", None)
        if not isinstance(vha_rank, int) or isinstance(vha_rank, bool) or vha_rank <= 0:
            raise ValueError(f"ERNIE 5.5 H layers need a positive integer ernie_vha_rank, got {vha_rank!r}.")
        self.vha_bf16_mix = bool(getattr(config, "ernie_vha_bf16_mix", True))

        device = _param_device(config)
        num_heads = config.num_attention_heads
        self.vha_postmix_u = torch.nn.Parameter(torch.zeros(num_heads, vha_rank, dtype=torch.float32, device=device))
        self.vha_postmix_v = torch.nn.Parameter(torch.zeros(num_heads, vha_rank, dtype=torch.float32, device=device))
        mark_keep_in_fp32(self.vha_postmix_u)
        mark_keep_in_fp32(self.vha_postmix_v)
        # The throwaway window-only core Attention.__init__ builds first is discarded: no draws for it.
        if config.perform_initialization and self.compress_ratio == ERNIE55_HCA_COMPRESS_RATIO:
            with torch.no_grad():
                config.init_method(self.vha_postmix_u)
            _seed_zero_attention_sink_(self.attn_sink)

    def forward(self, *args, **kwargs) -> Tensor:
        """Run Megatron-Core's core attention, then mix its heads with ``I + V U^T`` (out of place).

        Returns:
            Same shape as the Megatron-Core core output: ``[s, b, heads * v_head_dim]``, or
            ``[t, 1, heads * v_head_dim]`` for packed THD input.
        """
        output = super().forward(*args, **kwargs)
        num_heads = self.vha_postmix_u.size(0)
        mixed = vha_postmix(
            output.reshape(*output.shape[:-1], num_heads, -1),
            u=self.vha_postmix_u,
            v=self.vha_postmix_v,
            bf16_mix=self.vha_bf16_mix,
        )
        return mixed.reshape(output.shape)


@dataclass
class Ernie55HCASelfAttentionSubmodules(DSv4HybridSelfAttentionSubmodules):
    """``DSv4HybridSelfAttentionSubmodules`` plus the ERNIE output gate.

    Args:
        linear_output_gate: Linear ``q_lora_rank -> output_projection_groups *
            output_projection_lora_rank`` for the q-latent gate, built like the compressor linears
            (``parallel_mode="duplicated"``, no bias).
    """

    linear_output_gate: ModuleSpec | type | None = None


class Ernie55HCASelfAttention(DSv4HybridSelfAttention):
    """``DSv4HybridSelfAttention`` for ERNIE 5.5 H layers: plain RoPE or YaRN, and a q-latent output gate.

    Only the constructor and the ``_post_o_group_proj`` hook differ from Megatron-Core;
    ``forward`` is inherited. The constructor validates the config
    (:func:`validate_ernie55_hca_config`), builds the DSv4 layer, swaps in plain RoPE
    (``ernie_hca_rope_type="rope"``) or checks the YaRN holders (``"yarn"``), keeps the compressor
    eager, builds ``linear_output_gate`` and checks the spec: ratio 128, an
    :class:`Ernie55HCACoreAttention` core and explicit q/kv latent norms.

    Parameters added on top of Megatron-Core (names are pinned by the checkpoint mappings):
    ``linear_output_gate.weight`` ``[output_projection_groups * output_projection_lora_rank,
    q_lora_rank]`` in ``params_dtype``, and the core's ``attn_sink`` / ``vha_postmix_u`` /
    ``vha_postmix_v``.

    Arguments after ``submodules`` and ``layer_number`` are passed through to Megatron-Core.

    Raises:
        RuntimeError: If Megatron-Core's ``DSv4HybridAttention`` has no ``_post_o_group_proj`` hook.
        TypeError: If the core is not an :class:`Ernie55HCACoreAttention`.
        ValueError: If the config is outside the H envelope, the layer's compression ratio is not
            128, the core has an indexer or no compressor, a latent norm is missing (``IdentityOp``)
            or the spec has no output gate.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Ernie55HCASelfAttentionSubmodules,
        layer_number: int,
        *args,
        **kwargs,
    ) -> None:
        validate_ernie55_hca_config(config)
        if not callable(getattr(DSv4HybridAttention, "_post_o_group_proj", None)):
            raise RuntimeError(
                "ERNIE 5.5 H layers need Megatron-Core's DSv4HybridAttention._post_o_group_proj hook, which this "
                "Megatron-Core build lacks. Use Megatron-Core dev at or after f78809f6d."
            )
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        if self._dsv4_compress_ratio != ERNIE55_HCA_COMPRESS_RATIO:
            raise ValueError(
                f"ERNIE 5.5 H layers need compress_ratio={ERNIE55_HCA_COMPRESS_RATIO}, got "
                f"{self._dsv4_compress_ratio}; ERNIE 5.5 has no C or W layers."
            )
        core = self.core_attention
        if not isinstance(core, Ernie55HCACoreAttention):
            raise TypeError(
                f"ERNIE 5.5 H layers need Ernie55HCACoreAttention in the core_attention slot, got {type(core).__name__}."
            )
        if core.compress_ratio != ERNIE55_HCA_COMPRESS_RATIO or core.compressor is None or core.indexer is not None:
            raise ValueError(
                f"ERNIE 5.5 H-layer core attention needs compress_ratio={ERNIE55_HCA_COMPRESS_RATIO}, a compressor and "
                f"no indexer; got compress_ratio={core.compress_ratio}, compressor={type(core.compressor).__name__}, "
                f"indexer={type(core.indexer).__name__}. Check the CompressedSparseAttentionSubmodules of the spec."
            )
        for norm_name in ("q_layernorm", "kv_layernorm"):
            norm = getattr(self, norm_name, None)
            if norm is None or isinstance(norm, IdentityOp):
                raise ValueError(
                    f"ERNIE 5.5 H layers need an explicit {norm_name} module in the spec (got {norm!r}); "
                    "use ernie55_hybrid_stack_spec()."
                )
        if getattr(self.config, "ernie_hca_rope_type", "rope") == "rope":
            self._use_plain_rope()
        else:
            self._check_rope_holders(self.rotary_pos_emb)
        self._use_eager_compressor()

        gate_spec = getattr(submodules, "linear_output_gate", None)
        if gate_spec is None:
            raise ValueError(
                "ERNIE 5.5 H layers need submodules.linear_output_gate (Ernie55HCASelfAttentionSubmodules); "
                "use ernie55_hybrid_stack_spec()."
            )
        self.output_gate_fp32 = bool(getattr(self.config, "ernie_output_gate_fp32", False))
        self.linear_output_gate = build_module(
            gate_spec,
            self.config.q_lora_rank,
            self.config.output_projection_groups * self.config.output_projection_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _use_plain_rope(self) -> None:
        """Replace Megatron-Core's YaRN rotary embedding with plain RoPE at the same base, everywhere.

        Megatron-Core builds one ``YarnRotaryEmbedding`` per compressed layer and hands it to the
        core (dev) and the compressor, which keep their own references. Every holder of a rotary
        embedding is rebound to one ``RotaryEmbedding(qk_pos_emb_head_dim,
        rotary_base=csa_compress_rotary_base, cp_group=<the layer's CP group>)``, so the forward,
        the compressor and the core's inverse RoPE all read the same plain table. The context-parallel
        path reads the same holders: it applies the table at the rows' global contiguous positions
        (``cp_utils.apply_thd_cp_local_rope_*``), the compressor rotates its blocks with its own
        ``rotary_pos_emb`` and the core's inverse RoPE (``_OutputInverseRope``) reads the core's.

        Afterwards no rotary embedding other than the plain one may be reachable from the layer, as a
        registered submodule or as a plain attribute (also inside a list, tuple, dict or partial), so a
        Megatron-Core change that adds another holder raises here instead of silently keeping YaRN.

        Raises:
            RuntimeError: If another rotary embedding is still reachable after the rebinding.
        """
        rope = RotaryEmbedding(
            self.config.qk_pos_emb_head_dim,
            rotary_percent=1.0,
            rotary_base=self._dsv4_rope_base,
            cp_group=self.pg_collection.cp,
        )
        holders = [
            module for module in self.modules() if isinstance(getattr(module, "rotary_pos_emb", None), RotaryEmbedding)
        ]
        for module in holders:
            module.rotary_pos_emb = rope
        self.rotary_pos_emb = rope
        self._dsv4_uses_yarn_rope = False
        self._check_rope_holders(rope)

    def _check_rope_holders(self, rope: RotaryEmbedding) -> None:
        """Raise ``RuntimeError`` if a rotary embedding other than ``rope`` is reachable from the layer."""
        stale = [path for path, embedding in rotary_embedding_references(self) if embedding is not rope]
        if stale:
            raise RuntimeError(
                f"ERNIE 5.5 H layer references rotary embeddings other than its {type(rope).__name__} at {stale}."
            )

    def _use_eager_compressor(self) -> None:
        """Keep the compressor's fp32 pooling eager, as the reference implementation does.

        Megatron-Core dev gates cudnn-frontend's fused ``Compressor`` kernels (``cudnn.csa.compressor``,
        which cudnn-frontend 1.29.1 ships) with ``use_fused_dsa_kernels(config)``. That is the same switch
        that selects the fused FlashMLA/cuDNN attention, so ``dsa_kernel_backend="cudnn"`` would also
        replace the eager pooling on SM100+ GPUs. The fused ratio-128 gradients match the eager pooling
        only within a tolerance, and the fused ``ape`` gradient is accumulated with atomics.
        """
        compressor = self.core_attention.compressor
        if hasattr(compressor, "use_fused_compressor"):
            compressor.use_fused_compressor = False

    def _post_o_group_proj(self, attn_out: Tensor, q_compressed: Tensor) -> Tensor:
        """Gate the grouped ``wo_a`` output with the q latent: ``attn_out * sigmoid(c_q W_g^T)``.

        ``ernie_output_gate_fp32`` (Hugging Face ``sigmoid_gate_fusion``) runs the sigmoid and the
        product in fp32 with one rounding; otherwise both run in the activation dtype. The result is
        a new tensor, and nothing saved for backward aliases it (the hook's offload contract).

        Args:
            attn_out: ``[s, b, groups * o_lora_rank]`` (``[t, 1, ...]`` for packed THD input).
            q_compressed: ``q_layernorm`` output with the leading dims of ``attn_out``.

        Returns:
            Gated tensor with the shape and dtype of ``attn_out``.
        """
        gate_logits, _ = self.linear_output_gate(q_compressed)
        return sigmoid_output_gate(attn_out, gate_logits=gate_logits, fp32=self.output_gate_fp32)

    def backward_dw(self) -> None:
        """Compute the deferred weight gradients of Megatron-Core's linears and of the output gate."""
        super().backward_dw()
        self.linear_output_gate.backward_dw()


def check_ernie55_attention_layers(model: torch.nn.Module) -> dict[str, int]:
    """Check that every attention module of an ERNIE 5.5 model is an ERNIE H or D layer.

    Megatron-Core picks the module of a hybrid ``C``/``H``/``W`` layer from several spec slots
    (``hca_layer`` on dev takes precedence over ``csa_qk_layernorm_layer``), so a spec that sets a
    slot :func:`~megatron.bridge.models.ernie.modeling_ernie55.ernie55_hybrid_stack_spec` does not
    know about would silently build a layer without the ERNIE gate, VHA and sink. The ERNIE 5.5
    model builder must call this on the built model; it walks every submodule, including the
    nested MTP stack and hyper-connection wrappers.

    Args:
        model: Built model, stack or layer.

    Returns:
        ``{"H": number of H layers, "D": number of D layers}``.

    Raises:
        TypeError: If a Megatron-Core attention module is neither an
            :class:`Ernie55HCASelfAttention` nor an ``Ernie55LatentMQASelfAttention``.
    """
    counts = {"H": 0, "D": 0}
    for name, module in model.named_modules():
        if isinstance(module, Ernie55HCASelfAttention):
            counts["H"] += 1
        elif isinstance(module, Ernie55LatentMQASelfAttention):
            counts["D"] += 1
        elif isinstance(module, Attention):
            raise TypeError(
                f"ERNIE 5.5 models only contain Ernie55HCASelfAttention (H) and Ernie55LatentMQASelfAttention (D) "
                f"layers, but {name or type(model).__name__!r} is a {type(module).__name__}; build the model from "
                "ernie55_hybrid_stack_spec()."
            )
    return counts
