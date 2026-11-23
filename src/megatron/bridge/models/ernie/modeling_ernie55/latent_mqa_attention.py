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

"""ERNIE 5.5 D-layer self attention: Megatron-Core ``AbsorbedMLASelfAttention`` on a D-layer config.

The D layer reuses AbsorbedMLA's forward unchanged (projections, K absorption, RoPE,
``linear_proj``). The subclass

* validates the supported envelope (:func:`validate_ernie55_config`),
* builds the module from a shallow config copy with the D-layer geometry and the settings the
  ERNIE architecture fixes (:func:`ernie55_d_layer_config`),
* stores the K and V up-projections as the reference implementation's two matrices
  ``k_b_proj`` and ``v_b_proj`` instead of Megatron-Core's combined ``linear_kv_up_proj``,
  because Muon orthogonalizes every 2-D parameter as one matrix,
* sets the core's indexer-loss tracker slot for MTP layers itself (the same slot Megatron-Core dev
  derives from the ``is_mtp_layer`` it forwards to the core), and
* runs context parallelism (CP > 1) in the zigzag partition inside a contiguous stack: the
  forward converts its input rows to zigzag with Megatron-Core dev's module-local converter, runs
  AbsorbedMLA's forward unchanged and converts the output back (see :meth:`forward` and
  :mod:`~megatron.bridge.models.ernie.modeling_ernie55.context_parallel`).

Because the rewrite lives in the attention module, it also reaches the nested MTP stack,
which reuses the decoder's ``dsa_layer`` spec.
"""

from __future__ import annotations

import copy

import torch
from megatron.core.packed_seq_params import resolve_cp_group
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import AbsorbedMLASelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_size

from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core import (
    Ernie55LatentMQACoreAttention,
    indexer_loss_layer_number,
)


__all__ = ["Ernie55LatentMQASelfAttention", "ernie55_d_layer_config", "validate_ernie55_config"]

# ERNIE field -> Megatron-Core field it overrides on the D-layer copy; None inherits the stack value.
_D_LAYER_OVERRIDES = (
    ("ernie_d_num_attention_heads", "num_attention_heads"),
    ("ernie_d_q_lora_rank", "q_lora_rank"),
    ("ernie_d_kv_lora_rank", "kv_lora_rank"),
    ("ernie_d_qk_head_dim", "qk_head_dim"),
    ("ernie_d_qk_pos_emb_head_dim", "qk_pos_emb_head_dim"),
    ("ernie_d_v_head_dim", "v_head_dim"),
    ("ernie_d_rotary_base", "rotary_base"),
)

_SUPPORTED_DSA_KERNEL_BACKENDS = ("cudnn", "none")
_LOSS_SCALED_ROOT_VARIANTS = ("dsa", "dsv4_hybrid")
# Indexer settings fixed by the ERNIE 5.5 architecture: field -> required value.
_FIXED_INDEXER_SETTINGS = (
    ("dsa_indexer_rope_interleaved", False),
    ("dsa_indexer_rotate_activation", True),
    ("dsa_indexer_scoring_relu", True),
    ("dsa_indexer_k_norm_fp32", True),
    ("dsa_indexer_weights_proj_output_dtype", "bf16"),
)


def ernie55_d_layer_config(config: TransformerConfig) -> TransformerConfig:
    """Return a shallow copy of ``config`` configured for an ERNIE 5.5 D layer.

    The copy is not re-finalized; derived fields of ``config`` are kept. Changes:

    * Geometry: every non-``None`` ``ernie_d_*`` field replaces its Megatron-Core counterpart
      (``ernie_d_rotary_base`` replaces ``rotary_base``); ``None`` inherits the stack value.
    * ``num_query_groups = num_attention_heads`` and ``kv_channels`` is set to ``v_head_dim``
      when it is ``None``.
    * Plain RoPE without YaRN scaling: ``rope_type="rope"``, ``rotary_percent=1.0``,
      ``rotary_interleaved=False``, ``rotary_scaling_factor=1.0``, ``mscale=1.0`` and
      ``mscale_all_dim=0.0``, so AbsorbedMLA's softmax scale is
      ``1 / sqrt(qk_head_dim + qk_pos_emb_head_dim)``. AbsorbedMLA always rotates adjacent
      channel pairs (``mla_rotary_interleaved=True``).
    * ``apply_rope_fusion=False``: Megatron-Core has no fused RoPE path for ``rope_type="rope"``.
    * ``qk_layernorm=True`` and ``attention_output_gate=False`` (the ERNIE gate lives in the
      core and is driven by the q latent).
    * ``attention_latent_norm_epsilon`` falls back to ``layernorm_epsilon`` when unset
      (:func:`validate_ernie55_config` rejects any other value, because Megatron-Core main
      normalizes the q/kv latents with ``layernorm_epsilon`` and dev with
      ``attention_latent_norm_epsilon``).

    ``experimental_attention_variant`` is not changed; the q/kv latent norms come from the
    layer spec instead.

    Args:
        config: Finalized stack or per-layer transformer config.

    Returns:
        A new config object; ``config`` is not modified.
    """
    d_config = copy.copy(config)
    for ernie_field, mcore_field in _D_LAYER_OVERRIDES:
        value = getattr(config, ernie_field, None)
        if value is not None:
            setattr(d_config, mcore_field, value)
    d_config.num_query_groups = d_config.num_attention_heads
    if d_config.kv_channels is None:
        d_config.kv_channels = d_config.v_head_dim
    d_config.rope_type = "rope"
    d_config.rotary_percent = 1.0
    d_config.rotary_interleaved = False
    d_config.rotary_scaling_factor = 1.0
    d_config.mscale = 1.0
    d_config.mscale_all_dim = 0.0
    d_config.apply_rope_fusion = False
    d_config.qk_layernorm = True
    d_config.attention_output_gate = False
    if getattr(d_config, "attention_latent_norm_epsilon", 0.0) is None:
        d_config.attention_latent_norm_epsilon = d_config.layernorm_epsilon
    return d_config


def validate_ernie55_config(config: TransformerConfig) -> None:
    """Raise ``ValueError`` if ``config`` is outside the supported ERNIE 5.5 D-layer envelope.

    Supported:

    * TP = 1.
    * CP >= 1. CP > 1 needs Megatron-Core's module-local CP partition converter (dev; main raises),
      ``cp_partition_mode`` ``"contiguous"`` or ``"zigzag"`` and static CP
      (:func:`~megatron.bridge.models.ernie.modeling_ernie55.context_parallel.validate_ernie55_context_parallel_config`);
      every micro-batch must then be packed THD with documents padded to a multiple of 2 * CP and
      prebuilt partition routes (checked in :meth:`Ernie55LatentMQASelfAttention.forward`).
    * BF16 only: ``bf16=True``, no FP16 (Megatron-Core's DSA indexer rotation needs bf16
      activations), no FP8/FP4, no ``quant_recipe`` and no Kitchen quantization.
    * No CUDA graphs, no delayed weight-gradient computation, and no selective recompute of
      ``core_attn`` or ``mla_up_proj`` (full recompute is fine).
    * Per-token top-k every layer and the ERNIE indexer settings.
    * ``dsa_kernel_backend`` ``"cudnn"`` (with a fused ``attention_backend``; an unfused
      backend would silently run the PyTorch reference) or ``"none"``.
    * q/kv latent norms with ``layernorm_epsilon`` (``attention_latent_norm_epsilon`` unset
      or equal to it).
    * No attention dropout and a positive ``ernie_vha_rank``.

    With an indexer loss (``dsa_indexer_loss_coeff > 0``) the loss must be sparse, and the
    pipeline schedule must scale it through ``DSAIndexerLossAutoScaler``, the only way the D
    layer injects the KL gradient: ``experimental_attention_variant`` ``"dsa"`` or
    ``"dsv4_hybrid"`` without ``experimental_attention_variant_loss_scale_func``, or that function
    set to ``DSAIndexerLossAutoScaler.set_loss_scale``. Any other function replaces the schedule's
    hook, ``DSAIndexerLossAutoScaler`` keeps its stale scale, and the indexer gradients are off by
    ``num_microbatches * loss_scale``.

    Args:
        config: Stack-level or per-layer transformer config.

    Raises:
        RuntimeError: At CP > 1 if Megatron-Core has no CP partition converter.
        ValueError: On the first violated requirement.
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"ERNIE 5.5 D layer: {message}")

    require(getattr(config, "multi_latent_attention", False), "multi_latent_attention must be True.")
    require(config.tensor_model_parallel_size == 1, "tensor_model_parallel_size must be 1.")
    context_parallel.validate_ernie55_context_parallel_config(config, layer="D")
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
    if config.recompute_granularity == "selective":
        modules = config.recompute_modules if config.recompute_modules is not None else ["core_attn"]
        unsupported = sorted({"core_attn", "mla_up_proj"} & set(modules))
        require(not unsupported, f"selective recompute of {unsupported} is not supported; use full recompute.")
    require(not config.add_bias_linear, "add_bias_linear must be False.")
    require(not config.attention_dropout, "attention_dropout must be 0.")
    require(config.normalization == "RMSNorm", "normalization must be 'RMSNorm'.")
    latent_norm_eps = getattr(config, "attention_latent_norm_epsilon", None)
    require(
        latent_norm_eps is None or latent_norm_eps == config.layernorm_epsilon,
        f"attention_latent_norm_epsilon must be None or equal to layernorm_epsilon={config.layernorm_epsilon}, "
        f"got {latent_norm_eps}.",
    )
    require(not getattr(config, "cache_mla_latents", False), "cache_mla_latents is not supported.")
    q_lora_rank = getattr(config, "ernie_d_q_lora_rank", None) or config.q_lora_rank
    require(q_lora_rank is not None, "q_lora_rank (or ernie_d_q_lora_rank) must be set.")

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
    for field_name in ("dsa_indexer_n_heads", "dsa_indexer_head_dim", "dsa_indexer_topk"):
        value = getattr(config, field_name)
        require(value is not None and value > 0, f"{field_name} must be a positive integer, got {value!r}.")
    require(config.csa_window_size >= 1, f"csa_window_size must be at least 1, got {config.csa_window_size}.")
    require(getattr(config, "dsa_indexer_mode", "standard") == "standard", "dsa_indexer_mode must be 'standard'.")
    require((config.dsa_indexer_topk_freq or 1) == 1, "dsa_indexer_topk_freq must be 1 (no top-k sharing).")
    require((config.dsa_indexer_kpool or 1) == 1, "dsa_indexer_kpool must be 1.")
    for field_name, required in _FIXED_INDEXER_SETTINGS:
        value = getattr(config, field_name)
        require(value == required, f"{field_name} must be {required!r}, got {value!r}.")
    k_norm_eps = config.dsa_indexer_k_norm_epsilon
    require(
        k_norm_eps is None or k_norm_eps == config.layernorm_epsilon,
        "dsa_indexer_k_norm_epsilon must be None (inherit layernorm_epsilon).",
    )

    if (config.dsa_indexer_loss_coeff or 0.0) > 0:
        require(config.dsa_indexer_use_sparse_loss, "an indexer loss requires dsa_indexer_use_sparse_loss=True.")
        loss_scale_func = config.experimental_attention_variant_loss_scale_func
        if loss_scale_func is None:
            require(
                config.experimental_attention_variant in _LOSS_SCALED_ROOT_VARIANTS,
                "an indexer loss needs experimental_attention_variant in "
                f"{_LOSS_SCALED_ROOT_VARIANTS} or experimental_attention_variant_loss_scale_func="
                "DSAIndexerLossAutoScaler.set_loss_scale; otherwise the schedule never scales the indexer gradients.",
            )
        else:
            require(
                loss_scale_func is dsa.DSAIndexerLossAutoScaler.set_loss_scale,
                f"experimental_attention_variant_loss_scale_func={loss_scale_func!r} replaces the schedule's "
                "DSAIndexerLossAutoScaler.set_loss_scale hook, but the D layer scales its indexer loss only "
                "through DSAIndexerLossAutoScaler; leave it unset or set it to "
                "DSAIndexerLossAutoScaler.set_loss_scale.",
            )


class Ernie55LatentMQASelfAttention(AbsorbedMLASelfAttention):
    """``AbsorbedMLASelfAttention`` for ERNIE 5.5 D layers.

    The constructor validates ``config``, builds the module from
    ``ernie55_d_layer_config(config)``, then checks that the spec produced real q/kv latent
    norms, an :class:`Ernie55LatentMQACoreAttention` core (which consumes the V
    up-projection) and no AbsorbedMLA output gate.

    The K and V up-projections are two parameters in the reference implementation's layout,
    with ``H`` heads, ``C = kv_lora_rank``, ``nope = qk_head_dim`` and ``v = v_head_dim``:

    * ``k_b_proj`` ``[H * C, nope]``: rows ``h * C`` to ``(h + 1) * C`` hold ``W_UK[h]^T``;
    * ``v_b_proj`` ``[H * v, C]``: rows ``h * v`` to ``(h + 1) * v`` hold ``W_UV[h]``;

    where head ``h`` computes ``k_nope = W_UK[h] c_kv`` and ``value = W_UV[h] c_kv``. Megatron-Core
    builds one ``linear_kv_up_proj`` ``[H * (nope + v), C]`` whose head ``h`` block is
    ``[W_UK[h]; W_UV[h]]`` (dev's split-projection specs stack ``W_UK[h]``, the transpose of the
    reference's blocks). The math is the same, but Muon orthogonalizes each 2-D parameter as one
    matrix, and the polar factor of a stack of blocks depends on how the blocks are laid out, so
    only the reference layout gives the reference's optimizer updates. Hugging Face checkpoints
    store both projections as one ``kv_b_proj`` in Megatron-Core's combined layout, so converting
    a checkpoint needs a split on import and a join on export, not a rename.

    AbsorbedMLA's forward runs unchanged: :meth:`_get_kv_up_weights`, the accessor AbsorbedMLA
    reads both up-projections through, returns per-head views of the two parameters.
    :meth:`forward` only adds the zigzag context-parallel conversion around it.

    Positional and keyword arguments after ``config`` are passed through unchanged to
    ``AbsorbedMLASelfAttention``.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        validate_ernie55_config(config)
        super().__init__(ernie55_d_layer_config(config), *args, **kwargs)
        if not isinstance(self.core_attention, Ernie55LatentMQACoreAttention):
            raise TypeError(
                "ERNIE 5.5 D layers need Ernie55LatentMQACoreAttention in the core_attention slot, got "
                f"{type(self.core_attention).__name__}."
            )
        for norm_name in ("q_layernorm", "kv_layernorm"):
            norm = getattr(self, norm_name, None)
            if norm is None or isinstance(norm, IdentityOp):
                raise ValueError(
                    f"ERNIE 5.5 D layers need an explicit {norm_name} module in the spec (got {norm!r}); "
                    "use ernie55_hybrid_stack_spec()."
                )
        if getattr(self, "linear_gate", None) is not None:
            raise ValueError("ERNIE 5.5 D layers gate inside the core; AbsorbedMLA's linear_gate must not be built.")
        self._split_kv_up_projection()
        # Set the core's indexer-loss tracker slot from this layer's position, MTP layers included.
        self.core_attention.layer_number = indexer_loss_layer_number(self.config, self.layer_number, self.is_mtp_layer)

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """AbsorbedMLA's forward, run in the zigzag CP partition when context parallelism is active.

        Without context parallelism this is AbsorbedMLA's forward. At CP > 1 the stack hands the
        layer its rows in ``config.cp_partition_mode`` (``"contiguous"`` in the DSv4 hybrid stack,
        whose H layers accept nothing else), while AbsorbedMLA's RoPE and the D core need the zigzag
        partition. The layer therefore:

        1. checks the packed micro-batch on the host
           (:func:`~megatron.bridge.models.ernie.modeling_ernie55.context_parallel.check_thd_cp_metadata`:
           packed THD, documents padded to a multiple of 2 * CP, prebuilt partition route and host
           ``cu_seqlens``, ``packed_seq_params.cp_partition_mode`` equal to ``config.cp_partition_mode``);
        2. converts ``hidden_states`` to zigzag with Megatron-Core's
           ``convert_module_input_tensors_cp_partition_mode`` (one all-to-all over the CP group,
           routed by the prebuilt route), which also sets ``packed_seq_params.cp_partition_mode``
           to ``"zigzag"``;
        3. runs AbsorbedMLA's forward unchanged on the zigzag rows;
        4. converts the ``linear_proj`` output back to the stack partition (the narrowest tensor of
           the layer, ``[rows, hidden]``);
        5. restores ``packed_seq_params.cp_partition_mode`` in a ``finally`` block, so the next
           layer, a raised exception or a full-recompute replay of this layer all see the stack
           partition again.

        The conversions are differentiable all-to-alls; their backward runs the reverse exchange.
        A zigzag stack (``cp_partition_mode="zigzag"``, a D-only ``"dsa"`` stack) needs no conversion.

        Returns:
            ``(output, bias)`` like AbsorbedMLA; ``output`` is in the stack partition.

        Raises:
            NotImplementedError: For SBHD input at CP > 1.
            RuntimeError: At CP > 1 without Megatron-Core's CP partition converter.
            ValueError: If the micro-batch's packed metadata violates the CP requirements.
        """
        forward_kwargs = dict(
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            position_ids=position_ids,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )
        cp_group = resolve_cp_group(self.pg_collection.cp, packed_seq_params)
        if get_pg_size(cp_group) <= 1:
            return super().forward(hidden_states, attention_mask, **forward_kwargs)

        what = "ERNIE 5.5 D layers"
        api = context_parallel.require_cp_layout_api(what)
        source_mode = self.config.cp_partition_mode
        context_parallel.check_thd_cp_metadata(
            packed_seq_params,
            cp_size=cp_group.size(),
            local_rows=hidden_states.size(0),
            expected_mode=source_mode,
            need_route=source_mode != "zigzag",
            what=what,
        )
        stack_mode = packed_seq_params.cp_partition_mode
        try:
            zigzag_hidden_states, to_stack_partition = api.convert_module_input_tensors_cp_partition_mode(
                hidden_states=hidden_states,
                packed_seq_params=packed_seq_params,
                target_partition_mode="zigzag",
                sequence_parallel=self.config.sequence_parallel,
                config=self.config,
                cp_group=cp_group,
                tp_group=self.pg_collection.tp,
                tp_cp_group=getattr(self.pg_collection, "tp_cp", None),
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                key_value_states=key_value_states,
            )
            if packed_seq_params.cp_partition_mode != "zigzag":
                raise RuntimeError(
                    f"{what}: the CP partition conversion left cp_partition_mode="
                    f"{packed_seq_params.cp_partition_mode!r}, expected 'zigzag'."
                )
            output, bias = super().forward(zigzag_hidden_states, attention_mask, **forward_kwargs)
            if to_stack_partition is not None:
                output = to_stack_partition.convert(output, seq_dim=0)
            return output, bias
        finally:
            packed_seq_params.cp_partition_mode = stack_mode

    def _split_kv_up_projection(self) -> None:
        """Replace the K/V up-projection Megatron-Core built with ``k_b_proj`` and ``v_b_proj``.

        Megatron-Core main always builds the combined ``linear_kv_up_proj``; dev builds it or, for
        split-projection specs, ``linear_k_up_proj`` and ``linear_v_up_proj``. Megatron-Core's own
        accessor reads either. The new parameters take the values Megatron-Core initialized
        (``config.init_method``), copied into the reference layout, so the layer computes the
        same function and no random numbers are drawn.
        """
        heads = self.num_attention_heads_per_partition
        latent, nope, v_head_dim = self.config.kv_lora_rank, self.config.qk_head_dim, self.config.v_head_dim
        k_up, v_up = super()._get_kv_up_weights()  # [H, nope, C] and [H, v, C]
        with torch.no_grad():
            k_b_proj = torch.empty(heads * latent, nope, dtype=k_up.dtype, device=k_up.device)
            k_b_proj.view(heads, latent, nope).copy_(k_up.transpose(1, 2))
            v_b_proj = torch.empty(heads * v_head_dim, latent, dtype=v_up.dtype, device=v_up.device)
            v_b_proj.view(heads, v_head_dim, latent).copy_(v_up)
        for name in ("linear_kv_up_proj", "linear_k_up_proj", "linear_v_up_proj"):
            if name in self._modules:
                delattr(self, name)
        self.k_b_proj = torch.nn.Parameter(k_b_proj)
        self.v_b_proj = torch.nn.Parameter(v_b_proj)

    def _get_kv_up_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head K and V up-projections ``[H, nope, C]`` and ``[H, v, C]``, as views of the parameters.

        AbsorbedMLA absorbs the K view into the query (``einsum("...nd,ndk->...nk")`` takes the
        strided view as is) and hands the V view to the core, which applies it.
        """
        heads = self.num_attention_heads_per_partition
        latent = self.config.kv_lora_rank
        k_up = self.k_b_proj.view(heads, latent, self.config.qk_head_dim).transpose(1, 2)
        v_up = self.v_b_proj.view(heads, self.config.v_head_dim, latent)
        return k_up, v_up

    def _backward_kv_proj(self) -> None:
        """Deferred weight gradients of the K/V projections: only ``linear_kv_down_proj`` defers its own.

        Autograd computes the ``k_b_proj`` and ``v_b_proj`` gradients during the backward pass.
        Megatron-Core main would also call the deleted ``linear_kv_up_proj``.
        """
        self.linear_kv_down_proj.backward_dw()
