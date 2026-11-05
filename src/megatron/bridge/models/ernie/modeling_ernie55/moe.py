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

"""ERNIE 5.5 MoE layer (hybrid symbol ``E``) on Megatron-Core's ``MoELayer``.

Per token, with ``x_r = fp32(pre_mlp_layernorm(x))``, ``E`` experts and top-``k`` routing:

1. Split-feature scores ``s = sigmoid(x_r W^T) + sigmoid(x_r W_1^T)``, two fp32 ``[E, h]`` GEMMs,
   each entry in ``(0, 2)`` (:meth:`Ernie55SplitFeatureRouter.gating`).
2. noaux_tc selection: ``I = top_k(s + b)`` with the fp32 expert bias ``b``; the gate values are the
   unbiased ``p = s[I]``, so the bias only selects.
3. With ``ernie_norm_topk_prob`` and ``k > 1``: ``p = p / (sum(p) + 1e-20)``.
4. A learned fp32 per-expert scale: ``p = p * rho[I]`` (``routed_scaling_factor_param``, initialized
   to ``moe_router_topk_scaling_factor``, the Hugging Face ``routed_scaling_factor``).
5. Dropless dispatch to the grouped routed experts and an ungated shared expert on the full-width
   input; the probabilities scale each expert's activation before its fc2, and the layer output is
   the combined routed output plus the shared output.

Megatron-Core reused unchanged: ``MoELayer`` and its dispatchers, ``TEGroupedMLP``,
``SharedExpertMLP(gate=False)``, ``TopKRouter.apply_z_loss`` (on the scores ``s``),
``TopKRouter._apply_expert_bias`` (token counting) and the ``expert_bias`` buffer, and the
``finalize_model_grads`` expert-bias update ``b <- b + rate * sign(mean(c) - c)`` with
``moe_router_bias_update_rate``. Bridge adds only what Megatron-Core cannot express: the second
gate matrix, the per-expert learned scale and the unbiased-gate selection
(:class:`Ernie55SplitFeatureRouter`), and fp32 latent projections (:class:`Ernie55MoELayer`).

Precision contract (the reference implementation's):

* ``weight``, ``weight_1`` and ``routed_scaling_factor_param`` are fp32 parameters marked
  ``keep_in_fp32``; ``weight`` and ``weight_1`` stay 2-D and are tagged ``use_muon=False`` under
  ``moe_router_skip_muon`` (which the ERNIE 5.5 bridge sets), so AdamW updates them like the reference
  (``muon_exclude_patterns`` ``"mlp.gate"``) does.
* ``expert_bias`` is marked ``keep_in_fp32`` too, as in the reference. Bridge's model provider
  already restores it around the ``Float16Module`` wrap (``model_provider._apply_mixed_precision_wrapper``);
  the mark also covers code that wraps a router with ``Float16Module`` directly, where Megatron-Core's
  ``_maintain_float32_expert_bias`` would only restore the dtype of a bias already rounded to bf16.
* Gating, z-loss, selection, normalization and scaling run in fp32; ``moe_router_dtype`` is
  ignored, as in the reference. Experts compute in the activation dtype.
* Tokens are counted only while autograd is enabled, so full activation recompute (no-grad forward,
  grad-enabled replay) counts each token once. With ``moe_router_bias_update_rate == 0`` the bias is
  frozen: no counting and no update collective.

Latent MoE (``moe_latent_size``, unused by the 4-layer proxy): Megatron-Core's data path is reused
(router and shared expert on the full-width input, ``fc1_latent_proj`` before dispatch, experts at
latent width, ``fc2_norm`` when ``moe_use_norm_before_up_proj``, then ``fc2_latent_proj``).
:class:`Ernie55MoELayer` replaces the two latent projections with :class:`Ernie55LatentProjection`,
whose fp32 weight is cast to the activation dtype at use: the forward equals Megatron-Core's bf16
``TELinear``, but the Hugging Face checkpoint stores these weights in fp32, and a bf16 parameter
would round them at load and give the optimizer a rounded master copy.

MLP activation (dense ``-`` layers, routed and shared experts): ``hidden_act="silu"`` is
Megatron-Core's SwiGLU (``gated_linear_unit=True``, ``activation_func=F.silu``). ``hidden_act="situ"``
uses Megatron-Core's SiTU-GLU knob: ``activation_func_tanh_clamp_scale = beta`` and
``activation_func_tanh_clamp_scale_linear = linear_beta`` (``None`` leaves the up half unclamped,
exactly the reference's ``linear_beta=None`` branch). No other candidate covers the formula:
Megatron-Core's ``situlu`` always clamps the up half, and Kimi K3's ``situ_and_mul`` has no
``linear_beta=None`` branch.

Rounding against the reference, which runs with ``bias_activation_fusion=False``: SwiGLU is
Megatron-Core's unfused path, SiTU is computed in fp32 and rounded once, and routed experts round the
activation and then its product with the fp32 routing probability. Measured on CPU (torch inductor) on
bf16 fc1 outputs (4096 x 2 x 512, sigma 1 and 4, beta 1 and 4, both linear-beta modes):

* ``bias_activation_fusion=False`` (the reference setting and the default) is bitwise equal in every
  MLP: for SwiGLU it runs the reference's own code, and Megatron-Core's compiled ``situ_glu`` rounds
  once, as the reference does.
* ``bias_activation_fusion=True`` rounds once inside the fused kernels. For SiTU this changes only
  the routed experts, where 25-26% of outputs differ by 1 ulp. For SwiGLU it changes every MLP:
  26-28% of dense and shared outputs differ by 1 ulp, and 35-36% of routed outputs by up to 2 ulps.
  A parity run must therefore keep ``bias_activation_fusion=False``.
* Without the jit fuser (eager mode), ``situ_glu`` rounds its intermediates: 31-44% of outputs differ
  by up to 3 ulps.
* Input gradients of the compiled SiTU kernels differ from the reference's in at most 0.06% of elements.

GPU kernels lower ``tanh`` and ``sigmoid`` differently; ``test_ernie55_moe_gpu.py`` rechecks the SiTU
kernels there. Bridge's rule is that rounding differences alone do not justify replacing a
Megatron-Core block.

With ``ernie_fused_ops`` (the default), SiTU-GLU on CUDA runs the reference implementation's own Triton
kernel instead (fp32 inside, one rounding), in the dense MLPs, the shared
expert and the grouped routed experts.
:mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch` installs it around the ``situ_glu``
that Megatron-Core's MLP modules call, so the Megatron-Core blocks stay; the MLP ``activation_func`` slot
(``use_te_activation_func=True``) is not usable, because Megatron-Core rejects it together with
``activation_func_tanh_clamp_scale``.

Known differences from the reference implementation (not bugs):

* Expert token counts are exact int64 in Megatron-Core. The reference implementation registers an fp32
  buffer that ``Float16Module`` casts to bf16, so its counts are rounded while they accumulate (the
  spacing is 8 between 1024 and 2048 tokens). The first bias update can then differ for an expert whose
  count is within rounding of the mean, so routing can differ from the second iteration on. Bridge does
  not emulate this rounding.
* Random initialization only: the reference rounds ``weight`` to bf16 at init (Megatron-Core casts it
  to ``params_dtype`` before ERNIE re-casts it to fp32) and initializes ``fc2_latent_proj`` with
  ``init_method``; Bridge draws ``weight`` in fp32 and keeps Megatron-Core's
  ``output_layer_init_method`` for ``fc2_latent_proj``. Checkpoint loads are unaffected.
* Logging only: with ``moe_z_loss_coeff=0`` (z-loss tracked, no gradient), Megatron-Core records the
  mean z-loss; the gradient is zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core import _param_device


__all__ = [
    "Ernie55LatentProjection",
    "Ernie55MoELayer",
    "Ernie55SplitFeatureRouter",
    "validate_ernie55_moe_config",
]

# Added to the top-k sum before normalization, as in the reference implementation.
_TOPK_NORM_EPS = 1e-20
# Load-balancing modes that replace or bypass top-k-with-bias selection.
_UNSUPPORTED_BALANCING = ("sinkhorn", "quantile_balancing")


def _as_list(value: object) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def validate_ernie55_moe_config(config: TransformerConfig) -> None:
    """Raise ``ValueError`` if ``config`` is outside the supported ERNIE 5.5 MLP and MoE envelope.

    Supported:

    * Gated MLPs shared by dense and expert layers: ``gated_linear_unit=True`` and
      ``activation_func=F.silu`` (SwiGLU, or SiTU-GLU through Megatron-Core's
      ``activation_func_tanh_clamp_scale[_linear]``), no ``activation_func_clamp_value`` and no
      ``glu_linear_offset``.
    * The split-feature router: ``moe_router_score_function="sigmoid"``, the noaux_tc expert bias
      (``moe_router_enable_expert_bias=True``, ``moe_router_bias_update_rate >= 0``), no expert
      groups, no auxiliary loss (every ``moe_aux_loss_coeff`` is 0; sinkhorn and quantile balancing
      are rejected), no hash routing, no routing replay, no router fusion, ``moe_router_dtype`` unset
      or ``"fp32"`` (the router always computes in fp32), a finite
      ``moe_router_topk_scaling_factor`` (or ``None`` for 1.0) and a boolean
      ``ernie_norm_topk_prob``. The benchmark knobs ``moe_router_force_load_balancing`` and
      ``moe_router_force_biased`` are rejected: they replace the scores with random logits before
      routing, and this router uses the scores directly as gate values.
    * Dropless experts (``moe_expert_capacity_factor=None``) without biases, an ungated shared
      expert, and no fused RMSNorm-linear latent up-projection.

    Args:
        config: Transformer config of the MoE layers.

    Raises:
        ValueError: On the first violated requirement.
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"ERNIE 5.5 MoE: {message}")

    require(
        config.gated_linear_unit and config.activation_func is F.silu,
        "the MLPs are gated: set gated_linear_unit=True and activation_func=F.silu (SwiGLU; SiTU-GLU adds "
        "activation_func_tanh_clamp_scale).",
    )
    require(getattr(config, "activation_func_clamp_value", None) is None, "activation_func_clamp_value must be None.")
    require(config.glu_linear_offset == 0.0, f"glu_linear_offset must be 0, got {config.glu_linear_offset}.")

    require(bool(config.num_moe_experts), "num_moe_experts must be set.")
    require(config.moe_router_score_function == "sigmoid", "moe_router_score_function must be 'sigmoid'.")
    require(
        config.moe_router_enable_expert_bias,
        "the noaux_tc expert bias selects the experts; set moe_router_enable_expert_bias=True.",
    )
    rate = config.moe_router_bias_update_rate
    require(
        isinstance(rate, (int, float)) and math.isfinite(rate) and rate >= 0,
        f"moe_router_bias_update_rate must be finite and non-negative, got {rate!r}.",
    )
    require(
        config.moe_router_num_groups is None and config.moe_router_group_topk is None,
        "group-limited routing is not supported (moe_router_num_groups and moe_router_group_topk must be None).",
    )
    balancing = set(_as_list(config.moe_router_load_balancing_type))
    require(
        not balancing & set(_UNSUPPORTED_BALANCING),
        f"moe_router_load_balancing_type must not include {_UNSUPPORTED_BALANCING}, got {sorted(balancing)}.",
    )
    require(
        all(coeff == 0 for coeff in _as_list(config.moe_aux_loss_coeff)),
        f"routing is auxiliary-loss free; moe_aux_loss_coeff must be 0, got {config.moe_aux_loss_coeff!r}.",
    )
    require(not getattr(config, "moe_num_hash_layers", 0), "hash routing (moe_num_hash_layers) is not supported.")
    require(not config.moe_enable_routing_replay, "moe_enable_routing_replay is not supported.")
    require(not config.moe_router_fusion, "moe_router_fusion is not supported; the split-feature router is unfused.")
    require(
        not config.moe_router_force_load_balancing and config.moe_router_force_biased is None,
        "moe_router_force_load_balancing and moe_router_force_biased are not supported; they replace the "
        "split-feature scores, which are the gate values, with random logits.",
    )
    require(
        config.moe_router_dtype in (None, "fp32"),
        f"the router always computes in fp32; moe_router_dtype must be None or 'fp32', got {config.moe_router_dtype!r}.",
    )
    scale = config.moe_router_topk_scaling_factor
    require(
        scale is None or (isinstance(scale, (int, float)) and math.isfinite(scale)),
        f"moe_router_topk_scaling_factor initializes routed_scaling_factor_param and must be finite, got {scale!r}.",
    )
    require(
        isinstance(getattr(config, "ernie_norm_topk_prob", None), bool),
        "ernie_norm_topk_prob (Hugging Face norm_topk_prob) must be a bool; use Ernie55TransformerConfig.",
    )
    require(config.moe_expert_capacity_factor is None, "routing is dropless; moe_expert_capacity_factor must be None.")
    require(not config.add_bias_linear, "add_bias_linear must be False.")
    require(not config.moe_shared_expert_gate, "the shared expert is ungated; moe_shared_expert_gate must be False.")
    require(
        not getattr(config, "moe_latent_up_projection_rmsnorm", False),
        "moe_latent_up_projection_rmsnorm is not supported; use moe_use_norm_before_up_proj for the latent norm.",
    )


class Ernie55SplitFeatureRouter(TopKRouter):
    """Megatron-Core ``TopKRouter`` with ERNIE 5.5's split-feature gating and learned per-expert scale.

    ``gating`` returns the fp32 scores ``sigmoid(x W^T) + sigmoid(x W_1^T)`` instead of logits, and
    ``routing`` selects ``top_k(scores + expert_bias)``, gathers the unbiased scores, normalizes them
    (``ernie_norm_topk_prob``, ``k > 1``) and multiplies them by ``routed_scaling_factor_param``. The
    inherited ``forward`` (input jitter, observation hooks), ``apply_z_loss`` and
    ``_apply_expert_bias`` are unchanged, and ``finalize_model_grads`` updates ``expert_bias`` as for
    any ``TopKRouter``.

    Because ``gating`` returns scores, Megatron-Core's ``router_logits`` tensor observation records the
    scores, and its ``router_scores`` observation (the configured score function applied to those
    values) is not meaningful for this router.

    Positional and keyword arguments after ``config`` are passed through unchanged to ``TopKRouter``.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        validate_ernie55_moe_config(config)
        super().__init__(config, *args, **kwargs)
        num_experts, hidden_size = config.num_moe_experts, config.hidden_size
        # Allocated like Megatron-Core's ``weight``: fp32, on its device (CPU until the model moves).
        self.weight_1 = torch.nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=torch.float32, device=self.weight.device)
        )
        self.routed_scaling_factor_param = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32, device=self.weight.device)
        )
        if config.moe_router_skip_muon:
            setattr(self.weight_1, "use_muon", False)
        self._reset_split_feature_parameters()
        mark_keep_in_fp32(self.expert_bias)
        # A zero update rate freezes the bias: no token counting and no update collective.
        self.frozen_expert_bias = config.moe_router_bias_update_rate == 0
        if self._dense_route_indices_dtype() is not None:
            raise ValueError(
                "ERNIE 5.5 MoE: the split-feature router returns a [tokens, experts] routing map, but this "
                "dispatcher configuration expects dense top-k indices (virtual expert load balancing or fused "
                "flex routing)."
            )

    def _reset_fp32_matrix(self, param: torch.nn.Parameter) -> None:
        """Initialize a gate matrix with ``init_method`` and keep it in fp32."""
        if self.config.perform_initialization:
            with torch.no_grad():
                self.config.init_method(param)
        param.data = param.data.to(torch.float32)
        mark_keep_in_fp32(param)
        setattr(param, "sequence_parallel", self.config.sequence_parallel)

    def _reset_split_feature_parameters(self) -> None:
        """Initialize ``weight_1`` and set every per-expert scale to ``moe_router_topk_scaling_factor``."""
        self._reset_fp32_matrix(self.weight_1)
        scale = self.config.moe_router_topk_scaling_factor
        with torch.no_grad():
            self.routed_scaling_factor_param.fill_(1.0 if scale is None else float(scale))
        mark_keep_in_fp32(self.routed_scaling_factor_param)
        setattr(self.routed_scaling_factor_param, "sequence_parallel", self.config.sequence_parallel)

    def reset_parameters(self) -> None:
        """Initialize the gate parameters in fp32 (Megatron-Core casts ``weight`` to ``params_dtype``)."""
        self._reset_fp32_matrix(self.weight)
        # Router.__init__ calls this method before weight_1 and the per-expert scale exist.
        if "weight_1" in self._parameters:
            self._reset_split_feature_parameters()

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """Return the fp32 split-feature scores ``sigmoid(x W^T) + sigmoid(x W_1^T)`` (not logits).

        Args:
            input: Router input ``[..., hidden]`` (the pre-MLP norm output).

        Returns:
            Scores ``[..., num_experts]`` in fp32, each in ``(0, 2)``.
        """
        # Like Router.gating: the gate parameters are allocated on CPU and follow the activations lazily
        # if the model was not moved before its first forward.
        for param in (self.weight, self.weight_1, self.routed_scaling_factor_param):
            if param.device.type == "cpu" and input.device.type != "cpu":
                param.data = param.data.to(input.device)
        hidden = input.float()
        # The two GEMMs are part of the fp32 contract; an enclosing autocast must not lower them.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return torch.sigmoid(F.linear(hidden, self.weight)) + torch.sigmoid(F.linear(hidden, self.weight_1))

    def routing(
        self,
        logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        packed_seq_params: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts on the biased scores and weight them with the unbiased ones.

        Args:
            logits: Scores from :meth:`gating`, ``[..., num_experts]``.
            padding_mask: Optional ``True``-is-padding mask over the tokens; padding is excluded from
                the z-loss and from the token counts, as in ``TopKRouter``.
            input_ids: Unused (no hash routing).
            packed_seq_params: Unused; ERNIE routing does not depend on sequence boundaries.

        Returns:
            ``(probs, routing_map)``: fp32 probabilities ``[tokens, num_experts]`` with ``k`` non-zero
            entries per row, and the matching boolean map.
        """
        del input_ids, packed_seq_params
        scores = logits.reshape(-1, self.config.num_moe_experts).float()
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)
        scores = self.apply_z_loss(scores, padding_mask=padding_mask)
        # noaux_tc: the expert bias selects the experts but never enters the gate values.
        top_indices = torch.topk(scores + self.expert_bias, k=self.topk, dim=-1).indices
        probs = scores.gather(-1, top_indices)
        if self.config.ernie_norm_topk_prob and self.topk > 1:
            probs = probs / (probs.sum(dim=-1, keepdim=True) + _TOPK_NORM_EPS)
        probs = probs * self.routed_scaling_factor_param[top_indices]
        routing_probs = torch.zeros_like(scores).scatter(-1, top_indices, probs)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, top_indices, True)
        if not self.frozen_expert_bias:
            # Megatron-Core counts only while autograd is enabled (once per token under full recompute).
            self._apply_expert_bias(routing_map, padding_mask=padding_mask)
        return routing_probs, routing_map


class Ernie55LatentProjection(MegatronModule):
    """Latent-MoE projection ``y = x W^T`` whose fp32 weight is cast to the input dtype at use.

    It replaces Megatron-Core's duplicated ``TELinear`` latent projections and keeps their call
    contract: ``forward`` returns ``(y, None)`` and ``backward_dw`` is a no-op.

    Args:
        config: Transformer config.
        input_size: Input width.
        output_size: Output width.
        init_method: Weight initializer, applied when ``config.perform_initialization`` is set.
    """

    def __init__(self, config: TransformerConfig, input_size: int, output_size: int, *, init_method) -> None:
        super().__init__(config=config)
        self.weight = torch.nn.Parameter(
            torch.empty((output_size, input_size), dtype=torch.float32, device=_param_device(config))
        )
        if config.perform_initialization:
            with torch.no_grad():
                init_method(self.weight)
        mark_keep_in_fp32(self.weight)
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Project ``hidden_states`` with the weight rounded to their dtype."""
        return F.linear(hidden_states, self.weight.to(hidden_states.dtype)), None

    def backward_dw(self) -> None:
        """No delayed weight-gradient computation."""


class Ernie55MoELayer(MoELayer):
    """Megatron-Core ``MoELayer`` with the ERNIE 5.5 router and fp32 latent projections.

    The routing, dispatch, expert, shared-expert and combine code is Megatron-Core's. The constructor
    checks that the spec installed :class:`Ernie55SplitFeatureRouter` and, for latent MoE, replaces
    ``fc1_latent_proj`` (``init_method``) and ``fc2_latent_proj`` (``output_layer_init_method``) with
    :class:`Ernie55LatentProjection`. ``fc2_norm`` (``moe_use_norm_before_up_proj``) stays
    Megatron-Core's ``TENorm``.

    Arguments after ``config`` are passed through to ``MoELayer``.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        if not isinstance(self.router, Ernie55SplitFeatureRouter):
            raise TypeError(
                "ERNIE 5.5 MoE layers need Ernie55SplitFeatureRouter in MoESubmodules.router, got "
                f"{type(self.router).__name__}; use ernie55_hybrid_stack_spec()."
            )
        latent_size = self.config.moe_latent_size
        if latent_size:
            self.fc1_latent_proj = Ernie55LatentProjection(
                self.config, self.config.hidden_size, latent_size, init_method=self.config.init_method
            )
            self.fc2_latent_proj = Ernie55LatentProjection(
                self.config, latent_size, self.config.hidden_size, init_method=self.config.output_layer_init_method
            )
