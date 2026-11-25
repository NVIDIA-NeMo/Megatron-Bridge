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

Routed SiTU-GLU fused with the probability multiply (``ernie_fused_situ_prob``, off by default, needs
``ernie_fused_ops``). Megatron-Core's ``TEGroupedMLP`` runs the routed activation as ``situ_glu``, a product with
the fp32 probabilities and a cast back to bf16: three kernels forward and six backward, with the two roundings the
reference has. With the flag, :class:`Ernie55MoELayer` turns its ``TEGroupedMLP`` into :class:`Ernie55GroupedMLP`,
which computes the same two roundings in one Bridge Triton kernel per direction
(:mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_situ_prob`): output and input gradient bitwise equal to
the unfused path, the probability gradient summed over each row in the kernel's order. ``bias_activation_fusion``
stays ``False``; Megatron-Core's fused bias-SiTU kernels round once and are not used. Megatron-Core's activation is a
closure inside ``TEGroupedMLP._unfused_forward``, so :class:`Ernie55GroupedMLP` carries a copy of that method; the
copy is tied to Megatron-Core's source by a fingerprint (:data:`TE_GROUPED_MLP_FORWARD_FINGERPRINTS`), and converting
the experts under an unknown version raises instead of running a stale copy.

Forced-balance benchmark routing (``ernie_router_force_balance``, off by default; BENCHMARK ONLY: it changes the
training math, and every rank logs a warning). Megatron-Core's MoE benchmarks force a balanced load with
``moe_router_force_load_balancing``, which replaces the router logits with random ones. That knob stays rejected here:
the split-feature scores are the gate values, and standard-normal gates could be negative, with a normalization sum
near zero. The flag replaces only the selection of step 2: ``I = top_k(r)`` with ``r = apply_random_logits(s)``,
Megatron-Core's ``RandomSTE`` and the knob's own draw, a fresh standard-normal ``[tokens, experts]`` tensor from the
expert-parallel RNG tracker on every call (seeded per expert-parallel rank). Every token gets ``k`` distinct experts
drawn uniformly, so an expert's load over ``N`` tokens is Binomial(``N``, ``k / E``). Neither the scores nor the expert
bias select. Everything else is unchanged: the gate values are still the unbiased ``s[I]``, normalized and scaled
(finite and non-negative), so ``W``, ``W_1`` and ``rho`` get their usual gradients; the z-loss is taken on ``s``; the
forced routing map is counted and ``finalize_model_grads`` updates the bias with the usual collective, so the bias
random-walks by ``moe_router_bias_update_rate`` per step without selecting anything.

``RandomSTE`` caches nothing. A recompute (``"moe"`` in ``recompute_modules``, or full recompute; both through
Megatron-Core's ``precision_aware_checkpoint``) replays the forward's experts only because the checkpoint restores the
RNG tracker's expert-parallel state before the replay. In BF16 that is ``tensor_parallel.checkpoint``, which snapshots
every tracker state. The states of a graph-safe tracker (Transformer Engine's, or Megatron-Core's with CUDA-graph RNG)
are generator objects that advance in place, so only a copy of their contents can be restored: Megatron-Core dev takes
one since dev 2f2f8ebae (#5841); main does not. In FP8 and FP4 it is Transformer Engine's checkpoint,
which keeps live handles. :func:`validate_ernie55_moe_config` rejects the flag where a replay would draw from an
advanced state.

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

import ast
import functools
import hashlib
import inspect
import io
import logging
import math
import textwrap
import tokenize
from collections.abc import Callable

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel import random as tensor_parallel_random
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.moe import experts as mcore_experts
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import apply_random_logits
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55 import fused_dispatch
from megatron.bridge.models.ernie.modeling_ernie55.expert_load import (
    attach_expert_load_probe,
    expert_load_logging_enabled,
)
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core import _param_device
from megatron.bridge.models.ernie.modeling_ernie55.norm import Ernie55RMSNorm, uses_reference_rmsnorm


__all__ = [
    "TE_GROUPED_MLP_FORWARD_FINGERPRINTS",
    "Ernie55GroupedMLP",
    "Ernie55LatentProjection",
    "Ernie55MoELayer",
    "Ernie55SplitFeatureRouter",
    "te_grouped_mlp_forward_fingerprint",
    "use_ernie55_fused_situ_prob",
    "validate_ernie55_moe_config",
]

# Added to the top-k sum before normalization, as in the reference implementation.
_TOPK_NORM_EPS = 1e-20
# Load-balancing modes that replace or bypass top-k-with-bias selection.
_UNSUPPORTED_BALANCING = ("sinkhorn", "quantile_balancing")


def _as_list(value: object) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _recompute_replays_router(config: TransformerConfig) -> bool:
    """Whether activation recompute runs the MoE router again in the backward (MoE or full recompute)."""
    granularity = config.recompute_granularity
    return granularity == "full" or (granularity == "selective" and "moe" in (config.recompute_modules or ()))


def _graph_safe_rng_tracker(config: TransformerConfig) -> bool:
    """Whether training builds a graph-safe RNG tracker, whose states are generator objects that advance in place.

    Bridge's ``initialize`` builds Transformer Engine's tracker for ``rng.te_rng_tracker``, which its config keeps
    equal to ``use_te_rng_tracker``, and Megatron-Core's with CUDA-graph RNG for any ``cuda_graph_impl`` other than
    ``"none"``.
    """
    return bool(getattr(config, "use_te_rng_tracker", False)) or getattr(config, "cuda_graph_impl", None) not in (
        None,
        "none",
    )


def _checkpoint_rewinds_graph_safe_rng() -> bool:
    """Whether Megatron-Core's ``tensor_parallel.checkpoint`` rewinds a graph-safe RNG tracker before the recompute.

    Since dev 2f2f8ebae (#5841), ``_get_all_rng_states`` copies the contents of graph-safe states (``clone_state``) and
    tags the snapshot, and ``_set_all_rng_states`` takes the tag (``kind``). Before, main included, the snapshot keeps
    the generator handles, which the forward has advanced by the time the recompute restores them.
    """
    try:
        return "kind" in inspect.signature(tensor_parallel_random._set_all_rng_states).parameters
    except (AttributeError, TypeError, ValueError):
        return False


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
      routing, and this router uses the scores directly as gate values. ``ernie_router_force_balance``
      (a bool) is the supported way to force balanced routing. With it, a recompute that replays the
      router (MoE or full) under a graph-safe RNG tracker (``use_te_rng_tracker`` or CUDA graphs) needs
      BF16 and a Megatron-Core whose checkpoint rewinds that tracker (dev 2f2f8ebae, #5841), or the
      replay would select other experts than the forward.
    * Dropless experts (``moe_expert_capacity_factor=None``) without biases, an ungated shared
      expert, and no fused RMSNorm-linear latent up-projection.
    * With ``ernie_fused_situ_prob``: ``ernie_fused_ops=True``, SiTU-GLU (``activation_func_tanh_clamp_scale``
      set), and the unfused routed activation it replaces: ``bias_activation_fusion=False``,
      ``use_te_activation_func=False``, contiguous gate and up halves (``moe_mlp_glu_interleave_size=None``) and the
      probabilities applied after the activation (``moe_apply_probs_on_input=False``).

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
        "split-feature scores, which are the gate values, with random logits. For a benchmark with balanced expert "
        "load, set ernie_router_force_balance=True, which takes only the expert selection from random logits.",
    )
    force_balance = getattr(config, "ernie_router_force_balance", False)
    require(isinstance(force_balance, bool), f"ernie_router_force_balance must be a bool, got {force_balance!r}.")
    if force_balance and _recompute_replays_router(config) and _graph_safe_rng_tracker(config):
        # RandomSTE draws anew on every call: the recompute selects the forward's experts only if its checkpoint
        # rewinds the expert-parallel RNG state, which a graph-safe tracker advances in place.
        require(
            not (config.fp8 or config.fp4),
            "ernie_router_force_balance=True with the MoE or full recompute and a graph-safe RNG tracker "
            "(use_te_rng_tracker or CUDA graphs) needs BF16: in FP8/FP4 the recompute runs Transformer Engine's "
            "checkpoint, which restores live handles of the tracker states, so the replayed router would draw new "
            "random logits and select other experts than the forward. Use BF16, or turn off the MoE/full recompute.",
        )
        require(
            _checkpoint_rewinds_graph_safe_rng(),
            "ernie_router_force_balance=True with the MoE or full recompute and a graph-safe RNG tracker "
            "(use_te_rng_tracker or CUDA graphs) needs a Megatron-Core whose checkpoint rewinds that tracker: this "
            "one restores live handles of the tracker states, so the replayed router would draw new random logits "
            "and select other experts than the forward. Use Megatron-Core dev 2f2f8ebae (#5841) or later, or turn "
            "off the MoE/full recompute or the graph-safe tracker.",
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
    if getattr(config, "ernie_fused_situ_prob", False):
        require(
            getattr(config, "ernie_fused_ops", False),
            "ernie_fused_situ_prob=True fuses the SiTU-GLU kernel of ernie_fused_ops with the probability multiply; "
            "set ernie_fused_ops=True.",
        )
        require(
            getattr(config, "activation_func_tanh_clamp_scale", None) is not None,
            "ernie_fused_situ_prob=True needs SiTU-GLU (activation_func_tanh_clamp_scale set, hidden_act='situ').",
        )
        require(
            not getattr(config, "bias_activation_fusion", False),
            "ernie_fused_situ_prob=True replaces the unfused routed activation and keeps its two roundings; "
            "bias_activation_fusion must be False.",
        )
        require(
            not getattr(config, "use_te_activation_func", False),
            "ernie_fused_situ_prob=True replaces the unfused routed activation; use_te_activation_func must be False.",
        )
        require(
            getattr(config, "moe_mlp_glu_interleave_size", None) is None,
            "ernie_fused_situ_prob=True reads contiguous gate and up halves; moe_mlp_glu_interleave_size must be "
            "None.",
        )
        require(
            not getattr(config, "moe_apply_probs_on_input", False),
            "ernie_fused_situ_prob=True applies the probabilities after the activation; moe_apply_probs_on_input "
            "must be False.",
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

    With ``ernie_router_force_balance`` (``force_balance``; benchmark only) ``routing`` selects
    ``top_k`` of Megatron-Core's random logits (``apply_random_logits``, its ``RandomSTE``) instead of
    ``scores + expert_bias``; the gates are still the gathered scores, and the forced selection is
    counted for the bias update. A recompute replays the same draw because its checkpoint restores
    the RNG tracker (see the module docstring). Every rank logs a warning once when such a router is
    built.

    Positional and keyword arguments after ``config`` are passed through unchanged to ``TopKRouter``.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        validate_ernie55_moe_config(config)
        super().__init__(config, *args, **kwargs)
        self.force_balance = getattr(config, "ernie_router_force_balance", False)
        if self.force_balance:
            fused_dispatch.log_once(
                "router_force_balance",
                "ERNIE 5.5 MoE (rank %d): ernie_router_force_balance=True, BENCHMARK ONLY. Every MoE router selects "
                "its top-k experts from Megatron-Core's random logits (RandomSTE) instead of the split-feature scores "
                "plus the expert bias. This forces a balanced expert load and CHANGES THE TRAINING MATH. Do not use "
                "this run for convergence, parity or checkpoints.",
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                level=logging.WARNING,
                all_ranks=True,
            )
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

        With ``force_balance`` (benchmark only), the experts are the top-k of Megatron-Core's random logits instead;
        the weights are the unbiased scores of those experts, normalized and scaled the same way.

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
        if self.force_balance:
            # Benchmark only: moe_router_force_load_balancing's RandomSTE draw (expert-parallel RNG state) selects;
            # the scores and the bias do not. Detached: the draw only selects, the gates below stay the scores.
            top_indices = torch.topk(apply_random_logits(scores.detach()), k=self.topk, dim=-1).indices
        else:
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


TE_GROUPED_MLP_FORWARD_FINGERPRINTS: dict[str, str] = {
    "d3fc97c39397908b55f73c970fd57e9ea98e11c827039fcee5ddc605f6ac4068": "Megatron-Core dev e2148f486d",
}
"""Fingerprints (:func:`te_grouped_mlp_forward_fingerprint`) of the Megatron-Core ``TEGroupedMLP._unfused_forward``
versions that :meth:`Ernie55GroupedMLP._unfused_forward` reproduces, with the commit each was taken from."""


def _normalized_source_text(source: str) -> str:
    """Python source as a token string without comments, docstrings and line breaks inside statements.

    Tokens keep their text; ``INDENT``, ``DEDENT`` and ``NEWLINE`` keep the block structure. Line breaks inside
    statements, comments and docstrings do not change the result; any change to the other tokens does.
    """
    source = textwrap.dedent(source)
    docstrings = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    docstrings.add((first.lineno, first.col_offset))
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER) or token.start in docstrings:
            continue
        if token.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE):
            tokens.append(tokenize.tok_name[token.type])
        else:
            tokens.append(token.string)
    return " ".join(tokens)


def _source_fingerprint(function: Callable[..., object]) -> str:
    """SHA-256 of ``function``'s source without comments, docstrings and layout (:func:`_normalized_source_text`)."""
    return hashlib.sha256(_normalized_source_text(inspect.getsource(function)).encode()).hexdigest()


@functools.cache
def te_grouped_mlp_forward_fingerprint() -> str:
    """Fingerprint of Megatron-Core's ``TEGroupedMLP._unfused_forward``: SHA-256 of its source without layout.

    :class:`Ernie55GroupedMLP` carries a copy of that method, so any change to its code changes the fingerprint and
    makes :func:`use_ernie55_fused_situ_prob` raise until the copy is checked against the new source.

    Returns:
        The hex digest; ``"unavailable"`` if the source cannot be read.
    """
    try:
        return _source_fingerprint(TEGroupedMLP._unfused_forward)
    except (AttributeError, OSError, TypeError):
        return "unavailable"


class Ernie55GroupedMLP(TEGroupedMLP):
    """Megatron-Core's ``TEGroupedMLP`` with the routed SiTU-GLU and probability multiply fused into one kernel.

    :class:`Ernie55MoELayer` switches its routed experts to this class with ``ernie_fused_situ_prob``
    (:func:`use_ernie55_fused_situ_prob`); no parameter, buffer, hook or checkpoint key changes.

    :meth:`forward` is Megatron-Core's own. :meth:`_unfused_forward` is Megatron-Core's
    ``TEGroupedMLP._unfused_forward`` (:data:`TE_GROUPED_MLP_FORWARD_FINGERPRINTS`) with one change: the activation,
    Megatron-Core's ``bias_act_func`` closure, is :meth:`_situ_glu_prob`. For the ERNIE 5.5 configuration that
    closure computes ``(situ_glu(fc1_output) * probs).to(dtype)``; :meth:`_situ_glu_prob` hands the same computation
    to one fused kernel per direction. Everything around it is Megatron-Core's code: the fine-grained offloading
    groups, ``moe_act`` recompute, fc1, fc2 and the bias; padding, paged stashing and the op-fuser path
    (``_with_fused_impl``, FP8 and FP4 only) stay in Megatron-Core's ``forward``. Probabilities applied on the input
    (``moe_apply_probs_on_input``) run Megatron-Core's ``_unfused_forward``.
    """

    def _unfused_forward(
        self,
        permuted_local_hidden_states: Tensor,
        tokens_per_expert: Tensor | list[int],
        permuted_probs: Tensor,
    ) -> Tensor:
        """Megatron-Core's ``TEGroupedMLP._unfused_forward`` with :meth:`_situ_glu_prob` as the activation.

        Args:
            permuted_local_hidden_states: The permuted (and padded) input rows of the local experts.
            tokens_per_expert: Rows per local expert.
            permuted_probs: The routing probability of every row, ``[rows, 1]``.

        Returns:
            The local experts' output.
        """
        if self.config.moe_apply_probs_on_input:
            return super()._unfused_forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)

        # From here on Megatron-Core's TEGroupedMLP._unfused_forward, statement for statement, except the activation.
        off_interface = mcore_experts.off_interface
        apply_module = mcore_experts.apply_module
        setup_activation_checkpoint = self.activation_recompute and torch.is_grad_enabled()
        offload_group_mlp = self.offload_expert_fc1 and self.offload_moe_act
        group_includes_fc1 = offload_group_mlp and not self._use_grouped_tensor
        offload_expert_fc1 = self.offload_expert_fc1 and not offload_group_mlp
        offload_moe_act = (self.offload_moe_act and not offload_group_mlp) or (
            offload_group_mlp and not group_includes_fc1
        )
        moe_act_group_name = "fused_group_mlp" if offload_group_mlp else "moe_act"
        group_mlp_manager = off_interface(group_includes_fc1, permuted_local_hidden_states, "fused_group_mlp")
        with group_mlp_manager as permuted_local_hidden_states:
            expert_fc1_manager = off_interface(offload_expert_fc1, permuted_local_hidden_states, "expert_fc1")
            with expert_fc1_manager as permuted_local_hidden_states:
                fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                    permuted_local_hidden_states, tokens_per_expert
                )
            fc1_output = expert_fc1_manager.group_offload(
                fc1_output,
                forced_released_tensors=[permuted_local_hidden_states],
                delay_offload=self.config.delay_offload_until_cuda_graph,
            )

            moe_act_manager = off_interface(offload_moe_act, fc1_output, moe_act_group_name)

            if setup_activation_checkpoint:
                self.activation_checkpoint = mcore_experts.tensor_parallel.CheckpointWithoutOutput()
                with moe_act_manager as fc1_output:
                    bias_act_output = self.activation_checkpoint.checkpoint(
                        self._situ_glu_prob, fc1_output, bias_parallel, permuted_probs
                    )
            else:
                with moe_act_manager as fc1_output:
                    bias_act_output = self._situ_glu_prob(fc1_output, bias_parallel, permuted_probs)
        output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
        if setup_activation_checkpoint:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        output = moe_act_manager.group_offload(
            output,
            forced_released_tensors=[fc1_output],
            delay_offload=self.config.delay_offload_until_cuda_graph,
        )
        output = group_mlp_manager.group_offload(
            output,
            forced_released_tensors=[permuted_local_hidden_states, fc1_output],
            delay_offload=self.config.delay_offload_until_cuda_graph,
        )
        return self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

    def _situ_glu_prob(self, fc1_output: Tensor, bias_parallel: Tensor | None, permuted_probs: Tensor) -> Tensor:
        """The routed activation ``(situ_glu(fc1_output) * permuted_probs).to(dtype)``, fused where it can be.

        Takes the place of Megatron-Core's ``bias_act_func(intermediate_parallel, bias_parallel, permuted_probs)``,
        whose SiTU-GLU branch it reproduces. The fused kernel runs for the calls that
        :func:`~megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch.situ_glu_prob_unfit_reason` accepts:
        those whose SiTU-GLU Megatron-Core's ``situ_glu`` binding would hand to Bridge's SiTU-GLU kernel, with fp32
        probabilities. Every other call runs Megatron-Core's operations; a CUDA call among them logs its reason once
        per process on its rank.

        Args:
            fc1_output: The fc1 output ``[rows, 2 * moe_ffn_hidden_size]``.
            bias_parallel: The fc1 bias, always ``None`` for ERNIE 5.5 (``add_bias_linear=False``).
            permuted_probs: The probabilities ``[rows, 1]``.

        Returns:
            The activation ``[rows, moe_ffn_hidden_size]`` in the dtype of ``fc1_output``.
        """
        if bias_parallel is not None:
            raise RuntimeError("ERNIE 5.5 routed experts have no fc1 bias; add_bias_linear must be False.")
        if self.config.moe_paged_stash:
            mcore_experts._mark_te_grouped_tensor(fc1_output, permuted_probs)
        gate_scale = self.config.activation_func_tanh_clamp_scale
        linear_scale = self.config.activation_func_tanh_clamp_scale_linear
        linear_offset = self.config.glu_linear_offset
        reason = fused_dispatch.situ_glu_prob_unfit_reason(
            fc1_output, permuted_probs, gate_scale, linear_scale, linear_offset
        )
        if reason is None:
            return fused_dispatch.fused_situ_glu_prob(fc1_output, permuted_probs, gate_scale, linear_scale)
        if fc1_output.is_cuda:
            fused_dispatch.log_once(
                f"situ_glu_prob_fallback:{reason}",
                "ERNIE 5.5 fused SiTU x prob (rank %d): ernie_fused_situ_prob=True, but the routed activation runs "
                "unfused: %s.",
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                reason,
                level=logging.WARNING,
                all_ranks=True,
            )
        # Megatron-Core's bias_act_func on this path: situ_glu through the module's binding, then the probabilities.
        intermediate_parallel = mcore_experts.situ_glu(fc1_output, gate_scale, linear_scale, linear_offset)
        original_dtype = intermediate_parallel.dtype
        intermediate_parallel = intermediate_parallel * permuted_probs
        return intermediate_parallel.to(original_dtype)


def use_ernie55_fused_situ_prob(moe_layer: torch.nn.Module) -> bool:
    """Switch ``moe_layer.experts`` from Megatron-Core's ``TEGroupedMLP`` to :class:`Ernie55GroupedMLP` (in place).

    Only the class changes; no parameter, buffer, hook or checkpoint key does. Idempotent. Experts that are not a
    ``TEGroupedMLP`` (``SequentialMLP`` without Transformer Engine or grouped GEMMs) keep their class and the unfused
    activation, and every rank that holds them logs one warning.

    Args:
        moe_layer: An MoE layer (anything with an ``experts`` module).

    Returns:
        Whether the experts are an :class:`Ernie55GroupedMLP` afterwards.

    Raises:
        TypeError: If the experts are a ``TEGroupedMLP`` subclass this function does not know.
        RuntimeError: If Megatron-Core's ``TEGroupedMLP._unfused_forward`` is not a version
            :class:`Ernie55GroupedMLP` copies (its fingerprint is not in :data:`TE_GROUPED_MLP_FORWARD_FINGERPRINTS`).
    """
    experts = moe_layer.experts
    if type(experts) is Ernie55GroupedMLP:
        return True
    if not isinstance(experts, TEGroupedMLP):
        fused_dispatch.log_once(
            f"situ_glu_prob_experts:{type(experts).__name__}",
            "ERNIE 5.5 fused SiTU x prob (rank %d): ernie_fused_situ_prob=True, but the routed experts are %s; only "
            "Megatron-Core's TEGroupedMLP takes the fused kernel, so they run the unfused activation.",
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            type(experts).__name__,
            level=logging.WARNING,
            all_ranks=True,
        )
        return False
    if type(experts) is not TEGroupedMLP:
        raise TypeError(
            f"ERNIE 5.5 cannot fuse the routed SiTU-GLU of {type(experts).__name__}; ernie_fused_situ_prob converts "
            "Megatron-Core's TEGroupedMLP only."
        )
    fingerprint = te_grouped_mlp_forward_fingerprint()
    if fingerprint not in TE_GROUPED_MLP_FORWARD_FINGERPRINTS:
        raise RuntimeError(
            "ernie_fused_situ_prob=True: Megatron-Core's TEGroupedMLP._unfused_forward is not a version that "
            f"Ernie55GroupedMLP._unfused_forward copies (fingerprint {fingerprint}; known: "
            f"{TE_GROUPED_MLP_FORWARD_FINGERPRINTS}). Compare the copy in modeling_ernie55/moe.py with this "
            "Megatron-Core's method and add the fingerprint, or set ernie_fused_situ_prob=False."
        )
    experts.__class__ = Ernie55GroupedMLP
    fused_dispatch.log_once(
        "situ_glu_prob_on",
        "ERNIE 5.5 fused SiTU x prob: on (ernie_fused_situ_prob=True); the routed experts run SiTU-GLU and the "
        "probability multiply as one Triton kernel per direction on CUDA.",
    )
    if experts._with_fused_impl:
        fused_dispatch.log_once(
            "situ_glu_prob_op_fuser",
            "ERNIE 5.5 fused SiTU x prob (rank %d): these routed experts run Megatron-Core's op-fuser path "
            "(use_transformer_engine_op_fuser with FP8 or FP4), which keeps its own activation.",
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            level=logging.WARNING,
            all_ranks=True,
        )
    return True


class Ernie55MoELayer(MoELayer):
    """Megatron-Core ``MoELayer`` with the ERNIE 5.5 router and fp32 latent projections.

    The routing, dispatch, expert, shared-expert and combine code is Megatron-Core's. The constructor
    checks that the spec installed :class:`Ernie55SplitFeatureRouter` and, for latent MoE, replaces
    ``fc1_latent_proj`` (``init_method``) and ``fc2_latent_proj`` (``output_layer_init_method``) with
    :class:`Ernie55LatentProjection`. Under ``ernie_reference_rmsnorm``, the optional routed latent
    norm uses :class:`Ernie55RMSNorm` before its up projection. With ``ernie_fused_situ_prob``, the routed ``TEGroupedMLP`` becomes
    :class:`Ernie55GroupedMLP` (:func:`use_ernie55_fused_situ_prob`).

    With ``ERNIE55_LOG_EXPERT_LOAD=1`` (a diagnostic, off by default) the constructor also attaches the layer to
    :mod:`~megatron.bridge.models.ernie.modeling_ernie55.expert_load`, which records the tokens its router sends to
    every expert and its local experts receive.

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
            if self.config.moe_use_norm_before_up_proj and uses_reference_rmsnorm(self.config):
                # Preserve Megatron-Core's parameter, including its distributed optimizer attributes.
                weight = self.fc2_norm.weight
                self.fc2_norm = Ernie55RMSNorm(self.config, latent_size, eps=self.config.layernorm_epsilon)
                self.fc2_norm.weight = weight
        if getattr(self.config, "ernie_fused_situ_prob", False):
            use_ernie55_fused_situ_prob(self)
        if expert_load_logging_enabled():
            attach_expert_load_probe(self)
