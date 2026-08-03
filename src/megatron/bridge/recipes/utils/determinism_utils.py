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

"""Config-level overrides for deterministic training."""

from megatron.bridge.training.config import ConfigContainer


def apply_determinism_overrides(cfg: ConfigContainer) -> None:
    """Apply determinism config overrides to an existing ConfigContainer in-place.

    Sets the model-level flags required for bit-exact reproducibility and
    disables TP comm overlap (which uses non-deterministic NCCL collectives).
    Attention backend selection is a separate concern and is not touched here.

    The deterministic env vars are placed in ``cfg.env_vars`` so they take effect
    at process start. ``MAMBA_DETERMINISTIC`` in particular must be set before the
    first kernel launch: for Mamba/hybrid models (e.g. Nemotron-H) the SSD
    selective-scan Triton kernel cold-autotunes on its first launch, and two
    processes can otherwise pick different configs -> divergent scan output from
    iteration 1, before ``deterministic_mode`` takes effect. It is a no-op for
    non-Mamba models.

    The matching validator that enforces these flags at training time is
    :meth:`megatron.bridge.training.config.ConfigContainer._validate_and_apply_deterministic_mode`.

    This function is idempotent and is safe to call on configs with
    ``comm_overlap = None``.

    KNOWN GAP -- MoE + Transformer Engine grouped MLP. TE computes the
    routing-probability gradient (``dscales``/``dprob_tensor``) inside the cuDNN fused
    grouped-GEMM backward, with an atomic reduction and NO determinism guard. TE guards the
    equivalent *Triton* kernel (``pytorch/triton/grouped_dbias_dscales.py`` raises under
    ``NVTE_ALLOW_NONDETERMINISTIC_ALGO=0``, "uses non-deterministic atomic adds") but the
    cuDNN implementation of the same quantity is unguarded, so that env var gives no
    protection. There is no config lever: ``moe_apply_probs_on_input`` requires
    ``moe_router_topk == 1``, and ``_grouped_mlp_unit_activation_scale`` is never set and is
    forced False whenever ``num_groups != 1``. Disabling ``moe_grouped_gemm`` avoids the path
    but is not viable at scale (OOM). Until TE provides a deterministic dscales path, MoE
    models using the TE grouped MLP are NOT bit-reproducible in the backward pass, even with
    everything below applied.

    Args:
        cfg: Recipe config to modify.
    """
    cfg.model.deterministic_mode = True
    cfg.model.cross_entropy_loss_fusion = False
    # OVERBROAD, DELIBERATELY -- this is an interim workaround, not the right fix.
    # `moe_router_fusion` gates SEVEN call sites in megatron/core/transformer/moe/router.py:
    # the aux-loss ones (switch_load_balancing_loss_func, compute_routing_scores_for_aux_loss)
    # AND the top-k routing ones (topk_routing_with_score_function, topk_routing). Only the
    # aux-loss path is nondeterministic; routing decisions matched bit-for-bit on every rank in
    # every trace. So this also unfuses top-k routing, costing throughput for no determinism
    # benefit, and Megatron exposes no narrower flag (transformer_config.py:898 is a single
    # bool).
    # THE PROPER FIX IS UPSTREAM: gate only the aux-loss call sites, i.e.
    #     fused=self.config.moe_router_fusion and not self.config.deterministic_mode
    # in router.py's _apply_aux_loss / _apply_seq_aux_loss / _apply_global_aux_loss. Once that
    # lands, drop this line -- deterministic_mode already propagates and top-k routing keeps
    # its fused kernel.
    #
    # MoE router fusion routes the load-balancing aux loss into TE's fused_moe_aux_loss,
    # which accumulates with atomicAdd (fused_moe_aux_loss.cu) -> the reduction order, and
    # therefore the last bits of the aux-loss scalar, vary run to run. Measured on
    # Nemotron-3-Ultra MXFP8 at 256 GPUs: with fusion ON, 222 of 256 ranks had their FIRST
    # divergence in the layer-1 router, every |delta| an integer multiple of 2**-37 (one fp32
    # ULP); with it OFF, 0 of 256 did and the whole forward pass became bit-identical.
    # The aux loss is NOT log-only -- MoEAuxLossAutoScaler puts it on the autograd graph, so
    # that perturbation becomes a router-weight gradient. The unfused path computes the same
    # quantity with ordered torch reductions.
    cfg.model.moe_router_fusion = False
    cfg.env_vars.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NCCL_ALGO": "Ring",
            # UNVALIDATED -- kept only because the cost is bounded. Do NOT cite as a fix.
            # NCCL_ALGO alone fixes the ring order but leaves the protocol (Simple/LL/LL128)
            # and the channel count free, and both change how a buffer is chunked, hence the
            # summation order of a reduce-scatter. That mechanism is real in general.
            #
            # It was NOT shown to matter here. The evidence that once supported it was a
            # root-cause test applied to a collective: "this rank's input matched, its output
            # differed". That test is invalid for collectives -- a reduce-scatter output
            # depends on EVERY group member's input, not the local one. Checked directly on
            # the traces: for the first collective flagged this way, 16 of 256 participating
            # ranks had a differing local input, so the output difference was inherited from
            # an upstream divergence, not produced by the collective. The apparent
            # origin counts (12/18, 2/1, 19/26, 25/5) track which peers were contaminated.
            # Any fixed channel count works -- 4 is a compromise, not a tuned value.
            "NCCL_PROTO": "Simple",
            "NCCL_MIN_NCHANNELS": 4,
            "NCCL_MAX_NCHANNELS": 4,
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
            "MAMBA_DETERMINISTIC": 1,
        }
    )

    if cfg.comm_overlap is not None:
        cfg.comm_overlap.tp_comm_overlap = False
