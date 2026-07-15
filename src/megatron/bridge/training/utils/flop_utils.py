# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_hybrid_layer_counts,
    parse_hybrid_pattern,
)
from megatron.core.utils import get_attr_wrapped_model

from megatron.bridge.data.packing.algorithms import calculate_avg_seqlen
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


_lora_seq_stats_cache: dict = {}


def get_model_chunk_vp_stage(model: torch.nn.Module) -> int | None:
    """Return the virtual-pipeline stage assigned to a model chunk, if any.

    Args:
        model: Model chunk, possibly wrapped by mixed precision or DDP.

    Returns:
        The integer virtual-pipeline stage, or ``None`` for an unchunked model
        or a model that does not expose the stage.
    """
    try:
        vp_stage = get_attr_wrapped_model(model, "vp_stage", allow_none=False)
    except RuntimeError:
        return None
    return vp_stage if isinstance(vp_stage, int) else None


def _accumulator_to_int(value) -> int:
    """Coerce a FLOPs accumulator (``int`` or scalar ``Tensor``) to ``int``."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return int(value.detach().cpu().item())
    return 0


def resolve_global_flops_seqlen_stats(
    state,
    *,
    data_parallel_size: int,
    vp_size: int | None = None,
    dp_group=None,
) -> tuple[int | None, int | None, int]:
    """Resolve data-parallel-global FLOPS sequence stats from per-rank accumulators.

    Reads the three accumulators populated by the forward step
    (``_flops_seqlen_sum`` = Σ padded tokens, ``_flops_seqlen_sq_sum`` = Σᵢ sᵢ²
    over real sub-sequences, ``_flops_vision_patches``) and reduces them to
    global totals across the data-parallel group.

    Under variable-length (THD packed) training the per-rank ``Σᵢ sᵢ²`` can
    differ across DP ranks, so a single SUM all-reduce over ``dp_group`` is used
    to get the exact global sum. Dense BSHD training never requests this reduce:
    every DP rank contributes the same fixed sequence statistics, so
    extrapolating ``local * data_parallel_size`` is exact and avoids an
    unnecessary collective.

    Args:
        state: Object carrying the ``_flops_*`` accumulators (``GlobalState``).
        data_parallel_size: Size of the data-parallel group (used for the
            extrapolation fallback).
        vp_size: Virtual pipeline size. Kept for call-site compatibility; VPP
            does not rescale these accumulators because they already represent
            the executed training step consumed by the full-model FLOPS formula.
        dp_group: Data-parallel process group to SUM-reduce over. Must be the
            pure DP group (excluding CP) matching ``data_parallel_size`` — CP
            ranks share the same ``cu_seqlens`` and would double-count.

    Returns:
        ``(seqlen_sum, seqlen_squared_sum, num_vision_patches)``. The first two
        are ``None`` when no accumulation happened, signalling the caller to fall
        back to a fixed-length estimate. ``num_vision_patches`` is ``0`` when no
        vision tokens were seen.
    """
    local_seqlen_sum = _accumulator_to_int(getattr(state, "_flops_seqlen_sum", 0))
    local_seqlen_sq_sum = _accumulator_to_int(getattr(state, "_flops_seqlen_sq_sum", 0))
    local_vision_patches = _accumulator_to_int(getattr(state, "_flops_vision_patches", 0))
    _ = vp_size

    use_all_reduce = (
        bool(getattr(state, "_flops_requires_global_reduce", False))
        and dp_group is not None
        and data_parallel_size > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if use_all_reduce:
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        stats = torch.tensor(
            [local_seqlen_sum, local_seqlen_sq_sum, local_vision_patches],
            dtype=torch.long,
            device=device,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        seqlen_sum, seqlen_squared_sum, num_vision_patches = (int(x) for x in stats.tolist())
    else:
        # No process group: extrapolate from the local rank (approximation).
        seqlen_sum = local_seqlen_sum * data_parallel_size
        seqlen_squared_sum = local_seqlen_sq_sum * data_parallel_size
        num_vision_patches = local_vision_patches * data_parallel_size

    if seqlen_sum <= 0:
        return None, None, max(num_vision_patches, 0)
    return seqlen_sum, seqlen_squared_sum, max(num_vision_patches, 0)


def _add_flops_accumulator(state, name: str, delta) -> None:
    """Add an int or scalar tensor to a state accumulator."""
    current = getattr(state, name, 0)
    if not isinstance(current, (int, torch.Tensor)):
        current = 0
    setattr(state, name, current + delta)


def _scalar_sum_for_accumulator(value: torch.Tensor) -> int | torch.Tensor:
    """Return a scalar sum without forcing a CUDA host sync inside forward_step."""
    total = value.sum()
    if total.device.type == "cuda":
        return total
    return int(total.item())


def _real_subseq_lengths(
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_argmin: torch.Tensor | None = None,
    cu_seqlens_unpadded: torch.Tensor | None = None,
    cu_seqlens_unpadded_argmin: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Extract sub-sequence lengths from cu_seqlens metadata.

    Prefers ``cu_seqlens_unpadded`` (true sub-sequence boundaries when
    ``pad_seq_to_mult > 1``) over the padded ``cu_seqlens``. Truncates by the
    corresponding ``*_argmin`` when provided. Returns ``None`` when no
    cu_seqlens info is available.

    Runs once per micro-batch, so it must stay free of GPU→CPU syncs:
    ``cu_seqlens`` is a (monotonic non-decreasing) cumulative sum, so the diffs
    are always ``>= 0`` and we do **not** filter them — a boolean mask like
    ``sub_seq_lens[sub_seq_lens > 0]`` would force a data-dependent-size device
    sync every micro-batch (the cause of a ~7% throughput regression). Zero-length
    entries (padding) contribute ``0`` to ``Σᵢ sᵢ²`` so dropping them is
    unnecessary; the result is identical.
    """
    if cu_seqlens_unpadded is not None:
        cu = cu_seqlens_unpadded.squeeze()
        argmin = cu_seqlens_unpadded_argmin
    elif cu_seqlens is not None:
        cu = cu_seqlens.squeeze()
        argmin = cu_seqlens_argmin
    else:
        return None

    if argmin is not None:
        cu = cu[: int(argmin.item())]

    if cu.numel() < 2:
        return cu.new_empty(0, dtype=torch.long)

    # No boolean mask here on purpose (see docstring): keep this sync-free.
    return (cu[1:] - cu[:-1]).long()


def accumulate_flops_metadata(
    state,
    tokens: torch.Tensor | None,
    *,
    vp_stage: int | None = None,
    config_seq_len: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_argmin: torch.Tensor | None = None,
    cu_seqlens_unpadded: torch.Tensor | None = None,
    cu_seqlens_unpadded_argmin: torch.Tensor | None = None,
    num_vision_patches: int | torch.Tensor | None = None,
) -> None:
    """Accumulate per-microbatch FLOPS metadata onto ``state``.

    Under interleaved pipeline parallelism, the forward step runs once per
    virtual model chunk for the same logical data microbatch. Only virtual stage
    0 contributes metadata so model chunking does not multiply the full-model
    FLOPS estimate. ``None`` and ``0`` both represent the primary/only chunk.

    Writes three accumulators consumed by ``train.py`` at end of step:

    - ``_flops_seqlen_sum``: ``mbs * tokens.shape[1]`` (padded total tokens
      this microbatch contributes), or ``mbs * config_seq_len`` for dense
      non-packed batches whose tensors were already context-parallel sliced.
      Drives the linear MLP/proj/logit terms.
    - ``_flops_seqlen_sq_sum``: the THD attention term Σᵢ sᵢ², computed inline from
      ``cu_seqlens`` (preferring ``cu_seqlens_unpadded``). The per-pack sub-sequence
      lengths are reduced via :func:`_scalar_sum_for_accumulator`, which keeps the
      result **on-device** (no ``.item()``) — so the per-microbatch path stays
      sync-free and the single host sync happens once per step in
      :func:`resolve_global_flops_seqlen_stats`. When ``cu_seqlens`` is absent
      (dense / non-packed) or degenerate, the host-int BSHD fallback
      ``mbs * dense_seq_len²`` is accumulated instead (bit-exact with the
      pre-fix value). ``dense_seq_len`` is ``config_seq_len`` when provided,
      otherwise ``tokens.shape[1]``.
    - ``_flops_vision_patches``: running total of ``num_vision_patches``.

    ``num_vision_patches`` is the precomputed number of vision patches in this
    microbatch (drives the ViT term). It is kept model-agnostic on purpose: the
    caller — which knows its own encoder's layout — computes the count and passes
    a scalar (e.g. Qwen-VL sums ``grid_thw.prod(-1)`` over images and videos). May
    be an ``int`` or a scalar ``Tensor`` (a device tensor avoids a host sync here).

    For THD packed training (offline packed LLM SFT or VLM in-batch packing),
    treating the whole pack as one length-``seq_len`` sequence over-counts
    attention FLOPS by a large factor: actual attention work is Σᵢ sᵢ²,
    not (Σᵢ sᵢ)². Using ``cu_seqlens`` here closes that gap.
    """
    if vp_stage not in (None, 0) or tokens is None:
        return

    mbs = tokens.shape[0]
    tensor_seq_len = tokens.shape[1]
    dense_seq_len = config_seq_len if isinstance(config_seq_len, int) and config_seq_len > 0 else tensor_seq_len

    # THD attention term Σᵢ sᵢ², computed inline from cu_seqlens. The squared
    # sub-sequence lengths stay on-device (``_scalar_sum_for_accumulator`` returns a
    # device tensor, no host sync) so the launch-bound forward path is not stalled; the
    # single sync is deferred to the per-step reduce. cu_seqlens is monotonic, so the
    # diffs are >= 0 and zero-length padding entries contribute 0 — no boolean mask
    # (which would force a data-dependent-size sync) is needed.
    sub_seq_lens = _real_subseq_lengths(cu_seqlens, cu_seqlens_argmin, cu_seqlens_unpadded, cu_seqlens_unpadded_argmin)
    if sub_seq_lens is not None and sub_seq_lens.numel() > 0:
        _add_flops_accumulator(state, "_flops_seqlen_sum", mbs * tensor_seq_len)
        setattr(state, "_flops_requires_global_reduce", True)
        _add_flops_accumulator(state, "_flops_seqlen_sq_sum", _scalar_sum_for_accumulator(sub_seq_lens.long() ** 2))
    else:
        # No cu_seqlens (dense / non-packed) or a degenerate pack with no real
        # sub-sequences → BSHD fallback (single pack-length sequence).
        _add_flops_accumulator(state, "_flops_seqlen_sum", mbs * dense_seq_len)
        _add_flops_accumulator(state, "_flops_seqlen_sq_sum", mbs * dense_seq_len**2)

    if num_vision_patches is not None:
        _add_flops_accumulator(state, "_flops_vision_patches", num_vision_patches)


def vit_flops(
    cfg: ConfigContainer,
    batch_size: int,
    num_patches: int,
):
    """Calculate FLOPs for a Vision Transformer (ViT) encoder + patch merger.

    Includes:
    - ViT transformer layers (bidirectional full attention, not causal)
    - Patch merger (spatial merge + MLP projection to LLM hidden size)

    Args:
        cfg: Configuration container. ViT hyper-parameters are read from
            ``cfg.model.vision_config`` (``depth``, ``hidden_size``,
            ``num_heads``, ``intermediate_size``, ``spatial_merge_size``,
            ``out_hidden_size``). Passing the whole config keeps the public
            signature stable as the list of required ViT attributes grows.
        batch_size: Batch size.
        num_patches: Per-image number of vision patches (before spatial
            merge). Callers that track the total patch count across the
            batch should divide by ``batch_size`` before invoking, because
            ViT attention is per-image (not cross-image) and scales
            quadratically with the per-image patch count.

    Returns:
        Total training FLOPs (forward * 3 for fwd+bwd). Returns 0 when
        no ``vision_config`` is attached or ``num_patches`` is non-positive.
    """
    vision_config = getattr(cfg.model, "vision_config", None)
    if vision_config is None or num_patches <= 0:
        return 0

    depth = getattr(vision_config, "depth", 0)
    hidden_size = getattr(vision_config, "hidden_size", 0)
    intermediate_size = getattr(vision_config, "intermediate_size", 0)
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
    out_hidden_size = getattr(vision_config, "out_hidden_size", cfg.model.hidden_size)

    # ViT Transformer layers (bidirectional attention)
    per_token_per_layer = (
        # QKV + O projections: 4 matmuls of h x h => 4 * 2 * h^2 FMA = 8h^2
        # but standard counting: Q,K,V each h->h (3 * 2h^2) + O h->h (2h^2) = 8h^2
        8 * hidden_size**2
        # Attention core (full bidirectional, not causal): QK^T + attn*V
        # = 2 * 2 * h * num_patches = 4 * h * num_patches
        + 4 * hidden_size * num_patches
        # MLP (GELU, 2 matmuls): fc1 h->intermediate + fc2 intermediate->h
        # = 2 * 2 * h * intermediate = 4 * h * intermediate
        + 4 * hidden_size * intermediate_size
    )
    transformer_flops_val = per_token_per_layer * num_patches * depth

    # Patch Merger: spatial merge (2x2) + MLP projection
    merge_unit = spatial_merge_size**2
    merged_hidden = hidden_size * merge_unit  # concatenated hidden dim
    num_merged_tokens = num_patches // merge_unit if merge_unit > 0 else num_patches
    merger_flops_val = num_merged_tokens * (
        2 * merged_hidden * merged_hidden  # fc1: merged_hidden -> merged_hidden
        + 2 * merged_hidden * out_hidden_size  # fc2: merged_hidden -> out_hidden_size
    )

    return (transformer_flops_val + merger_flops_val) * batch_size * 3  # 3x for training (fwd + bwd)


def num_floating_point_operations(
    cfg: ConfigContainer,
    batch_size: int = 1,
    seqlen_sum: int | None = None,
    seqlen_squared_sum: int | None = None,
    num_vision_patches: int = 0,
):
    """Return the number of floating point operations.

    Args:
        cfg: Configuration container.
        batch_size: Batch size.
        seqlen_sum: Sum of actual sequence lengths across the batch
            (batch_size * actual_seq_length). When provided, overrides
            cfg.model.seq_length for more accurate FLOPS estimation with
            dynamic-length sequences (e.g., VLM with dynamic padding).
        seqlen_squared_sum: Sum of squared sequence lengths across the batch
            (sum_i actual_seq_length_i^2). Used for attention core FLOPS
            which scale quadratically with sequence length; when omitted,
            falls back to ``batch_size * effective_seq_length^2`` so the
            result matches the legacy constant-length estimate.
        num_vision_patches: Total number of vision patches in the batch
            (before spatial merge). Used to compute ViT encoder FLOPS.
    """
    # Compute effective sequence length from actual values or fall back to config.
    if seqlen_sum is not None and batch_size > 0:
        effective_seq_length = seqlen_sum / batch_size
    else:
        effective_seq_length = cfg.model.seq_length
        seqlen_sum = batch_size * cfg.model.seq_length

    # Per-layer attention core FLOPS scale as sum_i(s_i^2), while the outer
    # formula multiplies every per-layer term by ``seqlen_sum``. To account
    # for the quadratic scaling (and variance) of core attention we replace
    # the linear ``effective_seq_length`` factor in core-attn expressions
    # with ``core_attn_seq_factor``. With ``seqlen_sum * core_attn_seq_factor
    # == sum_i(s_i^2)`` this reproduces the correct quadratic sum; when the
    # squared sum is unavailable we fall back to ``effective_seq_length`` so
    # the result matches the legacy constant-length estimate.
    if seqlen_squared_sum is not None and seqlen_sum > 0:
        core_attn_seq_factor = seqlen_squared_sum / seqlen_sum
    else:
        core_attn_seq_factor = effective_seq_length

    peft = getattr(cfg, "peft", None)
    is_lora = isinstance(peft, LoRA)
    # If the model provider has a custom TFLOPS calculation method, use it (non-LoRA only).
    if not is_lora and hasattr(cfg.model, "_get_num_floating_point_operations"):
        return cfg.model._get_num_floating_point_operations(batch_size)

    def calculate_layer_counts():
        """Calculate the number of attention, Mamba, MLP, MoE, and GDN layers."""
        hybrid_pattern = getattr(cfg.model, "hybrid_layer_pattern", None)
        if hybrid_pattern:
            layer_counts = get_hybrid_layer_counts(hybrid_pattern)
            return (
                layer_counts[Symbols.ATTENTION],
                layer_counts[Symbols.MAMBA],
                layer_counts[Symbols.MLP],
                layer_counts[Symbols.MOE],
                layer_counts[Symbols.GDN],
            )

        num_attn_layers = round(cfg.model.num_layers * getattr(cfg.model, "hybrid_attention_ratio", 0))
        num_mlp_layers = round(cfg.model.num_layers * getattr(cfg.model, "hybrid_mlp_ratio", 0))
        num_mamba_layers = cfg.model.num_layers - num_attn_layers - num_mlp_layers
        num_moe_layers = 0
        num_gdn_layers = 0
        return num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers, num_gdn_layers

    def mlp_layer_flops(batch_size, seq_len, hidden_size, expansion=4.0, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        return 4 * expansion * scale_factor * batch_size * seq_len * hidden_size**2

    def moe_layer_flops(
        batch_size,
        seq_len,
        hidden_size,
        moe_ffn_hidden_size,
        shared_expert_ffn_hidden_size,
        num_experts_routed_to,
        moe_latent_size=None,
        swiglu=False,
    ):
        """Calculate FLOPs for an MoE layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        if moe_latent_size is None:
            routed_flops = (
                4 * batch_size * seq_len * hidden_size * moe_ffn_hidden_size * num_experts_routed_to * scale_factor
            )
        else:
            # Routed experts run on moe_latent_size.
            routed_flops = (
                4 * batch_size * seq_len * moe_latent_size * moe_ffn_hidden_size * num_experts_routed_to * scale_factor
            )
            # Up proj and down proj.
            routed_flops += 4 * batch_size * seq_len * hidden_size * moe_latent_size
        shared_flops = 4 * batch_size * seq_len * hidden_size * shared_expert_ffn_hidden_size * scale_factor
        return routed_flops + shared_flops

    def attn_layer_flops(
        batch_size,
        seq_len,
        hidden_size,
        num_heads,
        gqa_groups=8,
        kv_channels=None,
        core_attn_seq_factor=None,
    ):
        """Calculate FLOPs for an attention layer."""
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups
        core_seq = seq_len if core_attn_seq_factor is None else core_attn_seq_factor
        return (
            4 * batch_size * seq_len * hidden_size * p * (hidden_size + (hidden_size * (g / num_heads)))
            + 2 * batch_size * seq_len * hidden_size * p * core_seq
        )

    def calculate_swa_context():
        """Calculate the average causal sliding-window context length."""
        window_size = getattr(cfg.model, "window_size", None)
        if window_size is None:
            return None
        if isinstance(window_size, (list, tuple)):
            effective_window = window_size[0] + window_size[1] + 1
        else:
            effective_window = window_size

        if effective_window < effective_seq_length:
            return effective_window - effective_window * (effective_window - 1) / (2 * effective_seq_length)
        return core_attn_seq_factor / 2

    def is_swa_layer(layer_number: int) -> bool:
        """Return whether the 1-indexed physical layer uses sliding-window attention."""
        window_size = getattr(cfg.model, "window_size", None)
        if window_size is None:
            return False
        window_attn_skip_freq = getattr(cfg.model, "window_attn_skip_freq", None)
        if window_attn_skip_freq is None:
            return True
        if isinstance(window_attn_skip_freq, int):
            return layer_number % window_attn_skip_freq != 0
        if isinstance(window_attn_skip_freq, list):
            return layer_number <= len(window_attn_skip_freq) and bool(window_attn_skip_freq[layer_number - 1])
        return False

    def count_hybrid_swa_attention_layers(num_attn_layers: int) -> int:
        """Count SWA attention symbols in the main physical hybrid pattern."""
        hybrid_pattern = getattr(cfg.model, "hybrid_layer_pattern", None)
        if not hybrid_pattern or getattr(cfg.model, "window_size", None) is None:
            return 0
        parsed = parse_hybrid_pattern(hybrid_pattern)
        main_pattern = parsed.main_pattern or ""

        num_swa_layers = 0
        num_main_attn_layers = 0
        for layer_number, layer_type in enumerate(main_pattern.replace(Symbols.PIPE, ""), start=1):
            if layer_type != Symbols.ATTENTION:
                continue
            num_main_attn_layers += 1
            if is_swa_layer(layer_number):
                num_swa_layers += 1

        # MTP attention layers are counted in num_attn_layers, but they do not
        # correspond to main-pattern layer numbers for window_attn_skip_freq.
        return min(num_swa_layers, num_attn_layers, num_main_attn_layers)

    def mamba_layer_flops(
        batch_size,
        seq_len,
        hidden_size,
        state_dim=16,
        head_dim=64,
        num_groups=1,
        num_heads=128,
    ):
        """Calculate FLOPs for a Mamba layer."""
        # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
        # but small percent of overall layer flops
        d_in = 2 * hidden_size
        if num_heads:
            nheads = num_heads
        else:
            nheads = d_in // head_dim
        return (
            (2 * batch_size * seq_len * hidden_size * (2 * d_in + 2 * num_groups * state_dim + nheads))  # in_proj
            + (7 * batch_size * seq_len * d_in * state_dim)  # scan
            + (2 * batch_size * seq_len * d_in * hidden_size)  # out_proj
        )

    def gdn_layer_flops(
        batch_size,
        seq_len,
        hidden_size,
        qk_head_dim=128,
        v_head_dim=128,
        num_qk_heads=16,
        num_v_heads=32,
        conv_kernel_dim=4,
    ):
        """Calculate FLOPs for a Gated Delta Net (GDN) layer."""
        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads
        return (
            2
            * batch_size
            * seq_len
            * (
                hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                + conv_kernel_dim * (2 * qk_dim + v_dim)
                + num_v_heads * (v_head_dim**2) * 4
                + hidden_size * v_dim
            )
        )

    def hybrid_flops(
        batch_size,
        seq_len,
        hidden_size,
        num_attn_layers,
        num_mamba_layers,
        num_mlp_layers,
        num_moe_layers,
        num_gdn_layers=0,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=128,
        num_attn_heads=32,
        gqa_groups=8,
        kv_channels=None,
        mlp_expansion=4.0,
        swiglu=False,
        moe_latent_size=None,
        moe_ffn_hidden_size=2048,
        shared_expert_ffn_hidden_size=2048,
        num_experts_routed_to=1,
        gdn_qk_head_dim=128,
        gdn_v_head_dim=128,
        gdn_num_qk_heads=16,
        gdn_num_v_heads=32,
        gdn_conv_kernel_dim=4,
        vocab_size=256000,
        mtp_num_layers=0,
        num_swa_attn_layers=0,
        swa_context=None,
        core_attn_seq_factor=None,
    ):
        """Calculate total FLOPs for the hybrid model."""
        num_full_attn_layers = num_attn_layers - num_swa_attn_layers
        full_attn_flops = num_full_attn_layers * attn_layer_flops(
            batch_size,
            seq_len,
            hidden_size,
            num_attn_heads,
            gqa_groups,
            kv_channels,
            core_attn_seq_factor=core_attn_seq_factor,
        )
        swa_attn_flops = 0
        if num_swa_attn_layers > 0:
            if swa_context is None:
                raise ValueError("swa_context must be set when num_swa_attn_layers > 0")
            swa_attn_flops = num_swa_attn_layers * attn_layer_flops(
                batch_size,
                seq_len,
                hidden_size,
                num_attn_heads,
                gqa_groups,
                kv_channels,
                core_attn_seq_factor=2 * swa_context,
            )

        flops_fwd = (
            full_attn_flops
            + swa_attn_flops
            + num_mlp_layers * mlp_layer_flops(batch_size, seq_len, hidden_size, mlp_expansion, swiglu)
            + num_mamba_layers
            * mamba_layer_flops(
                batch_size,
                seq_len,
                hidden_size,
                mamba_state_dim,
                mamba_head_dim,
                mamba_num_groups,
                mamba_num_heads,
            )
            + num_moe_layers
            * moe_layer_flops(
                batch_size,
                seq_len,
                hidden_size,
                moe_ffn_hidden_size,
                shared_expert_ffn_hidden_size,
                num_experts_routed_to,
                moe_latent_size,
                swiglu,
            )
            + num_gdn_layers
            * gdn_layer_flops(
                batch_size,
                seq_len,
                hidden_size,
                gdn_qk_head_dim,
                gdn_v_head_dim,
                gdn_num_qk_heads,
                gdn_num_v_heads,
                gdn_conv_kernel_dim,
            )
            + (2 * batch_size * seq_len * hidden_size * vocab_size * (1 + mtp_num_layers))  # logits computation
        )
        return flops_fwd * 3

    def transformer_flops():
        """Calculate FLOPs for a standard Transformer model."""
        # TODO(helenn/dnarayanan): Refactor this to reuse the helper methods.
        # Attention projection size.
        query_projection_size = cfg.model.kv_channels * cfg.model.num_attention_heads
        # GQA or MHA
        num_query_groups = (
            cfg.model.num_attention_heads if cfg.model.num_query_groups is None else cfg.model.num_query_groups
        )

        dataset_cfg = getattr(cfg, "dataset", None)
        hf_dataset_cfg = getattr(dataset_cfg, "hf_dataset", None)
        hf_dataset_name = getattr(hf_dataset_cfg, "dataset_name", None) or getattr(
            hf_dataset_cfg, "path_or_dataset", None
        )
        is_squad = getattr(dataset_cfg, "dataset_name", hf_dataset_name) in ("squad", "rajpurkar/squad")
        hf_model_id = getattr(cfg.model, "hf_model_id", None)
        is_llama3_70b = hf_model_id is not None and "Meta-Llama-3-70B" in hf_model_id
        packed_specs = (
            getattr(dataset_cfg, "offline_packing_specs", None)
            if getattr(dataset_cfg, "enable_offline_packing", False)
            else None
        )
        packed_data_path = getattr(packed_specs, "packed_train_data_path", None)
        # If not explicitly set, try to find the file via dataset_root (the GPTSFTDatasetBuilder
        # computes this path dynamically, but dataset_root is available from the config).
        if packed_data_path is None and packed_specs is not None:
            dataset_root = getattr(dataset_cfg, "dataset_root", None) or getattr(dataset_cfg, "hf_output_root", None)
            seq_size = getattr(packed_specs, "packed_sequence_size", None)
            if dataset_root is not None and seq_size is not None:
                matches = sorted(Path(dataset_root).glob(f"packed/*/training_{seq_size}.npy"))
                if matches:
                    packed_data_path = str(matches[0])
        if is_lora and is_squad and is_llama3_70b and packed_data_path is not None and Path(packed_data_path).exists():
            gbs = cfg.train.global_batch_size
            seq_len = cfg.model.seq_length
            cache_key = (packed_data_path, gbs, seq_len)
            if cache_key not in _lora_seq_stats_cache:
                _lora_seq_stats_cache[cache_key] = calculate_avg_seqlen(
                    packed_data_path, gbs, seq_len, drop_remainder=True
                )
            _, avg_tokens, _, avg_seqlen2 = _lora_seq_stats_cache[cache_key]

            hs = cfg.model.hidden_size
            n_layers = cfg.model.num_layers
            n_heads = cfg.model.num_attention_heads
            ffn_hs = cfg.model.ffn_hidden_size
            vocab_size = cfg.model.vocab_size

            model_flops_frozen = (
                avg_tokens
                * n_layers
                * hs**2
                * (12 + 12 * num_query_groups / n_heads + 18 * ffn_hs / hs + 6 * vocab_size / (n_layers * hs))
            )
            model_flops_unfrozen = n_layers * hs**2 * (12 * avg_seqlen2 / hs)

            return batch_size * (model_flops_frozen * (2.0 / 3.0) + model_flops_unfrozen)
        # MoE.
        if cfg.model.num_moe_experts is None:
            # Every Transformer MLP is dense.
            num_dense_layers = cfg.model.num_layers
            num_moe_layers = 0
            num_experts_routed_to = 0
            last_layer_is_moe = 0
        else:
            # Calculate number of dense and MoE Transformer MLPs.
            moe_layer_freq = getattr(cfg.model, "moe_layer_freq", 1)
            if isinstance(moe_layer_freq, int):
                moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(cfg.model.num_layers)]
            elif isinstance(moe_layer_freq, list):
                moe_layer_pattern = moe_layer_freq
            else:
                raise RuntimeError("Illegal --moe-layer-freq argument provided!")
            assert len(moe_layer_pattern) == cfg.model.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {cfg.model.num_layers}, "
                f"current moe layer pattern: {moe_layer_freq}"
            )
            num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
            num_dense_layers = cfg.model.num_layers - num_moe_layers
            num_experts_routed_to = getattr(cfg.model, "moe_router_topk", 1)
            last_layer_is_moe = moe_layer_pattern[-1]

        if cfg.model.mtp_num_layers is not None:
            mtp_num_layers = cfg.model.mtp_num_layers
            num_moe_layers += last_layer_is_moe * mtp_num_layers
            num_dense_layers += (1 - last_layer_is_moe) * mtp_num_layers
            num_layers = cfg.model.num_layers + mtp_num_layers
        else:
            mtp_num_layers = 0
            num_layers = cfg.model.num_layers

        # 'moe_ffn_hidden_size' is set only for MoE models.
        moe_ffn_hidden_size = (
            cfg.model.ffn_hidden_size if cfg.model.moe_ffn_hidden_size is None else cfg.model.moe_ffn_hidden_size
        )
        moe_latent_size = getattr(cfg.model, "moe_latent_size", None)
        shared_expert_ffn_hidden_size = (
            0
            if cfg.model.moe_shared_expert_intermediate_size is None
            else cfg.model.moe_shared_expert_intermediate_size
        )
        # GLU: h->2*ffn_h and ffn_h->h = 3 projections; non-GLU: h->ffn_h and ffn_h->h = 2 projections.
        ffn_expansion_factor = 3 if cfg.model.gated_linear_unit is True else 2

        experimental_attention_variant = getattr(cfg.model, "experimental_attention_variant", None)

        if cfg.model.multi_latent_attention:
            """
            Basic arithmetic
            let B is batch size, s is seq_len, h is embedding dim,
            for one self_attnetion block (prenorm is not included)
            qkv projection:  6Bsh^2
            attn:            2Bs^2h
            attn over value: 2Bs^2h
            oproj:           2Bsh^2

            references
            https://arxiv.org/abs/2305.10403
            https://arxiv.org/abs/2205.05198
            """
            if experimental_attention_variant == "dsv4_hybrid":
                # DeepSeek-V4 hybrid MLA uses sparse attention instead of the full
                # core-attention terms used by DeepSeek-V2/V3 MLA. Projection costs
                # are accounted here; sparse attention, compressor, and indexer
                # costs are added below.
                q_lora_rank = getattr(cfg.model, "q_lora_rank", None)
                if q_lora_rank is None:
                    raise ValueError("q_lora_rank must be set for dsv4_hybrid FLOPs calculation")

                qk_head_dim = getattr(cfg.model, "qk_head_dim", 64)
                qk_pos_emb_head_dim = getattr(cfg.model, "qk_pos_emb_head_dim", 0)
                v_head_dim = getattr(cfg.model, "v_head_dim", 64)
                o_lora_rank = getattr(cfg.model, "o_lora_rank", 0)
                o_groups = getattr(cfg.model, "o_groups", 1)

                q_term = q_lora_rank * (
                    cfg.model.hidden_size
                    + cfg.model.num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)
                    + 1  # q norm
                )
                kv_term = cfg.model.hidden_size * v_head_dim + v_head_dim  # kv projection + kv norm
                o_term = (
                    cfg.model.num_attention_heads * v_head_dim * o_lora_rank
                    + o_groups * o_lora_rank * cfg.model.hidden_size
                )
                self_attn_term = 3 * 2 * num_layers * (q_term + kv_term + o_term)

                compress_ratios = getattr(cfg.model, "csa_compress_ratios", None)
                if compress_ratios is None:
                    raise ValueError("csa_compress_ratios must be set for dsv4_hybrid FLOPs calculation")
                if len(compress_ratios) != num_layers:
                    raise ValueError(
                        f"Invalid length of csa_compress_ratios: {len(compress_ratios)}, "
                        f"expected {num_layers} "
                        f"(num_layers={cfg.model.num_layers}, mtp_num_layers={mtp_num_layers})."
                    )

                supported_compress_ratios = {0, 4, 128}
                unsupported_compress_ratios = [
                    ratio for ratio in compress_ratios if ratio not in supported_compress_ratios
                ]
                if unsupported_compress_ratios:
                    raise ValueError(
                        "csa_compress_ratios contains unsupported values: "
                        f"{unsupported_compress_ratios}. Only 0, 4, and 128 are supported."
                    )

                n_layers_r0 = sum(1 for ratio in compress_ratios if ratio == 0)
                n_layers_r4 = sum(1 for ratio in compress_ratios if ratio == 4)
                n_layers_r128 = sum(1 for ratio in compress_ratios if ratio == 128)
                window = getattr(cfg.model, "csa_window_size", 128)

                sparse_attn_r0 = n_layers_r0 * cfg.model.num_attention_heads * window * v_head_dim * 2
                avg_comp_128 = (core_attn_seq_factor // 128) / 2
                sparse_attn_r128 = (
                    n_layers_r128 * cfg.model.num_attention_heads * (window + avg_comp_128) * v_head_dim * 2
                )

                main_compressor_term = (
                    n_layers_r4 * cfg.model.hidden_size * (2 * v_head_dim) * 2
                    + n_layers_r128 * cfg.model.hidden_size * v_head_dim * 2
                )

                if n_layers_r4 > 0:
                    idx_n_heads = getattr(cfg.model, "dsa_indexer_n_heads", None)
                    idx_head_dim = getattr(cfg.model, "dsa_indexer_head_dim", None)
                    idx_topk = getattr(cfg.model, "dsa_indexer_topk", None)
                    if idx_n_heads is None:
                        raise ValueError("dsa_indexer_n_heads must be set for dsv4_hybrid ratio==4 layers")
                    if idx_head_dim is None:
                        raise ValueError("dsa_indexer_head_dim must be set for dsv4_hybrid ratio==4 layers")
                    if idx_topk is None:
                        raise ValueError("dsa_indexer_topk must be set for dsv4_hybrid ratio==4 layers")

                    effective_topk_4 = min(idx_topk, core_attn_seq_factor // 4)
                    avg_comp_4 = effective_topk_4 * (1 - effective_topk_4 * 4 / (2 * core_attn_seq_factor))
                    sparse_attn_r4 = (
                        n_layers_r4 * cfg.model.num_attention_heads * (window + avg_comp_4) * v_head_dim * 2
                    )
                    indexer_term = (
                        n_layers_r4 * cfg.model.hidden_size * (2 * idx_head_dim) * 2
                        + n_layers_r4 * q_lora_rank * idx_n_heads * idx_head_dim
                        + n_layers_r4 * cfg.model.hidden_size * idx_n_heads
                        + n_layers_r4 * idx_n_heads * idx_head_dim * (core_attn_seq_factor // 4)
                    )
                else:
                    sparse_attn_r4 = 0
                    indexer_term = 0

                sparse_attn_term = sparse_attn_r0 + sparse_attn_r4 + sparse_attn_r128
                self_attn_term += 3 * 2 * (sparse_attn_term + main_compressor_term + indexer_term)
            else:
                ## MLA
                if not hasattr(cfg.model, "q_lora_rank") or cfg.model.q_lora_rank is None:
                    q_term = (
                        cfg.model.hidden_size
                        * cfg.model.num_attention_heads
                        * (getattr(cfg.model, "qk_head_dim", 64) + getattr(cfg.model, "qk_pos_emb_head_dim", 0))
                    )
                else:
                    q_term = cfg.model.q_lora_rank * (
                        cfg.model.hidden_size
                        + cfg.model.num_attention_heads
                        * (getattr(cfg.model, "qk_head_dim", 64) + getattr(cfg.model, "qk_pos_emb_head_dim", 0))
                        + 1
                    )
                self_attn_term = (
                    3
                    * 2  # fwd(1) + bwd(2) *FMA
                    * num_layers
                    * (
                        ## q lora + rope + q norm
                        q_term
                        ## kv lora + rope + kv norm
                        + getattr(cfg.model, "kv_lora_rank", 0)
                        * (
                            cfg.model.hidden_size
                            + cfg.model.num_attention_heads
                            * (getattr(cfg.model, "qk_head_dim", 64) + getattr(cfg.model, "v_head_dim", 64))
                            + 1
                        )
                        + cfg.model.hidden_size * getattr(cfg.model, "qk_pos_emb_head_dim", 0)
                        ## o proj
                        + (cfg.model.num_attention_heads * getattr(cfg.model, "v_head_dim", 64))
                        * cfg.model.hidden_size
                        ## core attn
                        + core_attn_seq_factor
                        * (
                            cfg.model.num_attention_heads
                            * (getattr(cfg.model, "qk_head_dim", 64) + getattr(cfg.model, "qk_pos_emb_head_dim", 0))
                        )
                        / 2
                        + core_attn_seq_factor
                        * cfg.model.num_attention_heads
                        * getattr(cfg.model, "v_head_dim", 64)
                        / 2
                    )
                )

        else:
            ## MHA or GQA
            key_projection_size = cfg.model.kv_channels * num_query_groups
            value_projection_size = cfg.model.kv_channels * num_query_groups
            gate_projection_size = query_projection_size if getattr(cfg.model, "attention_output_gate", False) else 0
            proj_per_layer = (
                cfg.model.hidden_size
                * (query_projection_size + key_projection_size + value_projection_size + gate_projection_size)
                + query_projection_size * cfg.model.hidden_size
            )

            window_size = getattr(cfg.model, "window_size", None)
            window_attn_skip_freq = getattr(cfg.model, "window_attn_skip_freq", None)

            if window_size is not None:
                if isinstance(window_size, (list, tuple)):
                    effective_window = window_size[0] + window_size[1] + 1
                else:
                    effective_window = window_size
                # Exact average causal SWA context:
                # W * (W + 1) / 2 + (T - W) * W if W < T, else T * (T + 1) / 2.
                # Both expressions are divided by T because the multiplication with T happens later.
                if effective_window < effective_seq_length:
                    swa_context = effective_window - effective_window * (effective_window - 1) / (
                        2 * effective_seq_length
                    )
                else:
                    swa_context = core_attn_seq_factor / 2

                if window_attn_skip_freq is None:
                    num_swa_layers = num_layers
                    num_full_attn_layers = 0
                elif isinstance(window_attn_skip_freq, int):
                    swa_pattern = [0 if ((i + 1) % window_attn_skip_freq == 0) else 1 for i in range(num_layers)]
                    num_swa_layers = sum(swa_pattern)
                    num_full_attn_layers = num_layers - num_swa_layers
                elif isinstance(window_attn_skip_freq, list):
                    swa_pattern = window_attn_skip_freq[:num_layers]
                    num_swa_layers = sum(swa_pattern)
                    num_full_attn_layers = num_layers - num_swa_layers
                else:
                    num_swa_layers = 0
                    num_full_attn_layers = num_layers

                # Full attention is quadratic in seq_len -> use core_attn_seq_factor.
                full_core = query_projection_size * core_attn_seq_factor / 2 * 2
                swa_core = query_projection_size * swa_context * 2

                self_attn_term = (
                    3
                    * 2
                    * (
                        num_full_attn_layers * (proj_per_layer + full_core)
                        + num_swa_layers * (proj_per_layer + swa_core)
                    )
                )
            else:
                full_core = query_projection_size * core_attn_seq_factor / 2 * 2
                self_attn_term = 3 * 2 * num_layers * (proj_per_layer + full_core)

        # Handle GDN (Gated DeltaNet) hybrid attention variant.
        # When experimental_attention_variant is "gated_delta_net", a fraction of the
        # layers use GDN instead of standard attention. Override self_attn_term with a
        # weighted sum of GDN and standard-attention per-layer costs.
        if experimental_attention_variant == "gated_delta_net":
            linear_attention_freq = cfg.model.linear_attention_freq
            decoder_num_layers = cfg.model.num_layers
            if linear_attention_freq is None:
                raise ValueError(
                    "linear_attention_freq must be set when experimental_attention_variant='gated_delta_net'"
                )
            if isinstance(linear_attention_freq, int):
                linear_attention_pattern = [
                    0 if ((i + 1) % linear_attention_freq == 0) else 1 for i in range(decoder_num_layers)
                ]
            elif isinstance(linear_attention_freq, list):
                linear_attention_pattern = linear_attention_freq
                if len(linear_attention_pattern) != decoder_num_layers:
                    raise ValueError(
                        f"Invalid length of linear_attention_pattern: {len(linear_attention_pattern)}, "
                        f"expected {decoder_num_layers}, "
                        f"current linear_attention_freq: {linear_attention_freq}"
                    )
            else:
                raise TypeError(
                    f"linear_attention_freq must be int or list, got {type(linear_attention_freq).__name__}"
                )

            # MTP construction reuses the final decoder layer spec, so each MTP
            # layer has the same attention type as the final decoder layer.
            last_layer_is_gdn = linear_attention_pattern[-1] if linear_attention_pattern else 0
            num_gdn_layers = sum(linear_attention_pattern) + last_layer_is_gdn * mtp_num_layers
            num_standard_attn_layers = num_layers - num_gdn_layers

            standard_self_attn_per_layer = self_attn_term / num_layers if num_layers > 0 else 0

            qk_head_dim = cfg.model.linear_key_head_dim
            v_head_dim = cfg.model.linear_value_head_dim
            num_qk_heads = cfg.model.linear_num_key_heads
            num_v_heads = cfg.model.linear_num_value_heads
            conv_kernel_dim = cfg.model.linear_conv_kernel_dim

            qk_dim = qk_head_dim * num_qk_heads
            v_dim = v_head_dim * num_v_heads

            gdn_self_attn_per_layer = (
                3
                * 2
                * (
                    cfg.model.hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                    + conv_kernel_dim * (2 * qk_dim + v_dim)
                    + num_v_heads * (v_head_dim**2) * 4
                    + cfg.model.hidden_size * v_dim
                )
            )

            self_attn_term = (
                gdn_self_attn_per_layer * num_gdn_layers + standard_self_attn_per_layer * num_standard_attn_layers
            )

        padded_vocab_size = calculate_padded_vocab_size(
            cfg.model.vocab_size,
            cfg.model.make_vocab_size_divisible_by,
            cfg.model.tensor_model_parallel_size,
            logging_enabled=False,
        )

        # Routed expert MLP FLOPs per layer (accounts for latent compression).
        if moe_latent_size is None:
            routed_expert_term = moe_ffn_hidden_size * num_experts_routed_to * ffn_expansion_factor
        else:
            routed_expert_term = (
                moe_ffn_hidden_size
                * num_experts_routed_to
                * ffn_expansion_factor
                * moe_latent_size
                / cfg.model.hidden_size
            ) + 2 * moe_latent_size

        total_floating_point_operations = seqlen_sum * (
            # MLP
            3
            * 2
            * cfg.model.hidden_size
            * (
                # dense layers
                (cfg.model.ffn_hidden_size * ffn_expansion_factor) * num_dense_layers
                # routed experts
                + routed_expert_term * num_moe_layers
                # Shared Experts.
                + (shared_expert_ffn_hidden_size * ffn_expansion_factor) * num_moe_layers
            )
            # Self Attention
            + self_attn_term
            # MTP norms and proj
            + 3
            * 2
            * mtp_num_layers
            * (
                # MTP eh norm + final norm
                3 * cfg.model.hidden_size
                # MTP eh proj
                + 2 * cfg.model.hidden_size * cfg.model.hidden_size
            )
            # Logit.
            + 3 * 2 * cfg.model.hidden_size * padded_vocab_size * (mtp_num_layers + 1)
        )
        return total_floating_point_operations + _compute_vit_flops()

    def _compute_vit_flops():
        """Compute ViT encoder FLOPs if vision config is available.

        Note: num_vision_patches is the *total* patches across the batch.
        ViT attention is per-image (not cross-image), so we convert to
        per-image patch count before invoking ``vit_flops`` to get the
        correct quadratic attention scaling. ``vit_flops`` itself returns
        0 when ``cfg.model.vision_config`` is absent.
        """
        if num_vision_patches <= 0:
            return 0
        patches_per_image = num_vision_patches / batch_size if batch_size > 0 else num_vision_patches
        return vit_flops(cfg, batch_size, patches_per_image)

    # Main entrypoint for FLOPs calculation. Mirror MCore's hybrid detection:
    # a physical hybrid pattern is sufficient to select hybrid accounting.
    if getattr(cfg.model, "is_hybrid_model", False) or getattr(cfg.model, "hybrid_layer_pattern", None):
        # Calculate the number of each type of layer.
        num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers, num_gdn_layers = calculate_layer_counts()
        mtp_num_layers = getattr(cfg.model, "mtp_num_layers", None)
        if mtp_num_layers is None:
            # When using unified hybrid patterns, infer MTP depth count from the pattern.
            hybrid_pattern = getattr(cfg.model, "hybrid_layer_pattern", None)
            if hybrid_pattern:
                parsed = parse_hybrid_pattern(hybrid_pattern)
                mtp_num_layers = parsed.mtp_num_depths if parsed.mtp_pattern else 0
            else:
                mtp_num_layers = 0
        padded_vocab_size = calculate_padded_vocab_size(
            cfg.model.vocab_size,
            cfg.model.make_vocab_size_divisible_by,
            cfg.model.tensor_model_parallel_size,
            logging_enabled=False,
        )
        num_query_groups = (
            cfg.model.num_attention_heads if cfg.model.num_query_groups is None else cfg.model.num_query_groups
        )
        num_swa_attn_layers = count_hybrid_swa_attention_layers(num_attn_layers)
        swa_context = calculate_swa_context() if num_swa_attn_layers > 0 else None

        # Compute hybrid model FLOPs.
        llm_flops = hybrid_flops(
            batch_size=batch_size,
            seq_len=effective_seq_length,
            hidden_size=cfg.model.hidden_size,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            num_mlp_layers=num_mlp_layers,
            num_moe_layers=num_moe_layers,
            num_gdn_layers=num_gdn_layers,
            mamba_state_dim=getattr(cfg.model, "mamba_state_dim", 128),
            mamba_head_dim=getattr(cfg.model, "mamba_head_dim", 64),
            mamba_num_groups=getattr(cfg.model, "mamba_num_groups", 8),
            mamba_num_heads=getattr(cfg.model, "mamba_num_heads", 128),
            num_attn_heads=cfg.model.num_attention_heads,
            gqa_groups=num_query_groups,
            kv_channels=getattr(cfg.model, "kv_channels", None),
            mlp_expansion=cfg.model.ffn_hidden_size / cfg.model.hidden_size,
            swiglu=getattr(cfg.model, "gated_linear_unit", False),
            moe_latent_size=getattr(cfg.model, "moe_latent_size", None),
            moe_ffn_hidden_size=(
                cfg.model.ffn_hidden_size
                if getattr(cfg.model, "moe_ffn_hidden_size", None) is None
                else cfg.model.moe_ffn_hidden_size
            ),
            shared_expert_ffn_hidden_size=(
                0
                if getattr(cfg.model, "moe_shared_expert_intermediate_size", None) is None
                else cfg.model.moe_shared_expert_intermediate_size
            ),
            num_experts_routed_to=getattr(cfg.model, "moe_router_topk", 1),
            gdn_qk_head_dim=getattr(cfg.model, "linear_key_head_dim", None) or 128,
            gdn_v_head_dim=getattr(cfg.model, "linear_value_head_dim", None) or 128,
            gdn_num_qk_heads=getattr(cfg.model, "linear_num_key_heads", None) or 16,
            gdn_num_v_heads=getattr(cfg.model, "linear_num_value_heads", None) or 32,
            gdn_conv_kernel_dim=getattr(cfg.model, "linear_conv_kernel_dim", None) or 4,
            vocab_size=padded_vocab_size,
            mtp_num_layers=mtp_num_layers,
            num_swa_attn_layers=num_swa_attn_layers,
            swa_context=swa_context,
            core_attn_seq_factor=core_attn_seq_factor,
        )
        return llm_flops + _compute_vit_flops()
    else:
        # Compute standard Transformer model FLOPs.
        return transformer_flops()
