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

import torch.nn.functional as F

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


def num_floating_point_operations(cfg: ConfigContainer, batch_size: int = 1):
    """Return the number of floating point operations"""
    # If the model provider has a custom TFLOPS calculation method, use it.
    if hasattr(cfg.model, "_get_num_floating_point_operations"):
        return cfg.model._get_num_floating_point_operations(batch_size)

    def vision_floating_point_operations(cfg: ConfigContainer, batch_size: int) -> float:
        """Estimate FLOPs for the vision tower + projector (patch merger).

        This complements the LM FLOPs estimate with a *vision-side* estimate for VLMs (e.g.,
        Qwen3-VL). Vision FLOPs depend on runtime inputs (image/video resolution, #frames),
        so this function prefers runtime statistics populated during `vlm_step.forward_step`.
        """
        vision_cfg = getattr(cfg.model, "vision_config", None)
        if vision_cfg is None:
            return 0.0

        depth = getattr(vision_cfg, "depth", None)
        d_model = getattr(vision_cfg, "hidden_size", None)
        d_ff = getattr(vision_cfg, "intermediate_size", None)
        n_heads = getattr(vision_cfg, "num_heads", None)
        spatial_merge_size = getattr(vision_cfg, "spatial_merge_size", None)
        out_hidden_size = getattr(vision_cfg, "out_hidden_size", None)
        num_position_embeddings = getattr(vision_cfg, "num_position_embeddings", None)
        in_channels = getattr(vision_cfg, "in_channels", 3)
        patch_size = getattr(vision_cfg, "patch_size", None)
        temporal_patch_size = getattr(vision_cfg, "temporal_patch_size", 1)

        critical = [
            depth,
            d_model,
            d_ff,
            n_heads,
            spatial_merge_size,
            out_hidden_size,
            num_position_embeddings,
        ]
        if any(v is None for v in critical):
            return 0.0

        merge_sq = int(spatial_merge_size) ** 2
        n_pre_runtime = getattr(cfg, "_runtime_vision_tokens_pre_per_sample", None)
        n_post_runtime = getattr(cfg, "_runtime_vision_tokens_post_per_sample", None)
        if n_pre_runtime is not None:
            n_pre = float(n_pre_runtime)
            n_post = float(n_post_runtime) if n_post_runtime is not None else n_pre / float(max(1, merge_sq))
        else:
            # Config-only fallback: assume one image per sample at the max grid size.
            n_pre = float(num_position_embeddings)
            n_post = n_pre / float(max(1, merge_sq))

        # Vision attention in Qwen3-VL is packed per-frame (each frame is a separate sequence
        # of length h*w). If runtime statistics are provided, prefer using sum(L^2) instead
        # of (sum L)^2 for the attention matmul term.
        sum_seqlen_sq_pre_runtime = getattr(cfg, "_runtime_vision_sum_seqlen_sq_pre_per_sample", None)

        # FLOPs are counted as 2 * MACs for GEMM/conv.
        # For training, we approximate backward as 2x forward -> total ≈ 3x forward.
        fwd_to_train_multiplier = 3.0

        # Patch embedding conv3d FLOPs:
        # FLOPs ≈ 2 * n_pre * d_model * (C * kt * kh * kw)
        flops_patch_embed_fwd = 0.0
        if patch_size is not None and temporal_patch_size is not None:
            kt = int(temporal_patch_size)
            kh = int(patch_size)
            kw = int(patch_size)
            flops_patch_embed_fwd = 2.0 * n_pre * float(d_model) * (int(in_channels) * kt * kh * kw)

        # Vision transformer block FLOPs (ViT-style, rough but standard):
        # - QKV projections + output projection: 8 * N * D^2
        # - Attention score + value: 4 * sum(L_i^2) * D (or 4 * N^2 * D if not packed)
        # - MLP: 4 * N * D * D_ff
        n = float(n_pre)
        if sum_seqlen_sq_pre_runtime is not None and float(sum_seqlen_sq_pre_runtime) > 0:
            attn_quad = 4.0 * float(sum_seqlen_sq_pre_runtime) * float(d_model)
        else:
            attn_quad = 4.0 * (n**2) * float(d_model)

        flops_vit_layer_fwd = (8.0 * n * (float(d_model) ** 2)) + attn_quad + (4.0 * n * float(d_model) * float(d_ff))
        flops_vit_fwd = float(depth) * flops_vit_layer_fwd

        # Patch merger / projector FLOPs:
        # First projection: (D*merge^2 -> D*merge^2)
        # Second projection: (D*merge^2 -> out_hidden_size)
        d_merge = float(d_model) * float(merge_sq)
        flops_merger_fwd = (2.0 * float(n_post) * d_merge * d_merge) + (
            2.0 * float(n_post) * d_merge * float(out_hidden_size)
        )

        # Deepstack mergers: approximate each as a patch merger.
        deepstack_idxs = getattr(vision_cfg, "deepstack_visual_indexes", None)
        num_deepstack = len(deepstack_idxs) if isinstance(deepstack_idxs, (list, tuple)) else 0
        flops_deepstack_mergers_fwd = float(num_deepstack) * flops_merger_fwd

        flops_vision_fwd_per_sample = (
            flops_patch_embed_fwd + flops_vit_fwd + flops_merger_fwd + flops_deepstack_mergers_fwd
        )
        flops_vision_fwd = float(batch_size) * float(flops_vision_fwd_per_sample)
        return flops_vision_fwd * fwd_to_train_multiplier

    def calculate_layer_counts():
        """Calculate the number of attention, Mamba, MLP, and MoE layers."""
        if hasattr(cfg.model, "hybrid_override_pattern") and cfg.model.hybrid_override_pattern:
            counts = {"M": 0, "*": 0, "-": 0, "E": 0}
            for layer_type in cfg.model.hybrid_override_pattern:
                if layer_type in counts:
                    counts[layer_type] += 1
            return counts["*"], counts["M"], counts["-"], counts["E"]
        else:
            num_attn_layers = round(cfg.model.num_layers * getattr(cfg.model, "hybrid_attention_ratio", 0))
            num_mlp_layers = round(cfg.model.num_layers * getattr(cfg.model, "hybrid_mlp_ratio", 0))
            num_mamba_layers = cfg.model.num_layers - num_attn_layers - num_mlp_layers
            num_moe_layers = 0
            return num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers

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
    ):
        """Calculate FLOPs for an attention layer."""
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups
        return (
            4
            * batch_size
            * seq_len
            * hidden_size
            * p
            * (hidden_size + (hidden_size * (g / num_heads)) + (seq_len / 2))
        )

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

    def hybrid_flops(
        batch_size,
        seq_len,
        hidden_size,
        num_attn_layers,
        num_mamba_layers,
        num_mlp_layers,
        num_moe_layers,
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
        vocab_size=256000,
    ):
        """Calculate total FLOPs for the hybrid model."""
        flops_fwd = (
            num_attn_layers
            * attn_layer_flops(
                batch_size,
                seq_len,
                hidden_size,
                num_attn_heads,
                gqa_groups,
                kv_channels,
            )
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
            + (2 * batch_size * seq_len * hidden_size * vocab_size)  # logits computation
        )
        return flops_fwd * 3

    def transformer_flops():
        """Calculate FLOPs for a standard Transformer model."""
        # TODO(helenn/dnarayanan): Refactor this to reuse the helper methods.
        # Attention projection size.
        query_projection_size = cfg.model.kv_channels * cfg.model.num_attention_heads
        query_projection_to_hidden_size_ratio = query_projection_size / cfg.model.hidden_size
        # GQA or MHA
        num_query_groups = (
            cfg.model.num_attention_heads if cfg.model.num_query_groups is None else cfg.model.num_query_groups
        )
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
        shared_expert_ffn_hidden_size = (
            0
            if cfg.model.moe_shared_expert_intermediate_size is None
            else cfg.model.moe_shared_expert_intermediate_size
        )
        # SwiGLU.
        gated_linear_multiplier = (
            3 / 2 if (cfg.model.gated_linear_unit is True and cfg.model.activation_func == F.silu) else 1
        )

        # The 12x term below comes from the following factors; for more details, see
        # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
        #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
        #       in MLP layer).
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        expansion_factor = 3 * 2 * 2

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
                    + (cfg.model.num_attention_heads * getattr(cfg.model, "v_head_dim", 64)) * cfg.model.hidden_size
                    ## core attn
                    + cfg.model.seq_length
                    * (
                        cfg.model.num_attention_heads
                        * (getattr(cfg.model, "qk_head_dim", 64) + getattr(cfg.model, "qk_pos_emb_head_dim", 0))
                    )
                    / 2
                    + cfg.model.seq_length * cfg.model.num_attention_heads * getattr(cfg.model, "v_head_dim", 64) / 2
                )
            )

        else:
            ## MHA or GQA
            self_attn_term = (
                expansion_factor
                * num_layers
                * cfg.model.hidden_size
                * cfg.model.hidden_size
                * (
                    (
                        1
                        + (num_query_groups / cfg.model.num_attention_heads)
                        # # Only half of the attention matrix is non-zero and needs to be multiplied with V.
                        + (cfg.model.seq_length / cfg.model.hidden_size / 2)
                    )
                    * query_projection_to_hidden_size_ratio
                )
            )

        padded_vocab_size = calculate_padded_vocab_size(
            cfg.model.vocab_size,
            cfg.model.make_vocab_size_divisible_by,
            cfg.model.tensor_model_parallel_size,
            logging_enabled=False,
        )

        # ---------------------------------------------------------------------
        # Runtime token statistics (packed / padded)
        # ---------------------------------------------------------------------
        tokens_padded_per_sample = getattr(cfg, "_runtime_lm_total_tokens_padded_per_sample", None)
        sum_seqlen_sq_padded_per_sample = getattr(cfg, "_runtime_lm_sum_seqlen_sq_padded_per_sample", None)

        if tokens_padded_per_sample is None or float(tokens_padded_per_sample) <= 0:
            tokens_padded_per_sample = float(cfg.model.seq_length)
        tokens_total = float(batch_size) * float(tokens_padded_per_sample)

        if sum_seqlen_sq_padded_per_sample is None or float(sum_seqlen_sq_padded_per_sample) <= 0:
            sum_seqlen_sq_total = float(batch_size) * float(cfg.model.seq_length) * float(cfg.model.seq_length)
        else:
            sum_seqlen_sq_total = float(batch_size) * float(sum_seqlen_sq_padded_per_sample)

        # MLP (linear in tokens_total).
        mlp_total = (
            tokens_total
            * expansion_factor
            * float(num_layers)
            * float(cfg.model.hidden_size)
            * (
                (float(cfg.model.ffn_hidden_size) * float(gated_linear_multiplier)) * (float(num_dense_layers) / float(num_layers))
                + (float(moe_ffn_hidden_size) * float(num_experts_routed_to) * float(gated_linear_multiplier))
                * (float(num_moe_layers) / float(num_layers))
                + (float(shared_expert_ffn_hidden_size) * float(gated_linear_multiplier)) * (float(num_moe_layers) / float(num_layers))
            )
        )

        # MTP norms and proj (linear in tokens_total).
        mtp_total = (
            tokens_total
            * 3
            * 2
            * float(mtp_num_layers)
            * (
                3 * float(cfg.model.hidden_size)
                + 2 * float(cfg.model.hidden_size) * float(cfg.model.hidden_size)
            )
        )

        # Logit (linear in tokens_total).
        logit_total = tokens_total * 3 * 2 * float(cfg.model.hidden_size) * float(padded_vocab_size) * float(mtp_num_layers + 1)

        # Self-attention:
        # - projection and output terms scale with sum(L_i)
        # - attention matmul terms scale with sum(L_i^2) (causal uses ~half, reflected by /2)
        if cfg.model.multi_latent_attention:
            # Keep the existing MLA approximation (uses cfg.model.seq_length).
            self_attn_total = float(batch_size) * float(cfg.model.seq_length) * float(self_attn_term)
        else:
            attn_linear_factor = 1.0 + (float(num_query_groups) / float(cfg.model.num_attention_heads))
            attn_base = (
                expansion_factor
                * float(num_layers)
                * float(cfg.model.hidden_size)
                * float(cfg.model.hidden_size)
                * float(query_projection_to_hidden_size_ratio)
            )
            self_attn_total = attn_base * (
                attn_linear_factor * tokens_total + (sum_seqlen_sq_total / (2.0 * float(cfg.model.hidden_size)))
            )

        total_floating_point_operations = mlp_total + self_attn_total + mtp_total + logit_total
        total_floating_point_operations += vision_floating_point_operations(cfg, batch_size)
        return total_floating_point_operations

    # Main entrypoint for FLOPs calculation.
    if getattr(cfg.model, "is_hybrid_model", False):
        # Calculate the number of each type of layer.
        num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers = calculate_layer_counts()
        padded_vocab_size = calculate_padded_vocab_size(
            cfg.model.vocab_size,
            cfg.model.make_vocab_size_divisible_by,
            cfg.model.tensor_model_parallel_size,
            logging_enabled=False,
        )
        num_query_groups = (
            cfg.model.num_attention_heads if cfg.model.num_query_groups is None else cfg.model.num_query_groups
        )

        # Compute hybrid model FLOPs.
        total_floating_point_operations = hybrid_flops(
            batch_size=batch_size,
            seq_len=cfg.model.seq_length,
            hidden_size=cfg.model.hidden_size,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            num_mlp_layers=num_mlp_layers,
            num_moe_layers=num_moe_layers,
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
            vocab_size=padded_vocab_size,
        )
        total_floating_point_operations += vision_floating_point_operations(cfg, batch_size)
        return total_floating_point_operations
    else:
        # Compute standard Transformer model FLOPs.
        return transformer_flops()
