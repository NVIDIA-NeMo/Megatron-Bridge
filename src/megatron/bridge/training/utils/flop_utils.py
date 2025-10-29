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

    def calculate_layer_counts():
        """Calculate the number of attention, Mamba, and MLP layers."""
        if hasattr(cfg.model, "hybrid_override_pattern") and cfg.model.hybrid_override_pattern:
            counts = {"M": 0, "*": 0, "-": 0}
            for layer_type in cfg.model.hybrid_override_pattern:
                if layer_type in counts:
                    counts[layer_type] += 1
            return counts["*"], counts["M"], counts["-"]
        else:
            num_attn_layers = round(cfg.model.num_layers * getattr(cfg.model, "hybrid_attention_ratio", 0))
            num_mlp_layers = round(cfg.model.num_layers * getattr(cfg.model, "hybrid_mlp_ratio", 0))
            num_mamba_layers = cfg.model.num_layers - num_attn_layers - num_mlp_layers
            return num_attn_layers, num_mamba_layers, num_mlp_layers

    def mlp_layer_flops(batch_size, seq_len, hidden_size, expansion=4.0, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        return 4 * expansion * scale_factor * batch_size * seq_len * hidden_size**2

    def attn_layer_flops(
        batch_size,
        seq_len,
        hidden_size,
        num_heads,
        gqa=True,
        gqa_groups=8,
        kv_channels=None,
    ):
        """Calculate FLOPs for an attention layer."""
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups if gqa else num_heads
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
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=128,
        num_attn_heads=32,
        gqa=True,
        gqa_groups=8,
        kv_channels=None,
        mlp_expansion=4.0,
        swiglu=False,
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
                gqa,
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

        total_floating_point_operations = (
            batch_size
            * cfg.model.seq_length
            * (
                # MLP
                expansion_factor
                * num_layers
                * cfg.model.hidden_size
                * (
                    # dense layer (deepseek v2, v3 style)
                    (cfg.model.ffn_hidden_size * gated_linear_multiplier) * (num_dense_layers / num_layers)
                    # routed experts
                    + (moe_ffn_hidden_size * num_experts_routed_to * gated_linear_multiplier)
                    * (num_moe_layers / num_layers)
                    # Shared Experts.
                    + (shared_expert_ffn_hidden_size * gated_linear_multiplier) * (num_moe_layers / num_layers)
                )
                # Self Attention
                + self_attn_term
                # MTP norms and proj
                + 3
                * 2
                * mtp_num_layers
                * (
                    # MTP eh norm + final nrom
                    3 * cfg.model.hidden_size
                    # MTH eh proj
                    + 2 * cfg.model.hidden_size * cfg.model.hidden_size
                )
                # Logit.
                + 3 * 2 * cfg.model.hidden_size * padded_vocab_size * (mtp_num_layers + 1)
            )
        )
        return total_floating_point_operations

    # Main entrypoint for FLOPs calculation.
    if getattr(cfg.model, "is_hybrid_model", False):
        # TODO: Fix this when onboarding hybrid models
        # Calculate the number of each type of layer.
        # num_attn_layers, num_mamba_layers, num_mlp_layers = calculate_layer_counts()

        # # Compute hybrid model FLOPs.
        # return hybrid_flops(
        #     batch_size=batch_size,
        #     seq_len=cfg.model.seq_length,
        #     hidden_size=cfg.model.hidden_size,
        #     num_attn_layers=num_attn_layers,
        #     num_mamba_layers=num_mamba_layers,
        #     num_mlp_layers=num_mlp_layers,
        #     mamba_state_dim=getattr(cfg.model, 'mamba_state_dim', 128),
        #     mamba_head_dim=getattr(cfg.model, 'mamba_head_dim', 64),
        #     mamba_num_groups=getattr(cfg.model, 'mamba_num_groups', 8),
        #     mamba_num_heads=getattr(cfg.model, 'mamba_num_heads', 128),
        #     num_attn_heads=cfg.model.num_attention_heads,
        #     gqa=getattr(cfg.model, 'group_query_attention', False),
        #     gqa_groups=getattr(cfg.model, 'num_query_groups', 8),
        #     kv_channels=getattr(cfg.model, 'kv_channels', None),
        #     mlp_expansion=cfg.model.ffn_hidden_size / cfg.model.hidden_size,
        #     swiglu=getattr(cfg.model, 'gated_linear_unit', False),
        #     vocab_size=cfg.tokenizer.padded_vocab_size,
        # )
        return 0
    else:
        # Compute standard Transformer model FLOPs.
        return transformer_flops()
