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

import importlib
from pathlib import Path

import torch.nn.functional as F

from megatron.bridge.data.datasets.packing_utils import calculate_avg_seqlen
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


_lora_seq_stats_cache: dict = {}


def vit_flops(
    batch_size,
    num_patches,
    depth,
    hidden_size,
    num_heads,
    intermediate_size,
    spatial_merge_size=2,
    out_hidden_size=4096,
):
    """Calculate FLOPs for a Vision Transformer (ViT) encoder + patch merger.

    Includes:
    - ViT transformer layers (bidirectional full attention, not causal)
    - Patch merger (spatial merge + MLP projection to LLM hidden size)

    Args:
        batch_size: Batch size.
        num_patches: Total number of vision patches (before spatial merge).
        depth: Number of ViT transformer layers.
        hidden_size: ViT hidden dimension.
        num_heads: Number of attention heads in ViT.
        intermediate_size: ViT MLP intermediate dimension.
        spatial_merge_size: Spatial merge factor (2 means 2x2 merge).
        out_hidden_size: Output hidden size after patch merger (LLM hidden size).

    Returns:
        Total training FLOPs (forward * 3 for fwd+bwd).
    """
    if num_patches <= 0:
        return 0

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
    seqlen_sum: int = None,
    seqlen_squared_sum: int = None,
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
            (batch_size * actual_seq_length^2). Used for attention core FLOPS
            which scale quadratically with sequence length.
        num_vision_patches: Total number of vision patches in the batch
            (before spatial merge). Used to compute ViT encoder FLOPS.
    """
    # Compute effective sequence length from actual values or fall back to config.
    if seqlen_sum is not None and batch_size > 0:
        effective_seq_length = seqlen_sum / batch_size
    else:
        effective_seq_length = cfg.model.seq_length
        seqlen_sum = batch_size * cfg.model.seq_length

    if seqlen_squared_sum is not None and batch_size > 0:
        effective_seq_length_sq = seqlen_squared_sum / batch_size
    else:
        effective_seq_length_sq = effective_seq_length**2

    peft = getattr(cfg, "peft", None)
    is_lora = isinstance(peft, LoRA)
    # If the model provider has a custom TFLOPS calculation method, use it (non-LoRA only).
    if not is_lora and hasattr(cfg.model, "_get_num_floating_point_operations"):
        return cfg.model._get_num_floating_point_operations(batch_size)

    def calculate_layer_counts():
        """Calculate the number of attention, Mamba, MLP, MoE, and GDN layers."""
        if hasattr(cfg.model, "hybrid_layer_pattern") and cfg.model.hybrid_layer_pattern:
            counts = {"M": 0, "G": 0, "*": 0, "-": 0, "E": 0}
            try:
                parse_hybrid_pattern = importlib.import_module(
                    "megatron.core.ssm.mamba_hybrid_layer_allocation"
                ).parse_hybrid_pattern
                parsed = parse_hybrid_pattern(cfg.model.hybrid_layer_pattern)
                if parsed.main_pattern:
                    for layer_type in parsed.main_pattern:
                        if layer_type in counts:
                            counts[layer_type] += 1
                if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
                    for layer_type in parsed.mtp_pattern:
                        if layer_type in counts:
                            counts[layer_type] += parsed.mtp_num_depths
            except (ImportError, ModuleNotFoundError):
                for layer_type in cfg.model.hybrid_layer_pattern:
                    if layer_type in counts:
                        counts[layer_type] += 1
            return counts["*"], counts["M"], counts["-"], counts["E"], counts["G"]
        else:
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

        is_squad = getattr(getattr(cfg, "dataset", None), "dataset_name", None) == "squad"
        hf_model_id = getattr(cfg.model, "hf_model_id", None)
        is_llama3_70b = hf_model_id is not None and "Meta-Llama-3-70B" in hf_model_id
        packed_specs = getattr(getattr(cfg, "dataset", None), "packed_sequence_specs", None)
        packed_data_path = getattr(packed_specs, "packed_train_data_path", None)
        # If not explicitly set, try to find the file via dataset_root (the FinetuningDatasetBuilder
        # computes this path dynamically, but dataset_root is available from the config).
        if packed_data_path is None and packed_specs is not None:
            dataset_root = getattr(cfg.dataset, "dataset_root", None)
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
        # SwiGLU: h->2*ffn_h and ffn_h->h = 3 projections; non-SwiGLU: h->ffn_h and ffn_h->h = 2 projections.
        ffn_expansion_factor = (
            3 if (cfg.model.gated_linear_unit is True and cfg.model.activation_func == F.silu) else 2
        )

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
                    + effective_seq_length
                    * (
                        cfg.model.num_attention_heads
                        * (getattr(cfg.model, "qk_head_dim", 64) + getattr(cfg.model, "qk_pos_emb_head_dim", 0))
                    )
                    / 2
                    + effective_seq_length * cfg.model.num_attention_heads * getattr(cfg.model, "v_head_dim", 64) / 2
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
                swa_context = min(effective_window, effective_seq_length)

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

                full_core = query_projection_size * effective_seq_length / 2 * 2
                swa_core = query_projection_size * swa_context / 2 * 2

                self_attn_term = (
                    3
                    * 2
                    * (
                        num_full_attn_layers * (proj_per_layer + full_core)
                        + num_swa_layers * (proj_per_layer + swa_core)
                    )
                )
            else:
                full_core = query_projection_size * effective_seq_length / 2 * 2
                self_attn_term = 3 * 2 * num_layers * (proj_per_layer + full_core)

        # Handle GDN (Gated DeltaNet) hybrid attention variant.
        # When experimental_attention_variant is "gated_delta_net", a fraction of the
        # layers use GDN instead of standard attention. Override self_attn_term with a
        # weighted sum of GDN and standard-attention per-layer costs.
        experimental_attention_variant = getattr(cfg.model, "experimental_attention_variant", None)
        if experimental_attention_variant == "gated_delta_net":
            linear_attention_freq = cfg.model.linear_attention_freq
            if linear_attention_freq is None:
                raise ValueError(
                    "linear_attention_freq must be set when experimental_attention_variant='gated_delta_net'"
                )
            if isinstance(linear_attention_freq, int):
                linear_attention_pattern = [
                    0 if ((i + 1) % linear_attention_freq == 0) else 1 for i in range(num_layers)
                ]
            elif isinstance(linear_attention_freq, list):
                linear_attention_pattern = linear_attention_freq
                if len(linear_attention_pattern) != num_layers:
                    raise ValueError(
                        f"Invalid length of linear_attention_pattern: {len(linear_attention_pattern)}, "
                        f"expected {num_layers}, "
                        f"current linear_attention_freq: {linear_attention_freq}"
                    )
            else:
                raise TypeError(
                    f"linear_attention_freq must be int or list, got {type(linear_attention_freq).__name__}"
                )

            num_gdn_layers = sum(linear_attention_pattern)
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

        total_floating_point_operations = (
            seqlen_sum
            * (
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
        )
        return total_floating_point_operations + _compute_vit_flops()

    def _compute_vit_flops():
        """Compute ViT encoder FLOPs if vision config is available.

        Note: num_vision_patches is the *total* patches across the batch.
        ViT attention is per-image (not cross-image), so we compute FLOPs
        using per-image patch count to get correct quadratic attention scaling.
        """
        vision_config = getattr(cfg.model, "vision_config", None)
        if vision_config is None or num_vision_patches <= 0:
            return 0
        patches_per_image = num_vision_patches / batch_size if batch_size > 0 else num_vision_patches
        return vit_flops(
            batch_size=batch_size,
            num_patches=patches_per_image,
            depth=getattr(vision_config, "depth", 0),
            hidden_size=getattr(vision_config, "hidden_size", 0),
            num_heads=getattr(vision_config, "num_heads", 16),
            intermediate_size=getattr(vision_config, "intermediate_size", 0),
            spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
            out_hidden_size=getattr(vision_config, "out_hidden_size", cfg.model.hidden_size),
        )

    # Main entrypoint for FLOPs calculation.
    if getattr(cfg.model, "is_hybrid_model", False):
        # Calculate the number of each type of layer.
        num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers, num_gdn_layers = calculate_layer_counts()
        mtp_num_layers = getattr(cfg.model, "mtp_num_layers", None)
        if mtp_num_layers is None:
            # When using unified hybrid patterns, infer MTP depth count from the pattern.
            hybrid_pattern = getattr(cfg.model, "hybrid_layer_pattern", None)
            if hybrid_pattern:
                try:
                    parse_hybrid_pattern = importlib.import_module(
                        "megatron.core.ssm.mamba_hybrid_layer_allocation"
                    ).parse_hybrid_pattern
                    parsed = parse_hybrid_pattern(hybrid_pattern)
                    mtp_num_layers = parsed.mtp_num_depths if parsed.mtp_pattern else 0
                except (ImportError, ModuleNotFoundError):
                    mtp_num_layers = 0
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
        )
        return llm_flops + _compute_vit_flops()
    else:
        # Compute standard Transformer model FLOPs.
        return transformer_flops()
