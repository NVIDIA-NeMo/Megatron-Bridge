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

"""Computes theoretical memory footprint for model training."""

import math
from typing import Optional

import torch.nn.functional as F

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


NUM_BYTES_IN_MEGABYTE: int = 1024 * 1024


def compute_weight_and_optimizer_memory(config: ConfigContainer, verbose: bool = False) -> float:
    """Compute theoretical memory footprint for model weights and optimizer states.

    Splits parameters into three buckets — regular tensor-parallel-sharded,
    replicated, and routed-expert — and applies the correct sharding factor
    per bucket. Routed experts are sharded by ``expert_tensor_parallel_size *
    expert_model_parallel_size``, with distributed-optimizer state sized by the
    expert data-parallel domain. Other parameters use the regular TP and DP
    domains. Supports MoE with optional shared experts, multi-latent attention
    (DeepSeek), and Multi-Token Prediction (MTP) blocks.

    Ported from ``megatron/training/theoretical_memory_usage.py`` in
    Megatron-LM (NVIDIA/Megatron-LM PR #4687).

    Args:
        config (ConfigContainer): The main configuration container.
        verbose (bool, optional): If True, prints detailed parameter counts.
                                Defaults to False.

    Returns:
        float: Estimated memory footprint in bytes for weights and optimizer states
               on the most loaded GPU shard.
    """
    model_config = config.model
    # Attention projection size.
    query_projection_size = model_config.kv_channels * model_config.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / model_config.hidden_size
    # Group Query Attention.
    num_query_groups = (
        model_config.num_query_groups if model_config.num_query_groups else model_config.num_attention_heads
    )
    # MoE.
    num_experts = 1 if model_config.num_moe_experts is None else model_config.num_moe_experts
    gated_linear_multiplier = 3 / 2 if model_config.gated_linear_unit and model_config.activation_func == F.silu else 1

    shared_expert_ffn_hidden_size = (
        0
        if model_config.moe_shared_expert_intermediate_size is None
        else model_config.moe_shared_expert_intermediate_size
    )

    if model_config.num_moe_experts is not None:
        if isinstance(model_config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % model_config.moe_layer_freq == 0) else 0 for i in range(model_config.num_layers)
            ]
        elif isinstance(model_config.moe_layer_freq, list):
            moe_layer_pattern = model_config.moe_layer_freq
            assert len(moe_layer_pattern) == model_config.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {model_config.num_layers}, "
                f"current moe layer pattern: {model_config.moe_layer_freq}"
            )
        else:
            raise TypeError(f"moe_layer_freq must be int or list, got {type(model_config.moe_layer_freq).__name__}")

        num_dense_layers = model_config.num_layers - sum(moe_layer_pattern)
        num_moe_layers = sum(moe_layer_pattern)
        moe_ffn_hidden_size = model_config.moe_ffn_hidden_size
    else:
        moe_layer_pattern = [0] * model_config.num_layers
        num_dense_layers = model_config.num_layers
        num_moe_layers = 0
        moe_ffn_hidden_size = 0
    assert num_dense_layers + num_moe_layers == model_config.num_layers

    mtp_num_layers = getattr(model_config, "mtp_num_layers", None)
    if mtp_num_layers is not None:
        mtp_layer_is_moe = moe_layer_pattern[-1]
        mtp_num_moe_layers = mtp_layer_is_moe * mtp_num_layers
        mtp_num_dense_layers = (1 - mtp_layer_is_moe) * mtp_num_layers
    else:
        mtp_num_moe_layers = 0
        mtp_num_dense_layers = 0

    # RMSNorm does not have bias, but LayerNorm has.
    norm_size = 1 if model_config.normalization == "RMSNorm" else 2

    if getattr(model_config, "multi_latent_attention", False):
        q_lora_rank = getattr(model_config, "q_lora_rank", None)
        kv_lora_rank = model_config.kv_lora_rank
        qk_head_dim = model_config.qk_head_dim
        qk_pos_emb_head_dim = model_config.qk_pos_emb_head_dim
        v_head_dim = model_config.v_head_dim
        if q_lora_rank is None:
            q_term = model_config.hidden_size * model_config.num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)
        else:
            # q lora + rope + q norm
            q_term = q_lora_rank * (
                model_config.hidden_size
                + model_config.num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)
                + norm_size
            )

        self_attn_term = (
            q_term
            # kv lora + rope + kv norm
            + kv_lora_rank
            * (model_config.hidden_size + model_config.num_attention_heads * (qk_head_dim + v_head_dim) + norm_size)
            + model_config.hidden_size * qk_pos_emb_head_dim
            # o proj
            + (model_config.num_attention_heads * v_head_dim) * model_config.hidden_size
        )
    else:
        # Self-attention linear weights: fused QKV plus output projection.
        self_attn_term = (
            2
            * model_config.hidden_size
            * model_config.hidden_size
            *
            # Attention.
            ((1 + (num_query_groups / model_config.num_attention_heads)) * query_projection_to_hidden_size_ratio)
        )

    # Per-layer attention linear parameters, sharded by the regular tensor-parallel group.
    num_parameters_in_attention = self_attn_term
    # Per-layer dense MLP parameters
    num_parameters_in_dense_mlp = 2 * model_config.hidden_size * model_config.ffn_hidden_size * gated_linear_multiplier
    # Per-layer routed expert MLP parameters across all experts
    num_parameters_in_routed_experts = (
        2 * model_config.hidden_size * moe_ffn_hidden_size * num_experts * gated_linear_multiplier
    )
    # Per-layer routed expert parameters active for one token; top-k experts are used per token.
    num_active_parameters_in_routed_experts = (
        2 * model_config.hidden_size * moe_ffn_hidden_size * model_config.moe_router_topk * gated_linear_multiplier
        if model_config.num_moe_experts is not None
        else 0
    )
    # Per-layer shared expert MLP parameters; shared experts use regular TP, not ETP.
    num_parameters_in_shared_experts = (
        2 * model_config.hidden_size * shared_expert_ffn_hidden_size * gated_linear_multiplier
    )
    # Per-layer normalization parameters; the factor 2 counts input norm and pre-MLP norm.
    num_parameters_in_layernorms = 2 * model_config.hidden_size * norm_size
    # Per-layer optional shared expert gate weight, replicated across tensor-parallel ranks.
    num_parameters_in_shared_expert_gate = (
        model_config.hidden_size
        if shared_expert_ffn_hidden_size > 0 and getattr(model_config, "moe_shared_expert_gate", False)
        else 0
    )
    # Per-layer router gate parameters, replicated across tensor-parallel ranks.
    num_parameters_in_router = (
        (model_config.hidden_size * num_experts) + (num_experts if model_config.add_bias_linear else 0)
        if model_config.num_moe_experts is not None
        else 0
    )

    # Per dense transformer layer, parameters sharded by regular tensor parallelism.
    num_tp_sharded_parameters_in_transformer_layer_dense = num_parameters_in_attention + num_parameters_in_dense_mlp
    # Per dense transformer layer, parameters replicated across tensor-parallel ranks.
    num_replicated_parameters_in_transformer_layer_dense = num_parameters_in_layernorms
    # Per dense transformer layer, total logical parameters before any parallel sharding.
    num_parameters_in_transformer_layer_dense = (
        num_tp_sharded_parameters_in_transformer_layer_dense + num_replicated_parameters_in_transformer_layer_dense
    )

    # Per MoE transformer layer, non-routed parameters sharded by regular tensor parallelism.
    num_tp_sharded_parameters_in_transformer_layer_moe = num_parameters_in_attention + num_parameters_in_shared_experts
    # Per MoE transformer layer, non-routed parameters replicated across tensor-parallel ranks.
    num_replicated_parameters_in_transformer_layer_moe = (
        num_parameters_in_layernorms + num_parameters_in_router + num_parameters_in_shared_expert_gate
    )
    # Per MoE transformer layer, total logical parameters before any parallel sharding.
    num_parameters_in_transformer_layer_moe = (
        num_tp_sharded_parameters_in_transformer_layer_moe
        + num_replicated_parameters_in_transformer_layer_moe
        + num_parameters_in_routed_experts
    )
    # Per MoE transformer layer, logical parameters used by one routed token.
    num_active_parameters_in_transformer_layer_moe = (
        num_tp_sharded_parameters_in_transformer_layer_moe
        + num_replicated_parameters_in_transformer_layer_moe
        + num_active_parameters_in_routed_experts
    )
    # Input embedding table parameters.
    embedding_size = model_config.hidden_size * _get_vocab_size(model_config)
    # Final normalization parameters, replicated across tensor-parallel ranks.
    final_layernorm = norm_size * model_config.hidden_size
    if not model_config.share_embeddings_and_output_weights:
        # Untied embeddings have separate input embedding and output LM-head tables.
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        # Tied embeddings share the input embedding and output LM-head table.
        num_parameters_in_embedding_layers = embedding_size

    # Transformer block parameters that will be divided by regular tensor parallelism.
    num_tp_sharded_parameters_in_transformer_block = (
        num_tp_sharded_parameters_in_transformer_layer_dense * num_dense_layers
        + num_tp_sharded_parameters_in_transformer_layer_moe * num_moe_layers
    )
    # Transformer block parameters replicated across regular tensor-parallel ranks.
    num_replicated_parameters_in_transformer_block = (
        num_replicated_parameters_in_transformer_layer_dense * num_dense_layers
        + num_replicated_parameters_in_transformer_layer_moe * num_moe_layers
        + final_layernorm
    )
    # Transformer block routed expert parameters that will be divided by ETP and EP.
    num_routed_expert_parameters_in_transformer_block = num_parameters_in_routed_experts * num_moe_layers
    # Total logical transformer block parameters before model-parallel sharding.
    num_parameters_in_transformer_block = (
        num_parameters_in_transformer_layer_dense * num_dense_layers
        + num_parameters_in_transformer_layer_moe * num_moe_layers
        + final_layernorm
    )
    # Total logical active transformer block parameters before model-parallel sharding.
    num_active_parameters_in_transformer_block = (
        num_parameters_in_transformer_layer_dense * num_dense_layers
        + num_active_parameters_in_transformer_layer_moe * num_moe_layers
        + final_layernorm
    )

    # MTP block parameters that will be divided by regular tensor parallelism.
    num_tp_sharded_parameters_in_mtp_block = (
        num_tp_sharded_parameters_in_transformer_layer_dense * mtp_num_dense_layers
        + num_tp_sharded_parameters_in_transformer_layer_moe * mtp_num_moe_layers
    )
    # MTP block parameters replicated across regular tensor-parallel ranks.
    num_replicated_parameters_in_mtp_block = (
        num_replicated_parameters_in_transformer_layer_dense * mtp_num_dense_layers
        + num_replicated_parameters_in_transformer_layer_moe * mtp_num_moe_layers
    )
    # MTP block routed expert parameters that will be divided by ETP and EP.
    num_routed_expert_parameters_in_mtp_block = num_parameters_in_routed_experts * mtp_num_moe_layers
    # Total logical MTP block parameters before model-parallel sharding.
    num_parameters_in_mtp_block = (
        num_parameters_in_transformer_layer_dense * mtp_num_dense_layers
        + num_parameters_in_transformer_layer_moe * mtp_num_moe_layers
    )

    num_total_parameters = (
        num_parameters_in_transformer_block + num_parameters_in_mtp_block + num_parameters_in_embedding_layers
    )
    num_active_parameters = (
        num_active_parameters_in_transformer_block + num_parameters_in_mtp_block + num_parameters_in_embedding_layers
    )
    if verbose:
        print(
            f"Number of parameters in transformer block in billions: "
            f"{num_parameters_in_transformer_block / 10**9: .2f}"
        )
        print(
            f"Number of active parameters in transformer block in billions: "
            f"{num_active_parameters_in_transformer_block / 10**9: .2f}"
        )
        if mtp_num_layers is not None:
            print(f"Number of parameters in mtp block in billions: {num_parameters_in_mtp_block / 10**9: .2f}")
        print(
            f"Number of parameters in embedding layers in billions: {num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")
        print(f"Total number of active parameters in billions: {num_active_parameters / 10**9:.2f}")

    # Number of ranks that shard each routed expert's tensor dimensions.
    expert_tensor_parallel_size = (
        model_config.expert_tensor_parallel_size
        if model_config.expert_tensor_parallel_size is not None
        else model_config.tensor_model_parallel_size
    )
    # Number of ranks that split the global set of routed experts.
    expert_model_parallel_size = model_config.expert_model_parallel_size
    # Number of ranks in one expert tensor/expert/pipeline model-parallel group.
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * model_config.pipeline_model_parallel_size
    )
    # Bridge's world_size is DP * TP * PP * CP. Mirror that here so the EDP
    # derivation matches the global rank count Bridge expects.
    world_size = (
        config.data_parallel_size
        * model_config.tensor_model_parallel_size
        * model_config.pipeline_model_parallel_size
        * model_config.context_parallel_size
    )
    # Data-parallel size used by expert parameters and distributed optimizer state.
    expert_data_parallel_size = world_size // expert_tensor_model_pipeline_parallel_size

    # Most loaded model shard has 1/pp_size transformer layers, 1 mtp block, and 1 embedding layer.
    # TP-sharded dense/shared parameters use regular TP. Routed experts use ETP and EP. Router and
    # normalization parameters are replicated across TP/ETP ranks.
    num_tp_sharded_parameters_on_most_loaded_model_shard = (
        (num_tp_sharded_parameters_in_transformer_block / model_config.pipeline_model_parallel_size)
        + num_tp_sharded_parameters_in_mtp_block
        + embedding_size
    ) / model_config.tensor_model_parallel_size
    num_replicated_parameters_on_most_loaded_model_shard = (
        num_replicated_parameters_in_transformer_block / model_config.pipeline_model_parallel_size
    ) + num_replicated_parameters_in_mtp_block
    num_routed_expert_parameters_on_most_loaded_model_shard = (
        (num_routed_expert_parameters_in_transformer_block / model_config.pipeline_model_parallel_size)
        + num_routed_expert_parameters_in_mtp_block
    ) / (expert_tensor_parallel_size * expert_model_parallel_size)
    num_parameters_on_most_loaded_model_shard = (
        num_tp_sharded_parameters_on_most_loaded_model_shard
        + num_replicated_parameters_on_most_loaded_model_shard
        + num_routed_expert_parameters_on_most_loaded_model_shard
    )
    if not model_config.share_embeddings_and_output_weights and model_config.pipeline_model_parallel_size == 1:
        num_tp_sharded_parameters_on_most_loaded_model_shard += (
            embedding_size / model_config.tensor_model_parallel_size
        )
        num_parameters_on_most_loaded_model_shard += embedding_size / model_config.tensor_model_parallel_size
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if model_config.pipeline_model_parallel_size > 1:
        # Other shards just have 1/pp_size transformer layers.
        num_parameters_on_other_model_shards = (
            num_tp_sharded_parameters_in_transformer_block
            / (model_config.pipeline_model_parallel_size * model_config.tensor_model_parallel_size)
            + num_replicated_parameters_in_transformer_block / model_config.pipeline_model_parallel_size
            + num_routed_expert_parameters_in_transformer_block
            / (model_config.pipeline_model_parallel_size * expert_tensor_parallel_size * expert_model_parallel_size)
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: {num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    # Bf16 training bytes per logical parameter for the given data-parallel domain.
    # Assumes bf16 model params, fp32 main grads, fp32 main params, and fp32 Adam states.
    def num_bytes_per_parameter(data_parallel_size: int) -> float:
        if not config.optimizer.use_distributed_optimizer:
            return 18
        return 6 + (12 / data_parallel_size)

    # Per-rank memory for weights, gradients, main params, and optimizer state.
    weight_and_optimizer_memory = (
        num_tp_sharded_parameters_on_most_loaded_model_shard + num_replicated_parameters_on_most_loaded_model_shard
    ) * num_bytes_per_parameter(config.data_parallel_size) + (
        num_routed_expert_parameters_on_most_loaded_model_shard * num_bytes_per_parameter(expert_data_parallel_size)
    )

    return weight_and_optimizer_memory


def compute_activation_memory(
    config: ConfigContainer, num_microbatches: Optional[int], verbose: bool = False
) -> float:
    """Compute theoretical memory footprint for activations.

    Estimates activation memory based on the formula from the Megatron-LM paper
    (Table 2, https://arxiv.org/pdf/2205.05198.pdf), accounting for sequence length,
    batch size, hidden size, number of layers, parallelism degrees (TP, PP, virtual PP),
    and other model specifics.

    Note:
        Currently assumes selective activation recomputation and sequence parallelism.
        Calculations focus on the first pipeline stage, which typically has the
        highest activation memory footprint.

    Args:
        config (ConfigContainer): The main configuration container.
        num_microbatches (int, optional): The number of microbatches used in training.
        verbose (bool, optional): If True, prints intermediate memory calculations.
                                Defaults to False.

    Returns:
        float: Estimated activation memory footprint in bytes on a single GPU shard.
    """
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.

    model_config = config.model
    train_config = config.train

    # Memory footprint from transformer layer (self-attention and MLP).
    activation_memory = (model_config.seq_length * train_config.micro_batch_size * model_config.hidden_size) * (
        18 + (4 * (model_config.ffn_hidden_size / model_config.hidden_size))
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / model_config.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= model_config.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * model_config.seq_length * train_config.micro_batch_size * model_config.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        model_config.seq_length
        * train_config.micro_batch_size
        * model_config.hidden_size
        * model_config.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if model_config.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (model_config.pipeline_model_parallel_size - 1)
            / (model_config.pipeline_model_parallel_size * model_config.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * model_config.pipeline_model_parallel_size
        )
        if verbose:
            print(f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}")
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if model_config.virtual_pipeline_model_parallel_size is None and model_config.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / model_config.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, model_config.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = model_config.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if model_config.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            model_config.seq_length
            * train_config.micro_batch_size
            * model_config.hidden_size
            * 4
            * (1 + (_get_vocab_size(model_config) / model_config.hidden_size))
        )

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory / model_config.tensor_model_parallel_size


def report_theoretical_memory(
    config: ConfigContainer, num_microbatches: Optional[int] = None, verbose: bool = False
) -> None:
    """Compute and print the theoretical memory footprint components.

    Calls `compute_weight_and_optimizer_memory` and `compute_activation_memory`
    (if applicable based on config) and prints the results in MB.

    Args:
        config (ConfigContainer): The main configuration container.
        num_microbatches (int, optional): The number of microbatches. Required for
                                        accurate activation memory estimation with PP.
                                        Defaults to None.
        verbose (bool, optional): If True, passes verbosity flag to helper functions.
                                Defaults to False.
    """
    # Skip for MegatronMIMO: MegatronMIMOProvider is not a TransformerConfig, so it lacks
    # kv_channels/num_attention_heads/etc. needed for the calculation.
    # (Other providers like GPTModelProvider inherit TransformerConfig and work fine.)
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider

    if isinstance(config.model, MegatronMIMOProvider):
        return

    weight_and_optimizer_memory = compute_weight_and_optimizer_memory(config, verbose=verbose) / NUM_BYTES_IN_MEGABYTE

    # Formulae here assume sequence parallelism and selective activation recomputation.
    if not config.model.sequence_parallel or config.model.recompute_granularity != "selective":
        print(f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB")
        return

    activation_memory = (
        compute_activation_memory(config, num_microbatches=num_microbatches, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )


def _get_vocab_size(model_cfg) -> int:
    """Get the potentially padded vocabulary size for the given configuration.

    Args:
        cfg: The model provider configuration.

    Returns:
        int: The vocabulary size used.
    """
    if model_cfg.should_pad_vocab:
        return calculate_padded_vocab_size(
            model_cfg.vocab_size,
            model_cfg.make_vocab_size_divisible_by,
            model_cfg.tensor_model_parallel_size,
            logging_enabled=False,
        )
    else:
        return model_cfg.vocab_size
