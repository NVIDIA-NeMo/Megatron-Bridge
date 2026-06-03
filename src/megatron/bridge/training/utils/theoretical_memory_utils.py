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

"""Formula-based theoretical memory estimates for model training.

The estimator logic is adapted for Megatron Bridge from the public ISEEKYAN
Megatron memory estimator implementation:
https://github.com/ISEEKYAN/mbridge/tree/main/memory_estimator
"""

import math
from dataclasses import dataclass

import torch.nn.functional as F

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


NUM_BYTES_IN_MEGABYTE: int = 1024 * 1024
NUM_BYTES_IN_GIGABYTE: int = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MemoryComponentEstimate:
    """Estimated memory for one per-GPU training memory component.

    Args:
        name: Human-readable component name.
        parameter_count: Global parameter count covered by this component.
        parameter_count_per_gpu: Parameter count on the most-loaded GPU shard.
        bytes_per_parameter: Per-parameter bytes for weights, gradients, and optimizer states.
        memory_bytes: Estimated memory on the most-loaded GPU shard.
    """

    name: str
    parameter_count: float = 0.0
    parameter_count_per_gpu: float = 0.0
    bytes_per_parameter: float = 0.0
    memory_bytes: float = 0.0

    @property
    def memory_mb(self) -> float:
        """Memory in MiB."""
        return self.memory_bytes / NUM_BYTES_IN_MEGABYTE

    @property
    def memory_gb(self) -> float:
        """Memory in GiB."""
        return self.memory_bytes / NUM_BYTES_IN_GIGABYTE


@dataclass(frozen=True, slots=True)
class TrainingMemoryEstimate:
    """Structured theoretical per-GPU memory estimate for Bridge training.

    Args:
        model_state_components: Weight, gradient, and optimizer-state components.
        activation: Activation component, if activation estimation was requested.
        total_parameters: Global model parameter count covered by the estimator.
        assumptions: Estimator assumptions and intentionally unsupported details.
    """

    model_state_components: tuple[MemoryComponentEstimate, ...]
    activation: MemoryComponentEstimate | None
    total_parameters: float
    assumptions: tuple[str, ...]

    @property
    def weight_and_optimizer_bytes(self) -> float:
        """Estimated per-GPU memory for weights, gradients, and optimizer states."""
        return sum(component.memory_bytes for component in self.model_state_components)

    @property
    def total_memory_bytes(self) -> float:
        """Estimated per-GPU training memory for all available components."""
        activation_bytes = 0.0 if self.activation is None else self.activation.memory_bytes
        return self.weight_and_optimizer_bytes + activation_bytes

    @property
    def total_memory_mb(self) -> float:
        """Total estimated per-GPU memory in MiB."""
        return self.total_memory_bytes / NUM_BYTES_IN_MEGABYTE

    @property
    def total_memory_gb(self) -> float:
        """Total estimated per-GPU memory in GiB."""
        return self.total_memory_bytes / NUM_BYTES_IN_GIGABYTE


@dataclass(frozen=True, slots=True)
class _LayerCounts:
    dense: int
    moe: int
    total: int


@dataclass(frozen=True, slots=True)
class _ParameterCounts:
    dense_transformer: float
    routed_experts: float
    embeddings: float

    @property
    def total(self) -> float:
        return self.dense_transformer + self.routed_experts + self.embeddings


def estimate_training_memory(
    config: ConfigContainer,
    num_microbatches: int | None = None,
    *,
    include_activation: bool = True,
) -> TrainingMemoryEstimate:
    """Estimate per-GPU training memory for a Bridge GPT-like model config.

    The estimator is intentionally formula-based. It does not instantiate a Megatron
    model or import UI/debug dependencies from the external prototype linked in
    issue #1673. The returned structure separates dense/embedding model state,
    routed expert model state, and activation memory so callers can display or
    post-process the breakdown.

    The estimator logic is adapted from the public ISEEKYAN Megatron memory
    estimator implementation.

    Args:
        config: Bridge training configuration container.
        num_microbatches: Number of microbatches in the pipeline schedule.
            Supplying this improves activation estimates when PP is enabled.
        include_activation: Include the activation-memory estimate. The activation
            formula assumes sequence parallelism and selective recomputation, matching
            the legacy training-time report.

    Returns:
        Structured per-GPU theoretical memory estimate.
    """
    model_config = config.model
    parameter_counts = _count_parameters(model_config)

    tensor_parallel_size = _positive_int_attr(model_config, "tensor_model_parallel_size", 1)
    pipeline_parallel_size = _positive_int_attr(model_config, "pipeline_model_parallel_size", 1)
    context_parallel_size = _positive_int_attr(model_config, "context_parallel_size", 1)
    expert_parallel_size = _positive_int_attr(model_config, "expert_model_parallel_size", 1)
    expert_tensor_parallel_size = _positive_int_attr(model_config, "expert_tensor_parallel_size", 1)

    dense_parameters_per_gpu = parameter_counts.dense_transformer / (pipeline_parallel_size * tensor_parallel_size)
    dense_parameters_per_gpu += _embedding_parameters_on_most_loaded_shard(model_config)
    dense_optimizer_shard_size = _positive_int_attr(config, "data_parallel_size", 1) * context_parallel_size
    dense_bytes_per_parameter = _bytes_per_parameter(config, dense_optimizer_shard_size)

    components = [
        MemoryComponentEstimate(
            name="dense parameters and optimizer",
            parameter_count=parameter_counts.dense_transformer + parameter_counts.embeddings,
            parameter_count_per_gpu=dense_parameters_per_gpu,
            bytes_per_parameter=dense_bytes_per_parameter,
            memory_bytes=dense_parameters_per_gpu * dense_bytes_per_parameter,
        )
    ]

    if parameter_counts.routed_experts > 0:
        routed_expert_parameters_per_gpu = parameter_counts.routed_experts / (
            pipeline_parallel_size * expert_parallel_size * expert_tensor_parallel_size
        )
        expert_optimizer_shard_size = _expert_optimizer_shard_size(
            config,
            tensor_parallel_size=tensor_parallel_size,
            context_parallel_size=context_parallel_size,
            expert_parallel_size=expert_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
        )
        expert_bytes_per_parameter = _bytes_per_parameter(config, expert_optimizer_shard_size)
        components.append(
            MemoryComponentEstimate(
                name="routed expert parameters and optimizer",
                parameter_count=parameter_counts.routed_experts,
                parameter_count_per_gpu=routed_expert_parameters_per_gpu,
                bytes_per_parameter=expert_bytes_per_parameter,
                memory_bytes=routed_expert_parameters_per_gpu * expert_bytes_per_parameter,
            )
        )

    activation = None
    if include_activation:
        activation = MemoryComponentEstimate(
            name="activation",
            memory_bytes=_compute_activation_memory_bytes(config, num_microbatches=num_microbatches),
        )

    assumptions = _estimate_assumptions(
        config,
        include_activation=include_activation,
        has_routed_experts=parameter_counts.routed_experts > 0,
    )
    return TrainingMemoryEstimate(
        model_state_components=tuple(components),
        activation=activation,
        total_parameters=parameter_counts.total,
        assumptions=assumptions,
    )


def format_training_memory_estimate(estimate: TrainingMemoryEstimate, *, unit: str = "MB") -> str:
    """Format a theoretical memory estimate as a compact single-line summary.

    Args:
        estimate: Structured estimate returned by :func:`estimate_training_memory`.
        unit: Either ``"MB"`` for MiB output or ``"GB"`` for GiB output.

    Returns:
        Human-readable summary string.

    Raises:
        ValueError: If ``unit`` is not ``"MB"`` or ``"GB"``.
    """
    unit = unit.upper()
    if unit == "MB":
        scale = NUM_BYTES_IN_MEGABYTE
    elif unit == "GB":
        scale = NUM_BYTES_IN_GIGABYTE
    else:
        raise ValueError(f"Unsupported memory unit: {unit}")

    component_parts = [
        f"{component.name}={component.memory_bytes / scale:.2f} {unit}"
        for component in estimate.model_state_components
    ]
    if estimate.activation is not None:
        component_parts.append(f"{estimate.activation.name}={estimate.activation.memory_bytes / scale:.2f} {unit}")
    component_parts.append(f"total={estimate.total_memory_bytes / scale:.2f} {unit}")
    return "Theoretical memory footprints: " + ", ".join(component_parts)


def compute_weight_and_optimizer_memory(config: ConfigContainer, verbose: bool = False) -> float:
    """Compute theoretical memory footprint for model weights and optimizer states.

    Calculates the number of parameters for the model based on the configuration,
    determines the number of parameters on the most loaded shard considering
    pipeline and tensor parallelism, and estimates the memory needed based on
    bytes per parameter (considering precision and optimizer type).

    Args:
        config (ConfigContainer): The main configuration container.
        verbose (bool, optional): If True, prints detailed parameter counts.
                                Defaults to False.

    Returns:
        float: Estimated memory footprint in bytes for weights and optimizer states
               on the most loaded GPU shard.
    """
    estimate = estimate_training_memory(config, include_activation=False)
    if verbose:
        _print_parameter_summary(estimate)

    return estimate.weight_and_optimizer_bytes


def compute_activation_memory(config: ConfigContainer, num_microbatches: int | None, verbose: bool = False) -> float:
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
    return _compute_activation_memory_bytes(config, num_microbatches=num_microbatches, verbose=verbose)


def report_theoretical_memory(
    config: ConfigContainer, num_microbatches: int | None = None, verbose: bool = False
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

    should_report_activation = config.model.sequence_parallel and config.model.recompute_granularity == "selective"
    estimate = estimate_training_memory(
        config,
        num_microbatches=num_microbatches,
        include_activation=should_report_activation,
    )
    if verbose:
        _print_parameter_summary(estimate)

    # Formulae here assume sequence parallelism and selective activation recomputation.
    if not should_report_activation:
        print(
            "Theoretical memory footprints: "
            f"weight and optimizer={estimate.weight_and_optimizer_bytes / NUM_BYTES_IN_MEGABYTE:.2f} MB"
        )
        return

    activation_memory = 0.0 if estimate.activation is None else estimate.activation.memory_mb
    total_memory = estimate.weight_and_optimizer_bytes / NUM_BYTES_IN_MEGABYTE + activation_memory

    print(
        f"Theoretical memory footprints: "
        f"weight and optimizer={estimate.weight_and_optimizer_bytes / NUM_BYTES_IN_MEGABYTE:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )


def _compute_activation_memory_bytes(
    config: ConfigContainer,
    *,
    num_microbatches: int | None,
    verbose: bool = False,
) -> float:
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.
    model_config = config.model
    train_config = config.train

    tensor_parallel_size = _positive_int_attr(model_config, "tensor_model_parallel_size", 1)
    pipeline_parallel_size = _positive_int_attr(model_config, "pipeline_model_parallel_size", 1)
    context_parallel_size = _positive_int_attr(model_config, "context_parallel_size", 1)
    virtual_pipeline_parallel_size = getattr(model_config, "virtual_pipeline_model_parallel_size", None)

    layer_counts = _get_layer_counts(model_config)
    ffn_activation_ratio = _ffn_activation_ratio(model_config, layer_counts)

    # Memory footprint from transformer layer (self-attention and MLP).
    activation_memory = (
        model_config.seq_length
        * train_config.micro_batch_size
        * model_config.hidden_size
        * (18 + (4 * ffn_activation_ratio))
    )
    if verbose:
        print(
            "Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / tensor_parallel_size / context_parallel_size:.1f} MB"
        )
    activation_memory *= layer_counts.total

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += 8 * model_config.seq_length * train_config.micro_batch_size * pipeline_parallel_size
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        model_config.seq_length * train_config.micro_batch_size * model_config.hidden_size * pipeline_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if virtual_pipeline_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (pipeline_parallel_size - 1) / (pipeline_parallel_size * virtual_pipeline_parallel_size)
        )
        in_flight_microbatches = math.ceil(interleaved_schedule_memory_penalty * pipeline_parallel_size)
        if verbose:
            print(f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}")
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if virtual_pipeline_parallel_size is None and pipeline_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / pipeline_parallel_size)
            in_flight_microbatches = min(num_microbatches, pipeline_parallel_size)
        else:
            in_flight_microbatches = pipeline_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if pipeline_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            model_config.seq_length
            * train_config.micro_batch_size
            * model_config.hidden_size
            * 4
            * (1 + (_get_vocab_size(model_config) / model_config.hidden_size))
        )

    # Activation memory is partitioned by TP/SP and CP.
    return activation_memory / (tensor_parallel_size * context_parallel_size)


def _count_parameters(model_config: object) -> _ParameterCounts:
    hidden_size = model_config.hidden_size
    layer_counts = _get_layer_counts(model_config)
    ffn_projection_factor = _ffn_projection_factor(model_config)

    query_projection_size = model_config.kv_channels * model_config.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_size
    num_query_groups = (
        model_config.num_query_groups if model_config.num_query_groups else model_config.num_attention_heads
    )

    attention_parameters_per_layer = (
        2
        * hidden_size
        * hidden_size
        * ((1 + (num_query_groups / model_config.num_attention_heads)) * query_projection_to_hidden_size_ratio)
    )
    layernorm_parameters = (4 * hidden_size * layer_counts.total) + (2 * hidden_size)
    dense_mlp_parameters = ffn_projection_factor * hidden_size * model_config.ffn_hidden_size * layer_counts.dense

    shared_expert_intermediate_size = _optional_positive_int_attr(
        model_config,
        "moe_shared_expert_intermediate_size",
        0,
    )
    shared_expert_parameters = ffn_projection_factor * hidden_size * shared_expert_intermediate_size * layer_counts.moe

    routed_expert_parameters = 0.0
    latent_projection_parameters = 0.0
    if _has_moe(model_config) and layer_counts.moe > 0:
        moe_ffn_hidden_size = _optional_positive_int_attr(
            model_config,
            "moe_ffn_hidden_size",
            model_config.ffn_hidden_size,
        )
        moe_latent_size = getattr(model_config, "moe_latent_size", None)
        if moe_latent_size is None:
            routed_expert_parameters = (
                ffn_projection_factor
                * hidden_size
                * moe_ffn_hidden_size
                * model_config.num_moe_experts
                * layer_counts.moe
            )
        else:
            routed_expert_parameters = (
                ffn_projection_factor
                * moe_latent_size
                * moe_ffn_hidden_size
                * model_config.num_moe_experts
                * layer_counts.moe
            )
            latent_projection_parameters = 2 * hidden_size * moe_latent_size * layer_counts.moe

    dense_transformer_parameters = (
        (attention_parameters_per_layer * layer_counts.total)
        + layernorm_parameters
        + dense_mlp_parameters
        + shared_expert_parameters
        + latent_projection_parameters
    )

    embedding_size = hidden_size * _get_vocab_size(model_config)
    embedding_parameters = embedding_size
    if not model_config.share_embeddings_and_output_weights:
        embedding_parameters += embedding_size

    return _ParameterCounts(
        dense_transformer=dense_transformer_parameters,
        routed_experts=routed_expert_parameters,
        embeddings=embedding_parameters,
    )


def _get_layer_counts(model_config: object) -> _LayerCounts:
    num_layers = _positive_int_attr(model_config, "num_layers", 1)
    if not _has_moe(model_config):
        return _with_mtp_layers(model_config, dense=num_layers, moe=0)

    moe_layer_freq = getattr(model_config, "moe_layer_freq", 1)
    if isinstance(moe_layer_freq, int):
        if moe_layer_freq <= 0:
            raise ValueError("moe_layer_freq must be positive")
        moe_layer_pattern = tuple(1 if layer_idx % moe_layer_freq == 0 else 0 for layer_idx in range(num_layers))
    elif isinstance(moe_layer_freq, (list, tuple)):
        if len(moe_layer_freq) != num_layers:
            raise ValueError(f"Invalid moe_layer_freq length: expected {num_layers}, got {len(moe_layer_freq)}")
        moe_layer_pattern = tuple(1 if layer else 0 for layer in moe_layer_freq)
    else:
        raise ValueError(f"Unsupported moe_layer_freq value: {moe_layer_freq}")

    num_moe_layers = sum(moe_layer_pattern)
    return _with_mtp_layers(model_config, dense=num_layers - num_moe_layers, moe=num_moe_layers)


def _with_mtp_layers(model_config: object, *, dense: int, moe: int) -> _LayerCounts:
    mtp_num_layers = getattr(model_config, "mtp_num_layers", None) or 0
    if mtp_num_layers > 0:
        last_layer_is_moe = 1 if moe > 0 and dense == 0 else 0
        if _has_moe(model_config):
            moe_layer_freq = getattr(model_config, "moe_layer_freq", 1)
            if isinstance(moe_layer_freq, int):
                last_layer_is_moe = 1 if ((model_config.num_layers - 1) % moe_layer_freq == 0) else 0
            elif isinstance(moe_layer_freq, (list, tuple)):
                last_layer_is_moe = 1 if moe_layer_freq[-1] else 0
        moe += last_layer_is_moe * mtp_num_layers
        dense += (1 - last_layer_is_moe) * mtp_num_layers
    return _LayerCounts(dense=dense, moe=moe, total=dense + moe)


def _embedding_parameters_on_most_loaded_shard(model_config: object) -> float:
    embedding_size = model_config.hidden_size * _get_vocab_size(model_config)
    tensor_parallel_size = _positive_int_attr(model_config, "tensor_model_parallel_size", 1)
    pipeline_parallel_size = _positive_int_attr(model_config, "pipeline_model_parallel_size", 1)

    embedding_parameters = embedding_size / tensor_parallel_size
    if not model_config.share_embeddings_and_output_weights and pipeline_parallel_size == 1:
        embedding_parameters += embedding_size / tensor_parallel_size
    return embedding_parameters


def _bytes_per_parameter(config: ConfigContainer, optimizer_shard_size: int) -> float:
    if not getattr(config.optimizer, "use_distributed_optimizer", False):
        return 18.0
    return 6.0 + (12.0 / max(1, optimizer_shard_size))


def _expert_optimizer_shard_size(
    config: ConfigContainer,
    *,
    tensor_parallel_size: int,
    context_parallel_size: int,
    expert_parallel_size: int,
    expert_tensor_parallel_size: int,
) -> int:
    data_parallel_size = _positive_int_attr(config, "data_parallel_size", 1)
    shard_size = data_parallel_size * tensor_parallel_size * context_parallel_size
    shard_size //= max(1, expert_parallel_size * expert_tensor_parallel_size)
    return max(1, shard_size)


def _estimate_assumptions(
    config: ConfigContainer,
    *,
    include_activation: bool,
    has_routed_experts: bool,
) -> tuple[str, ...]:
    model_config = config.model
    assumptions = [
        "Reports the most-loaded GPU shard; runtime allocator, kernel workspace, and CUDA graph buffers are excluded.",
        "Weight/optimizer bytes use the Megatron Bridge Adam estimate: 18 B/parameter without distributed "
        "optimizer, otherwise 6 B plus 12 B sharded over the relevant optimizer group.",
        "Embedding parameters are assigned to first/last pipeline stages; untied embeddings double-count only when PP=1.",
    ]
    if has_routed_experts:
        assumptions.append(
            "Routed MoE expert parameters are sharded by expert_model_parallel_size and "
            "expert_tensor_parallel_size; routed expert optimizer state uses the expert data-parallel shard size."
        )
    if _positive_int_attr(model_config, "context_parallel_size", 1) > 1:
        assumptions.append("Dense optimizer state and activations are divided across context parallel ranks.")
    if include_activation:
        assumptions.append(
            "Activation estimates assume sequence parallelism and selective activation recomputation; full "
            "recompute, CPU offload, token imbalance, and dispatcher workspace are not modeled."
        )
    return tuple(assumptions)


def _print_parameter_summary(estimate: TrainingMemoryEstimate) -> None:
    dense_component = estimate.model_state_components[0]
    routed_component = None
    if len(estimate.model_state_components) > 1:
        routed_component = estimate.model_state_components[1]

    print(f"Number of dense and embedding parameters in billions: {dense_component.parameter_count / 10**9:.2f}")
    if routed_component is not None:
        print(f"Number of routed expert parameters in billions: {routed_component.parameter_count / 10**9:.2f}")
    print(f"Total number of parameters in billions: {estimate.total_parameters / 10**9:.2f}")
    print(
        "Number of dense and embedding parameters in most loaded shard in billions: "
        f"{dense_component.parameter_count_per_gpu / 10**9:.4f}"
    )
    if routed_component is not None:
        print(
            "Number of routed expert parameters in most loaded shard in billions: "
            f"{routed_component.parameter_count_per_gpu / 10**9:.4f}"
        )


def _ffn_projection_factor(model_config: object) -> float:
    # SwiGLU: h->2*ffn_h and ffn_h->h = 3 projections; otherwise two projections.
    if getattr(model_config, "gated_linear_unit", False) and getattr(model_config, "activation_func", None) == F.silu:
        return 3.0
    return 2.0


def _ffn_activation_ratio(model_config: object, layer_counts: _LayerCounts) -> float:
    if layer_counts.total == 0:
        return 0.0
    dense_ffn = layer_counts.dense * model_config.ffn_hidden_size
    moe_ffn = 0.0
    if _has_moe(model_config) and layer_counts.moe > 0:
        routed_to = _optional_positive_int_attr(
            model_config,
            "moe_router_topk",
            _optional_positive_int_attr(model_config, "num_experts_routed_to", 1),
        )
        moe_ffn_hidden_size = _optional_positive_int_attr(
            model_config,
            "moe_ffn_hidden_size",
            model_config.ffn_hidden_size,
        )
        shared_expert_intermediate_size = _optional_positive_int_attr(
            model_config,
            "moe_shared_expert_intermediate_size",
            0,
        )
        moe_ffn = layer_counts.moe * ((moe_ffn_hidden_size * routed_to) + shared_expert_intermediate_size)
    return (dense_ffn + moe_ffn) / (layer_counts.total * model_config.hidden_size)


def _has_moe(model_config: object) -> bool:
    num_experts = getattr(model_config, "num_moe_experts", None)
    return num_experts is not None and num_experts > 0


def _positive_int_attr(config: object, name: str, default: int) -> int:
    value = getattr(config, name, default)
    if value is None:
        return default
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_positive_int_attr(config: object, name: str, default: int) -> int:
    value = getattr(config, name, None)
    if value is None:
        return default
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _get_vocab_size(model_cfg) -> int:
    """Get the potentially padded vocabulary size for the given configuration.

    Args:
        cfg: The model provider configuration.

    Returns:
        int: The vocabulary size used.
    """
    if getattr(model_cfg, "should_pad_vocab", True):
        return calculate_padded_vocab_size(
            model_cfg.vocab_size,
            model_cfg.make_vocab_size_divisible_by,
            model_cfg.tensor_model_parallel_size,
            logging_enabled=False,
        )
    else:
        return model_cfg.vocab_size
