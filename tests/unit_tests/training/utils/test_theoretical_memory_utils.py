from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch.nn.functional as F

from megatron.bridge.training.utils.theoretical_memory_utils import compute_weight_and_optimizer_memory


@dataclass
class MockModelConfig:
    # Core dims
    num_layers: int = 4
    hidden_size: int = 128
    ffn_hidden_size: int = 512
    num_attention_heads: int = 4
    num_query_groups: int | None = 4
    kv_channels: int = 32
    vocab_size: int = 1024
    make_vocab_size_divisible_by: int = 128
    should_pad_vocab: bool = True
    share_embeddings_and_output_weights: bool = True
    normalization: str = "LayerNorm"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    activation_func: object = field(default=None)
    # Parallelism
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None
    virtual_pipeline_model_parallel_size: int | None = None
    sequence_parallel: bool = False
    recompute_granularity: str | None = None
    # MoE
    num_moe_experts: int | None = None
    moe_layer_freq: int | list = 1
    moe_router_topk: int = 1
    moe_ffn_hidden_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_shared_expert_gate: bool = False
    # MTP / MLA
    mtp_num_layers: int | None = None
    multi_latent_attention: bool = False

    def __post_init__(self):
        if self.activation_func is None:
            self.activation_func = F.silu


def _make_config(*, model: MockModelConfig, world_size: int, distributed_opt: bool = False):
    """Build a minimal SimpleNamespace mimicking ConfigContainer for the estimator."""
    total_model_size = (
        model.tensor_model_parallel_size * model.pipeline_model_parallel_size * model.context_parallel_size
    )
    if world_size % total_model_size != 0:
        raise ValueError(f"world_size {world_size} not divisible by TP*PP*CP={total_model_size}")
    return SimpleNamespace(
        model=model,
        optimizer=SimpleNamespace(use_distributed_optimizer=distributed_opt),
        data_parallel_size=world_size // total_model_size,
    )


def test_dense_model_returns_positive_memory():
    cfg = _make_config(model=MockModelConfig(), world_size=1)
    assert compute_weight_and_optimizer_memory(cfg) > 0


def test_moe_routes_experts_through_ep_etp():
    """With EP > 1 the per-rank routed-expert weight should drop vs. EP = 1."""
    base_kwargs = dict(
        num_layers=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        tensor_model_parallel_size=2,
        expert_tensor_parallel_size=1,
    )
    ep1 = MockModelConfig(**base_kwargs, expert_model_parallel_size=1)
    ep4 = MockModelConfig(**base_kwargs, expert_model_parallel_size=4)
    # world = TP * PP * CP * DP = 2 * 1 * 1 * 4 = 8
    mem_ep1 = compute_weight_and_optimizer_memory(_make_config(model=ep1, world_size=8))
    mem_ep4 = compute_weight_and_optimizer_memory(_make_config(model=ep4, world_size=8))
    assert mem_ep4 < mem_ep1, (
        f"Increasing EP should reduce per-rank routed-expert memory; got EP=1 -> {mem_ep1}, EP=4 -> {mem_ep4}"
    )


def test_distributed_optimizer_scales_with_dp():
    """Distributed optimizer state should shrink as data-parallel size grows."""
    model = MockModelConfig()
    cfg_dp1 = _make_config(model=model, world_size=1, distributed_opt=True)
    cfg_dp8 = _make_config(model=model, world_size=8, distributed_opt=True)
    assert compute_weight_and_optimizer_memory(cfg_dp8) < compute_weight_and_optimizer_memory(cfg_dp1)


def test_moe_layer_freq_pattern_list():
    """List form of moe_layer_freq must be honored (one MoE layer at index 1, dense elsewhere)."""
    model = MockModelConfig(
        num_layers=4,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_layer_freq=[0, 1, 0, 0],
    )
    mem = compute_weight_and_optimizer_memory(_make_config(model=model, world_size=1))
    # Compare to all-MoE; less aggregate routed-expert weight should cost less memory.
    model_all_moe = MockModelConfig(
        num_layers=4,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_layer_freq=[1, 1, 1, 1],
    )
    mem_all = compute_weight_and_optimizer_memory(_make_config(model=model_all_moe, world_size=1))
    assert mem < mem_all


def test_invalid_moe_layer_freq_type_raises():
    model = MockModelConfig(num_moe_experts=4, moe_ffn_hidden_size=256, moe_layer_freq="every-other")
    with pytest.raises(TypeError):
        compute_weight_and_optimizer_memory(_make_config(model=model, world_size=1))
