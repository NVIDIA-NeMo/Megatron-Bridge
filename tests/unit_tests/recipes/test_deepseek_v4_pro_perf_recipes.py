"""Full-model and precision contracts for the DeepSeek V4 Pro BF16 benchmark."""

import importlib
from dataclasses import asdict

import pytest
import torch

from megatron.bridge.models.deepseek.deepseek_v4_config import DeepSeekV4ModelConfig
from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config,
    deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config,
)
from tests.unit_tests.recipes.recipe_test_utils import (
    DeepSeekV4BuilderOnlyBridge,
    patch_recipe_construction_dependencies,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline_recipe_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)
    module = importlib.import_module("megatron.bridge.recipes.deepseek.gb300.deepseek_v4")
    monkeypatch.setattr(module, "AutoBridge", DeepSeekV4BuilderOnlyBridge)


def test_bf16_preserves_full_pro_workload_and_optimizer() -> None:
    bf16 = deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config()
    mxfp8 = deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config()

    assert isinstance(bf16.model, DeepSeekV4ModelConfig)
    assert bf16.model.hidden_size == 7168
    assert bf16.model.num_moe_experts == 384
    assert bf16.model.num_layers == 122  # 61 logical attention/MoE pairs.
    assert bf16.model.hybrid_layer_pattern == mxfp8.model.hybrid_layer_pattern
    assert bf16.model.pipeline_model_parallel_layout == mxfp8.model.pipeline_model_parallel_layout
    assert bf16.model.seq_length == bf16.dataset.seq_length == 4096
    assert bf16.train.global_batch_size == 4096
    assert bf16.train.micro_batch_size == 1
    assert bf16.model.tensor_model_parallel_size == 1
    assert bf16.model.pipeline_model_parallel_size == 4
    assert bf16.model.virtual_pipeline_model_parallel_size == 4
    assert bf16.model.expert_model_parallel_size == 64
    assert bf16.model.expert_tensor_parallel_size == bf16.model.context_parallel_size == 1
    assert bf16.model.moe_router_force_load_balancing is True
    assert asdict(bf16.optimizer) == asdict(mxfp8.optimizer)
    assert asdict(bf16.scheduler) == asdict(mxfp8.scheduler)
    assert bf16.optimizer.main_params_dtype == torch.float32
    assert bf16.optimizer.main_grads_dtype == torch.bfloat16
    assert bf16.optimizer.exp_avg_dtype == bf16.optimizer.exp_avg_sq_dtype == torch.bfloat16


def test_bf16_effective_precision_disables_quantized_compute_and_gather() -> None:
    cfg = deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config()
    # Exercise precision propagation on the strict builder-backed recipe config,
    # including clearing pre-existing FP8 settings.
    model = cfg.model
    model.fp8 = "e4m3"
    model.fp8_param = True
    cfg.mixed_precision.finalize()
    cfg.mixed_precision.setup(model, cfg.optimizer, cfg.ddp)

    assert model.bf16 is True
    assert model.params_dtype == model.pipeline_dtype == torch.bfloat16
    assert model.fp8 is None
    assert model.fp4 is None
    assert cfg.model.quant_recipe is None
    assert model.fp8_param is False
    assert cfg.ddp.fp8_param_gather is False
    assert cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag is False
    assert model.fp8_dot_product_attention is False
    assert model.fp8_multi_head_attention is False
    assert cfg.model.moe_router_padding_for_fp8 is False
    assert cfg.model.moe_router_padding_for_quantization is False
    assert cfg.ddp.grad_reduce_in_fp32 is False


def test_bf16_uses_supported_eager_recompute_combination() -> None:
    cfg = deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config()
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["mla_up_proj", "mhc", "moe"]
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.use_te_rng_tracker is cfg.rng.te_rng_tracker is False
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules == []
    assert cfg.model.moe_paged_stash is False
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is False
