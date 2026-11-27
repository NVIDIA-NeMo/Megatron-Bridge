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

"""Guard the Kimi K3 proxy model, serialization and native FLOPs contract."""

from dataclasses import fields
from pathlib import Path

import pytest

from megatron.bridge.models.hybrid import BridgeHybridMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig
from megatron.bridge.perf_recipes.kimi.gb300.kimi_k3 import (
    kimi_k3_pretrain_64gpu_gb300_fp8mx_proxy_config,
)
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        "attn_res_norm_fusion" not in {f.name for f in fields(MLATransformerConfig)},
        reason="Kimi K3 benchmarks require the pinned MCore dev configuration.",
    ),
]


def test_model_serialization_preserves_situ_and_backend() -> None:
    model = kimi_k3_pretrain_64gpu_gb300_fp8mx_proxy_config().model
    serialized = model.as_dict()
    restored = BridgeHybridMLAModelConfig.from_dict(serialized)
    assert serialized["transformer"]["activation_func"] == "situlu"
    assert callable_to_str(restored.activation_func) == "situlu"
    assert restored.activation_func is str_to_callable("situlu")
    assert restored.hybrid_layer_pattern == model.hybrid_layer_pattern
    assert restored.moe_megakernel_backend == "mok"
    assert restored.moe_megakernel_backend_config == {"shared_expert_mode": "native"}
    assert restored.attention_output_gate_fusion


def test_proxy_retains_width_but_has_one_stage_and_no_mtp() -> None:
    cfg = kimi_k3_pretrain_64gpu_gb300_fp8mx_proxy_config()
    assert cfg.model.num_layers == 48
    assert cfg.model.hybrid_layer_pattern.count("K") == 18
    assert cfg.model.hybrid_layer_pattern.count("+") == 6
    assert cfg.model.hybrid_layer_pattern.count("E") == 24
    assert (
        cfg.model.tensor_model_parallel_size,
        cfg.model.pipeline_model_parallel_size,
        cfg.model.expert_model_parallel_size,
        cfg.model.expert_tensor_parallel_size,
        cfg.model.context_parallel_size,
    ) == (1, 1, 64, 1, 1)
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.mtp_num_layers is None
    assert cfg.model.hidden_size == 7168
    assert cfg.model.num_moe_experts == 896
    assert cfg.model.moe_router_topk == 16
    assert cfg.model.moe_ffn_hidden_size == 3072
    assert (cfg.train.global_batch_size, cfg.train.micro_batch_size) == (2048, 1)
    assert cfg.optimizer.optimizer == "muon"
    assert cfg.optimizer.chunked_optimizer_state_offload
    assert cfg.optimizer.optimizer_state_offload_chunk_size_mb == 256
    assert cfg.optimizer.adam_beta2 == 0.999
    assert cfg.mixed_precision.grad_reduce_in_fp32
    assert cfg.ddp.grad_reduce_in_fp32
    assert not cfg.ddp.average_in_collective
    assert cfg.scheduler.start_weight_decay == cfg.scheduler.end_weight_decay == 0.1
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert cfg.mixed_precision.fp8_param_gather
    assert cfg.dataset.seq_length == cfg.model.seq_length == 8192
    assert not cfg.dataset.create_attention_mask
    assert cfg.model.moe_router_force_load_balancing


def test_native_hybrid_flops_forwards_actual_token_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    from megatron.training import training

    from megatron.bridge.training.utils.flop_utils import num_floating_point_operations

    cfg = kimi_k3_pretrain_64gpu_gb300_fp8mx_proxy_config()
    calls: list[tuple[object, int, dict[str, int | None]]] = []

    def native_counter(model: object, batch_size: int, **kwargs: int | None) -> int:
        calls.append((model, batch_size, kwargs))
        return 123456

    monkeypatch.setattr(training, "num_floating_point_operations", native_counter)
    actual = num_floating_point_operations(cfg, batch_size=2, seqlen_sum=1000, seqlen_squared_sum=520000)
    assert actual == 123456
    assert calls == [
        (
            cfg.model,
            2,
            {"sequence_length": 8192, "total_real_tokens_in_batch": 1000, "seqlen_squared_sum_in_batch": 520000},
        )
    ]


def test_public_recipe_loader_selects_mxfp8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts" / "training"))
    from recipe_metadata import resolved_benchmark_recipe_metadata
    from recipe_runner import load_recipe

    name = "kimi_k3_pretrain_64gpu_gb300_fp8mx_proxy_config"
    cfg = load_recipe(name)
    metadata = resolved_benchmark_recipe_metadata(name)
    assert metadata.family == "kimi"
    assert metadata.num_gpus == 64
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert cfg.optimizer.optimizer == "muon"
