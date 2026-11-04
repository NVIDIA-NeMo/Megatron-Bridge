# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise uncaptured Qwen3.5 MTP after capturing the Hybrid decoder with MXFP8.

Run with ``uv run python -m torch.distributed.run --standalone --nproc_per_node=1
-m pytest tests/functional_tests/test_groups/models/qwen/test_qwen35_hybrid_cuda_graphs.py``.
"""

import datetime
import gc
import os
import warnings

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper, VisionTECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformers import Qwen3_5MoeTextConfig

from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge
from megatron.bridge.models.qwen_vl.modeling_qwen35.hybrid_model import (
    Qwen35VLHybridAttentionLayer,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionTransformerLayer
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


pytestmark = [pytest.mark.integration, pytest.mark.run_only_on("GPU")]


@pytest.fixture(scope="session")
def ensure_test_data():
    """Use generated tensors without downloading functional-suite datasets."""
    return None


@pytest.fixture
def qwen35_graph_process_groups():
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell or newer")

    original_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(original_stream)
    torch.cuda.set_stream(stream)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
    parallel_state.initialize_model_parallel()
    init_num_microbatches_calculator(rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1)
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, use_cudagraphable_rng=True, force_reset_rng=True)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        gc.collect()
        torch.cuda.synchronize()
        destroy_num_microbatches_calculator()
        get_cuda_rng_tracker().reset()
        parallel_state.destroy_model_parallel()
        if owns_group:
            dist.destroy_process_group()
        torch.cuda.set_stream(original_stream)


@pytest.mark.parametrize("vision", [False, True])
def test_rotary_graph_inputs_without_eager_warmup(qwen35_graph_process_groups, vision):
    """Capture must consume changing positions without observing an eager batch first."""
    config = Qwen3VLTransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.mlp],
        cuda_graph_warmup_steps=0,
        use_te_rng_tracker=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        apply_rope_fusion=False,
        apply_rotary_pos_emb_in_fp32=vision,
        init_method_std=0.1,
    )
    batch_size, rotary_dim = (1, 16) if vision else (2, 8)
    spec = get_vit_layer_with_transformer_engine_spec() if vision else get_gpt_layer_with_transformer_engine_spec()
    spec.module = Qwen3VLVisionTransformerLayer if vision else Qwen35VLHybridAttentionLayer
    spec.params["rotary_dim"] = rotary_dim
    spec.submodules.self_attention.module = Qwen3VLSelfAttention
    decoder = MegatronModule(config)
    decoder.decoder = TransformerBlock(config, spec, pg_collection=qwen35_graph_process_groups).cuda().bfloat16()
    decoder.position_embedding_type = "mrope"
    container = MegatronModule(config) if vision else decoder
    if vision:
        container.vision_model = decoder
    model = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(),
        container,
        disable_bucketing=True,
        pg_collection=qwen35_graph_process_groups,
    )
    helper = (
        VisionTECudaGraphHelper([model], config, 32, batch_size, pg_collection=qwen35_graph_process_groups)
        if vision
        else TECudaGraphHelper(
            [model], config, 32, batch_size, optimizers=[], pg_collection=qwen35_graph_process_groups
        )
    )
    layer = decoder.decoder.layers[0]
    layer.current_microbatch = 0
    hidden = torch.randn(32, batch_size, 64, device="cuda", dtype=torch.bfloat16)
    positions = [torch.randn(32, batch_size, 1, rotary_dim, device="cuda") for _ in range(2)]

    try:
        helper.create_cudagraphs()
        assert helper.graphs_created()
        graphs = layer.cuda_graphs
        eager_outputs = []
        for rotary in positions:
            layer.cuda_graphs = []
            try:
                model.zero_grad_buffer()
                x = hidden.clone().requires_grad_(True)
                expected, _ = layer(x, attention_mask=None, rotary_pos_emb=rotary)
                expected.float().square().mean().backward()
                expected = expected.detach().clone()
                expected_input_grad = x.grad.detach().clone()
                expected_param_grads = {name: p.main_grad.detach().clone() for name, p in layer.named_parameters()}
                eager_outputs.append(expected)
            finally:
                layer.cuda_graphs = graphs

            model.zero_grad_buffer()
            x = hidden.clone().requires_grad_(True)
            actual, _ = layer(x, attention_mask=None, rotary_pos_emb=rotary)
            actual.float().square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(x.grad, expected_input_grad, rtol=1e-2, atol=1e-5)
            for name, param in layer.named_parameters():
                torch.testing.assert_close(param.main_grad, expected_param_grads[name], rtol=1e-2, atol=1e-4)
        assert not torch.allclose(eager_outputs[0], eager_outputs[1], rtol=1e-2, atol=1e-3)
        torch.cuda.synchronize()
    finally:
        if helper.graphs_created():
            helper.delete_cuda_graphs()


def test_hybrid_graph_replay_preserves_mtp_fp8_gradients(qwen35_graph_process_groups):
    """MTP must retain FP8 compute and fused weight gradients after decoder capture."""
    pg = qwen35_graph_process_groups
    hf = Qwen3_5MoeTextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=256,
        linear_num_key_heads=2,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
    )
    config = Qwen35MoEBridge().hf_config_to_model_config(hf)
    config.seq_length = 128
    config.mtp_num_layers = 1
    config.mtp_loss_scaling_factor = 0.1
    config.params_dtype = torch.bfloat16
    config.bf16 = True
    config.fp8 = "e4m3"
    config.fp8_recipe = "mxfp8"
    config.fp8_param = True
    config.gradient_accumulation_fusion = True
    config.gdn_kernel_backend = "transformer_engine"
    config.apply_rope_fusion = False
    config.moe_grouped_gemm = True
    config.moe_use_grouped_tensor = True
    config.moe_router_load_balancing_type = "none"
    config.moe_aux_loss_coeff = 0.0
    config.moe_router_fusion = False
    config.cuda_graph_impl = "transformer_engine"
    config.cuda_graph_warmup_steps = 1
    config.use_te_rng_tracker = True
    set_cuda_graph_modules(config, ["attn", "moe_router", "moe_preprocess"])
    config.finalize()
    module = config.get_builder_cls()(config).build_model(pg, pre_process=True, post_process=True).cuda()
    model = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True),
        Float16Module(config, module),
        pg_collection=pg,
    )
    helper = TECudaGraphHelper([model], config, 128, 1, pg_collection=pg)
    mtp_weights = {
        name: p for name, p in module.named_parameters() if name.startswith("mtp.") and isinstance(p, MXFP8Tensor)
    }
    assert mtp_weights, "The regression requires quantized MTP parameters"
    positions = torch.arange(128, device="cuda").unsqueeze(0)
    tokens = (positions * 7) % 256
    reference_loss = None
    reference_grads = None

    try:
        for step in range(3):
            if step == 1:
                helper.create_cudagraphs()
                assert helper.graphs_created()
                assert any(layer.cuda_graphs for layer in helper.flattened_callables)

            model.zero_grad_buffer()
            model.set_is_first_microbatch()
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="You are using quantized weights without quantized compute.*")
                loss = (
                    model(tokens, positions, None, labels=(tokens + 5) % 256, loss_mask=torch.ones_like(tokens))
                    .float()
                    .mean()
                )
                loss.backward()
            model.finish_grad_sync()
            assert torch.isfinite(loss)
            grads = {name: p.main_grad.detach().clone() for name, p in mtp_weights.items()}
            assert all(torch.isfinite(grad).all() for grad in grads.values())
            assert any(grad.count_nonzero() for grad in grads.values())
            if step == 0:
                reference_loss = loss.detach().clone()
                reference_grads = grads
            else:
                torch.testing.assert_close(loss.detach(), reference_loss, rtol=1e-3, atol=1e-3)
                for name, grad in grads.items():
                    torch.testing.assert_close(grad, reference_grads[name], rtol=2e-2, atol=2e-4)
            del loss
    finally:
        if helper.graphs_created():
            helper.delete_cuda_graphs()
