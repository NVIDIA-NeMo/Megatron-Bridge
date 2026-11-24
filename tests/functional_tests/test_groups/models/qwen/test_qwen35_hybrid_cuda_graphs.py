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

"""Exercise Qwen3.5 Hybrid CUDA graphs.

Covers the rotary graph inputs of the language and vision layers (with and without fused RoPE), uncaptured MTP after
capturing the Hybrid decoder with MXFP8, and one graph per Hybrid block against one graph per position.

Run with ``uv run python -m torch.distributed.run --standalone --nproc_per_node=1
-m pytest tests/functional_tests/test_groups/models/qwen/test_qwen35_hybrid_cuda_graphs.py``.
"""

import dataclasses
import datetime
import gc
import os
import warnings

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper, VisionTECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_model_config
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge
from megatron.bridge.models.qwen_vl.modeling_qwen35.hybrid_model import (
    Qwen35VLHybridAttentionLayer,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import rope
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionTransformerLayer
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
from megatron.bridge.utils.cuda_graph import enable_hybrid_cuda_graph_pairs, set_cuda_graph_modules


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


@pytest.mark.parametrize("rope_fusion", [False, True])
@pytest.mark.parametrize("vision", [False, True])
def test_rotary_graph_inputs_without_eager_warmup(qwen35_graph_process_groups, monkeypatch, vision, rope_fusion):
    """Capture must consume changing positions without observing an eager batch first."""
    # Count the fused RoPE kernel's calls: with fusion on, capture and eager runs must take it without falling back.
    fused_rope_calls = []
    fused_rope_kernel = rope.fused_apply_rotary_pos_emb

    def counted_fused_rope_kernel(*args, **kwargs):
        fused_rope_calls.append(tuple(args[0].shape))
        return fused_rope_kernel(*args, **kwargs)

    monkeypatch.setattr(rope, "fused_apply_rotary_pos_emb", counted_fused_rope_kernel)
    monkeypatch.setattr(rope, "_FUSED_ROPE_FALLBACK_REASONS", set())
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
        apply_rope_fusion=rope_fusion,
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
        graphs = layer.te_cuda_graphs
        eager_outputs = []
        for rotary in positions:
            layer.te_cuda_graphs = []
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
                layer.te_cuda_graphs = graphs

            model.zero_grad_buffer()
            x = hidden.clone().requires_grad_(True)
            actual, _ = layer(x, attention_mask=None, rotary_pos_emb=rotary)
            actual.float().square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(x.grad, expected_input_grad, rtol=1e-2, atol=1e-5)
            for name, param in layer.named_parameters():
                torch.testing.assert_close(param.main_grad, expected_param_grads[name], rtol=1e-2, atol=1e-4)
        assert not torch.allclose(eager_outputs[0], eager_outputs[1], rtol=1e-2, atol=1e-3)
        assert bool(fused_rope_calls) is rope_fusion
        assert not rope._FUSED_ROPE_FALLBACK_REASONS
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
                assert any(layer.te_cuda_graphs for layer in helper.flattened_callables)

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


_PAIR_FIELD = "cuda_graph_pair_hybrid_layers"


def _set_existing_fields(config, **values):
    """Override config fields; an unknown name would only create an unused attribute, so it raises."""
    for name, value in values.items():
        if not hasattr(config, name):
            raise ValueError(f"{type(config).__name__} has no field {name!r}")
        setattr(config, name, value)


def _tiny_qwen35_vl_moe_config():
    """A Qwen3.5-VL MoE with three GDN blocks and one attention block (eight decoder positions) and one MTP layer."""
    hf = Qwen3_5MoeConfig(
        text_config=dict(
            hidden_size=256,
            num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            vocab_size=2048,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            mtp_num_hidden_layers=1,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [3, 3, 2],
                "mrope_interleaved": True,
            },
        ),
        vision_config=dict(depth=1, hidden_size=256, intermediate_size=512, num_heads=4, out_hidden_size=256),
        image_token_id=2040,
        video_token_id=2041,
        vision_start_token_id=2042,
        vision_end_token_id=2043,
    )
    return Qwen35VLMoEBridge().hf_config_to_model_config(hf)


@pytest.mark.skipif(
    not any(field.name == _PAIR_FIELD for field in dataclasses.fields(TransformerConfig)),
    reason="Megatron-Core predates TransformerConfig.cuda_graph_pair_hybrid_layers",
)
@pytest.mark.parametrize("rope_fusion", [False, True])
def test_paired_hybrid_graphs_match_per_position_graphs(qwen35_graph_process_groups, monkeypatch, rope_fusion):
    """One TE graph per Qwen3.5-VL Hybrid block trains bitwise like one graph per position.

    The model graphs the performance recipes' modules (attention, MoE router and preprocess) and takes the MRoPE
    rotary input as a static graph input of each attention mixer. After one shared eager iteration, each arm restores
    the same weights, captures its graphs and runs two iterations of two microbatches with SGD updates. Losses and
    every gradient must be bitwise equal, and pairing must halve the decoder's graphed callables and graph replays.
    Per-position graphs leave the MTP block eager; a Megatron-Core that also graphs Hybrid MTP layers under the same
    option captures it as one more pair, which the counts allow for.
    """
    pg = qwen35_graph_process_groups
    # Deterministic kernels, so that the two arms can be compared bitwise: Transformer Engine attention without
    # nondeterministic algorithms, PyTorch's deterministic scatter for the MoE unpermute, torch GDN.
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setattr(rope, "_FUSED_ROPE_FALLBACK_REASONS", set())
    deterministic = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    micro_batch_size, seq_length, num_microbatches = 2, 256, 2
    destroy_num_microbatches_calculator()
    init_num_microbatches_calculator(
        rank=0,
        global_batch_size=micro_batch_size * num_microbatches,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    try:
        config = _tiny_qwen35_vl_moe_config()
        _set_existing_fields(
            config,
            seq_length=seq_length,
            params_dtype=torch.bfloat16,
            bf16=True,
            apply_rope_fusion=rope_fusion,
            gradient_accumulation_fusion=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            deterministic_mode=True,
            gdn_kernel_backend="torch",
            gdn_pre_gated_delta_rule_fusion=False,
            moe_router_aux_loss_fusion=False,
            cross_entropy_loss_fusion=False,
            cuda_graph_impl="transformer_engine",
            cuda_graph_warmup_steps=1,
            use_te_rng_tracker=True,
        )
        set_cuda_graph_modules(config, ["attn", "moe_router", "moe_preprocess"])
        config.finalize()
        module = config.get_builder_cls()(config).build_model(pg, pre_process=True, post_process=True).cuda()
        module.model_type = ModelType.encoder_or_decoder  # read by Megatron-Core's training schedule
        model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True),
            Float16Module(config, module),
            pg_collection=pg,
        )
        # The graph helper reads the pairing flag from the model config it is given, as in training; the decoder layers
        # hold the finalized copy made at build time. Both are switched together.
        model_config = get_model_config(model)
        pairing_configs = [model_config, module.language_model.config]
        params = [(name, param) for name, param in model.module.named_parameters() if param.requires_grad]
        generator = torch.Generator(device="cuda").manual_seed(7)
        batches = [
            {
                "input_ids": torch.randint(
                    0, 2000, (micro_batch_size, seq_length), device="cuda", generator=generator
                ),
                "labels": torch.randint(0, 2000, (micro_batch_size, seq_length), device="cuda", generator=generator),
                "loss_mask": torch.ones(micro_batch_size, seq_length, device="cuda"),
            }
            for _ in range(num_microbatches)
        ]

        def forward_step(data_iterator, model_chunk):
            batch = next(data_iterator)
            output = model_chunk(
                input_ids=batch["input_ids"],
                position_ids=None,
                attention_mask=None,
                labels=batch["labels"],
                loss_mask=batch["loss_mask"],
            )

            def loss_func(output_tensor):
                loss = output_tensor.float().sum()
                return loss, {"loss": loss.detach()}

            return output, loss_func

        replays = [0]
        replay = torch.cuda.CUDAGraph.replay

        def counted_replay(graph):
            replays[0] += 1
            return replay(graph)

        monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)
        forward_backward = get_forward_backward_func()

        def train_iteration():
            model.zero_grad_buffer()
            replays[0] = 0
            losses = forward_backward(
                forward_step_func=forward_step,
                data_iterator=iter(batches),
                model=[model],
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
            )
            model.finish_grad_sync()
            torch.cuda.synchronize()
            grads = {name: param.main_grad.detach().clone() for name, param in params}
            with torch.no_grad():
                for _, param in params:
                    param.add_(param.main_grad.to(param.dtype), alpha=-1e-3)
            return float(sum(loss["loss"] for loss in losses)), grads, replays[0]

        train_iteration()  # eager, shared by both arms
        weights = [param.detach().clone() for _, param in params]

        def graphed_iterations(paired):
            with torch.no_grad():
                for (_, param), weight in zip(params, weights):
                    param.copy_(weight)
            for pairing_config in pairing_configs:
                if paired:
                    assert enable_hybrid_cuda_graph_pairs(pairing_config)
                else:
                    pairing_config.cuda_graph_pair_hybrid_layers = False
            helper = TECudaGraphHelper(
                [model], model_config, seq_length, micro_batch_size, optimizers=[], pg_collection=pg
            )
            try:
                helper.create_cudagraphs()
                assert helper.graphs_created()
                callables = len(helper.flattened_callables), sum(helper.flattened_callables_is_mtp)
                return callables, [train_iteration() for _ in range(2)]
            finally:
                if helper.graphs_created():
                    helper.delete_cuda_graphs()

        (per_position_callables, per_position_mtp_callables), per_position = graphed_iterations(paired=False)
        (paired_callables, paired_mtp_callables), paired = graphed_iterations(paired=True)

        # Per-position graphs leave the Hybrid MTP block eager. A Megatron-Core that graphs its layers with pairing
        # captures the block's two positions as one more pair, which then also replays.
        assert per_position_mtp_callables == 0
        assert paired_mtp_callables in (0, 1)
        paired_decoder_callables = paired_callables - paired_mtp_callables
        assert per_position_callables == 2 * paired_decoder_callables > 0
        # One forward and one backward replay per graphed callable and microbatch.
        paired_mtp_replays = 2 * num_microbatches * paired_mtp_callables
        for iteration, ((loss, grads, replay_count), (paired_loss, paired_grads, paired_replay_count)) in enumerate(
            zip(per_position, paired)
        ):
            assert replay_count == 2 * (paired_replay_count - paired_mtp_replays) > 0, iteration
            assert paired_loss == loss, (iteration, loss, paired_loss)
            assert paired_grads.keys() == grads.keys()
            different = [name for name in grads if not torch.equal(paired_grads[name], grads[name])]
            assert not different, (iteration, different)
        assert not rope._FUSED_ROPE_FALLBACK_REASONS
        torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(deterministic[0], warn_only=deterministic[1])
