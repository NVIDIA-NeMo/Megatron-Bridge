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
import re
import warnings

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import reset_model_temporary_tensors
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.gpt_provider import mtp_block_spec
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge
from megatron.bridge.models.qwen_vl.modeling_qwen35.hybrid_model import (
    Qwen35VLHybridModel,
    qwen35_vl_hybrid_stack_spec,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen35VLMoEModelProvider,
    _patch_standard_attention_specs,
)
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


class _GPTQwen35ReferenceBridge(Qwen35MoEBridge):
    MODEL_CONFIG_CLASS = BridgeGPTModelConfig

    def hf_config_to_model_config_kwargs(self, hf_config):
        kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        for key in ("hybrid_layer_pattern", "hybrid_stack_spec", "mtp_hybrid_override_pattern"):
            kwargs.pop(key, None)
        kwargs.update(
            num_layers=hf_config.num_hidden_layers,
            is_hybrid_model=False,
            linear_attention_freq=full_attention_interval_from_hf(hf_config),
            moe_layer_freq=1,
        )
        return kwargs


def _build_vl_parity_pair(pg, precision):
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
    models = []
    for hybrid in (False, True):
        config = Qwen35VLMoEModelProvider(**_GPTQwen35ReferenceBridge().hf_config_to_model_config_kwargs(hf))
        config.seq_length = 128
        config.language_max_sequence_length = 128
        config.position_embedding_type = "mrope"
        config.mrope_section = [3, 3, 2]
        config.mtp_num_layers = 1
        config.mtp_loss_scaling_factor = 0.1
        config.params_dtype = torch.bfloat16
        config.bf16 = True
        config.gradient_accumulation_fusion = False
        config.fp8 = "e4m3" if precision == "fp8mx" else None
        config.fp8_recipe = "mxfp8" if precision == "fp8mx" else "delayed"
        config.moe_grouped_gemm = True
        config.moe_use_grouped_tensor = True
        config.gdn_kernel_backend = "transformer_engine"
        config.apply_rope_fusion = False
        config.moe_router_load_balancing_type = "global_aux_loss"
        config.moe_aux_loss_coeff = 0.001
        config.moe_router_fusion = False
        config.finalize()
        if hybrid:
            spec = qwen35_vl_hybrid_stack_spec()
            model_class = Qwen35VLHybridModel
            mtp_spec = None
        else:
            config.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
            spec = get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=None)
            _patch_standard_attention_specs(spec, Qwen3VLSelfAttention)
            model_class = Qwen3VLModel
            mtp_spec = mtp_block_spec(config, vp_stage=None)
            _patch_standard_attention_specs(mtp_spec, Qwen3VLSelfAttention)
        config.image_token_id = 250
        config.video_token_id = 251
        config.vision_start_token_id = 252
        config.vision_config = Qwen3_5VisionConfig(
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            deepstack_visual_indexes=[],
        )
        config.vision_config.torch_dtype = torch.bfloat16
        module = model_class(
            language_transformer_config=config,
            language_transformer_layer_spec=spec,
            vision_transformer_config=config.vision_config,
            mtp_block_spec=mtp_spec,
            pre_process=True,
            post_process=True,
            pg_collection=pg,
        ).cuda()
        models.append(Float16Module(module.config, module))
    return models


def _gpt_parameter_name(name):
    name = re.sub(r"(mtp_model_layer|transformer_layer)\.layers\.[01]\.", r"\1.", name)
    name = name.replace("language_model.decoder.final_norm.", "language_model.decoder.final_layernorm.")
    return re.sub(
        r"language_model\.decoder\.layers\.(\d+)\.",
        lambda match: f"language_model.decoder.layers.{int(match[1]) // 2}.",
        name,
    )


@pytest.mark.parametrize("precision", ["bf16", "fp8mx"])
def test_hybrid_vl_mtp_preserves_gpt_training(qwen35_graph_process_groups, precision):
    """Middle-of-sequence images and MTP boundaries must preserve losses and updates."""
    models = _build_vl_parity_pair(qwen35_graph_process_groups, precision)
    parameters = [dict(model.named_parameters()) for model in models]
    assert len(parameters[0]) == len(parameters[1])
    assert set(parameters[0]) == {_gpt_parameter_name(name) for name in parameters[1]}
    with torch.no_grad():
        for name, parameter in parameters[1].items():
            parameter.copy_(parameters[0][_gpt_parameter_name(name)])
            assert torch.equal(parameter, parameters[0][_gpt_parameter_name(name)])

    masters = [
        {name: torch.nn.Parameter(parameter.detach().float().clone()) for name, parameter in params.items()}
        for params in parameters
    ]
    optimizers = [
        torch.optim.AdamW(list(params.values()), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False)
        for params in masters
    ]
    logits = [[], []]
    hooks = [
        model.module.language_model.output_layer.register_forward_hook(
            lambda module, inputs, output, target=logits[index]: target.append(output[0].detach().clone())
        )
        for index, model in enumerate(models)
    ]
    trackers = [MoEMetricsTracker(), MoEMetricsTracker()]
    try:
        for step in range(3):
            torch.manual_seed(7000 + step)
            tokens = torch.randint(0, 240, (2, 128), device="cuda")
            tokens[:, 64] = 250
            positions = (
                torch.stack(
                    [
                        torch.arange(128, device="cuda"),
                        torch.arange(128, device="cuda") // 8,
                        torch.arange(128, device="cuda") % 8,
                    ]
                )
                .unsqueeze(1)
                .expand(3, 2, 128)
            )
            pixels = torch.randn(8, 3 * 2 * 16 * 16, device="cuda", dtype=torch.bfloat16)
            grids = torch.tensor([[1, 2, 2], [1, 2, 2]], device="cuda")
            losses, mtp_losses = [], []
            for index, model in enumerate(models):
                model.zero_grad(set_to_none=True)
                reset_model_temporary_tensors(model.config, [model])
                set_moe_metrics_tracker(trackers[index])
                trackers[index].clear()
                MTPLossLoggingHelper.clean_metrics_in_tracker()
                logits[index].clear()
                loss = model(
                    tokens,
                    positions,
                    None,
                    labels=(tokens + 5) % 256,
                    loss_mask=torch.ones_like(tokens),
                    pixel_values=pixels,
                    image_grid_thw=grids,
                )
                losses.append(loss.detach().clone())
                mtp_losses.append(MTPLossLoggingHelper.tracker["loss_values"].clone())
                loss.float().mean().backward()

            torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
            torch.testing.assert_close(mtp_losses[0], mtp_losses[1], rtol=0, atol=0)
            assert len(logits[0]) == len(logits[1]) == 2
            for reference, candidate in zip(logits[0], logits[1]):
                torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
            for name, parameter in parameters[1].items():
                reference = parameters[0][_gpt_parameter_name(name)]
                assert (reference.grad is None) == (parameter.grad is None), name
                if parameter.grad is not None:
                    torch.testing.assert_close(reference.grad, parameter.grad, rtol=0, atol=0, msg=name)
            for index, params in enumerate(parameters):
                for name, parameter in params.items():
                    masters[index][name].grad = None if parameter.grad is None else parameter.grad.float().clone()
                torch.nn.utils.clip_grad_norm_(list(masters[index].values()), 1.0)
                optimizers[index].step()
                with torch.no_grad():
                    for name, parameter in params.items():
                        parameter.copy_(masters[index][name])
            for name, parameter in parameters[1].items():
                torch.testing.assert_close(
                    parameters[0][_gpt_parameter_name(name)], parameter, rtol=0, atol=0, msg=name
                )
    finally:
        for hook in hooks:
            hook.remove()
        set_moe_metrics_tracker(MoEMetricsTracker())
        MTPLossLoggingHelper.clean_metrics_in_tracker()
