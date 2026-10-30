"""Two-GPU TE regression checks for vision sequence parallelism.

Run with ``uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest
 tests/unit_tests/models/qwen_vl/test_vision_sequence_parallel_distributed.py``.
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import get_vision_model_config
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import PatchMergerSubmodules
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel


@pytest.fixture(scope="module", params=[1, 2])
def pg(request):
    if int(os.environ.get("WORLD_SIZE", "1")) != 2 or not torch.cuda.is_available():
        pytest.skip("requires two CUDA ranks")
    owns_group = not dist.is_initialized()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if owns_group:
        dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=request.param)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        parallel_state.destroy_model_parallel()
        if owns_group:
            dist.destroy_process_group()


@pytest.mark.parametrize("merge,deepstack,recompute", [(2, [], None), (1, [], None), (2, [0], None), (2, [0], "full")])
def test_vision_sp_outputs_gradients_and_updates(pg, merge, deepstack, recompute):
    hf = SimpleNamespace(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        patch_size=2,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=merge,
        num_position_embeddings=16,
        out_hidden_size=64,
        deepstack_visual_indexes=deepstack,
    )
    grid = torch.tensor([[1, 2, 2], [2, 2, 4]] if merge == 2 else [[1, 1, 3], [2, 1, 2]], device="cuda")
    torch.manual_seed(4321)
    x = torch.randn(int(grid.prod(-1).sum()), 24, device="cuda", dtype=torch.bfloat16)
    reference = None
    for sp in [False, True] if pg.tp.size() > 1 else [False]:
        model_parallel_cuda_manual_seed(1234)
        parent = SimpleNamespace(
            recompute_granularity=recompute,
            recompute_method="uniform" if recompute else None,
            recompute_num_layers=1 if recompute else None,
            tensor_model_parallel_size=pg.tp.size(),
            vision_sequence_parallel=sp,
            enable_cuda_graph=False,
            cuda_graph_use_single_mempool=False,
            cuda_graph_retain_backward_graph=False,
            cuda_graph_warmup_steps=0,
            external_cuda_graph=False,
            use_cpu_initialization=False,
        )
        cfg = get_vision_model_config(hf, parent)
        cfg.gradient_accumulation_fusion = False
        cfg.params_dtype = torch.bfloat16
        cfg.bf16 = True
        spec = get_vit_layer_with_transformer_engine_spec()
        spec.submodules.self_attention.module = Qwen3VLSelfAttention
        model = (
            Qwen3VLVisionModel(
                cfg,
                spec,
                PatchMergerSubmodules(
                    patch_norm=TENorm, linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
                pg_collection=pg,
            )
            .cuda()
            .bfloat16()
        )
        if reference is None:
            weights = copy.deepcopy(model.state_dict())
        else:
            model.load_state_dict(weights)
        shapes = []
        handle = model.decoder.layers[0].register_forward_pre_hook(
            lambda m, a, k: shapes.append(tuple(k["hidden_states"].shape)), with_kwargs=True
        )
        wrapped = DistributedDataParallel(
            cfg,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False
            ),
            model,
            pg_collection=pg,
        )
        wrapped.zero_grad_buffer()
        inputs = x.clone().requires_grad_()
        y, extras = wrapped(inputs, grid)
        loss = y.float().square().sum()
        for z in extras:
            loss = loss + z.float().square().sum()
        loss.backward()
        finalize_model_grads([wrapped], pg_collection=pg)
        grads = {n: p.main_grad.detach().float().clone() for n, p in model.named_parameters()}
        assert y.isfinite().all()
        assert all(grad.isfinite().all() for grad in grads.values())
        current = (y.detach().float(), [z.detach().float() for z in extras], grads, inputs.grad.float().clone())
        assert shapes[0][0] == (x.shape[0] + int(sp and x.shape[0] % 2)) // (2 if sp else 1), shapes
        assert not model.merger.config.sequence_parallel
        if reference is None:
            reference = current
        else:
            for n, g in grads.items():
                base = reference[2][n]
                scale = base.abs().max().clamp_min(1e-8)
                torch.testing.assert_close(g / scale, base / scale, rtol=0.02, atol=0.01)
                # An SGD step must preserve the same update within BF16 gradient error
                # plus two FP32 rounding steps when subtracting nearby weights.
                initial = dict(model.named_parameters())[n].detach().float()
                torch.testing.assert_close(
                    (initial - 0.1 * g) - initial,
                    (initial - 0.1 * base) - initial,
                    rtol=0.02,
                    atol=float(0.001 * scale + 2 * torch.finfo(torch.float32).eps * initial.abs().max()),
                )
            torch.testing.assert_close(current[0], reference[0], rtol=0.02, atol=0.002)
            torch.testing.assert_close(current[3], reference[3], rtol=0.04, atol=0.002)
            for z, b in zip(current[1], reference[1]):
                torch.testing.assert_close(z, b, rtol=0.02, atol=0.002)
        handle.remove()
        del wrapped, model


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("overlap,distributed_optimizer,per_token", [(False, False, False), (True, True, True)])
def test_qwen35_independent_sp_flag_matrix(pg, moe, overlap, distributed_optimizer, per_token):
    """Exercise real vision-to-language insertion and root-config gradient finalization."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider, Qwen35VLMoEModelProvider
    from megatron.bridge.training.setup import _update_model_config_funcs

    torch.manual_seed(4321)
    pixels = torch.randn(20, 24, device="cuda", dtype=torch.bfloat16)
    tokens = torch.randint(0, 100, (1, 32), device="cuda")
    tokens[0, 2] = 101
    # Each video frame has its own run of vision tokens.
    tokens[0, 5:7] = 102
    tokens[0, 9:11] = 102
    positions = torch.arange(32, device="cuda").view(1, 1, -1).expand(3, 1, -1).contiguous()
    combinations = (
        [(False, False)] if pg.tp.size() == 1 else [(False, False), (True, False), (True, True), (False, True)]
    )
    if moe and pg.tp.size() > 1:
        # Core MoE training requires language SP when language TP is enabled.
        combinations = [(True, False), (True, True)]
    provider_type = Qwen35VLMoEModelProvider if moe else Qwen35VLModelProvider
    moe_config = (
        dict(
            num_moe_experts=4,
            moe_router_topk=2,
            moe_ffn_hidden_size=64,
            moe_shared_expert_intermediate_size=64,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_permute_fusion=False,
        )
        if moe
        else {}
    )
    reference = None
    for language_sp, vision_sp in combinations:
        model_parallel_cuda_manual_seed(1234)
        config = provider_type(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            ffn_hidden_size=128,
            linear_attention_freq=[0, 1],
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            vocab_size=128,
            make_vocab_size_divisible_by=64,
            seq_length=32,
            language_max_sequence_length=32,
            tensor_model_parallel_size=pg.tp.size(),
            sequence_parallel=language_sp,
            vision_sequence_parallel=vision_sp,
            params_dtype=torch.bfloat16,
            bf16=True,
            use_cpu_initialization=False,
            gradient_accumulation_fusion=True,
            calculate_per_token_loss=per_token,
            activation_func=torch.nn.functional.silu,
            mrope_section=[2, 3, 3],
            rotary_percent=1.0,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            image_token_id=101,
            video_token_id=102,
            vision_start_token_id=103,
            vision_end_token_id=104,
            vision_config=Qwen3_5VisionConfig(
                depth=2,
                hidden_size=64,
                num_heads=4,
                intermediate_size=128,
                patch_size=2,
                temporal_patch_size=2,
                spatial_merge_size=2,
                num_position_embeddings=16,
                out_hidden_size=64,
            ),
            **moe_config,
        )
        config.finalize()
        raw_models = []

        def record_models(models):
            raw_models.extend(models)
            return models

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=overlap,
            use_distributed_optimizer=distributed_optimizer,
        )
        models = config.provide_distributed_model(
            ddp_config=ddp_config,
            pg_collection=pg,
            pre_wrap_hook=record_models,
        )
        model = models[0]
        raw_model = raw_models[0]
        _update_model_config_funcs(
            models, config, ddp_config, optimizer=SimpleNamespace(scale_loss=lambda loss: loss), pg_collection=pg
        )
        shapes = {"vision": [], "language": []}
        handles = [
            raw_model.vision_model.decoder.layers[0].register_forward_pre_hook(
                lambda m, a, k: shapes["vision"].append(k["hidden_states"].shape[0]), with_kwargs=True
            ),
            raw_model.language_model.decoder.layers[0].register_forward_pre_hook(
                lambda m, a, k: shapes["language"].append(k["hidden_states"].shape[0]), with_kwargs=True
            ),
        ]
        if reference is None:
            weights = copy.deepcopy(model.state_dict())
        else:
            model.load_state_dict(weights)
        model.zero_grad_buffer()
        output = model(
            input_ids=tokens,
            position_ids=positions,
            attention_mask=None,
            pixel_values=pixels[:4],
            pixel_values_videos=pixels[4:],
            image_grid_thw=torch.tensor([[1, 2, 2]], device="cuda"),
            video_grid_thw=torch.tensor([[2, 2, 4]], device="cuda"),
            runtime_gather_output=True,
        )
        output.float().square().mean().backward()
        num_tokens = torch.tensor(32, device="cuda", dtype=torch.int64) if per_token else None
        config.finalize_model_grads_func(models, num_tokens, pg_collection=pg, force_all_reduce=False)
        grads = {name: parameter.main_grad.detach().float().clone() for name, parameter in model.named_parameters()}
        vision_grad = next(
            grad for name, grad in grads.items() if name.endswith("vision_model.patch_embed.proj.weight")
        )
        assert torch.count_nonzero(vision_grad) > 0
        assert shapes["vision"] == [20 // (pg.tp.size() if vision_sp else 1)]
        assert shapes["language"] == [32 // (pg.tp.size() if language_sp else 1)]
        assert output.isfinite().all()
        assert all(grad.isfinite().all() for grad in grads.values())
        if reference is None:
            reference = (output.detach().float(), grads)
        else:
            torch.testing.assert_close(output.float(), reference[0], rtol=0.03, atol=0.003)
            for name, grad in grads.items():
                baseline = reference[1][name]
                scale = baseline.abs().max().clamp_min(1e-10)
                torch.testing.assert_close(grad / scale, baseline / scale, rtol=0.03, atol=0.01, msg=name)
                initial = dict(model.named_parameters())[name].detach().float()
                torch.testing.assert_close(
                    (initial - 0.1 * grad) - initial,
                    (initial - 0.1 * baseline) - initial,
                    rtol=0.03,
                    atol=float(0.001 * scale + 2 * torch.finfo(torch.float32).eps * initial.abs().max()),
                    msg=name,
                )
        for handle in handles:
            handle.remove()
        raw_models.clear()
        del models, model, raw_model
