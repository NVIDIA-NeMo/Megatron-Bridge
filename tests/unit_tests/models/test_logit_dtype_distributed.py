# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Two-rank runtime coverage for Bridge output-logit dtype propagation.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/models/test_logit_dtype_distributed.py
"""

import inspect
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig


_TP_SIZE = 2


def _assert_fp32_head_contract(model: torch.nn.Module) -> None:
    output_layer = model.output_layer
    generator = torch.Generator(device="cuda").manual_seed(1234 + dist.get_rank())
    hidden = torch.randint(
        -8,
        9,
        (4, 2, model.config.hidden_size),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    ).to(torch.bfloat16)
    hidden.requires_grad_(True)
    with torch.no_grad():
        output_layer.weight.copy_(
            torch.randint(
                -8,
                9,
                output_layer.weight.shape,
                generator=generator,
                device="cuda",
                dtype=torch.int32,
            ).to(torch.bfloat16)
        )

    logits, _ = output_layer(hidden)
    reference = F.linear(hidden.detach().float(), output_layer.weight.detach().float())
    logits.sum().backward()

    assert output_layer.output_dtype is torch.float32
    assert logits.dtype is torch.float32
    assert hidden.grad is not None and hidden.grad.dtype is torch.bfloat16
    assert output_layer.weight.grad is not None and output_layer.weight.grad.dtype is torch.bfloat16
    assert torch.equal(logits, reference)


def _assert_default_and_bf16_heads_match(default_model: torch.nn.Module, bf16_model: torch.nn.Module) -> None:
    default_layer = default_model.output_layer
    bf16_layer = bf16_model.output_layer
    generator = torch.Generator(device="cuda").manual_seed(4321 + dist.get_rank())
    weight = torch.randint(
        -8,
        9,
        default_layer.weight.shape,
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    ).to(torch.bfloat16)
    with torch.no_grad():
        default_layer.weight.copy_(weight)
        bf16_layer.weight.copy_(weight)

    hidden = torch.randint(
        -8,
        9,
        (4, 2, default_model.config.hidden_size),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    ).to(torch.bfloat16)
    default_hidden = hidden.detach().clone().requires_grad_(True)
    bf16_hidden = hidden.detach().clone().requires_grad_(True)
    default_logits, _ = default_layer(default_hidden)
    bf16_logits, _ = bf16_layer(bf16_hidden)
    default_logits.sum().backward()
    bf16_logits.sum().backward()

    assert default_layer.output_dtype is None
    assert bf16_layer.output_dtype is torch.bfloat16
    assert default_logits.dtype is torch.bfloat16
    assert bf16_logits.dtype is torch.bfloat16
    assert torch.equal(default_logits, bf16_logits)
    assert torch.equal(default_hidden.grad, bf16_hidden.grad)
    assert torch.equal(default_layer.weight.grad, bf16_layer.weight.grad)


@pytest.mark.gpu
def test_tp2_standard_hybrid_and_qwen_wrapper_use_true_fp32_logits() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if "logit_dtype" not in inspect.signature(GPTModel).parameters:
        pytest.skip("installed MCore predates logit_dtype")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=_TP_SIZE)
    model_parallel_cuda_manual_seed(1234)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    common = {
        "num_layers": 1,
        "hidden_size": 128,
        "ffn_hidden_size": 256,
        "num_attention_heads": 4,
        "num_query_groups": 4,
        "kv_channels": 32,
        "vocab_size": 128,
        "tensor_model_parallel_size": _TP_SIZE,
        "params_dtype": torch.bfloat16,
        "bf16": True,
        "attention_backend": AttnBackend.fused,
        "use_cpu_initialization": False,
        "perform_initialization": True,
        "gradient_accumulation_fusion": False,
        "init_method": torch.nn.init.normal_,
        "output_layer_init_method": torch.nn.init.normal_,
        "share_embeddings_and_output_weights": False,
    }

    try:
        gpt_provider = GPTModelProvider(seq_length=8, logit_dtype=torch.float32, **common)
        gpt_provider.finalize()
        gpt_provider._pg_collection = pg_collection
        gpt_model = gpt_provider.provide(pre_process=True, post_process=True).cuda()
        _assert_fp32_head_contract(gpt_model)

        hybrid_provider = MambaModelProvider(
            seq_length=8,
            hybrid_layer_pattern="*",
            position_embedding_type="rope",
            logit_dtype=torch.float32,
            **common,
        )
        hybrid_provider.finalize()
        hybrid_provider._pg_collection = pg_collection
        hybrid_model = hybrid_provider.provide(pre_process=True, post_process=True).cuda()
        _assert_fp32_head_contract(hybrid_model)

        qwen_config = Qwen3VLTransformerConfig(
            language_max_sequence_length=8,
            mrope_section=[4, 6, 6],
            image_token_id=120,
            video_token_id=121,
            vision_start_token_id=122,
            logit_dtype=torch.float32,
            **common,
        )
        qwen_model = Qwen3VLModel(
            language_transformer_config=qwen_config,
            language_transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vision_transformer_config=SimpleNamespace(spatial_merge_size=1, deepstack_visual_indexes=[]),
            pre_process=True,
            post_process=True,
            add_encoder=False,
            add_decoder=True,
            pg_collection=pg_collection,
        ).cuda()
        _assert_fp32_head_contract(qwen_model.language_model)

        default_provider = GPTModelProvider(seq_length=8, **common)
        default_provider.finalize()
        default_provider._pg_collection = pg_collection
        default_model = default_provider.provide(pre_process=True, post_process=True).cuda()
        bf16_provider = GPTModelProvider(seq_length=8, logit_dtype=torch.bfloat16, **common)
        bf16_provider.finalize()
        bf16_provider._pg_collection = pg_collection
        bf16_model = bf16_provider.provide(pre_process=True, post_process=True).cuda()
        _assert_default_and_bf16_heads_match(default_model, bf16_model)
    finally:
        parallel_state.destroy_model_parallel()
        dist.barrier()
        dist.destroy_process_group()
