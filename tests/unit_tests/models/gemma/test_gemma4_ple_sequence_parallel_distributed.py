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

"""Two-rank Gemma4 PLE sequence-parallel parity and backward regression test.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/models/gemma/test_gemma4_ple_sequence_parallel_distributed.py
"""

import os
from types import SimpleNamespace

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)

from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider
from megatron.bridge.models.gemma.modeling_gemma4 import (
    _attach_ple_modules,
    _compute_per_layer_inputs,
)


_TP_SIZE = 2
_RTOL = 1e-5
_ATOL = 1e-6


def _full_weights(
    *, vocab_size: int, packed_ple_size: int, hidden_size: int, ple_dim: int
) -> tuple[torch.Tensor, ...]:
    embedding = torch.arange(vocab_size * packed_ple_size, device="cuda", dtype=torch.float32).reshape(
        vocab_size, packed_ple_size
    )
    projection = torch.arange(packed_ple_size * hidden_size, device="cuda", dtype=torch.float32).reshape(
        packed_ple_size, hidden_size
    )
    norm = torch.arange(ple_dim, device="cuda", dtype=torch.float32) / 16 + 0.75
    output = torch.arange(5 * packed_ple_size, device="cuda", dtype=torch.float32).reshape(5, packed_ple_size)
    return (
        (embedding.remainder(17) - 8) / 64,
        (projection.remainder(13) - 6) / 32,
        norm,
        (output.remainder(11) - 5) / 32,
    )


def _build_ple_model(
    *,
    tp_size: int,
    sequence_parallel: bool,
    tp_group: dist.ProcessGroup,
    num_layers: int,
    ple_dim: int,
    hidden_size: int,
    vocab_size: int,
) -> SimpleNamespace:
    config = Gemma4DenseProvider(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=16,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=4,
        global_kv_channels=4,
        per_layer_embed_vocab_size=vocab_size,
        per_layer_embed_dim=ple_dim,
        vocab_size=vocab_size,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sequence_parallel,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
        params_dtype=torch.float32,
        bf16=False,
    )
    model = SimpleNamespace(config=config, pg_collection=ProcessGroupCollection(tp=tp_group))
    _attach_ple_modules(model, config, config)
    model.per_layer_proj_norm.cuda()
    return model


def _gather_sequence(local_tensor: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
    gathered = gather_from_sequence_parallel_region(
        local_tensor.transpose(0, 1).contiguous(),
        tensor_parallel_output_grad=False,
        group=tp_group,
    )
    return gathered.transpose(0, 1).contiguous()


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.gpu
def test_gemma4_ple_tp2_sp_matches_tp1_forward_backward() -> None:
    """TP2/SP PLE values and gradients must match TP1 on identical tensors."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    # Avoid TF32 so the tolerance measures only FP32 collective/reduction ordering.
    torch.set_float32_matmul_precision("highest")

    if owns_process_group:
        dist.init_process_group(backend="nccl")

    try:
        if owns_model_parallel:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=_TP_SIZE,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
            )
        tp_group = ProcessGroupCollection.use_mpu_process_groups().tp
        singleton_groups = [dist.new_group(ranks=[rank]) for rank in range(_TP_SIZE)]
        reference_tp_group = singleton_groups[dist.get_rank()]

        num_layers = 3
        ple_dim = 4
        packed_ple_size = num_layers * ple_dim
        hidden_size = 8
        vocab_size = 16
        model = _build_ple_model(
            tp_size=_TP_SIZE,
            sequence_parallel=True,
            tp_group=tp_group,
            num_layers=num_layers,
            ple_dim=ple_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        reference_model = _build_ple_model(
            tp_size=1,
            sequence_parallel=False,
            tp_group=reference_tp_group,
            num_layers=num_layers,
            ple_dim=ple_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

        full_embedding, full_projection, norm_weight, output_weight = _full_weights(
            vocab_size=vocab_size,
            packed_ple_size=packed_ple_size,
            hidden_size=hidden_size,
            ple_dim=ple_dim,
        )
        with torch.no_grad():
            vocab_start = model.per_layer_embedding.vocab_start_index
            vocab_end = model.per_layer_embedding.vocab_end_index
            model.per_layer_embedding.weight.copy_(full_embedding[vocab_start:vocab_end])
            output_partition = model.per_layer_model_proj.output_size_per_partition
            output_start = dist.get_rank(group=tp_group) * output_partition
            output_end = output_start + output_partition
            model.per_layer_model_proj.weight.copy_(full_projection[output_start:output_end])
            model.per_layer_proj_norm.weight.copy_(norm_weight)
            reference_model.per_layer_embedding.weight.copy_(full_embedding)
            reference_model.per_layer_model_proj.weight.copy_(full_projection)
            reference_model.per_layer_proj_norm.weight.copy_(norm_weight)

        for batch_size, seq_length in ((1, 4), (2, 6)):
            input_ids = torch.arange(batch_size * seq_length, device="cuda").reshape(batch_size, seq_length)
            input_ids = input_ids.remainder(vocab_size)
            decoder_values = torch.arange(
                seq_length * batch_size * hidden_size,
                device="cuda",
                dtype=torch.float32,
            ).reshape(seq_length, batch_size, hidden_size)
            decoder_values = (decoder_values.remainder(19) - 9) / 32
            coefficients = torch.arange(
                batch_size * seq_length * output_weight.shape[0], device="cuda", dtype=torch.float32
            ).reshape(batch_size, seq_length, output_weight.shape[0])
            coefficients = (coefficients.remainder(7) - 3) / 16

            for parameter in (
                reference_model.per_layer_embedding.weight,
                reference_model.per_layer_model_proj.weight,
                reference_model.per_layer_proj_norm.weight,
            ):
                parameter.grad = None
            output_ref = output_weight.detach().clone().requires_grad_(True)
            decoder_ref = decoder_values.detach().clone().requires_grad_(True)
            ple_ref = _compute_per_layer_inputs(reference_model, input_ids, decoder_ref)
            logits_ref = torch.nn.functional.linear(ple_ref.flatten(-2), output_ref)
            loss_ref = (logits_ref * coefficients).sum()
            loss_ref.backward()

            for parameter in (
                model.per_layer_embedding.weight,
                model.per_layer_model_proj.weight,
                model.per_layer_proj_norm.weight,
            ):
                parameter.grad = None
            decoder_dist = decoder_values.detach().clone().requires_grad_(True)
            local_decoder = scatter_to_sequence_parallel_region(decoder_dist, group=tp_group)
            output_dist = output_weight.detach().clone().requires_grad_(True)
            ple_local = _compute_per_layer_inputs(model, input_ids, local_decoder)
            logits_local = torch.nn.functional.linear(ple_local.flatten(-2), output_dist)
            coefficients_local = scatter_to_sequence_parallel_region(
                coefficients.transpose(0, 1).contiguous(), group=tp_group
            ).transpose(0, 1)
            loss_local = (logits_local * coefficients_local).sum()

            _assert_close(_gather_sequence(ple_local.detach(), tp_group), ple_ref.detach())
            _assert_close(_gather_sequence(logits_local.detach(), tp_group), logits_ref.detach())
            loss_dist = loss_local.detach().clone()
            dist.all_reduce(loss_dist, group=tp_group)
            _assert_close(loss_dist, loss_ref.detach())

            loss_local.backward()
            _assert_close(decoder_dist.grad, decoder_ref.grad)
            _assert_close(
                model.per_layer_embedding.weight.grad,
                reference_model.per_layer_embedding.weight.grad[vocab_start:vocab_end],
            )
            _assert_close(
                model.per_layer_model_proj.weight.grad,
                reference_model.per_layer_model_proj.weight.grad[output_start:output_end],
            )

            norm_grad = model.per_layer_proj_norm.weight.grad.detach().clone()
            dist.all_reduce(norm_grad, group=tp_group)
            _assert_close(norm_grad, reference_model.per_layer_proj_norm.weight.grad)
            output_grad = output_dist.grad.detach().clone()
            dist.all_reduce(output_grad, group=tp_group)
            _assert_close(output_grad, output_ref.grad)

        with pytest.raises(
            AssertionError,
            match="First dimension of the tensor should be divisible by tensor parallel size",
        ):
            scatter_to_sequence_parallel_region(torch.zeros(5, 1, hidden_size, device="cuda"), group=tp_group)
        with pytest.raises(AssertionError, match="9 is not divisible by 2"):
            _build_ple_model(
                tp_size=_TP_SIZE,
                sequence_parallel=True,
                tp_group=tp_group,
                num_layers=3,
                ple_dim=3,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
            )
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
