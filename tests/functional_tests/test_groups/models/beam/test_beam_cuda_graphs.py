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

"""Two-GPU Beam sandwich-norm parameter gathering during real TE graph replay.

Run with ``python -m torch.distributed.run --standalone --nproc_per_node=2
-m pytest tests/functional_tests/test_groups/models/beam/test_beam_cuda_graphs.py``.
No model download or dataset is needed.
"""

import datetime
import gc
import os
from collections import Counter
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.module import Float16Module

from megatron.bridge.models.beam.beam_provider import BeamModelProvider
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


pytestmark = [pytest.mark.integration, pytest.mark.run_only_on("GPU")]


@pytest.fixture(scope="session")
def ensure_test_data():
    """Override the functional suite's download fixture: this test uses generated tensors."""
    return None


@pytest.fixture(scope="module")
def beam_process_groups():
    assert int(os.environ.get("WORLD_SIZE", "1")) == 2, "Launch this regression with exactly two GPU ranks"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Match initialize._initialize_distributed for TE graphs: construct DDP
    # and run warmup on a non-default stream so retained gradient nodes can
    # participate in capture.
    original_stream = torch.cuda.current_stream()
    training_stream = torch.cuda.Stream()
    training_stream.wait_stream(original_stream)
    torch.cuda.set_stream(training_stream)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    assert dist.get_world_size() == 2
    assert not parallel_state.model_parallel_is_initialized()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    assert pg.dp_cp.size() == 2
    try:
        yield pg
    finally:
        gc.collect()
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        if owns_group:
            dist.destroy_process_group()
        torch.cuda.set_stream(original_stream)


@pytest.fixture
def beam_dp_group(beam_process_groups):
    # Keep the process groups alive across scopes: default NCCL group
    # destruction/reinitialization in one process needs external rendezvous.
    init_num_microbatches_calculator(
        rank=dist.get_rank(), global_batch_size=2, micro_batch_size=1, data_parallel_size=2
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, use_cudagraphable_rng=True, force_reset_rng=True)
    try:
        yield beam_process_groups
    finally:
        gc.collect()
        torch.cuda.synchronize()
        get_cuda_rng_tracker().reset()
        destroy_num_microbatches_calculator()


def _make_model(pg, scope):
    config = BeamModelProvider(
        hybrid_layer_pattern="*-*-",
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=64,
        vocab_size=128,
        seq_length=32,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        position_embedding_type="rope",
        apply_rope_fusion=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        post_self_attn_layernorm=True,
        post_mlp_layernorm=True,
        depth_scaled_residual=True,
        attention_output_gate=True,
        attention_output_gate_bias=True,
        gradient_accumulation_fusion=True,
        cuda_graph_impl="transformer_engine" if scope else "none",
        cuda_graph_warmup_steps=1,
        use_te_rng_tracker=True,
        cross_entropy_loss_fusion=False,
    )
    set_cuda_graph_modules(config, [scope] if scope else [])
    config.finalize()
    config._pg_collection = pg
    model = config.provide(pre_process=True, post_process=True).cuda()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "post_self_attn_layernorm.weight" in name or "post_mlp_layernorm.weight" in name:
                parameter.copy_(torch.linspace(0.7, 1.3, parameter.numel(), device="cuda").reshape_as(parameter))
    model = Float16Module(config, model)
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            grad_reduce_in_fp32=True,
            bucket_size=1,
        ),
        model,
        pg_collection=pg,
    )
    return ddp


def _make_optimizer(model, pg):
    return get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=2e-3,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            main_grads_dtype=torch.float32,
            main_params_dtype=torch.float32,
            use_distributed_optimizer=True,
            overlap_param_gather=True,
            clip_grad=1.0,
        ),
        [model],
        use_gloo_process_groups=False,
        pg_collection=pg,
    )


def _assert_parameters_close(eager, graphed):
    for (name, expected), (other_name, actual) in zip(
        eager.named_parameters(), graphed.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3, msg=lambda message: f"{name}: {message}")


def _assert_reduced_gradients_close(eager, graphed, pg):
    # Reduce-scatter only promises valid reduced values in this rank's shard.
    # Comparing the rest of main_grad would compare unreduced scratch storage.
    for expected_group, actual_group in zip(eager.bucket_groups, graphed.bucket_groups, strict=True):
        for expected_bucket, actual_bucket in zip(expected_group.buckets, actual_group.buckets, strict=True):
            expected = expected_bucket.grad_data.chunk(pg.dp_cp.size())[pg.dp_cp.rank()]
            actual = actual_bucket.grad_data.chunk(pg.dp_cp.size())[pg.dp_cp.rank()]
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-4)


@pytest.mark.parametrize("scope", ["attn", "mlp"])
def test_beam_graph_replay_gathers_updated_sandwich_norms(beam_dp_group, scope):
    """Compare eager and TE replay with native DP2 sharded Adam and gather overlap."""
    pg = beam_dp_group
    eager = _make_model(pg, None)
    graphed = _make_model(pg, scope)
    with torch.no_grad():
        for expected, actual in zip(eager.parameters(), graphed.parameters(), strict=True):
            actual.copy_(expected)
    eager_optimizer = _make_optimizer(eager, pg)
    graph_optimizer = _make_optimizer(graphed, pg)
    helper = TECudaGraphHelper([graphed], graphed.config, 32, 1, optimizers=[graph_optimizer], pg_collection=pg)
    norms = [
        layer.post_self_attn_layernorm if scope == "attn" else layer.post_mlp_layernorm
        for layer in helper.flattened_callables
    ]
    assert len(norms) == 2 and all(norm is not None for norm in norms)
    initial_norms = [norm.weight.detach().clone() for norm in norms]
    hook_calls = Counter()
    make_native_hook = graphed._make_forward_pre_hook

    def make_observed_hook():
        native_hook = make_native_hook()

        def hook(module, *args):
            native_hook(module, *args)
            if module in norms:
                hook_calls[id(module)] += 1
                bucket_group = graphed.param_to_bucket_group[module.weight]
                assert bucket_group.param_gather_dispatched and bucket_group.param_gather_handle is None

        return hook

    try:
        # Bridge disables gathering during the initial forward/backward pass,
        # then enables it after the first successful optimizer update.
        for model in (eager, graphed):
            model.disable_forward_pre_hook(param_sync=False)
        # One eager warmup records native DDP gradient-ready counts. Three replay
        # updates then exercise two already-updated states, without gathering all
        # parameters between the optimizer step and the following forward pass.
        for step in range(4):
            if step == 1:
                graphed.disable_forward_pre_hook(param_sync=False)
                try:
                    helper.create_cudagraphs()
                finally:
                    graphed.enable_forward_pre_hook()
                assert helper.graphs_created()
                assert all(layer.cuda_graphs for layer in helper.flattened_callables)
                # Observe real native gather hooks, without replacing their work.
                with patch.object(graphed, "_make_forward_pre_hook", make_observed_hook):
                    helper.cuda_graph_set_manual_hooks()

            for model, optimizer in ((eager, eager_optimizer), (graphed, graph_optimizer)):
                model.zero_grad_buffer()
                optimizer.zero_grad()

            positions = torch.arange(32, device="cuda").unsqueeze(0)
            tokens = (positions * 7 + pg.dp_cp.rank() * 11 + step * 3) % 128
            labels = (tokens + 5 + step) % 128
            outputs = []
            for model in (eager, graphed):
                model.set_is_first_microbatch()
                outputs.append(model(input_ids=tokens, position_ids=positions, attention_mask=None))
                assert all(group.param_gather_handle is None for group in model.bucket_groups)
            torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-2, atol=2e-3)
            # At this point the forward hooks must have gathered the previous
            # update. A force_sync before forward would conceal missing hooks.
            _assert_parameters_close(eager, graphed)
            if step > 0:
                assert all(hook_calls[id(norm)] == step for norm in norms)
            for model, output in zip((eager, graphed), outputs, strict=True):
                loss = F.cross_entropy(output.float().reshape(-1, 128), labels.reshape(-1))
                assert torch.isfinite(loss)
                loss.backward()
                model.finish_grad_sync()
            _assert_reduced_gradients_close(eager, graphed, pg)
            for optimizer in (eager_optimizer, graph_optimizer):
                success, grad_norm, _ = optimizer.step()
                assert success and grad_norm is not None and 0 < grad_norm < float("inf")
            if step == 0:
                for model in (eager, graphed):
                    model.enable_forward_pre_hook()
            # The training loop releases the completed step's autograd graph
            # before capture. Keep the same lifetime in this direct layer test.
            del loss, output, outputs

        # Check the final update too, after all replay comparisons are complete.
        for model in (eager, graphed):
            model.start_param_sync(force_sync=True)
        _assert_parameters_close(eager, graphed)
        assert all(not torch.equal(norm.weight, initial) for norm, initial in zip(norms, initial_norms, strict=True))
    finally:
        if helper.graphs_created():
            helper.delete_cuda_graphs()
        for model in (eager, graphed):
            if model.remove_forward_pre_hook_handles:
                model.disable_forward_pre_hook(param_sync=False)
