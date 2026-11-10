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

"""GPU tests: Bridge's dispatch runs the ERNIE 5.5 Triton kernels in the real Megatron-Core modules.

* SiTU-GLU reaches the kernel from every MLP kind of an ERNIE 5.5 layer: the dense ``-`` MLP and the shared expert
  (``MLP.forward``, told apart by their fc1 widths) and the grouped routed experts (TEGroupedMLP, Transformer
  Engine). With ``ernie_fused_ops=False`` nothing reaches it.
* The multimax head runs the kernel on CUDA logits with ``fused=True`` and the chunked path without.

Each run is compared with the eager run by cosine similarity (at least 0.9999), a sanity check only: the kernels'
own accuracy tests are in ``test_ernie55_fused_ops_gpu.py``.

Needs a CUDA device, Triton and Transformer Engine with grouped linear layers; without them the tests skip. Runs on
one GPU, or on every rank of a ``torch.distributed.run`` launch (expert parallelism over all ranks)::

    python -m pytest -rA tests/unit_tests/models/ernie/test_ernie55_fused_dispatch_gpu.py
"""

from __future__ import annotations

import collections
import os

import pytest
import torch

from megatron.bridge.models.ernie.modeling_ernie55 import install_ernie55_fused_ops
from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import triton_available
from tests.unit_tests.models.ernie.ernie55_test_utils import isolate_fused_dispatch


pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or not triton_available(), reason="the Triton kernels need CUDA and Triton"
    ),
]

COSINE = 0.9999
MLP_SITE = "megatron.core.transformer.mlp"
EXPERTS_SITE = "megatron.core.transformer.moe.experts"


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach().double().flatten(), expected.detach().double().flatten()
    return (actual @ expected / (actual.norm() * expected.norm()).clamp_min(1e-300)).item()


@pytest.fixture(scope="module")
def expert_parallel(tmp_path_factory):
    """Expert parallelism over every rank (one rank without a launcher), and the process groups."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import EXPERTS, _missing_dependencies

    missing = _missing_dependencies()
    if missing:
        pytest.skip("the SiTU-GLU dispatch GPU test needs " + "; ".join(missing))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        if "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
        else:
            store = tmp_path_factory.mktemp("ernie55_fused_dispatch") / "store"
            dist.init_process_group("nccl", init_method=f"file://{store}", rank=0, world_size=1)
    # The default group stays in place: creating it again in the same worker reuses stale NCCL ids.
    world = dist.get_world_size()
    if EXPERTS % world:
        pytest.skip(f"{EXPERTS} experts do not divide over {world} ranks")
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=world
    )
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups(), world
    parallel_state.destroy_model_parallel()


def test_every_mlp_kind_runs_the_situ_kernel(expert_parallel, monkeypatch):
    import torch.distributed as dist
    from megatron.core.transformer.spec_utils import build_module

    from megatron.bridge.models.ernie.modeling_ernie55 import ernie55_hybrid_stack_spec
    from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import (
        HIDDEN,
        MOE_FFN,
        SHARED_FFN,
        TOKENS,
        build_moe,
        make_config,
    )

    pg_collection, world = expert_parallel
    dispatch = isolate_fused_dispatch(monkeypatch)
    config = make_config(
        activation_func_tanh_clamp_scale=1.0,
        activation_func_tanh_clamp_scale_linear=4.0,
        expert_model_parallel_size=world,
    )
    device = torch.cuda.current_device()
    dense = build_module(
        ernie55_hybrid_stack_spec(config).submodules.mlp_layer,
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    ).to(device)
    moe = build_moe(config, pg_collection, device=device)
    generator = torch.Generator().manual_seed(7 + dist.get_rank())
    hidden = torch.randn(TOKENS, 1, HIDDEN, generator=generator).to(device=device, dtype=torch.bfloat16)

    launches = collections.Counter()
    kernel = dispatch.fused_situ_glu

    def counting_kernel(x, gate_scale, linear_scale=None, *, site):
        launches[(site, x.shape[-1])] += 1
        return kernel(x, gate_scale, linear_scale, site=site)

    monkeypatch.setattr(dispatch, "fused_situ_glu", counting_kernel)

    def run():
        x = hidden.clone().requires_grad_(True)
        dense_out, _ = dense.mlp(x)
        moe_out, _ = moe(x)
        (dense_out.float().square().sum() + moe_out.float().square().sum()).backward()
        return {"dense output": dense_out.detach(), "moe output": moe_out.detach(), "input grad": x.grad}

    config.ernie_fused_ops = False
    install_ernie55_fused_ops(config)
    eager = run()
    assert not launches and not dispatch.FUSED_OP_CALLS

    config.ernie_fused_ops = True
    install_ernie55_fused_ops(config)
    fused = run()
    # fc1 widths: the dense MLP 2 x 4 x HIDDEN, the shared expert 2 x SHARED_FFN, the grouped experts 2 x MOE_FFN.
    assert launches == {
        (MLP_SITE, 2 * 4 * HIDDEN): 1,
        (MLP_SITE, 2 * SHARED_FFN): 1,
        (EXPERTS_SITE, 2 * MOE_FFN): 1,
    }
    assert dispatch.FUSED_OP_CALLS == {f"situ_glu@{MLP_SITE}": 2, f"situ_glu@{EXPERTS_SITE}": 1}
    for name, value in fused.items():
        assert value.dtype == eager[name].dtype and value.shape == eager[name].shape, name
        assert _cosine(value, eager[name]) >= COSINE, name


def test_multimax_head_runs_the_kernel_on_cuda(monkeypatch):
    from megatron.bridge.models.ernie.modeling_ernie55 import ernie55_multimax

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dispatch = isolate_fused_dispatch(monkeypatch)
    generator = torch.Generator().manual_seed(11)
    logits = (8 * torch.randn(64, 2, 1031, generator=generator)).to(device="cuda", dtype=torch.bfloat16)
    cotangent = torch.randn(logits.shape, generator=generator).to(device="cuda", dtype=torch.bfloat16)
    ranges = torch.tensor([-6.0, 7.0, -2.0, 3.0], device="cuda", dtype=torch.bfloat16)
    scales = torch.tensor([0.5, -0.25, 0.01, -0.02], device="cuda", dtype=torch.bfloat16)

    def run(fused):
        x = logits.clone().requires_grad_(True)
        r, s = ranges.clone().requires_grad_(True), scales.clone().requires_grad_(True)
        out = ernie55_multimax(x, r, s, fused=fused)
        out.backward(cotangent)
        return {"output": out.detach(), "logits grad": x.grad, "ranges grad": r.grad, "scales grad": s.grad}

    eager = run(False)
    assert not dispatch.FUSED_OP_CALLS
    fused = run(True)
    assert dispatch.FUSED_OP_CALLS == {"multimax@output_layer": 1}
    for name, value in fused.items():
        assert value.dtype == eager[name].dtype and value.shape == eager[name].shape, name
        assert _cosine(value, eager[name]) >= COSINE, name
