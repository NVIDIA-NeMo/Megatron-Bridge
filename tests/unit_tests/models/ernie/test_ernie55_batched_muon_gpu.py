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

"""GPU tests of :class:`Ernie55BatchedMuon` (``ernie_muon_ns_batch_size``) on full-size ERNIE 5.5 shapes.

The reference is Megatron-Core's ``TensorParallelMuon`` with the ERNIE 5.5 recipe's Muon settings (bf16 Newton-Schulz,
``fp32_matmul_prec="medium"``, quintic, 5 steps, spectral scale times 0.2, blockwise, split QKV, Nesterov, decoupled
weight decay) on copies of the same fp32 weights, tagged as Megatron-Core tags the ERNIE 5.5 row blocks at TP 1:

* routed experts: fc1 ``[4096, 2048]`` as gate and up halves, fc2 ``[2048, 2048]`` whole (2048 x 2048 blocks);
* the D layers' ``k_b_proj`` per head (``512 x 448`` blocks, more rows than columns) and ``q_up`` with the no-RoPE and
  RoPE rows of every head apart (``448`` and ``64`` rows on 1024 columns);
* the H layers' ``q_up`` per head (``512 x 1024``);
* the dense MLP's fc1 as gate and up halves (``14336 x 4096``, tall) and a whole tall matrix.

Pass criteria:

* chunks of one block (the batched code path with 2-D Newton-Schulz) and ``ns_batch_size=1``: weights and momentum
  bitwise equal to the reference after 3 steps;
* ``ns_batch_size=32``: momentum bitwise equal after 3 steps; every parameter's update (weight change of one step)
  within a relative Frobenius difference of 1e-3 of the reference's, at every step; every block's update with a
  cosine similarity of at least 0.999 to the reference's (a block in the wrong place would fail it);
* the optimizer state dicts have the same keys, shapes and dtypes.

The step times of both optimizers on these shapes are printed (not asserted). Needs one CUDA device::

    ERNIE55_GPU_STRICT=1 python -m pytest -rA -s tests/unit_tests/models/ernie/test_ernie55_batched_muon_gpu.py
"""

import os

import pytest
import torch
from megatron.core.optimizer import emerging_optimizers
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

from megatron.bridge.models.ernie.modeling_ernie55.muon import Ernie55BatchedMuon


pytestmark = [pytest.mark.unit, pytest.mark.run_only_on("GPU")]

_STRICT = os.environ.get("ERNIE55_GPU_STRICT") == "1"
STEPS = 3
# (name, rows, cols, blocks or None for a whole matrix, copies)
SHAPES = [
    ("expert.fc1", 4096, 2048, [2048, 2048], 16),
    ("expert.fc2", 2048, 2048, None, 16),
    ("d.k_b_proj", 16 * 512, 448, [512] * 16, 2),
    ("d.q_up", 16 * 512, 1024, [448, 64] * 16, 2),
    ("h.q_up", 64 * 512, 1024, [512] * 64, 1),
    ("dense.fc1", 2 * 14336, 4096, [14336, 14336], 1),
    ("shared.fc2", 4096, 2048, None, 1),
]
MUON = dict(
    lr=3e-4,
    momentum=0.95,
    nesterov=True,
    weight_decay=0.1,
    use_decoupled_weight_decay=True,
    split_qkv=True,
    split_qkv_per_head=False,
    is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
    qkv_split_shapes=None,
    fp32_matmul_prec="medium",
    coefficient_type="quintic",
    num_ns_steps=5,
    scale_mode="spectral",
    extra_scale_factor=0.2,
    pg_collection=None,
    tp_mode="blockwise",
)


def _skip(reason: str) -> None:
    if _STRICT:
        pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def _device():
    if not torch.cuda.is_available():
        _skip("needs a CUDA device")
    if not emerging_optimizers.HAVE_EMERGING_OPTIMIZERS:
        _skip("needs the emerging-optimizers package")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    yield


def _params() -> list[tuple[str, torch.nn.Parameter, list[int] | None]]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    params = []
    for name, rows, cols, blocks, copies in SHAPES:
        for index in range(copies):
            param = torch.nn.Parameter(
                torch.randn(rows, cols, device="cuda", generator=generator) * cols**-0.5, requires_grad=True
            )
            if blocks is not None:
                param.is_qkv = True
                param.qkv_split_shapes = blocks
                param.qkv_split_shapes_global = blocks
                param.qkv_split_groups_are_complete = True
            params.append((f"{name}.{index}", param, blocks))
    return params


def _twin(params):
    twins = []
    for name, param, blocks in params:
        twin = torch.nn.Parameter(param.detach().clone(), requires_grad=True)
        twin.__dict__.update({k: v for k, v in param.__dict__.items() if not k.startswith("_")})
        twins.append((name, twin, blocks))
    return twins


def _grads(step: int, params) -> list[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(1000 + step)
    return [torch.randn(param.shape, device="cuda", generator=generator) for _, param, _ in params]


def _timed_step(optimizer) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    optimizer.step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _run(ns_batch_size: int, *, chunks_of_one: bool = False):
    """Three steps of the reference and the batched optimizer from the same weights and gradients."""
    params = _params()
    twins = _twin(params)
    reference = TensorParallelMuon([p for _, p, _ in params], **MUON)
    batched = Ernie55BatchedMuon([p for _, p, _ in twins], ns_batch_size=ns_batch_size, **MUON)
    updates, times = [], []
    for step in range(STEPS):
        grads = _grads(step, params)
        before = [(p.detach().clone(), t.detach().clone()) for (_, p, _), (_, t, _) in zip(params, twins)]
        for grad, (_, p, _), (_, t, _) in zip(grads, params, twins):
            p.grad, t.grad = grad.clone(), grad
        reference_ms = _timed_step(reference)
        if chunks_of_one:
            with torch.no_grad():
                batched._batched_step(1)
            batched_ms = float("nan")
        else:
            batched_ms = _timed_step(batched)
        times.append((reference_ms, batched_ms))
        updates.append(
            [
                (b_ref - p.detach(), b_bat - t.detach())
                for (b_ref, b_bat), (_, p, _), (_, t, _) in zip(before, params, twins)
            ]
        )
    return params, twins, reference, batched, updates, times


def test_chunks_of_one_and_batch_size_one_are_bitwise():
    for chunks_of_one, ns_batch_size in ((True, 32), (False, 1)):
        params, twins, reference, batched, _, _ = _run(ns_batch_size, chunks_of_one=chunks_of_one)
        for (name, p, _), (_, t, _) in zip(params, twins):
            assert torch.equal(p, t), f"{name} (chunks_of_one={chunks_of_one}, ns_batch_size={ns_batch_size})"
            assert torch.equal(reference.state[p]["momentum_buffer"], batched.state[t]["momentum_buffer"]), name


def test_batched_updates_match_the_reference():
    params, twins, reference, batched, updates, times = _run(32)
    worst_relative, worst_cosine = 0.0, 1.0
    for step, step_updates in enumerate(updates):
        for (name, _, blocks), (ref_update, bat_update) in zip(params, step_updates):
            relative = ((bat_update - ref_update).norm() / ref_update.norm()).item()
            worst_relative = max(worst_relative, relative)
            assert relative <= 1e-3, f"step {step + 1} {name}: relative update difference {relative:.3e}"
            for index, (ref_block, bat_block) in enumerate(
                zip(
                    torch.split(ref_update, blocks or [ref_update.shape[0]]),
                    torch.split(bat_update, blocks or [ref_update.shape[0]]),
                )
            ):
                cosine = torch.nn.functional.cosine_similarity(ref_block.flatten(), bat_block.flatten(), dim=0).item()
                worst_cosine = min(worst_cosine, cosine)
                assert cosine >= 0.999, f"step {step + 1} {name} block {index}: cosine {cosine:.6f}"
    for (name, p, _), (_, t, _) in zip(params, twins):
        assert torch.equal(reference.state[p]["momentum_buffer"], batched.state[t]["momentum_buffer"]), name
    print(
        f"\n[batched-muon] ns_batch_size=32: worst relative update difference {worst_relative:.3e}, worst block cosine "
        f"{worst_cosine:.6f}; step ms (TensorParallelMuon, Ernie55BatchedMuon): "
        + ", ".join(f"({ref:.1f}, {bat:.1f})" for ref, bat in times)
    )


def test_state_dicts_match():
    _, _, reference, batched, _, _ = _run(32)
    ref_state, bat_state = reference.state_dict(), batched.state_dict()
    assert ref_state["state"].keys() == bat_state["state"].keys()
    for key, state in ref_state["state"].items():
        assert state.keys() == bat_state["state"][key].keys()
        for name, value in state.items():
            other = bat_state["state"][key][name]
            assert (value.shape, value.dtype) == (other.shape, other.dtype), (key, name)
    ref_groups = [{k: v for k, v in group.items() if k != "params"} for group in ref_state["param_groups"]]
    bat_groups = [{k: v for k, v in group.items() if k != "params"} for group in bat_state["param_groups"]]
    assert ref_groups == bat_groups
