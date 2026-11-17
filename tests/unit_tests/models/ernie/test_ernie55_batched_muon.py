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

"""CPU tests of ``ernie_muon_ns_batch_size``: :class:`Ernie55BatchedMuon` and its registry installation.

The reference is Megatron-Core's ``TensorParallelMuon`` on copies of the same parameters, tagged the way Megatron-Core's
optimizer setup tags the ERNIE 5.5 row blocks (one-group layouts at TP 1), plus a two-group layout and whole matrices.

* With every chunk of one block (``_batched_step(1)``), the batched step must equal the parent's step bitwise: the same
  per-parameter weight decay, momentum and Nesterov input, the same blocks, the same 2-D Newton-Schulz calls, the same
  reassembly and update.
* With batches, only the GEMM kernels change (``baddbmm`` instead of ``addmm``), so the results match to fp32 rounding.
* ``ns_batch_size=1``, and any sharded tensor parallel group, run the parent's step itself.

The GPU counterpart (``test_ernie55_batched_muon_gpu.py``) checks the full-size shapes in bf16 Newton-Schulz.
"""

import logging
import types

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig, emerging_optimizers
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon, _localize_qkv_split_shapes

from megatron.bridge.models.ernie.modeling_ernie55 import muon
from megatron.bridge.models.ernie.modeling_ernie55.muon import Ernie55BatchedMuon, install_ernie55_batched_muon
from tests.unit_tests.models.ernie.test_ernie55_muon import _find_muon, _model, pg_collection  # noqa: F401


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not emerging_optimizers.HAVE_EMERGING_OPTIMIZERS, reason="needs the emerging-optimizers package"
    ),
]

COLS = 7
# (rows of each block, number of groups) of the split weights; None = a whole matrix of the given rows.
SPLIT_LAYOUTS = [
    pytest.param([5] * 4, id="uniform-wide"),  # an H-layer q per head
    pytest.param([6, 2] * 3, id="mixed"),  # a D-layer q: no-RoPE and RoPE rows of every head
    pytest.param([9] * 3, id="uniform-tall"),  # k_b_proj: blocks with more rows than columns
    pytest.param([9, 2] * 2, id="mixed-tall"),
]


@pytest.fixture(autouse=True)
def _restore_registry():
    """Every test leaves Megatron-Core's ``muon`` registry entry as it found it."""
    saved_entry = emerging_optimizers._EMERGING_OPTIMIZERS["muon"]
    saved_original = muon._MCORE_MUON_ENTRY
    yield
    emerging_optimizers._EMERGING_OPTIMIZERS["muon"] = saved_entry
    muon._MCORE_MUON_ENTRY = saved_original


def _tag_split(param: torch.nn.Parameter, blocks: list[int], *, groups: int = 1, per_head: bool = False) -> None:
    """The attributes Megatron-Core's optimizer setup gives a TP-1 weight with a layout of ``blocks``."""
    param.is_qkv = True
    param.qkv_split_shapes_global = blocks * groups if not per_head else blocks
    if per_head:
        param.qkv_split_shapes, param.qkv_split_heads_are_complete = _localize_qkv_split_shapes(
            blocks, local_start=0, local_rows=sum(blocks)
        )
    else:
        param.qkv_split_shapes = blocks
        param.qkv_split_groups_are_complete = True


def _params(*, per_head: bool = False) -> list[torch.nn.Parameter]:
    """Split weights of every layout, a two-group split, whole matrices of repeated shapes, and a grad-less weight."""
    generator = torch.Generator().manual_seed(3)
    params = []

    def new(rows: int, cols: int = COLS) -> torch.nn.Parameter:
        param = torch.nn.Parameter(torch.randn(rows, cols, generator=generator))
        params.append(param)
        return param

    for layout in SPLIT_LAYOUTS:
        blocks = layout.values[0]
        if per_head and len(set(blocks)) == 1 and blocks[0] > COLS:
            continue  # Megatron-Core's per-head path cannot reshape tall equal blocks (test_ernie55_muon.py)
        for _ in range(2):  # two weights of each layout, so their blocks share batches
            _tag_split(new(sum(blocks)), blocks, per_head=per_head)
    if not per_head:
        _tag_split(new(2 * 5), [3, 2], groups=2)  # two groups: each projection takes rows of both groups
    for rows in (6, 6, 6, 7, 12, 12, 3):  # whole matrices: wide, square and tall, some of one shape
        new(rows)
    new(4).requires_grad_(True)  # no gradient below: skipped, as by the parent
    return params


def _optimizers(params, *, ns_batch_size: int, **overrides):
    """A ``TensorParallelMuon`` on ``params`` and an ``Ernie55BatchedMuon`` on copies of them."""
    kwargs = dict(
        lr=0.05,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.1,
        use_decoupled_weight_decay=True,
        split_qkv=True,
        split_qkv_per_head=False,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=None,
        fp32_matmul_prec="highest",
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=0.2,
        pg_collection=None,
        tp_mode="blockwise",
    )
    kwargs.update(overrides)
    copies = []
    for param in params:
        twin = torch.nn.Parameter(param.detach().clone())
        twin.__dict__.update({k: v for k, v in param.__dict__.items() if not k.startswith("_")})
        copies.append(twin)
    reference = TensorParallelMuon(params, **kwargs)
    batched = Ernie55BatchedMuon(copies, ns_batch_size=ns_batch_size, **kwargs)
    return reference, batched, copies


def _set_grads(params, copies, step: int) -> None:
    generator = torch.Generator().manual_seed(100 + step)
    for param, twin in zip(params[:-1], copies[:-1]):
        grad = torch.randn(param.shape, generator=generator)
        param.grad, twin.grad = grad.clone(), grad.clone()


def _run(reference, batched, params, copies, *, steps: int = 3, batched_step=None) -> None:
    for step in range(steps):
        _set_grads(params, copies, step)
        reference.step()
        if batched_step is None:
            batched.step()
        else:
            with torch.no_grad():
                batched_step(batched)


def _assert_equal_states(reference, batched, params, copies, *, exact: bool) -> None:
    for index, (param, twin) in enumerate(zip(params, copies)):
        if exact:
            assert torch.equal(param, twin), f"parameter {index}"
        else:
            torch.testing.assert_close(twin, param, rtol=1e-5, atol=1e-6, msg=f"parameter {index}")
        ref_state, state = reference.state.get(param, {}), batched.state.get(twin, {})
        assert ref_state.keys() == state.keys(), f"state of parameter {index}"
        for key in ref_state:
            # The momentum depends only on the gradients, so it is bitwise equal in every mode.
            assert torch.equal(ref_state[key], state[key]), f"{key} of parameter {index}"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="nesterov-decoupled"),
        pytest.param(dict(nesterov=False, use_decoupled_weight_decay=False), id="momentum-l2"),
        pytest.param(dict(tp_mode="duplicated"), id="duplicated"),
    ],
)
def test_single_block_chunks_equal_the_parent_bitwise(overrides):
    """Chunks of one block take the parent's exact calls: three steps give bitwise-equal weights and momentum."""
    params = _params()
    reference, batched, copies = _optimizers(params, ns_batch_size=4, **overrides)
    _run(reference, batched, params, copies, batched_step=lambda opt: opt._batched_step(1))
    _assert_equal_states(reference, batched, params, copies, exact=True)
    assert params[-1].grad is None and copies[-1].grad is None and not batched.state.get(copies[-1])


@pytest.mark.parametrize("ns_batch_size", [2, 3, 32])
def test_batched_newton_schulz_matches_the_parent(ns_batch_size, caplog):
    """Batched chunks change only the GEMM kernels: weights match to fp32 rounding, the momentum bitwise."""
    params = _params()
    reference, batched, copies = _optimizers(params, ns_batch_size=ns_batch_size)
    with caplog.at_level(logging.INFO, logger=muon.__name__):
        _run(reference, batched, params, copies)
    _assert_equal_states(reference, batched, params, copies, exact=False)
    lines = [record.getMessage() for record in caplog.records if "batched Muon (rank" in record.getMessage()]
    assert len(lines) == 1, lines  # logged once, at the first step
    assert f"ns_batch_size={ns_batch_size};" in lines[0] and "0 parameters orthogonalized one at a time" in lines[0]


def test_every_block_is_orthogonalized_alone():
    """One batched step equals a loop over the blocks with the parent's own Newton-Schulz and scale."""
    params = _params()
    reference, batched, copies = _optimizers(params, ns_batch_size=32, weight_decay=0.0)
    _set_grads(params, copies, 0)
    before = [twin.detach().clone() for twin in copies]
    batched.step()
    group = batched.param_groups[0]
    for index, (twin, start) in enumerate(zip(copies[:-1], before[:-1])):
        effective = twin.grad.lerp(batched.state[twin]["momentum_buffer"], group["momentum"])
        if getattr(twin, "is_qkv", False):
            blocks, width = twin.qkv_split_shapes, sum(twin.qkv_split_shapes)
            grouped = effective.view(-1, width, COLS)
            pieces = torch.split(grouped, blocks, dim=1)
            update = torch.cat(
                [
                    reference.scaled_orthogonalize_fn(piece.reshape(-1, COLS), None).view(grouped.shape[0], -1, COLS)
                    for piece in pieces
                ],
                dim=1,
            ).view_as(twin)
        else:
            update = reference.scaled_orthogonalize_fn(effective, None)
        torch.testing.assert_close(twin.detach(), start - group["lr"] * update, rtol=1e-5, atol=1e-6, msg=str(index))


def test_ns_batch_size_one_runs_the_parent_step(monkeypatch):
    """``ns_batch_size=1`` never enters the batched path, and gives the parent's results bitwise."""
    params = _params()
    reference, batched, copies = _optimizers(params, ns_batch_size=1)
    monkeypatch.setattr(Ernie55BatchedMuon, "_batched_step", lambda *args: pytest.fail("batched path entered"))
    _run(reference, batched, params, copies)
    _assert_equal_states(reference, batched, params, copies, exact=True)


def test_sharded_groups_run_the_parent_step(monkeypatch):
    """With a tensor parallel group of two ranks the parent's step runs (its gathers and collectives)."""
    params = _params()
    collection = types.SimpleNamespace(tp="tp", expt_tp="expt_tp", gtp_remat=None, expt_gtp_remat=None)
    _, batched, copies = _optimizers(params, ns_batch_size=8)
    batched.pg_collection = collection
    monkeypatch.setattr(muon, "get_pg_size", lambda group: 2 if group == "tp" else 1)
    assert batched._is_unsharded() is False
    calls = []
    monkeypatch.setattr(TensorParallelMuon, "step", lambda self, closure=None: calls.append(self))
    monkeypatch.setattr(Ernie55BatchedMuon, "_batched_step", lambda *args: pytest.fail("batched path entered"))
    batched.step()
    assert calls == [batched]


def test_per_head_weights_go_to_the_parent():
    """Under ``split_qkv_per_head`` the split weights are orthogonalized by the parent, the rest batched."""
    params = _params(per_head=True)
    reference, batched, copies = _optimizers(params, ns_batch_size=4, split_qkv_per_head=True)
    _run(reference, batched, params, copies, batched_step=lambda opt: opt._batched_step(1))
    _assert_equal_states(reference, batched, params, copies, exact=True)
    plans = [batched._ns_blocks(index, twin) for index, twin in enumerate(copies)]
    assert [plan is None for plan in plans] == [getattr(twin, "is_qkv", False) for twin in copies]


def test_invalid_batch_size_is_rejected():
    for value in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="ns_batch_size must be an int of at least 1"):
            Ernie55BatchedMuon([torch.nn.Parameter(torch.zeros(2, 2))], ns_batch_size=value)


# ---------------------------------------------------------------------------------------------
# The registry entry
# ---------------------------------------------------------------------------------------------


def _model_chunk(batch_size: int) -> types.SimpleNamespace:
    """What ``muon``'s ``config_to_kwargs`` reads from a model chunk: its transformer config."""
    config = types.SimpleNamespace(
        ernie_muon_ns_batch_size=batch_size, num_attention_heads=4, num_query_groups=4, kv_channels=8
    )
    return types.SimpleNamespace(config=config)


def _optimizer_config(**overrides) -> OptimizerConfig:
    values = dict(optimizer="muon", lr=1e-3, bf16=False, use_distributed_optimizer=False, muon_tp_mode="blockwise")
    values.update(overrides)
    return OptimizerConfig(**values)


def test_flag_one_keeps_megatron_cores_entry():
    original = emerging_optimizers._EMERGING_OPTIMIZERS["muon"]
    install_ernie55_batched_muon(types.SimpleNamespace(ernie_muon_ns_batch_size=1))
    assert emerging_optimizers._EMERGING_OPTIMIZERS["muon"] is original
    assert original.config_to_cls(_optimizer_config()) is TensorParallelMuon


def test_flag_installs_and_selects_the_batched_class():
    original = emerging_optimizers._EMERGING_OPTIMIZERS["muon"]
    install_ernie55_batched_muon(types.SimpleNamespace(ernie_muon_ns_batch_size=32))
    entry = emerging_optimizers._EMERGING_OPTIMIZERS["muon"]
    assert entry is not original and entry.optimizer_cls is Ernie55BatchedMuon
    assert entry.default_param_overrides is original.default_param_overrides
    assert entry.init_state_fn is original.init_state_fn
    config = _optimizer_config()
    assert entry.config_to_cls(config) is Ernie55BatchedMuon
    kwargs = entry.config_to_kwargs(config, [_model_chunk(32)], None)
    expected = original.config_to_kwargs(config, [_model_chunk(32)], None)
    assert kwargs.pop("ns_batch_size") == 32
    assert kwargs.keys() == expected.keys()
    optimizer, _ = emerging_optimizers._create_emerging_optimizer(
        config, [{"params": [torch.nn.Parameter(torch.zeros(3, 3))]}], "muon", [_model_chunk(32)], None
    )
    assert type(optimizer) is Ernie55BatchedMuon and optimizer.ns_batch_size == 32
    # Installing again keeps one wrapper, and the flag at 1 restores Megatron-Core's own entry.
    install_ernie55_batched_muon(types.SimpleNamespace(ernie_muon_ns_batch_size=8))
    assert emerging_optimizers._EMERGING_OPTIMIZERS["muon"] is entry
    install_ernie55_batched_muon(types.SimpleNamespace(ernie_muon_ns_batch_size=1))
    assert emerging_optimizers._EMERGING_OPTIMIZERS["muon"] is original


def test_layer_sharded_and_state_offload_keep_their_classes():
    from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon

    install_ernie55_batched_muon(types.SimpleNamespace(ernie_muon_ns_batch_size=32))
    entry = emerging_optimizers._EMERGING_OPTIMIZERS["muon"]
    # The class selector reads only muon_tp_mode (OptimizerConfig itself also wants the layer-wise optimizer for it).
    assert entry.config_to_cls(types.SimpleNamespace(muon_tp_mode="layer_sharded")) is LayerShardedMuon
    offload = _optimizer_config()
    offload.chunked_optimizer_state_offload, offload.optimizer_state_offload_fraction = True, 0.5
    assert entry.config_to_cls(offload) is TensorParallelMuon
    assert "ns_batch_size" not in entry.config_to_kwargs(offload, [_model_chunk(32)], None)


def test_megatron_core_setup_builds_the_batched_muon(pg_collection, monkeypatch):
    """The toy model's flag reaches Megatron-Core's real optimizer setup, whose batched step matches the parent's.

    Two identical toy models: one built with ``ernie_muon_ns_batch_size=4`` (its construction installs the batched
    entry), then one with 1 (which restores Megatron-Core's entry). ``_get_megatron_emerging_optimizer`` builds
    ``Ernie55BatchedMuon`` for the first and exactly ``TensorParallelMuon`` for the second; one step with the same
    gradients moves every Muon weight the same way (fp32 Newton-Schulz on CPU, batched GEMMs within 1e-5).
    The CPU stand-ins are ``test_megatron_core_optimizer_setup_splits_the_reference_blocks``'s.
    """
    import megatron.core.optimizer as mcore_optimizer
    from megatron.core.optimizer import _get_megatron_emerging_optimizer, get_standard_config_overrides
    from megatron.core.optimizer.optimizer import FP32Optimizer, MegatronOptimizer

    class _CPUFP32Optimizer(FP32Optimizer):
        def __init__(self, optimizer, config, init_state_fn):
            MegatronOptimizer.__init__(self, optimizer, config, init_state_fn)
            self._scale = torch.tensor([1.0])
            self.is_stub_optimizer = optimizer is None

    monkeypatch.setattr(mcore_optimizer, "FP32Optimizer", _CPUFP32Optimizer)
    monkeypatch.setattr(mcore_optimizer, "USING_PYTORCH_OPTIMIZER", True)
    config = OptimizerConfig(
        optimizer="muon",
        lr=1e-2,
        weight_decay=0.1,
        bf16=False,
        use_distributed_optimizer=False,
        muon_split_qkv=True,
        muon_split_qkv_per_head=False,
        muon_split_mla_qkv=False,
        muon_tp_mode="blockwise",
        muon_nesterov=True,
        muon_fp32_matmul_prec="highest",
        muon_extra_scale_factor=0.2,
        muon_scalar_optimizer="adam",
    )

    def build(batch_size: int):
        model = _model(pg_collection, ernie_muon_ns_batch_size=batch_size).float()
        optimizer = _get_megatron_emerging_optimizer(
            config, [model], config_overrides=get_standard_config_overrides(config), pg_collection=pg_collection
        )
        return model, _find_muon(optimizer)

    model, batched = build(4)
    reference_model, reference = build(1)
    assert type(batched) is Ernie55BatchedMuon and batched.ns_batch_size == 4
    assert type(reference) is TensorParallelMuon
    names = {param: name for name, param in model.named_parameters()}
    reference_params = dict(reference_model.named_parameters())
    muon_params = [param for group in batched.param_groups for param in group["params"]]
    reference_ids = {id(param) for group in reference.param_groups for param in group["params"]}
    assert {names[param] for param in muon_params} == {
        name for name, param in reference_params.items() if id(param) in reference_ids
    }
    generator = torch.Generator().manual_seed(5)
    for param in muon_params:
        grad = torch.randn(param.shape, generator=generator, dtype=param.dtype)
        param.grad, reference_params[names[param]].grad = grad, grad.clone()
    batched.step()
    reference.step()
    split = [param for param in muon_params if getattr(param, "is_qkv", False)]
    assert split and len(split) < len(muon_params)  # both split and whole matrices took the batched path
    for param in muon_params:
        torch.testing.assert_close(param, reference_params[names[param]], rtol=1e-5, atol=1e-6, msg=names[param])
