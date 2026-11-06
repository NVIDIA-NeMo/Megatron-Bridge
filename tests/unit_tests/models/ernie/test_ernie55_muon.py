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

"""CPU tests of the reference implementation's Muon recipe on the toy ERNIE 5.5 model.

The expected row blocks are written from the reference's ``split_head`` and ``ffn_split`` rules and the
toy model's dimensions (``test_ernie55_mappings``), not from the Bridge implementation. The blocks reach
Megatron-Core's Muon as per-parameter ``qkv_layout``s; the tests read them back through Megatron-Core's own
split helpers, so they check what ``TensorParallelMuon`` orthogonalizes, not only what the model stores.
"""

import re

import pytest
import torch
from megatron.core.optimizer import emerging_optimizers
from megatron.core.transformer import attention as mcore_attention


# Needs Megatron-Core dev (QKVLayout, emerging_optimizers._localize_qkv_split_shapes): ERNIE 5.5 is developed
# against Megatron-Core dev, and CI pins Megatron-Core main, so this test is skipped there for now.
if not hasattr(mcore_attention, "QKVLayout") or not hasattr(emerging_optimizers, "_localize_qkv_split_shapes"):
    pytest.skip(
        "needs Megatron-Core dev's per-parameter QKVLayout and _localize_qkv_split_shapes", allow_module_level=True
    )

from megatron.core.optimizer.emerging_optimizers import (  # noqa: E402
    _get_qkv_split_shapes,
    _is_muon_excluded,
    _localize_qkv_split_shapes,
)

from megatron.bridge.models.ernie.modeling_ernie55 import tag_ernie55_muon_params
from tests.unit_tests.models.ernie import test_ernie55_mappings as toy
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import _build_model, pg_collection  # noqa: F401


_H, _D, _INDEXER = toy._H, toy._D, toy._INDEXER

# Layer symbol -> {parameter name suffix: the reference's Muon row blocks}.
_EXPECTED_BLOCKS = {
    # H layers: q per head; the output projection and its gate per output group.
    "H": {
        r"self_attention\.linear_q_up_proj\.weight": [_H["head_dim"]] * _H["heads"],
        r"self_attention\.linear_o_group_proj": [_H["o_lora"]] * _H["groups"],
        r"self_attention\.linear_output_gate\.weight": [_H["o_lora"]] * _H["groups"],
    },
    # D layers: q per head with its no-RoPE and RoPE rows apart, the KV latent and the RoPE key apart,
    # the absorbed K/V up projections and the output gate per head, the indexer query per indexer head.
    "D": {
        r"self_attention\.linear_q_up_proj\.weight": [_D["nope"], _D["rope"]] * _D["heads"],
        r"self_attention\.linear_kv_down_proj\.weight": [_D["kv_lora"], _D["rope"]],
        r"self_attention\.k_b_proj": [_D["kv_lora"]] * _D["heads"],
        r"self_attention\.v_b_proj": [_D["v"]] * _D["heads"],
        r"core_attention\.linear_output_gate\.weight": [_D["v"]] * _D["heads"],
        r"core_attention\.indexer\.linear_wq_b\.weight": [_INDEXER["head_dim"]] * _INDEXER["heads"],
    },
    # Every gated fc1: gate and up halves.
    "-": {r"mlp\.linear_fc1\.weight": [toy._DENSE_FFN] * 2},
    "E": {
        r"mlp\.experts\.local_experts\.\d+\.linear_fc1\.weight": [toy._MOE_FFN] * 2,
        r"mlp\.shared_experts\.linear_fc1\.weight": [toy._MOE_FFN] * 2,
    },
}


def _model(pg_collection, **overrides):
    return _build_model(pg_collection, pattern=toy._PATTERN, seed=0, **toy._megatron_overrides(**overrides))


def _expected_blocks(name: str):
    """The blocks of a decoder parameter, from its layer's symbol in the hybrid pattern."""
    match = re.match(r"decoder\.layers\.(\d+)\.", name)
    if match is None:
        return None
    kind = toy._PATTERN.split("/")[0][int(match.group(1))]
    for pattern, blocks in _EXPECTED_BLOCKS[kind].items():
        if re.search(rf"\.{pattern}$", name):
            return blocks
    return None


def _split_sizes(model) -> dict[str, list[int]]:
    """The row blocks Megatron-Core's per-head split-QKV Muon uses for every non-MLA ``qkv_layout``."""
    return {
        name: _get_qkv_split_shapes(param.qkv_layout, split_qkv_per_head=True)
        for name, param in model.named_parameters()
        if getattr(param, "qkv_layout", None) is not None and not param.qkv_layout.is_mla
    }


def _muon_split_parameters(model, *, split_mla: bool) -> dict[str, list[int]]:
    """The parameters Megatron-Core's emerging-optimizer setup marks ``is_qkv``, with their split.

    Megatron-Core's selection (``megatron/core/optimizer/__init__.py``, the tagging loop of
    ``_get_megatron_emerging_optimizer``) at TP 1: a trainable 2-D parameter with a ``qkv_layout``, or
    named ``linear_qkv.weight``; an MLA layout only under ``muon_split_mla_qkv``; the split is the
    layout's split, kept when it covers the rows. Our one-group layouts split the same way with and
    without ``muon_split_qkv_per_head``. ``test_megatron_core_optimizer_setup_splits_the_reference_blocks``
    runs the real loop.
    """
    selected = {}
    for name, param in model.named_parameters():
        layout = getattr(param, "qkv_layout", None)
        if not param.requires_grad or (layout is not None and layout.is_mla and not split_mla):
            continue
        if (layout is not None or "linear_qkv.weight" in name) and param.dim() == 2:
            assert layout is not None, f"{name}: a fused QKV weight without a layout"
            shapes = _get_qkv_split_shapes(layout, split_qkv_per_head=True)
            if sum(shapes) == param.shape[0]:
                selected[name] = shapes
    return selected


def _clear_ernie55_layouts(model) -> None:
    for param in model.parameters():
        layout = getattr(param, "qkv_layout", None)
        if layout is not None and not layout.is_mla:
            del param.qkv_layout


def test_muon_row_blocks_follow_the_reference(pg_collection):
    """Exactly the reference's split weights carry a non-MLA ``qkv_layout``, whose split is its blocks."""
    model = _model(pg_collection)
    expected = {
        name: blocks
        for name, _ in model.named_parameters()
        if name.startswith("decoder.") and (blocks := _expected_blocks(name)) is not None
    }
    # Every listed weight exists: 3 per H layer, 6 per D layer, 1 dense fc1, and each MoE layer's experts and
    # shared expert.
    layers = toy._PATTERN.split("/")[0]
    assert len(expected) == 3 * layers.count("H") + 6 * layers.count("D") + layers.count("-") + (
        toy._EXPERTS + 1
    ) * layers.count("E")
    assert _split_sizes(model) == expected
    for name, blocks in expected.items():
        param = model.get_parameter(name)
        layout = param.qkv_layout
        assert layout.num_groups == 1 and not layout.is_mla, name
        # The recipes' projection split (muon_split_qkv_per_head off) is the same blocks.
        assert _get_qkv_split_shapes(layout, split_qkv_per_head=False) == blocks, name
        assert sum(blocks) == param.shape[0], name
        # At TP 1 Megatron-Core keeps the split local and complete, so Muon splits this rank's rows directly.
        assert _localize_qkv_split_shapes(blocks, local_start=0, local_rows=param.shape[0]) == (blocks, True), name


def test_megatron_core_splits_exactly_the_reference_weights(pg_collection):
    """Under the recipe's flags Megatron-Core's Muon splits exactly the tagged weights, and nothing else.

    Megatron-Core's own layouts on other weights are MLA layouts, which stay whole matrices with
    ``muon_split_mla_qkv`` off; no ERNIE 5.5 weight is a fused ``linear_qkv``.
    """
    model = _model(pg_collection)
    assert not [name for name, _ in model.named_parameters() if "linear_qkv.weight" in name]
    assert _muon_split_parameters(model, split_mla=False) == _split_sizes(model)
    for name, param in model.named_parameters():
        layout = getattr(param, "qkv_layout", None)
        if layout is not None and name not in _split_sizes(model):
            assert layout.is_mla, name


def test_compressor_ape_goes_to_the_scalar_optimizer(pg_collection):
    """The compressors' APE (the reference's ``"ape"`` exclusion) is the only matrix tagged out of Muon."""
    model = _model(pg_collection)
    apes = [name for name, _ in model.named_parameters() if name.endswith("compressor.ape")]
    assert apes
    excluded = {
        name
        for name, param in model.named_parameters()
        if param.dim() == 2 and getattr(param, "use_muon", True) is False
    }
    assert excluded == set(apes)
    for name in apes:
        assert _is_muon_excluded(model.get_parameter(name)), name


def test_router_matrices_go_to_the_scalar_optimizer_under_moe_router_skip_muon(pg_collection):
    """With ``moe_router_skip_muon`` (the bridge's setting) both gate matrices leave Muon."""
    model = _model(pg_collection, moe_router_skip_muon=True)
    gates = [name for name, _ in model.named_parameters() if re.search(r"mlp\.router\.weight(_1)?$", name)]
    # The SD-MTP layer shares the last decoder layer's router.
    assert len(gates) == 2 * toy._PATTERN.split("/")[0].count("E")
    for name in gates:
        assert _is_muon_excluded(model.get_parameter(name)), name


def test_sharded_rows_keep_whole_matrices(pg_collection):
    """Under tensor parallelism the row blocks would not describe local rows, so none is set."""
    model = _model(pg_collection)
    _clear_ernie55_layouts(model)
    model.config.tensor_model_parallel_size = 2
    tag_ernie55_muon_params(model)
    assert _split_sizes(model) == {}


def test_glu_interleaved_experts_keep_whole_fc1(pg_collection):
    """Routed experts with GLU-interleaved fc1 rows are not split; the dense and shared fc1 still are."""
    model = _model(pg_collection)
    _clear_ernie55_layouts(model)
    model.config.moe_mlp_glu_interleave_size = 4
    tag_ernie55_muon_params(model)
    fc1 = {name for name in _split_sizes(model) if name.endswith(".weight") and "linear_fc1" in name}
    assert fc1
    assert not any(".experts.local_experts." in name for name in fc1)
    assert any(".shared_experts." in name for name in fc1)
    assert any(re.search(r"\.mlp\.linear_fc1\.weight$", name) for name in fc1)


# ---------------------------------------------------------------------------------------------
# Megatron-Core's per-head split Newton-Schulz
# ---------------------------------------------------------------------------------------------


def _polar(grad: torch.Tensor) -> torch.Tensor:
    """A shape-dependent stand-in for Muon's scaled orthogonalization; batched over leading dims."""
    u, _, vh = torch.linalg.svd(grad, full_matrices=False)
    return (u @ vh) * max(1.0, grad.shape[-2] / grad.shape[-1]) ** 0.5


class _PerHead:
    split_qkv_per_head = True


class _Projection:
    split_qkv_per_head = False


# Blocks on a 7-column gradient. Wide blocks (rows < 7) and tall ones (rows > 7, Newton-Schulz on the transpose):
# the recipe's k_b_proj blocks are tall ([kv_lora_rank, qk_head_dim]) and take the batched path.
_BLOCK_CASES = [
    pytest.param([6, 2] * 3, id="mixed"),  # a D-layer q: no-RoPE and RoPE rows of every head
    pytest.param([5] * 4, id="uniform"),  # per head: the batched 3-D Newton-Schulz path
    pytest.param([9, 2] * 2, id="mixed-tall"),  # one tall and one wide block per head
    pytest.param([9] * 3, id="uniform-tall"),  # the batched path on tall blocks
]


@pytest.mark.parametrize(
    "muon", [pytest.param(_Projection(), id="projection"), pytest.param(_PerHead(), id="per_head")]
)
@pytest.mark.parametrize("blocks", _BLOCK_CASES)
def test_split_newton_schulz_orthogonalizes_every_block_alone(blocks, muon):
    """``TensorParallelMuon._orthogonalize_split_qkv`` equals a loop over the blocks with the same function."""
    grad = torch.randn(sum(blocks), 7, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    actual = emerging_optimizers.TensorParallelMuon._orthogonalize_split_qkv(muon, grad, blocks, _polar)
    expected = torch.cat([_polar(block) for block in torch.split(grad, blocks, dim=0)], dim=0)
    torch.testing.assert_close(actual, expected)
    assert (actual - _polar(grad)).norm() > 0.1 * actual.norm()


def _tag_like_megatron_core(param, blocks, *, per_head: bool) -> None:
    """The attributes Megatron-Core's setup gives a TP-1 weight with a one-group layout of ``blocks``."""
    param.is_qkv = True
    param.qkv_split_shapes_global = blocks
    if per_head:
        param.qkv_split_shapes, param.qkv_split_heads_are_complete = _localize_qkv_split_shapes(
            blocks, local_start=0, local_rows=sum(blocks)
        )
    else:
        param.qkv_split_shapes = blocks
        param.qkv_split_groups_are_complete = True


# Megatron-Core's per-head path reshapes the batched Newton-Schulz of equal blocks with ``view_as``; for tall
# blocks Newton-Schulz returns its result transposed (non-contiguous), so the view fails. The ERNIE 5.5 recipes
# therefore keep muon_split_qkv_per_head off. Strict: this flips once Megatron-Core reshapes instead.
_PER_HEAD_TALL_BUG = pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="Megatron-Core's batched per-head Newton-Schulz cannot view a tall block's transposed result",
)


@pytest.mark.skipif(not emerging_optimizers.HAVE_EMERGING_OPTIMIZERS, reason="needs the emerging-optimizers package")
@pytest.mark.parametrize(
    "per_head",
    [pytest.param(False, id="projection"), pytest.param(True, id="per_head")],
)
@pytest.mark.parametrize("blocks", _BLOCK_CASES)
def test_muon_orthogonalize_splits_a_tagged_weight(request, blocks, per_head):
    """``TensorParallelMuon.orthogonalize`` on a weight tagged the way the optimizer setup tags our layouts.

    Each block is orthogonalized with Muon's scale of its own shape, the reference's ``split_head`` update,
    both without ``muon_split_qkv_per_head`` (the recipes) and with it.
    """
    if per_head and len(set(blocks)) == 1 and blocks[0] > 7:
        request.applymarker(_PER_HEAD_TALL_BUG)
    param = torch.nn.Parameter(torch.zeros(sum(blocks), 7))
    _tag_like_megatron_core(param, blocks, per_head=per_head)
    muon = emerging_optimizers.TensorParallelMuon(
        [param],
        split_qkv=True,
        split_qkv_per_head=per_head,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=None,
        fp32_matmul_prec="highest",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=0.2,
        pg_collection=None,
        tp_mode="blockwise",
    )
    grad = torch.randn(sum(blocks), 7, generator=torch.Generator().manual_seed(11))
    actual = muon.orthogonalize(param, grad)
    expected = torch.cat(
        [muon.scaled_orthogonalize_fn(block, tp_group=None) for block in torch.split(grad, blocks, dim=0)], dim=0
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    whole = muon.scaled_orthogonalize_fn(grad, tp_group=None)
    assert (actual - whole).norm() > 0.1 * whole.norm()


# ---------------------------------------------------------------------------------------------
# Megatron-Core's own optimizer setup on the toy model
# ---------------------------------------------------------------------------------------------


def _find_muon(optimizer):
    """The ``TensorParallelMuon`` inside the wrappers ``_get_megatron_emerging_optimizer`` returns."""
    pending = [optimizer]
    while pending:
        current = pending.pop()
        if isinstance(current, emerging_optimizers.TensorParallelMuon):
            return current
        if hasattr(current, "chained_optimizers"):
            # ChainedOptimizer.optimizer asserts a single child; walk the children instead.
            pending.extend(current.chained_optimizers)
        elif getattr(current, "optimizer", None) is not None:
            pending.append(current.optimizer)
    raise AssertionError(f"no TensorParallelMuon in {type(optimizer).__name__}")


@pytest.mark.skipif(not emerging_optimizers.HAVE_EMERGING_OPTIMIZERS, reason="needs the emerging-optimizers package")
def test_megatron_core_optimizer_setup_splits_the_reference_blocks(pg_collection, monkeypatch):
    """Megatron-Core's real optimizer setup, with the recipes' flags, tags exactly our weights and splits them.

    ``_get_megatron_emerging_optimizer`` runs its own tagging loop (``optimizer/__init__.py``), which the
    layer-wise distributed optimizer shares; the non-layer-wise wrapper keeps this CPU test free of DDP
    buffers. Then one ``TensorParallelMuon.step`` from zero momentum (no weight decay) moves every Muon
    weight by ``-lr`` times its orthogonalized effective gradient: block by block for our weights (with
    each block's own scale), as a whole matrix for every other weight.

    CPU stand-ins replace only what needs a GPU and plays no part in the split: ``FP32Optimizer``'s
    loss-scale tensor (allocated on CUDA) and the fused Adam of the scalar groups (Megatron-Core's own
    fallback when neither Transformer Engine nor Apex is installed, ``USING_PYTORCH_OPTIMIZER``).
    """
    import megatron.core.optimizer as mcore_optimizer
    from megatron.core.optimizer.optimizer import FP32Optimizer, MegatronOptimizer

    class _CPUFP32Optimizer(FP32Optimizer):
        def __init__(self, optimizer, config, init_state_fn):
            MegatronOptimizer.__init__(self, optimizer, config, init_state_fn)
            self._scale = torch.tensor([1.0])
            self.is_stub_optimizer = optimizer is None

    monkeypatch.setattr(mcore_optimizer, "FP32Optimizer", _CPUFP32Optimizer)
    monkeypatch.setattr(mcore_optimizer, "USING_PYTORCH_OPTIMIZER", True)
    from megatron.core.optimizer import (
        OptimizerConfig,
        _get_megatron_emerging_optimizer,
        get_standard_config_overrides,
    )

    # Muon steps the fp32 main parameters (under bf16 Megatron-Core's wrapper copies these attributes onto them).
    model = _model(pg_collection).float()
    expected = _split_sizes(model)
    assert expected
    config = OptimizerConfig(
        optimizer="muon",
        lr=1e-2,
        weight_decay=0.0,
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
    # As get_megatron_optimizer calls it.
    optimizer = _get_megatron_emerging_optimizer(
        config, [model], config_overrides=get_standard_config_overrides(config), pg_collection=pg_collection
    )

    tagged = {
        name: param.qkv_split_shapes_global
        for name, param in model.named_parameters()
        if getattr(param, "is_qkv", False)
    }
    assert tagged == expected
    for name in expected:
        param = model.get_parameter(name)
        assert param.qkv_split_shapes == expected[name] and param.qkv_split_groups_are_complete, name

    muon = _find_muon(optimizer)
    names = {param: name for name, param in model.named_parameters()}
    muon_params = [param for group in muon.param_groups for param in group["params"]]
    assert {names[param] for param in muon_params} >= set(expected)
    generator = torch.Generator().manual_seed(5)
    before = {}
    for param in muon_params:
        param.grad = torch.randn(param.shape, generator=generator, dtype=param.dtype)
        before[param] = param.detach().clone()
    muon.step()

    momentum = muon.param_groups[0]["momentum"]
    lr = muon.param_groups[0]["lr"]
    for param in muon_params:
        name = names[param]
        buffer = torch.zeros_like(param.grad).lerp_(param.grad, 1 - momentum)
        effective = param.grad.lerp(buffer, momentum) if muon.nesterov else buffer
        if name in expected:
            update = torch.cat(
                [
                    muon.scaled_orthogonalize_fn(block, tp_group=None)
                    for block in torch.split(effective, expected[name], dim=0)
                ],
                dim=0,
            )
        else:
            update = muon.scaled_orthogonalize_fn(effective, tp_group=None)
        torch.testing.assert_close(param.detach(), before[param] - lr * update, rtol=1e-5, atol=1e-6, msg=name)
        param.grad = None
