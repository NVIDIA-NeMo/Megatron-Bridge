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

"""CPU tests of the ERNIE 5.5 fused-op dispatch (``ernie_fused_ops``).

The Triton kernels need CUDA; ``test_ernie55_fused_ops_gpu.py`` and ``test_ernie55_fused_dispatch_gpu.py`` run
them. Here, on CPU:

* ``fused_ops`` imports without Triton, and only defining its kernels needs Triton;
* the flag reader, the kernel-support check and the Triton probe;
* the dispatchers in Megatron-Core: installed once, only with the flag on, around the right functions, and the
  SiTU-GLU scales they run the kernel for;
* the dispatch conditions of every kernel: CPU tensors run the eager code bitwise, and so does every input the
  kernel does not take; inputs it takes reach it (the kernel replaced by a recorder, the device check forced);
* the multimax head reads the flag when it is installed.

The model-level wiring (a built model's MLPs and output layer reach the kernels) is tested with the model
builders of ``test_ernie55_mhc_mtp.py``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from megatron.core import activations

from megatron.bridge.models.ernie.modeling_ernie55 import fused_ops, install_ernie55_fused_ops
from megatron.bridge.models.ernie.modeling_ernie55 import output_layer as output_layer_module
from megatron.bridge.models.ernie.modeling_ernie55.output_layer import ernie55_multimax, install_ernie55_multimax
from tests.unit_tests.models.ernie.ernie55_test_utils import CallRecorder, isolate_fused_dispatch


pytestmark = pytest.mark.unit

MLP_SITE = "megatron.core.transformer.mlp"
SITU_MODEL = dict(
    ernie_fused_ops=True, activation_func_tanh_clamp_scale=4.0, activation_func_tanh_clamp_scale_linear=25.0
)


@pytest.fixture
def dispatch(monkeypatch):
    """The ``fused_dispatch`` module with a clean dispatch state (restored after the test)."""
    return isolate_fused_dispatch(monkeypatch)


def _cp_utils():
    from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import CP_UTILS_MODULE

    return importlib.import_module(CP_UTILS_MODULE)


def _force_kernel_support(monkeypatch, dispatch) -> None:
    """Let every tensor pass the device, dtype and Triton check, as a CUDA tensor would."""
    monkeypatch.setattr(dispatch, "fused_kernels_support", lambda tensor: True)


# ---------------------------------------------------------------------------------------------
# Imports, the flag and the support checks
# ---------------------------------------------------------------------------------------------


def test_fused_ops_imports_without_triton():
    """``fused_ops`` imports only PyTorch: Triton is blocked, the module loads, and defining the kernels needs it."""
    code = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys


        class BlockTriton(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "triton" or name.startswith("triton."):
                    raise ImportError(f"{name} is blocked")
                return None


        sys.meta_path.insert(0, BlockTriton())
        spec = importlib.util.spec_from_file_location("ernie55_fused_ops_without_triton", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded = sorted(name for name in sys.modules if name == "triton" or name.startswith("triton."))
        assert not loaded, loaded
        assert module.triton is None and not module._KERNELS_LOADED
        try:
            module._load_kernels()
        except ImportError:
            pass
        else:
            raise AssertionError("defining the kernels must import Triton")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, fused_ops.__file__], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_flag_reader(dispatch):
    assert dispatch.ernie55_fused_ops_enabled(SimpleNamespace(ernie_fused_ops=True))
    assert not dispatch.ernie55_fused_ops_enabled(SimpleNamespace(ernie_fused_ops=False))
    # Configs of other models have no such field.
    assert not dispatch.ernie55_fused_ops_enabled(SimpleNamespace())


def test_kernel_support_check(dispatch, monkeypatch):
    def cuda(dtype):
        return SimpleNamespace(is_cuda=True, dtype=dtype)

    monkeypatch.setattr(dispatch, "triton_available", lambda: True)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        assert dispatch.fused_kernels_support(cuda(dtype)), dtype
    for dtype in (torch.float64, torch.int64, torch.bool):
        assert not dispatch.fused_kernels_support(cuda(dtype)), dtype
    assert not dispatch.fused_kernels_support(torch.zeros(2, dtype=torch.bfloat16))
    monkeypatch.setattr(dispatch, "triton_available", lambda: False)
    assert not dispatch.fused_kernels_support(cuda(torch.bfloat16))


def test_triton_probe_without_triton(dispatch):
    dispatch.triton_available.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(sys.modules, "triton", None)  # any import of triton now raises ImportError
            assert dispatch.triton_available() is False
    finally:
        dispatch.triton_available.cache_clear()


# ---------------------------------------------------------------------------------------------
# Installing the dispatchers
# ---------------------------------------------------------------------------------------------


def test_install_follows_the_flag_and_wraps_once(dispatch):
    sites = {name: importlib.import_module(name) for name in dispatch.SITU_GLU_SITES}
    cp_utils = _cp_utils()
    rope = cp_utils.apply_thd_cp_local_rope_unfused
    assert all(module.situ_glu is activations.situ_glu for module in sites.values())

    # ernie_fused_ops=False touches nothing.
    install_ernie55_fused_ops(SimpleNamespace(**{**SITU_MODEL, "ernie_fused_ops": False}))
    assert all(module.situ_glu is activations.situ_glu for module in sites.values())
    assert cp_utils.apply_thd_cp_local_rope_unfused is rope
    assert not dispatch._STATE.situ_glu_scales

    # A SwiGLU model: the RoPE dispatcher only.
    install_ernie55_fused_ops(SimpleNamespace(ernie_fused_ops=True, activation_func_tanh_clamp_scale=None))
    assert all(module.situ_glu is activations.situ_glu for module in sites.values())
    rope_dispatcher = cp_utils.apply_thd_cp_local_rope_unfused
    assert rope_dispatcher is not rope and rope_dispatcher.ernie55_unfused is rope
    assert not dispatch._STATE.situ_glu_scales

    # A SiTU-GLU model: every site, wrapped once around Megatron-Core's situ_glu, and the model's scales recorded.
    install_ernie55_fused_ops(SimpleNamespace(**SITU_MODEL))
    dispatchers = {name: module.situ_glu for name, module in sites.items()}
    for name, dispatcher in dispatchers.items():
        assert dispatcher.ernie55_site == name and dispatcher.ernie55_eager is activations.situ_glu
    install_ernie55_fused_ops(SimpleNamespace(**SITU_MODEL))
    assert {name: module.situ_glu for name, module in sites.items()} == dispatchers
    assert cp_utils.apply_thd_cp_local_rope_unfused is rope_dispatcher
    assert dispatch._STATE.situ_glu_scales == {(4.0, 25.0)}
    # A SiTU-GLU model without the tanh on the up half records (scale, None).
    install_ernie55_fused_ops(SimpleNamespace(ernie_fused_ops=True, activation_func_tanh_clamp_scale=2.0))
    assert dispatch._STATE.situ_glu_scales == {(4.0, 25.0), (2.0, None)}

    # Models built later with the flag off drop only their own scales: other scales and SwiGLU models change nothing.
    install_ernie55_fused_ops(SimpleNamespace(ernie_fused_ops=False, activation_func_tanh_clamp_scale=3.0))
    install_ernie55_fused_ops(SimpleNamespace(ernie_fused_ops=False))
    assert dispatch._STATE.situ_glu_scales == {(4.0, 25.0), (2.0, None)}
    install_ernie55_fused_ops(SimpleNamespace(**{**SITU_MODEL, "ernie_fused_ops": False}))
    assert dispatch._STATE.situ_glu_scales == {(2.0, None)}
    assert {name: module.situ_glu for name, module in sites.items()} == dispatchers


# ---------------------------------------------------------------------------------------------
# Dispatch conditions
# ---------------------------------------------------------------------------------------------


def test_situ_glu_dispatch_conditions(dispatch, monkeypatch):
    install_ernie55_fused_ops(SimpleNamespace(**SITU_MODEL))
    dispatcher = importlib.import_module(MLP_SITE).situ_glu
    x = torch.randn(6, 2, 16, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    expected = activations.situ_glu(x, 4.0, 25.0, 0.0)

    # CPU tensors run Megatron-Core's situ_glu.
    assert torch.equal(dispatcher(x, 4.0, 25.0), expected)
    assert not dispatch.FUSED_OP_CALLS

    kernel = CallRecorder(lambda x, beta, linear_beta: torch.full((*x.shape[:-1], x.shape[-1] // 2), 7.0))
    monkeypatch.setattr(fused_ops, "fused_situ_glu", kernel)
    _force_kernel_support(monkeypatch, dispatch)
    assert torch.equal(dispatcher(x, 4.0, 25.0), torch.full((6, 2, 8), 7.0))
    # A second SiTU-GLU model, with a plain up half: the scales of both models reach the kernel.
    install_ernie55_fused_ops(SimpleNamespace(ernie_fused_ops=True, activation_func_tanh_clamp_scale=4.0))
    dispatcher(x, 4.0)
    assert [(args[0] is x, args[1:]) for args, _ in kernel.calls] == [(True, (4.0, 25.0)), (True, (4.0, None))]
    assert dispatch.FUSED_OP_CALLS == {f"situ_glu@{MLP_SITE}": 2}

    # Inputs the kernel does not take run Megatron-Core's situ_glu, and so do scales that no model built with the
    # flag on recorded (another model's SiTU-GLU in the same process).
    transposed = x.transpose(0, 1)
    assert torch.equal(dispatcher(transposed, 4.0, 25.0), activations.situ_glu(transposed, 4.0, 25.0, 0.0))
    assert torch.equal(dispatcher(x, 4.0, 25.0, 0.5), activations.situ_glu(x, 4.0, 25.0, 0.5))
    for gate_scale, linear_scale in ((2.0, 25.0), (4.0, 5.0), (2.0, None)):
        assert torch.equal(
            dispatcher(x, gate_scale, linear_scale), activations.situ_glu(x, gate_scale, linear_scale, 0.0)
        )
    assert len(kernel.calls) == 2

    # A model built later with the first model's scales and the flag off: those calls run Megatron-Core's situ_glu
    # again, and the second model's keep the kernel.
    install_ernie55_fused_ops(SimpleNamespace(**{**SITU_MODEL, "ernie_fused_ops": False}))
    assert torch.equal(dispatcher(x, 4.0, 25.0), expected)
    assert len(kernel.calls) == 2
    dispatcher(x, 4.0)
    assert len(kernel.calls) == 3 and kernel.calls[-1][0][1:] == (4.0, None)


def test_thd_cp_local_rope_dispatch_conditions(dispatch, monkeypatch):
    cp_utils = _cp_utils()
    unfused = CallRecorder(lambda x, *args, **kwargs: x + 1)  # stands in for Megatron-Core's function
    monkeypatch.setattr(cp_utils, "apply_thd_cp_local_rope_unfused", unfused)
    install_ernie55_fused_ops(SimpleNamespace(ernie_fused_ops=True))
    rope = cp_utils.apply_thd_cp_local_rope_unfused
    assert rope.ernie55_unfused is unfused

    x, table, cu_seqlens = torch.zeros(10, 2, 12), torch.zeros(32, 1, 1, 4), torch.tensor([0, 7, 32])
    ernie = SimpleNamespace(ernie_fused_ops=True, rotary_interleaved=False)

    def unfused_call(index: int) -> tuple:
        args, kwargs = unfused.calls[index]
        assert args[0] is x and args[4] is cu_seqlens and not kwargs
        return args[2:4] + args[5:]

    # CPU tensors run Megatron-Core's function, with every argument.
    rope(x, table, 8, 4, cu_seqlens, 5, ernie)
    assert unfused.calls[-1][0][1] is table and unfused_call(-1) == (8, 4, 5, ernie, False)

    kernel = CallRecorder(lambda x, *args: x - 1)
    monkeypatch.setattr(fused_ops, "fused_thd_cp_local_rope", kernel)
    _force_kernel_support(monkeypatch, dispatch)
    assert torch.equal(rope(x, table, 8, 4, cu_seqlens, 5, ernie), x - 1)
    rope(x, table, 8, 4, cu_seqlens, 5, ernie, inverse=True)
    for (args, kwargs), inverse in zip(kernel.calls, (False, True)):
        assert args[0] is x and args[1] is table and args[4] is cu_seqlens and not kwargs
        assert args[2:4] + args[5:] == (8, 4, 5, inverse)
    assert dispatch.FUSED_OP_CALLS == {"thd_cp_local_rope@forward": 1, "thd_cp_local_rope@inverse": 1}

    # What the reference does not fuse keeps Megatron-Core's function: other models' configs, the flag off,
    # interleaved RoPE, and an angle table wider than the rotary channels.
    for config, angles in (
        (SimpleNamespace(rotary_interleaved=False), table),
        (SimpleNamespace(ernie_fused_ops=False, rotary_interleaved=False), table),
        (SimpleNamespace(ernie_fused_ops=True, rotary_interleaved=True), table),
        (ernie, torch.zeros(32, 1, 1, 8)),
    ):
        assert torch.equal(rope(x, angles, 8, 4, cu_seqlens, 5, config, inverse=True), x + 1)
        assert unfused_call(-1) == (8, 4, 5, config, True)
    assert len(kernel.calls) == 2 and len(unfused.calls) == 5

    # The kernel's other row layouts, [t, 1, heads, dim] and [t, dim], reach it too.
    for rows in (torch.zeros(10, 1, 2, 12), torch.zeros(10, 12)):
        assert torch.equal(rope(rows, table, 8, 4, cu_seqlens, 5, ernie), rows - 1)
        assert kernel.calls[-1][0][0] is rows
    # Layouts the kernel does not take keep Megatron-Core's function: a batch dimension other than 1, other ranks,
    # rows wider than nope_dim + pos_dim, an odd number of rotary channels, and an angle table with more than one
    # row of angles per position.
    for rows, angles, nope_dim, pos_dim in (
        (torch.zeros(10, 2, 2, 12), table, 8, 4),
        (torch.zeros(10, 1, 2, 2, 12), table, 8, 4),
        (torch.zeros(12), table, 8, 4),
        (torch.zeros(10, 2, 14), table, 8, 4),
        (torch.zeros(10, 2, 12), torch.zeros(32, 1, 1, 3), 9, 3),
        (torch.zeros(10, 2, 12), torch.zeros(32, 2, 1, 4), 8, 4),
    ):
        assert torch.equal(rope(rows, angles, nope_dim, pos_dim, cu_seqlens, 5, ernie), rows + 1)
        args, _ = unfused.calls[-1]
        assert args[0] is rows and args[1] is angles and args[2:4] == (nope_dim, pos_dim)
    assert len(kernel.calls) == 4 and len(unfused.calls) == 11
    assert dispatch.FUSED_OP_CALLS == {"thd_cp_local_rope@forward": 3, "thd_cp_local_rope@inverse": 1}


def test_thd_cp_rope_layout_fits(dispatch):
    """The layout check matches what the kernel launcher unpacks: the H layers' q, kv and output rows fit."""
    fits = dispatch.thd_cp_rope_layout_fits
    table = torch.zeros(4096, 1, 1, 64)
    # The H layers' q [t, heads, 512], kv [t, 1, 512] and core-attention output [t, heads, 512], nope 448 + rope 64.
    for shape in ((16, 8, 512), (16, 1, 512), (16, 1, 8, 512), (16, 512)):
        assert fits(torch.zeros(shape), table, 448, 64), shape
    assert fits(torch.zeros(16, 8, 512), torch.zeros(4096, 64), 448, 64)
    for shape, angles, nope_dim, pos_dim in (
        ((16, 2, 8, 512), table, 448, 64),  # a batch dimension
        ((16, 8, 448), table, 448, 64),  # rows narrower than nope_dim + pos_dim
        ((16, 8, 512), torch.zeros(4096, 1, 1, 32), 448, 64),  # a table narrower than pos_dim
        ((16, 8, 512), torch.zeros(4096, 1, 1, 63), 449, 63),  # odd rotary channels
        ((16, 8, 512), torch.zeros(4096, 1, 1, 0), 512, 0),  # no rotary channels
        ((16, 8, 512), torch.zeros(64), 448, 64),  # no position axis
    ):
        assert not fits(torch.zeros(shape), angles, nope_dim, pos_dim), (shape, tuple(angles.shape))


def test_rope_on_cpu_is_megatron_cores(dispatch):
    cp_utils = _cp_utils()
    original = cp_utils.apply_thd_cp_local_rope_unfused
    install_ernie55_fused_ops(SimpleNamespace(ernie_fused_ops=True))
    # A plain RoPE angle table [positions, 1, 1, pos_dim], as Megatron-Core's RotaryEmbedding builds it.
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
    angles = torch.outer(torch.arange(32, dtype=torch.float32), inv_freq).repeat(1, 2)[:, None, None, :]
    cu_seqlens = torch.tensor([0, 7, 32], dtype=torch.int32)
    config = SimpleNamespace(ernie_fused_ops=True, rotary_interleaved=False)
    for shape in ((10, 2, 12), (10, 1, 2, 12), (10, 12)):
        x = torch.randn(shape, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
        for inverse in (False, True):
            actual = cp_utils.apply_thd_cp_local_rope_unfused(x, angles, 8, 4, cu_seqlens, 5, config, inverse=inverse)
            assert torch.equal(actual, original(x, angles, 8, 4, cu_seqlens, 5, config, inverse=inverse))
    assert not dispatch.FUSED_OP_CALLS


def _multimax_inputs():
    generator = torch.Generator().manual_seed(2)
    logits = (4 * torch.randn(5, 2, 33, generator=generator)).to(torch.bfloat16)
    ranges = torch.tensor([-0.5, 0.75, -1.25, 1.5], dtype=torch.bfloat16)
    scales = torch.tensor([0.25, -0.5, 0.125, -0.0625], dtype=torch.bfloat16)
    return logits, ranges, scales


def test_multimax_dispatch_conditions(dispatch, monkeypatch):
    logits, ranges, scales = _multimax_inputs()
    eager = ernie55_multimax(logits, ranges, scales)
    # CPU logits keep the chunked path, flag or not.
    assert torch.equal(ernie55_multimax(logits, ranges, scales, fused=True), eager)
    assert not dispatch.FUSED_OP_CALLS

    kernel = CallRecorder(lambda logits, ranges, scales: 2 * logits)
    monkeypatch.setattr(fused_ops, "fused_multimax", kernel)
    _force_kernel_support(monkeypatch, dispatch)
    assert torch.equal(ernie55_multimax(logits, ranges, scales, fused=False), eager) and not kernel.calls
    transposed = logits.transpose(0, 1)
    assert torch.equal(ernie55_multimax(transposed, ranges, scales, fused=True), 2 * transposed)
    ((args, kwargs),) = kernel.calls
    # The kernel gets contiguous logits and the parameters themselves.
    assert args[0].is_contiguous() and torch.equal(args[0], transposed) and not kwargs
    assert args[1] is ranges and args[2] is scales
    assert dispatch.FUSED_OP_CALLS == {"multimax@output_layer": 1}
    with pytest.raises(ValueError, match="ranges and scales"):
        ernie55_multimax(logits, ranges[:3], scales, fused=True)


class _OutputLayer(torch.nn.Module):
    """A Megatron-Core-like output layer: ``forward`` returns ``(logits, bias)``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(11, 8, generator=torch.Generator().manual_seed(3)))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden @ self.weight.t(), None


@pytest.mark.parametrize("flag", [True, False, None], ids=["on", "off", "no-field"])
def test_multimax_head_reads_the_flag_at_install(dispatch, monkeypatch, flag):
    config = SimpleNamespace(tensor_model_parallel_size=1, params_dtype=torch.float32, use_cpu_initialization=True)
    if flag is not None:
        config.ernie_fused_ops = flag
    layer = _OutputLayer()
    install_ernie55_multimax(layer, config)
    assert layer.ernie55_fused_multimax is bool(flag)
    seen = []

    def multimax(logits, ranges, scales, *, fused):
        seen.append(fused)
        return logits

    monkeypatch.setattr(output_layer_module, "ernie55_multimax", multimax)
    layer(torch.randn(3, 8))
    assert seen == [bool(flag)]

    # Installing again keeps the parameters and takes the config's current flag.
    config.ernie_fused_ops = not flag
    install_ernie55_multimax(layer, config)
    assert layer.ernie55_fused_multimax is (not flag)
    assert [name for name, _ in layer.named_parameters()] == ["weight", "multimax_ranges", "multimax_ts"]
