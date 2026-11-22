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

"""Tests of the ERNIE 5.5 FP8 checkpoint import that build no model (``ernie55_mappings``).

``conftest.py`` leaves ``test_ernie55_*.py`` out of the default collection, because building an ERNIE 5.5 model needs
Megatron-Core dev and CI runs Megatron-Core main. The cases here build no model: they read a config, a list of tensor
names or a state dict. They live outside that pattern so that CI runs them, and can go back into
``test_ernie55_mappings.py`` once the exclusion is lifted. They cover:

* the block size a config's ``quantization_config`` declares, and the quantizations that are refused;
* the checkpoint key check, which accepts the ``*_scale_inv`` companions of an FP8 checkpoint;
* the dequantization of one tensor against the definition of the format, and the scales it refuses;
* the import hook of the registered bridge, for one tensor and for a group of tensors;
* what the registered bridge declares about the scales: an import task requests them with its weight, and a
  strict export does not expect them back.

The cases that load a toy model from an FP8 checkpoint are ``TestFP8Checkpoints`` in ``test_ernie55_mappings.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.models.ernie.ernie55_test_utils import (
    FP8_QUANTIZATION,
    allow_conversion_imports_without_te,
    fp8_quantize,
    fp8_reference,
)


# Bridge's conversion modules import its LoRA layers, which need Transformer Engine at import time.
allow_conversion_imports_without_te()

from megatron.bridge.models.ernie import Ernie55Bridge  # noqa: E402
from megatron.bridge.models.ernie import ernie55_mappings as mappings  # noqa: E402
from megatron.bridge.models.ernie.ernie55_mappings import check_ernie55_checkpoint_keys  # noqa: E402


pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parent / "fixtures"
_PROXY_CONFIG = json.loads((_FIXTURES / "ernie55_proxy_config.json").read_text())
_PROXY_DTYPES = json.loads((_FIXTURES / "ernie55_proxy_checkpoint_dtypes.json").read_text())["tensors"]
_FP8 = torch.float8_e4m3fn
_BF16, _FP32 = torch.bfloat16, torch.float32
_WEIGHT = "model.layers.0.mlp.down_proj.weight"


def _bridge(**hf_config) -> Ernie55Bridge:
    """The registered ERNIE 5.5 bridge with ``hf_config`` as its Hugging Face config; no model is built."""
    bridge = Ernie55Bridge()
    bridge.hf_config = SimpleNamespace(**hf_config)  # Bridge sets a config object, not a dict
    return bridge


def _quantization(block_size: int) -> dict:
    return dict(FP8_QUANTIZATION, weight_block_size=[block_size, block_size])


def _fp8_checkpoint(names: list[str], block_size: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A state dict with ``names`` as 20 x 12 FP8 weights next to their inverse scales, and the F32 sources.

    Every 8 rows are 4 x, every 8 columns 64 x and every tensor 4096 x the ones before. At block size 8 no two
    blocks, of one tensor or of two, then share an inverse scale, so a scale applied to the wrong block or taken
    from another tensor changes the result.
    """
    generator = torch.Generator().manual_seed(3)
    growth = torch.exp2(2.0 * (torch.arange(20) // 8))[:, None] * torch.exp2(6.0 * (torch.arange(12) // 8))[None, :]
    source = {
        name: torch.randn(20, 12, generator=generator) * growth * 4096.0**index for index, name in enumerate(names)
    }
    tensors = fp8_quantize(source, names, block_size)
    assert all(tensors[name].dtype == _FP8 for name in names)
    if block_size == 8:
        scales = torch.cat([tensors[f"{name}_scale_inv"].flatten() for name in names])
        assert scales.unique().numel() == scales.numel()
    return tensors, source


class TestBlockSize:
    def test_block_size(self):
        assert mappings.ernie55_fp8_block_size(_PROXY_CONFIG) is None
        assert mappings.ernie55_fp8_block_size(dict(_PROXY_CONFIG, quantization_config=FP8_QUANTIZATION)) == 128
        config = SimpleNamespace(quantization_config={"quant_method": "fp8", "weight_block_size": [64, 64]})
        assert mappings.ernie55_fp8_block_size(config) == 64
        # transformers may hand the quantization config over as an object: with to_dict(), or with attributes only
        as_object = SimpleNamespace(quant_method="fp8", weight_block_size=[16, 16])
        assert mappings.ernie55_fp8_block_size(SimpleNamespace(quantization_config=as_object)) == 16
        with_to_dict = SimpleNamespace(to_dict=lambda: {"quant_method": "fp8", "weight_block_size": [32, 32]})
        assert mappings.ernie55_fp8_block_size(SimpleNamespace(quantization_config=with_to_dict)) == 32
        for quantization in (
            {"quant_method": "awq"},
            {"quant_method": "fp8"},
            {"quant_method": "fp8", "weight_block_size": [128]},
            {"quant_method": "fp8", "weight_block_size": [8, 8, 8]},
            {"quant_method": "fp8", "weight_block_size": [128, 64]},
            {"quant_method": "fp8", "weight_block_size": [0, 0]},
            {"quant_method": "fp8", "weight_block_size": [1.5, 1.5]},
            {"quant_method": "fp8", "weight_block_size": [True, True]},
            {"quant_method": "fp8", "weight_block_size": [None, None]},
            {"quant_method": "fp8", "weight_block_size": "128"},
        ):
            with pytest.raises(NotImplementedError, match="only blockwise FP8"):
                mappings.ernie55_fp8_block_size(dict(_PROXY_CONFIG, quantization_config=quantization))


class TestCheckpointKeys:
    def test_checkpoint_key_check_accepts_the_inverse_scales(self):
        """A quantized checkpoint's ``*_scale_inv`` companions are read with their weights, the MTP copies' too."""
        config = dict(_PROXY_CONFIG, quantization_config=FP8_QUANTIZATION)
        scales = [f"{name}_scale_inv" for name in _PROXY_DTYPES if name.endswith(".weight")]
        copies = sorted(mappings.ernie55_mtp_body_copy_keys(_PROXY_CONFIG))
        assert scales and copies
        check_ernie55_checkpoint_keys(config, [*_PROXY_DTYPES, *scales, *copies, *(f"{c}_scale_inv" for c in copies)])
        stray = "model.layers.0.self_attn.unknown_scale_inv"
        with pytest.raises(ValueError, match=rf"1 that no mapping reads \['{re.escape(stray)}'\]"):
            check_ernie55_checkpoint_keys(config, [*_PROXY_DTYPES, *scales, stray])
        # Without quantization_config the scales are tensors no mapping reads.
        with pytest.raises(ValueError, match=rf"{len(scales)} that no mapping reads"):
            check_ernie55_checkpoint_keys(_PROXY_CONFIG, [*_PROXY_DTYPES, *scales])


class TestDequantization:
    @pytest.mark.parametrize("block_size", [8, 128], ids=["3x2_blocks", "one_partial_block"])
    @pytest.mark.parametrize("dtype", [_BF16, _FP32], ids=["bf16", "fp32"])
    def test_weight_is_its_blocks_times_their_inverse_scales(self, block_size, dtype):
        tensors, source = _fp8_checkpoint([_WEIGHT], block_size)
        weight, scale_inv = tensors[_WEIGHT], tensors[f"{_WEIGHT}_scale_inv"]
        assert tuple(scale_inv.shape) == {8: (3, 2), 128: (1, 1)}[block_size]  # partial blocks at the edges
        loaded = mappings._dequantize_fp8(weight, _WEIGHT, tensors, block_size, dtype)
        assert loaded.dtype == dtype
        assert torch.equal(loaded, fp8_reference(weight, scale_inv, block_size, dtype=dtype))
        # e4m3 keeps 3 mantissa bits: rounding to nearest is within 2^-4 relative, and the bound here is twice that;
        # below 2^-6 x scale the steps are 2^-9 x scale
        atol = 2.0**-9 * float(scale_inv.max())
        assert torch.allclose(loaded.float(), source[_WEIGHT], rtol=2.0**-3, atol=atol)

    def test_other_tensors_are_returned_unchanged(self):
        for dtype in (_BF16, _FP32):
            tensor = torch.ones(4, 4, dtype=dtype)
            # an unquantized tensor needs no scales
            assert mappings._dequantize_fp8(tensor, _WEIGHT, {}, 8, _BF16) is tensor

    def test_fp8_weight_without_its_scales_is_rejected(self):
        """A plain cast of an FP8 weight would load it without its scales; the import refuses instead."""
        block_size = 8
        tensors, _ = _fp8_checkpoint([_WEIGHT], block_size)
        weight, scale_inv = tensors[_WEIGHT], tensors[f"{_WEIGHT}_scale_inv"]
        name = re.escape(_WEIGHT)
        with pytest.raises(ValueError, match=rf"{name} is torch\.float8_e4m3fn .* shape \(3, 2\); got None"):
            mappings._dequantize_fp8(weight, _WEIGHT, {_WEIGHT: weight}, block_size, _BF16)
        # one scale for the whole weight, and the grid of the transposed weight
        for wrong in (scale_inv[:1, :1], scale_inv.T):
            scales = {f"{_WEIGHT}_scale_inv": wrong}
            shape = re.escape(str(tuple(wrong.shape)))
            with pytest.raises(ValueError, match=rf"{name} is torch\.float8_e4m3fn .* got {shape}"):
                mappings._dequantize_fp8(weight, _WEIGHT, scales, block_size, _BF16)
        # an FP8 tensor that is not a matrix has no blockwise scales to apply
        scales = {"model.norm.weight_scale_inv": torch.ones(1, 1)}
        with pytest.raises(ValueError, match=r"model\.norm\.weight is torch\.float8_e4m3fn .* shape None"):
            mappings._dequantize_fp8(torch.ones(12).to(_FP8), "model.norm.weight", scales, block_size, _BF16)


class TestImportHook:
    """``Ernie55Bridge.maybe_modify_loaded_hf_weight``, which Bridge calls for the tensor(s) of every mapping."""

    def test_unquantized_checkpoint_is_loaded_as_is(self):
        tensors = {_WEIGHT: torch.ones(4, 4, dtype=_BF16)}
        for bridge in (Ernie55Bridge(), _bridge()):  # no config yet; a config without quantization_config
            assert bridge.maybe_modify_loaded_hf_weight(_WEIGHT, tensors) is tensors[_WEIGHT]
            grouped = bridge.maybe_modify_loaded_hf_weight({"down": _WEIGHT}, tensors)
            assert list(grouped) == ["down"] and grouped["down"] is tensors[_WEIGHT]

    def test_one_tensor_is_dequantized_into_the_checkpoint_dtype(self):
        block_size = 8
        tensors, _ = _fp8_checkpoint([_WEIGHT], block_size)
        scale_inv = tensors[f"{_WEIGHT}_scale_inv"]
        # an ERNIE 5.5 config without ``dtype`` / ``torch_dtype`` is a bf16 checkpoint; the dtype may be a string
        # (plain JSON) or a torch.dtype
        configs = (
            ({}, _BF16),
            ({"dtype": "float32"}, _FP32),
            ({"dtype": _FP32}, _FP32),
            ({"torch_dtype": "float32"}, _FP32),
        )
        for config, dtype in configs:
            bridge = _bridge(quantization_config=_quantization(block_size), **config)
            loaded = bridge.maybe_modify_loaded_hf_weight(_WEIGHT, tensors)
            assert loaded.dtype == dtype
            assert torch.equal(loaded, fp8_reference(tensors[_WEIGHT], scale_inv, block_size, dtype=dtype))

    def test_a_group_of_tensors_is_dequantized_one_by_one(self):
        """Fused mappings (the experts' gate and up projections) ask for a dict of names; unquantized members stay."""
        block_size = 8
        experts = "model.layers.1.mlp.experts.0"
        names = {"gate": f"{experts}.gate_proj.weight", "up": f"{experts}.up_proj.weight"}
        tensors, _ = _fp8_checkpoint(list(names.values()), block_size)  # no two members share an inverse scale
        tensors["model.norm.weight"] = torch.ones(12, dtype=_BF16)
        bridge = _bridge(quantization_config=_quantization(block_size))
        loaded = bridge.maybe_modify_loaded_hf_weight({**names, "norm": "model.norm.weight"}, tensors)
        assert list(loaded) == ["gate", "up", "norm"]
        for part, name in names.items():
            expected = fp8_reference(tensors[name], tensors[f"{name}_scale_inv"], block_size)
            assert loaded[part].dtype == _BF16 and torch.equal(loaded[part], expected), part
        assert loaded["norm"] is tensors["model.norm.weight"]

    def test_fp8_weight_without_its_scales_is_rejected(self):
        block_size = 8
        tensors, _ = _fp8_checkpoint([_WEIGHT], block_size)
        del tensors[f"{_WEIGHT}_scale_inv"]
        bridge = _bridge(quantization_config=_quantization(block_size))
        with pytest.raises(ValueError, match=rf"{re.escape(_WEIGHT)} is torch\.float8_e4m3fn .* got None"):
            bridge.maybe_modify_loaded_hf_weight(_WEIGHT, tensors)
        with pytest.raises(ValueError, match=rf"{re.escape(_WEIGHT)} is torch\.float8_e4m3fn .* got None"):
            bridge.maybe_modify_loaded_hf_weight({"down": _WEIGHT}, tensors)

    def test_other_quantizations_are_refused(self):
        bridge = _bridge(quantization_config={"quant_method": "awq"})
        with pytest.raises(NotImplementedError, match="only blockwise FP8"):
            bridge.maybe_modify_loaded_hf_weight(_WEIGHT, {_WEIGHT: torch.ones(4, 4, dtype=_BF16)})


class TestScaleDeclarations:
    """What ``Ernie55Bridge`` declares about the inverse scales, besides reading them in its import hook."""

    def test_a_weight_is_requested_together_with_its_inverse_scales(self):
        """An incremental import waits for the tensors a task declares; the scales the hook reads are among them."""
        scales = f"{_WEIGHT}_scale_inv"
        assert Ernie55Bridge.get_hf_import_param_names(_WEIGHT, {_WEIGHT, scales}) == (_WEIGHT, scales)
        assert Ernie55Bridge.get_hf_import_param_names(_WEIGHT, {_WEIGHT}) == (_WEIGHT,)

    def test_export_does_not_expect_the_inverse_scales(self):
        """A strict export checks its tensors against the source shards, which also hold the consumed scales."""
        assert Ernie55Bridge.DEQUANTIZED_SOURCE_KEY_SUFFIXES == ("_scale_inv",)
