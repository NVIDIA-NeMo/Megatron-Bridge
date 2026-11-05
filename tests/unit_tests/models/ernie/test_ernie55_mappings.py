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

"""CPU tests of the ERNIE 5.5 Hugging Face <-> Megatron weight mappings (``ernie55_mappings``).

* Config parsing: the proxy config, a 43-layer config, every rejected config, the defaults of keys a
  config may omit, and the key spellings only the reference implementation reads.
* Key coverage: the registry reads and writes exactly the 439 tensors of the proxy checkpoint
  ``ernie55_small_random``, with their safetensors dtypes (``fixtures/``); an import checks the
  checkpoint's tensor names and dtypes against it.
* Transforms against their definitions: the ``kv_b_proj`` split against the Hugging Face latent
  attention, the D-layer RoPE rows against Megatron-Core's MLA rotation, the GLU interleave against
  Megatron-Core's grouped-expert de-interleave.
* A toy ERNIE 5.5 with the proxy's layout (``H-DEHEDE/DE``: H and D attention, a dense and three MoE
  MLPs, hyper-connections, multimax and an SD-MTP depth) built on CPU from
  ``ernie55_hybrid_stack_spec``: the Hugging Face tensor set and shapes are written out independently
  from the checkpoint definition, loaded through Bridge's conversion, checked where they land, and
  exported back bitwise, dtypes included. Variants: RoPE rows without adjacent pairing, latent MoE.
  Its D-layer dimensions are pairwise distinct, so a RoPE-row block, offset or width taken from the
  wrong dimension misplaces rows.
* Expert parallelism on 2 and 4 gloo ranks: every rank loads the global experts it owns, and the
  export reassembles the whole checkpoint on every rank. Pipeline parallelism on 2 gloo ranks.
"""

from __future__ import annotations

import contextlib
import json
import re
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# Bridge's conversion modules import its LoRA layers, which need Transformer Engine at import time.
allow_conversion_imports_without_te()

from megatron.core import parallel_state  # noqa: E402
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd  # noqa: E402
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.transformer.moe import moe_layer as mcore_moe_layer  # noqa: E402
from megatron.core.transformer.torch_norm import WrappedTorchNorm  # noqa: E402

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge  # noqa: E402
from megatron.bridge.models.ernie import ernie55_mappings as mappings  # noqa: E402
from megatron.bridge.models.ernie.ernie55_mappings import (  # noqa: E402
    Ernie55DLayerDims,
    Ernie55HFLayout,
    Ernie55WeightConversionMixin,
    check_ernie55_checkpoint_keys,
    ernie55_hf_param_dtypes,
    ernie55_mapping_registry,
    ernie55_param_mappings,
    join_kv_b_proj,
    rope_row_permutation,
    split_kv_b_proj,
)
from megatron.bridge.models.ernie.modeling_ernie55 import check_ernie55_mtp_aliases  # noqa: E402
from megatron.bridge.models.hf_pretrained.state import StateDict  # noqa: E402
from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases  # noqa: E402
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import (  # noqa: E402, F401 - fixtures
    VOCAB,
    _build_model,
    _cpu_nvtx,
    _fresh_trackers,
    pg_collection,
)
from tests.unit_tests.models.ernie.test_ernie55_moe import _DuplicatedLinear  # noqa: E402


pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parent / "fixtures"
_PROXY_CONFIG = json.loads((_FIXTURES / "ernie55_proxy_config.json").read_text())
_PROXY_DTYPES = json.loads((_FIXTURES / "ernie55_proxy_checkpoint_dtypes.json").read_text())["tensors"]
_SAFETENSORS_DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16"}
_BF16, _FP32 = torch.bfloat16, torch.float32

# The toy's layout is the proxy's: H + dense, D + MoE, H + MoE, D + MoE, and an SD-MTP depth D + MoE.
_PATTERN = "H-DEHEDE/DE"
_HIDDEN = 64
_STREAMS = 4
_EXPERTS = 8
_VHA_RANK = 2
_LATENT = 16
_H = dict(heads=4, q_lora=32, head_dim=16, rope=8, groups=2, o_lora=8)
# Pairwise distinct (and kv_lora != nope + rope), so no D-layer RoPE-row geometry fits by accident.
_D = dict(heads=4, q_lora=32, kv_lora=24, nope=12, rope=8, v=6)
_INDEXER = dict(heads=4, head_dim=16)
_DENSE_FFN = 24
_MOE_FFN = 8
# The H compressor pools 128-token blocks; its absolute position embedding has one row per position.
_COMPRESS_RATIO = 128


# ---------------------------------------------------------------------------------------------
# Toy checkpoint: config and tensors, from the Hugging Face definition
# ---------------------------------------------------------------------------------------------


def _toy_hf_config(pattern: str = _PATTERN, **overrides) -> dict:
    """A Hugging Face ERNIE 5.5 config for the toy model with ``pattern``'s layers."""
    main, *mtp = pattern.split("/")
    layers = [main[i : i + 2] for i in range(0, len(main), 2)] + mtp
    first_dense = next((index for index, layer in enumerate(layers) if layer[1] != "-"), len(layers))
    config = dict(
        architectures=["Ernie5_5_MoeForCausalLM"],
        model_type="ernie5_5_moe",
        dtype="bfloat16",
        vocab_size=VOCAB,
        hidden_size=_HIDDEN,
        tie_word_embeddings=False,
        num_hidden_layers=len(main) // 2,
        num_nextn_predict_layers=len(mtp),
        mtp_shared_last_layer=bool(mtp),
        compress_ratios=[128 if layer[0] == "H" else -2 for layer in layers],
        first_k_dense_replace=first_dense,
        hca_rope_type="rope",
        num_attention_heads=_H["heads"],
        q_lora_rank=_H["q_lora"],
        head_dim=_H["head_dim"],
        v_head_dim=_H["head_dim"],
        qk_rope_head_dim=_H["rope"],
        o_groups=_H["groups"],
        o_lora_rank=_H["o_lora"],
        hybrid_mla_attention="mqa_dsa",
        hybrid_mla_num_attention_heads=_D["heads"],
        hybrid_mla_q_lora_rank=_D["q_lora"],
        hybrid_mla_kv_lora_rank=_D["kv_lora"],
        hybrid_mla_qk_nope_head_dim=_D["nope"],
        hybrid_mla_qk_rope_head_dim=_D["rope"],
        hybrid_mla_v_head_dim=_D["v"],
        mqa_latent_rope_adjacent_pairing=True,
        index_n_heads=_INDEXER["heads"],
        index_head_dim=_INDEXER["head_dim"],
        intermediate_size=_DENSE_FFN,
        moe_intermediate_size=_MOE_FFN,
        n_routed_experts=_EXPERTS,
        n_shared_experts=1,
        hc_mult=_STREAMS,
        latent_moe_use_norm=False,
    )
    config.update(overrides)
    return config


def _toy_hf_spec(config: dict) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``{name: (shape, dtype)}`` of every tensor of an ERNIE 5.5 checkpoint, written from its definition.

    Independent of ``ernie55_mappings``: a Hugging Face ERNIE 5.5 checkpoint stores the router, the
    expert bias, hyper-connections, ``hc_head_*``, the H sink and compressor ``ape``, the indexer
    ``k_norm`` and VHA in F32 (as in the proxy checkpoint's safetensors header), and the latent-MoE
    projections in F32 (the keep-in-fp32 contract; the proxy has no latent MoE), the rest in the
    checkpoint dtype; ``mapping_proj`` and ``hc_head_fn`` are ``[n C, .]``.
    """
    hidden, vocab, streams = config["hidden_size"], config["vocab_size"], config["hc_mult"]
    spec: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}

    def add(name: str, shape: tuple[int, ...], dtype: torch.dtype = _BF16) -> None:
        assert name not in spec, name
        spec[name] = (tuple(shape), dtype)

    def hc_head(prefix: str) -> None:
        add(f"{prefix}.hc_head_fn", (streams * hidden, streams), _FP32)
        add(f"{prefix}.hc_head_base", (streams,), _FP32)
        add(f"{prefix}.hc_head_scale", (1,), _FP32)

    experts, expert_ffn = config["n_routed_experts"], config["moe_intermediate_size"]
    latent = config.get("moe_latent_size") or None

    def moe(prefix: str) -> None:
        add(f"{prefix}.gate.weight", (experts, hidden), _FP32)
        add(f"{prefix}.gate.weight_1", (experts, hidden), _FP32)
        add(f"{prefix}.gate.routed_scaling_factor_param", (experts,), _FP32)
        add(f"{prefix}.e_score_correction_bias", (experts,), _FP32)
        width = latent or hidden
        for expert in range(experts):
            add(f"{prefix}.experts.{expert}.w1.weight", (expert_ffn, width))
            add(f"{prefix}.experts.{expert}.w3.weight", (expert_ffn, width))
            add(f"{prefix}.experts.{expert}.w2.weight", (width, expert_ffn))
        shared = config["n_shared_experts"] * expert_ffn
        add(f"{prefix}.shared_experts.w1.weight", (shared, hidden))
        add(f"{prefix}.shared_experts.w3.weight", (shared, hidden))
        add(f"{prefix}.shared_experts.w2.weight", (hidden, shared))
        if latent:
            add(f"{prefix}.fc1_latent_proj.weight", (latent, hidden), _FP32)
            add(f"{prefix}.fc2_latent_proj.weight", (hidden, latent), _FP32)
            if config.get("latent_moe_use_norm"):
                add(f"{prefix}.latent_norm.weight", (latent,))

    add("model.embed_tokens.weight", (vocab, hidden))
    add("lm_head.weight", (vocab, hidden))
    add("model.norm.weight", (hidden,))
    add("model.lm_head.multimax_ranges", (4,))
    add("model.lm_head.multimax_ts", (4,))
    hc_head("model.mhc_contract")
    num_layers = config["num_hidden_layers"]
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        add(f"{prefix}.input_layernorm.weight", (hidden,))
        add(f"{prefix}.post_attention_layernorm.weight", (hidden,))
        for kind in ("self_attention_hyper_connection", "mlp_hyper_connection"):
            add(f"{prefix}.{kind}.mapping_proj.weight", (streams * hidden, streams * (streams + 2)), _FP32)
            add(f"{prefix}.{kind}.bias", (streams * (streams + 2),), _FP32)
            for alpha in ("alpha_pre", "alpha_post", "alpha_res"):
                add(f"{prefix}.{kind}.{alpha}", (1,), _FP32)
        attn = f"{prefix}.self_attn"
        if config["compress_ratios"][layer] == 128:
            heads, head_dim, q_lora = config["num_attention_heads"], config["head_dim"], config["q_lora_rank"]
            groups, o_lora = config["o_groups"], config["o_lora_rank"]
            add(f"{attn}.linear_q_down_proj.weight", (q_lora, hidden))
            add(f"{attn}.q_layernorm.weight", (q_lora,))
            add(f"{attn}.linear_q_up_proj.weight", (heads * head_dim, q_lora))
            add(f"{attn}.linear_kv_proj.weight", (head_dim, hidden))
            add(f"{attn}.kv_layernorm.weight", (head_dim,))
            add(f"{attn}.linear_o_group_proj", (groups * o_lora, heads * head_dim // groups))
            add(f"{attn}.core_attention.attn_sink", (heads,), _FP32)
            add(f"{attn}.core_attention.compressor.ape", (_COMPRESS_RATIO, head_dim), _FP32)
            add(f"{attn}.core_attention.compressor.linear_wkv.weight", (head_dim, hidden))
            add(f"{attn}.core_attention.compressor.linear_wgate.weight", (head_dim, hidden))
            add(f"{attn}.core_attention.compressor.norm.weight", (head_dim,))
            add(f"{attn}.gate_proj.weight", (groups * o_lora, q_lora))
            add(f"{attn}.o_proj.weight", (hidden, groups * o_lora))
        else:
            heads, q_lora = config["hybrid_mla_num_attention_heads"], config["hybrid_mla_q_lora_rank"]
            kv_lora, rope = config["hybrid_mla_kv_lora_rank"], config["hybrid_mla_qk_rope_head_dim"]
            nope, value = config["hybrid_mla_qk_nope_head_dim"], config["hybrid_mla_v_head_dim"]
            index_heads, index_dim = config["index_n_heads"], config["index_head_dim"]
            add(f"{attn}.q_a_proj.weight", (q_lora, hidden))
            add(f"{attn}.q_a_layernorm.weight", (q_lora,))
            add(f"{attn}.q_b_proj.weight", (heads * (nope + rope), q_lora))
            add(f"{attn}.kv_a_proj_with_mqa.weight", (kv_lora + rope, hidden))
            add(f"{attn}.kv_a_layernorm.weight", (kv_lora,))
            add(f"{attn}.kv_b_proj.weight", (heads * (nope + value), kv_lora))
            add(f"{attn}.o_proj.weight", (hidden, heads * value))
            add(f"{attn}.gate_proj.weight", (heads * value, q_lora))
            add(f"{attn}.core_attention.softmax_offset", (heads,))
            add(f"{attn}.core_attention.indexer.wq_b.weight", (index_heads * index_dim, q_lora))
            add(f"{attn}.core_attention.indexer.wk.weight", (index_dim, hidden))
            add(f"{attn}.core_attention.indexer.k_norm.weight", (index_dim,), _FP32)
            add(f"{attn}.core_attention.indexer.k_norm.bias", (index_dim,), _FP32)
            add(f"{attn}.core_attention.indexer.weights_proj.weight", (index_heads, hidden))
        add(f"{attn}.vha_postmix_U", (heads, _VHA_RANK), _FP32)
        add(f"{attn}.vha_postmix_V", (heads, _VHA_RANK), _FP32)
        if layer < config["first_k_dense_replace"]:
            ffn = config["intermediate_size"]
            add(f"{prefix}.mlp.gate_proj.weight", (ffn, hidden))
            add(f"{prefix}.mlp.up_proj.weight", (ffn, hidden))
            add(f"{prefix}.mlp.down_proj.weight", (hidden, ffn))
        else:
            moe(f"{prefix}.block_sparse_moe")
    for depth in range(config["num_nextn_predict_layers"]):
        prefix = f"model.layers.{num_layers + depth}"
        add(f"{prefix}.enorm.weight", (hidden,))
        add(f"{prefix}.hnorm.weight", (hidden,))
        add(f"{prefix}.e_proj.weight", (hidden, hidden))
        add(f"{prefix}.h_proj.weight", (hidden, hidden))
        add(f"{prefix}.shared_head.norm.weight", (hidden,))
        hc_head(prefix)
        add(f"{prefix}.block_sparse_moe.e_score_correction_bias", (experts,), _FP32)
    return spec


def _toy_hf_tensors(config: dict, *, seed: int) -> dict[str, torch.Tensor]:
    """Random checkpoint tensors; each name has its own generator, so every rank draws the same values."""
    tensors = {}
    for name, (shape, dtype) in _toy_hf_spec(config).items():
        generator = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
        tensors[name] = torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype)
    return tensors


def _megatron_overrides(**extra) -> dict:
    """``Ernie55TransformerConfig`` fields of the toy model that fix its tensor shapes (see ``_toy_hf_config``)."""
    return dict(
        hidden_size=_HIDDEN,
        mhc_num_residual_streams=_STREAMS,
        num_attention_heads=_H["heads"],
        q_lora_rank=_H["q_lora"],
        v_head_dim=_H["head_dim"],
        qk_pos_emb_head_dim=_H["rope"],
        output_projection_groups=_H["groups"],
        output_projection_lora_rank=_H["o_lora"],
        ernie_vha_rank=_VHA_RANK,
        ernie_d_num_attention_heads=_D["heads"],
        ernie_d_q_lora_rank=_D["q_lora"],
        ernie_d_kv_lora_rank=_D["kv_lora"],
        ernie_d_qk_head_dim=_D["nope"],
        ernie_d_qk_pos_emb_head_dim=_D["rope"],
        ernie_d_v_head_dim=_D["v"],
        dsa_indexer_n_heads=_INDEXER["heads"],
        dsa_indexer_head_dim=_INDEXER["head_dim"],
        ffn_hidden_size=_DENSE_FFN,
        num_moe_experts=_EXPERTS,
        moe_ffn_hidden_size=_MOE_FFN,
        moe_shared_expert_intermediate_size=_MOE_FFN,
        **extra,
    )


# ---------------------------------------------------------------------------------------------
# Conversion through Bridge
# ---------------------------------------------------------------------------------------------


class _Ernie55TestBridge(Ernie55WeightConversionMixin, MegatronModelBridge):
    """A bridge with only the ERNIE 5.5 weight conversion; the model is built by the test."""

    def provider_bridge(self, hf_pretrained):
        raise NotImplementedError("the tests build the model directly")


def _hf_pretrained(config: dict, tensors: dict[str, torch.Tensor]) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(**config), state=StateDict(dict(tensors)), model_name_or_path="ernie55-toy"
    )


def _load(model, config: dict, tensors: dict[str, torch.Tensor]) -> _Ernie55TestBridge:
    bridge = _Ernie55TestBridge()
    bridge.load_weights_hf_to_megatron(_hf_pretrained(config, tensors), [model])
    return bridge


def _export(bridge, model, config: dict, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    exported: dict[str, torch.Tensor] = {}
    for name, tensor in bridge.stream_weights_megatron_to_hf(
        [model], _hf_pretrained(config, tensors), cpu=True, show_progress=False
    ):
        assert name not in exported, f"{name} exported twice"
        exported[name] = tensor.detach().clone()
    return exported


def _assert_bitwise_equal(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert sorted(actual) == sorted(expected), (
        f"missing {sorted(set(expected) - set(actual))[:5]}, unexpected {sorted(set(actual) - set(expected))[:5]}"
    )
    mismatches = [
        f"{name}: {tuple(actual[name].shape)} {actual[name].dtype} vs {tuple(tensor.shape)} {tensor.dtype}"
        for name, tensor in expected.items()
        if actual[name].dtype != tensor.dtype
        or actual[name].shape != tensor.shape
        or not torch.equal(actual[name], tensor)
    ]
    assert not mismatches, "\n".join(mismatches[:10])


def _megatron_tensors(model) -> dict[str, torch.Tensor]:
    """Every parameter (shared ones once) and persistent buffer of ``model``."""
    tensors = dict(model.named_parameters())
    for module_name, module in model.named_modules():
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is not None and buffer_name not in module._non_persistent_buffers_set:
                tensors[f"{module_name}.{buffer_name}" if module_name else buffer_name] = buffer
    return tensors


@contextlib.contextmanager
def _latent_moe_without_te():
    """Megatron-Core's latent-MoE modules without Transformer Engine (as in ``test_ernie55_moe.py``)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mcore_moe_layer, "HAVE_TE", True)
        patch.setattr(mcore_moe_layer, "TELinear", _DuplicatedLinear, raising=False)
        patch.setattr(mcore_moe_layer, "TENorm", WrappedTorchNorm, raising=False)
        yield


# Toy variants: HF config overrides and Megatron config overrides.
_VARIANTS = {
    "proxy_layout": ({}, {}),
    "rope_rows_without_adjacent_pairing": (dict(mqa_latent_rope_adjacent_pairing=False), {}),
    "latent_moe": (
        dict(moe_latent_size=_LATENT, latent_moe_use_norm=True),
        dict(moe_latent_size=_LATENT, moe_use_norm_before_up_proj=True),
    ),
}


@pytest.fixture(scope="module", params=list(_VARIANTS))
def toy(request, pg_collection):
    """The toy model with a checkpoint loaded through Bridge, its export, and the source checkpoint."""
    hf_overrides, megatron_overrides = _VARIANTS[request.param]
    config = _toy_hf_config(**hf_overrides)
    tensors = _toy_hf_tensors(config, seed=1)
    latent = contextlib.nullcontext() if not megatron_overrides else _latent_moe_without_te()
    with latent:
        model = _build_model(pg_collection, pattern=_PATTERN, seed=0, **_megatron_overrides(**megatron_overrides))
    before = {name: tensor.detach().clone() for name, tensor in _megatron_tensors(model).items()}
    bridge = _load(model, config, tensors)
    exported = _export(bridge, model, config, tensors)
    return SimpleNamespace(
        name=request.param, config=config, tensors=tensors, model=model, before=before, exported=exported
    )


# ---------------------------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------------------------


class TestLayout:
    def test_proxy_config(self):
        layout = Ernie55HFLayout.from_hf_config(_PROXY_CONFIG)
        assert layout.hybrid_layer_pattern == "H-DEHEDE/DE"
        assert layout.attention == ("H", "D", "H", "D", "D")
        assert (layout.num_hidden_layers, layout.num_mtp_layers, layout.first_k_dense) == (4, 1, 1)
        assert (layout.num_experts, layout.shared_expert, layout.moe_latent, layout.moe_latent_norm) == (
            32,
            True,
            False,
            False,
        )
        assert layout.d_dims == Ernie55DLayerDims(
            num_heads=64, kv_lora_rank=512, qk_nope_head_dim=448, qk_rope_head_dim=64, v_head_dim=256
        )
        assert layout.rope_adjacent_pairing is True
        assert layout.dtype == torch.bfloat16
        assert [layout.is_moe(layer) for layer in range(5)] == [False, True, True, True, True]

    def test_full_size_config(self):
        ratios = [128, -2] * 21 + [-2, -2]
        config = dict(
            _PROXY_CONFIG, num_hidden_layers=43, compress_ratios=ratios, first_k_dense_replace=0, n_routed_experts=256
        )
        config.pop("dtype")
        config["torch_dtype"] = "bfloat16"
        layout = Ernie55HFLayout.from_hf_config(config)
        assert layout.hybrid_layer_pattern == "HEDE" * 21 + "DE" + "/DE"
        assert layout.num_experts == 256 and layout.dtype == torch.bfloat16

    @pytest.mark.parametrize(
        "overrides, error, message",
        [
            (dict(compress_ratios=[128, 4, 128, -2, -2]), ValueError, r"compress_ratios \[4\]"),
            (dict(compress_ratios=[128, -2, 128, -2]), ValueError, "entry for every decoder layer and MTP depth"),
            (dict(hybrid_mla_attention="mha"), ValueError, "mqa_dsa"),
            (dict(hca_rope_type="linear"), ValueError, "plain RoPE or YaRN"),
            (dict(mla_dense_layers=[1]), ValueError, "mla_dense_layers"),
            (dict(mtp_shared_last_layer=False), NotImplementedError, "mtp_shared_last_layer"),
            (dict(compress_ratios=[128, -2, 128, -2, 128]), ValueError, "attention type of the decoder's last layer"),
            (dict(first_k_dense_replace=4), ValueError, "must be MoE"),
            (dict(first_k_dense_replace=5), ValueError, "first_k_dense_replace=5"),
            (dict(n_routed_experts=0), ValueError, "n_routed_experts"),
            (dict(hybrid_mla_qk_rope_head_dim=63), ValueError, "must be even"),
            (dict(dtype="int8"), ValueError, "dtype"),
        ],
    )
    def test_rejects_configs_outside_the_envelope(self, overrides, error, message):
        with pytest.raises(error, match=message):
            Ernie55HFLayout.from_hf_config(dict(_PROXY_CONFIG, **overrides))

    @pytest.mark.parametrize(
        "dtype_keys, expected",
        [
            (dict(dtype="float32"), torch.float32),
            (dict(torch_dtype="torch.float16"), torch.float16),
            (dict(dtype=torch.bfloat16), torch.bfloat16),
            ({}, torch.bfloat16),
        ],
    )
    def test_checkpoint_dtype(self, dtype_keys, expected):
        config = {key: value for key, value in _PROXY_CONFIG.items() if key not in ("dtype", "torch_dtype")}
        assert Ernie55HFLayout.from_hf_config(dict(config, **dtype_keys)).dtype == expected

    def test_accepts_config_objects(self):
        layout = Ernie55HFLayout.from_hf_config(SimpleNamespace(**_PROXY_CONFIG))
        assert layout == Ernie55HFLayout.from_hf_config(_PROXY_CONFIG)

    def test_needs_a_config(self):
        """A bridge whose ``hf_config`` is unset (``Ernie55Bridge().mapping_registry()``) fails clearly."""
        with pytest.raises(ValueError, match="Hugging Face config is not set"):
            Ernie55HFLayout.from_hf_config(None)
        with pytest.raises(ValueError, match="Hugging Face config is not set"):
            _Ernie55TestBridge().mapping_registry()

    @pytest.mark.parametrize(
        "key, expected",
        [
            # D-layer RoPE rows in half-rotation order (the rows are permuted).
            ("mqa_latent_rope_adjacent_pairing", dict(rope_adjacent_pairing=False)),
            # Plain RoPE in the H layers: accepted.
            ("hca_rope_type", dict(attention=("H", "D", "H", "D", "D"))),
            ("first_k_dense_replace", dict(first_k_dense=0)),
            ("n_shared_experts", dict(shared_expert=False)),
            ("num_nextn_predict_layers", dict(num_mtp_layers=0)),
        ],
    )
    def test_omitted_keys_take_the_reference_defaults(self, key, expected):
        layout = Ernie55HFLayout.from_hf_config(_proxy_config_without(key))
        assert {name: getattr(layout, name) for name in expected} == expected

    def test_omitted_latent_norm_key_means_no_latent_norm(self):
        config = dict(_proxy_config_without("latent_moe_use_norm"), moe_latent_size=1024)
        layout = Ernie55HFLayout.from_hf_config(config)
        assert (layout.moe_latent, layout.moe_latent_norm) == (True, False)

    @pytest.mark.parametrize(
        "key, error, message",
        [
            # The reference default is plain MLA ("mha"), which Bridge does not implement.
            ("hybrid_mla_attention", ValueError, "mqa_dsa"),
            # The reference default is a separate MTP body.
            ("mtp_shared_last_layer", NotImplementedError, "mtp_shared_last_layer"),
        ],
    )
    def test_omitted_keys_whose_reference_default_is_unsupported(self, key, error, message):
        with pytest.raises(error, match=message):
            Ernie55HFLayout.from_hf_config(_proxy_config_without(key))

    @pytest.mark.parametrize(
        "spelling, value, dropped",
        [
            ("num_shared_experts", 1, "n_shared_experts"),
            ("moe_num_shared_experts", 1, "n_shared_experts"),
            ("num_shared_expert", 1, "n_shared_experts"),
            ("moe_layer_start_index", 1, "first_k_dense_replace"),
            ("moe_num_experts", 32, "n_routed_experts"),
            ("num_local_experts", 32, "n_routed_experts"),
            # The reference reads num_experts before n_routed_experts.
            ("num_experts", 64, None),
        ],
    )
    def test_rejects_key_spellings_only_the_reference_reads(self, spelling, value, dropped):
        """With these spellings the reference builds another model than Bridge, which reads only the key."""
        config = dict(_proxy_config_without(*([dropped] if dropped else [])), **{spelling: value})
        with pytest.raises(ValueError, match="which the reference implementation reads"):
            Ernie55HFLayout.from_hf_config(config)

    def test_accepts_key_spellings_that_agree(self):
        # num_experts agrees with n_routed_experts; the reference reads the others only when the key is absent.
        config = dict(_PROXY_CONFIG, num_experts=32, num_shared_experts=5, moe_layer_start_index=7)
        assert Ernie55HFLayout.from_hf_config(config) == Ernie55HFLayout.from_hf_config(_PROXY_CONFIG)


def _proxy_config_without(*keys: str) -> dict:
    """The proxy config without ``keys``."""
    return {key: value for key, value in _PROXY_CONFIG.items() if key not in keys}


# ---------------------------------------------------------------------------------------------
# Key coverage
# ---------------------------------------------------------------------------------------------


class TestProxyCheckpointKeys:
    def test_registry_covers_exactly_the_proxy_checkpoint_with_its_dtypes(self):
        dtypes = ernie55_hf_param_dtypes(_PROXY_CONFIG)
        assert len(_PROXY_DTYPES) == 439
        missing, unexpected = sorted(set(_PROXY_DTYPES) - set(dtypes)), sorted(set(dtypes) - set(_PROXY_DTYPES))
        assert not missing and not unexpected, f"missing {missing}, unexpected {unexpected}"
        wrong = {name: (_SAFETENSORS_DTYPES[dtype], _PROXY_DTYPES[name]) for name, dtype in dtypes.items()}
        wrong = {name: pair for name, pair in wrong.items() if pair[0] != pair[1]}
        assert not wrong, wrong
        # Every F32 tensor of the bf16 checkpoint is a keep-in-fp32 tensor, which must load into fp32.
        layout = Ernie55HFLayout.from_hf_config(_PROXY_CONFIG)
        keep_fp32 = {
            mapping.hf_param for mapping in ernie55_param_mappings(layout) if getattr(mapping, "keep_fp32", 0)
        }
        assert keep_fp32 == {name for name, dtype in _PROXY_DTYPES.items() if dtype == "F32"}

    def test_toy_checkpoint_definition_matches_the_registry(self):
        """The independent toy definition and the registry agree on every name and dtype, in every variant."""
        for hf_overrides, _ in _VARIANTS.values():
            config = _toy_hf_config(**hf_overrides)
            spec = {name: dtype for name, (_, dtype) in _toy_hf_spec(config).items()}
            assert ernie55_hf_param_dtypes(config) == dict(sorted(spec.items()))

    def test_proxy_key_structure_equals_the_toy_definition(self):
        """With experts collapsed, the toy definition has the proxy checkpoint's keys and dtypes."""

        def collapsed(names_to_dtypes: dict[str, str]) -> dict[str, str]:
            return {re.sub(r"\.experts\.\d+\.", ".experts.E.", name): dtype for name, dtype in names_to_dtypes.items()}

        toy = {name: _SAFETENSORS_DTYPES[dtype] for name, (_, dtype) in _toy_hf_spec(_toy_hf_config()).items()}
        assert collapsed(toy) == collapsed(_PROXY_DTYPES)

    def test_grouped_experts_resolve_to_global_checkpoint_experts(self):
        """Bridge renames local grouped experts to global ones; the registry maps those to the same HF expert."""
        registry = ernie55_mapping_registry(_PROXY_CONFIG)
        layer = "decoder.layers.3.inner_layer.mlp.experts"
        fc1 = registry.megatron_to_hf_lookup(f"{layer}.linear_fc1.weight31")
        assert fc1.hf_param == {
            "gate": "model.layers.1.block_sparse_moe.experts.31.w1.weight",
            "up": "model.layers.1.block_sparse_moe.experts.31.w3.weight",
        }
        assert fc1.glu_interleave == "routed"
        fc2 = registry.megatron_to_hf_lookup(f"{layer}.linear_fc2.weight17")
        assert fc2.hf_param == "model.layers.1.block_sparse_moe.experts.17.w2.weight"
        sequential = registry.megatron_to_hf_lookup(f"{layer}.local_experts.5.linear_fc1.weight")
        assert sequential.hf_param["up"] == "model.layers.1.block_sparse_moe.experts.5.w3.weight"
        assert sequential.glu_interleave is None
        # Only grouped experts may keep interleaved fc1 rows: the dense MLP never does.
        dense = registry.megatron_to_hf_lookup("decoder.layers.1.inner_layer.mlp.linear_fc1.weight")
        assert dense.hf_param == {
            "gate": "model.layers.0.mlp.gate_proj.weight",
            "up": "model.layers.0.mlp.up_proj.weight",
        }
        assert dense.glu_interleave is None
        shared = registry.megatron_to_hf_lookup("decoder.layers.3.inner_layer.mlp.shared_experts.linear_fc1.weight")
        assert shared.glu_interleave == "shared"
        bias = registry.megatron_to_hf_lookup(
            "mtp.layers.0.mtp_model_layer.layers.1.inner_layer.mlp.router.expert_bias"
        )
        assert bias.hf_param == "model.layers.4.block_sparse_moe.e_score_correction_bias"
        # The SD-MTP body has no mapping of its own.
        assert (
            registry.megatron_to_hf_lookup("mtp.layers.0.mtp_model_layer.layers.1.inner_layer.mlp.router.weight")
            is None
        )

    @pytest.mark.parametrize("adjacent_pairing", [True, False])
    def test_rope_row_geometry_of_the_proxy(self, adjacent_pairing):
        """Without adjacent pairing, the RoPE rows of q_b_proj (per head) and kv_a_proj_with_mqa are permuted.

        Proxy D layer: nope 448, rope 64, kv_lora 512, v 256. ``q_b_proj`` blocks are a head's
        ``nope + rope`` rows with the RoPE rows after ``nope``; ``kv_a_proj_with_mqa`` is one block of
        ``kv_lora + rope`` rows with the RoPE rows after ``kv_lora``.
        """
        registry = ernie55_mapping_registry(dict(_PROXY_CONFIG, mqa_latent_rope_adjacent_pairing=adjacent_pairing))
        attention = "decoder.layers.2.inner_layer.self_attention"
        q_up = registry.megatron_to_hf_lookup(f"{attention}.linear_q_up_proj.weight")
        kv_down = registry.megatron_to_hf_lookup(f"{attention}.linear_kv_down_proj.weight")
        assert (q_up.hf_param, kv_down.hf_param) == (
            "model.layers.1.self_attn.q_b_proj.weight",
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        )
        if adjacent_pairing:
            assert type(q_up) is type(kv_down) is mappings._Ernie55TensorMapping
            assert not q_up.transpose and not kv_down.transpose
        else:
            assert (q_up.block, q_up.offset, q_up.rope_dim) == (512, 448, 64)
            assert (kv_down.block, kv_down.offset, kv_down.rope_dim) == (576, 512, 64)

    def test_checkpoint_key_check(self):
        """An import needs exactly the registry's tensors; Bridge alone would ignore unread checkpoint tensors."""
        check_ernie55_checkpoint_keys(_PROXY_CONFIG, _PROXY_DTYPES)
        extra = "model.layers.0.mlp.shared_experts.w1.weight"
        with pytest.raises(ValueError, match=rf"1 that no mapping reads \['{re.escape(extra)}'\]"):
            check_ernie55_checkpoint_keys(_PROXY_CONFIG, [*_PROXY_DTYPES, extra])
        missing = "model.layers.3.self_attn.kv_b_proj.weight"
        with pytest.raises(ValueError, match=rf"1 missing \['{re.escape(missing)}'\]"):
            check_ernie55_checkpoint_keys(_PROXY_CONFIG, [name for name in _PROXY_DTYPES if name != missing])
        # A released checkpoint also stores the shared MTP body (the decoder's last layer) under the MTP layer.
        last, mtp = (
            f"model.layers.{_PROXY_CONFIG['num_hidden_layers'] - 1}.",
            f"model.layers.{_PROXY_CONFIG['num_hidden_layers']}.",
        )
        copies = {mtp + name[len(last) :] for name in _PROXY_DTYPES if name.startswith(last)} - set(_PROXY_DTYPES)
        copies = {
            name.replace(mtp, f"{mtp}transformer_layer.") if "_hyper_connection." in name else name for name in copies
        }
        assert copies and mappings.ernie55_mtp_body_copy_keys(_PROXY_CONFIG) >= copies
        check_ernie55_checkpoint_keys(_PROXY_CONFIG, [*_PROXY_DTYPES, *copies])
        with pytest.raises(ValueError, match="1 that no mapping reads"):
            check_ernie55_checkpoint_keys(_PROXY_CONFIG, [*_PROXY_DTYPES, *copies, f"{mtp}self_attn.unknown.weight"])
        # A config naming its shared experts as only the reference reads them is rejected before the keys.
        config = dict(_proxy_config_without("n_shared_experts"), num_shared_experts=1)
        with pytest.raises(ValueError, match="which the reference implementation reads"):
            check_ernie55_checkpoint_keys(config, _PROXY_DTYPES)


# ---------------------------------------------------------------------------------------------
# Transforms against their definitions
# ---------------------------------------------------------------------------------------------


_DIMS = Ernie55DLayerDims(num_heads=3, kv_lora_rank=5, qk_nope_head_dim=4, qk_rope_head_dim=6, v_head_dim=2)


def _interleaved_rows(gate: torch.Tensor, up: torch.Tensor, size: int) -> torch.Tensor:
    """Megatron-Core's block-interleaved fc1 rows ``[g_0:k, u_0:k, g_k:2k, u_k:2k, ...]``."""
    return torch.cat([torch.cat([gate[row : row + size], up[row : row + size]]) for row in range(0, len(gate), size)])


class TestTransforms:
    def test_kv_b_proj_split_reproduces_the_hf_latent_attention(self):
        """The absorbed K and the V up-projection on ``k_b_proj``/``v_b_proj`` equal the HF attention.

        Hugging Face: ``kv = c @ kv_b_proj^T`` viewed ``[S, H, nope + v]`` splits into ``k_nope`` and
        ``v``. Megatron-Core's absorbed MLA (``_get_kv_up_weights``) scores ``(q_nope W_UK) . c`` with
        ``W_UK = k_b_proj.view(H, C, nope).transpose(1, 2)`` and up-projects ``o v_b_proj.view(H, v, C)^T``.
        """
        heads, latent, nope, value = _DIMS.num_heads, _DIMS.kv_lora_rank, _DIMS.qk_nope_head_dim, _DIMS.v_head_dim
        generator = torch.Generator().manual_seed(0)
        kv_b_proj = torch.randn(heads * (nope + value), latent, generator=generator, dtype=torch.float64)
        latent_kv = torch.randn(7, latent, generator=generator, dtype=torch.float64)
        q_nope = torch.randn(5, heads, nope, generator=generator, dtype=torch.float64)
        probs = torch.softmax(torch.randn(5, heads, 7, generator=generator, dtype=torch.float64), dim=-1)

        kv = (latent_kv @ kv_b_proj.T).view(7, heads, nope + value)
        k_nope, v = kv[..., :nope], kv[..., nope:]
        hf_scores = torch.einsum("thd,shd->ths", q_nope, k_nope)
        hf_out = torch.einsum("ths,shv->thv", probs, v)

        k_b_proj, v_b_proj = split_kv_b_proj(kv_b_proj, _DIMS)
        assert k_b_proj.shape == (heads * latent, nope) and v_b_proj.shape == (heads * value, latent)
        w_uk = k_b_proj.view(heads, latent, nope).transpose(1, 2)
        absorbed = torch.einsum("...nd,ndk->...nk", q_nope, w_uk)
        megatron_scores = torch.einsum("thk,sk->ths", absorbed, latent_kv)
        latent_out = torch.einsum("ths,sk->thk", probs, latent_kv)
        megatron_out = torch.einsum("thk,hvk->thv", latent_out, v_b_proj.view(heads, value, latent))
        torch.testing.assert_close(megatron_scores, hf_scores, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(megatron_out, hf_out, rtol=1e-12, atol=1e-12)

    def test_kv_b_proj_split_and_join_are_inverse(self):
        generator = torch.Generator().manual_seed(1)
        kv_b_proj = torch.randn(_DIMS.num_heads * 6, _DIMS.kv_lora_rank, generator=generator).to(torch.bfloat16)
        k_b_proj, v_b_proj = split_kv_b_proj(kv_b_proj, _DIMS)
        assert torch.equal(join_kv_b_proj(k_b_proj, v_b_proj, _DIMS), kv_b_proj)
        with pytest.raises(ValueError, match="kv_b_proj must be"):
            split_kv_b_proj(kv_b_proj[1:], _DIMS)
        with pytest.raises(ValueError, match="k_b_proj/v_b_proj must be"):
            join_kv_b_proj(k_b_proj, v_b_proj[1:], _DIMS)

    @pytest.mark.parametrize("adjacent_pairing", [True, False])
    def test_rope_rows_follow_megatron_core_mla_rotation(self, adjacent_pairing):
        """Megatron-Core rotates ``q_b_proj``/``kv_a_proj_with_mqa`` rows like the HF checkpoint does.

        The reference rotates channel ``i`` with ``i + R/2`` (``rotate_half`` on the rows as stored), or,
        with ``mqa_latent_rope_adjacent_pairing``, de-interleaves first exactly as Megatron-Core's MLA
        (``mla_rotary_interleaved=True``). The mapping permutes the rows only without adjacent pairing.
        """
        rope = 8
        generator = torch.Generator().manual_seed(2)
        hf_rows = torch.randn(11, 2, rope, generator=generator, dtype=torch.float64)  # [positions, heads, R]
        freqs = torch.randn(11, 1, rope, generator=generator, dtype=torch.float64)
        hf_rotated = _apply_rotary_pos_emb_bshd(hf_rows, freqs, mla_rotary_interleaved=adjacent_pairing)
        megatron_rows = hf_rows if adjacent_pairing else hf_rows[..., rope_row_permutation(rope)]
        megatron_rotated = _apply_rotary_pos_emb_bshd(megatron_rows, freqs, mla_rotary_interleaved=True)
        assert torch.equal(megatron_rotated, hf_rotated)
        assert rope_row_permutation(64)[:4].tolist() == [0, 32, 1, 33]

    def test_rope_row_mapping_permutes_the_rope_block_of_every_head(self):
        nope, rope, heads = 3, 4, 2
        hf_weight = torch.arange(heads * (nope + rope) * 2, dtype=torch.float32).view(heads * (nope + rope), 2)
        mapping = mappings._Ernie55RopeRowsMapping(
            "q", "q_b_proj.weight", hf_dtype=torch.float32, block=nope + rope, offset=nope, rope_dim=rope
        )
        megatron = mapping._to_megatron_layout(hf_weight)
        perm = rope_row_permutation(rope)
        for head in range(heads):
            start = head * (nope + rope)
            assert torch.equal(megatron[start : start + nope], hf_weight[start : start + nope])
            assert torch.equal(megatron[start + nope : start + nope + rope], hf_weight[start + nope + perm])
        assert torch.equal(mapping._to_hf_layout(megatron), hf_weight)

    def test_glu_interleave_matches_megatron_core_grouped_experts(self):
        """Interleaved fc1 rows are what ``TEGroupedMLP._remove_glu_interleaving`` turns back into ``[gate; up]``."""
        from megatron.core.transformer.moe.experts import TEGroupedMLP

        remove = getattr(TEGroupedMLP, "_remove_glu_interleaving", None)
        if remove is None:
            pytest.skip("this Megatron-Core has no GLU interleave")
        # Three blocks per half: with two, interleaving and de-interleaving are the same permutation.
        size, ffn = 4, 12
        generator = torch.Generator().manual_seed(3)
        fc1 = torch.randn(2 * ffn, 6, generator=generator, dtype=torch.float64)
        tokens = torch.randn(5, 6, generator=generator, dtype=torch.float64)
        interleaved = mappings._glu_interleave(fc1, size)
        assert torch.equal(interleaved, _interleaved_rows(fc1[:ffn], fc1[ffn:], size))
        assert torch.equal(remove(tokens @ interleaved.T, size), tokens @ fc1.T)
        assert torch.equal(mappings._glu_deinterleave(interleaved, size), fc1)

    def test_grouped_expert_fc1_interleaves_on_import_and_restores_on_export(self, pg_collection):
        gate = torch.randn(12, 6).to(torch.bfloat16)
        up = torch.randn(12, 6).to(torch.bfloat16)
        module = torch.nn.Module()
        module.weight0 = torch.nn.Parameter(torch.empty(24, 6, dtype=torch.bfloat16))
        module.config = SimpleNamespace(moe_mlp_glu_interleave_size=4, num_moe_experts=1)
        mapping = mappings._Ernie55GatedMLPMapping(
            "decoder.layers.1.inner_layer.mlp.experts.linear_fc1.weight0",
            "experts.0.w1.weight",
            "experts.0.w3.weight",
            hf_dtype=torch.bfloat16,
            glu_interleave="routed",
        )
        local = mapping.hf_to_megatron({"gate": gate, "up": up}, module)
        assert torch.equal(local, _interleaved_rows(gate, up, 4))
        exported = mapping.megatron_to_hf(local, module)
        assert torch.equal(exported["experts.0.w1.weight"], gate) and torch.equal(exported["experts.0.w3.weight"], up)
        # Sequential experts and configs without the knob keep [gate; up].
        module.config.moe_mlp_glu_interleave_size = None
        assert torch.equal(mapping.hf_to_megatron({"gate": gate, "up": up}, module), torch.cat([gate, up]))

    def test_shared_expert_interleaves_only_with_grouped_gemm(self):
        config = SimpleNamespace(moe_shared_expert_glu_interleave_size=4, use_grouped_gemm_for_shared_expert=False)
        assert mappings._glu_interleave_size(config, "shared") is None
        config.use_grouped_gemm_for_shared_expert = True
        assert mappings._glu_interleave_size(config, "shared") == 4
        assert mappings._glu_interleave_size(config, None) is None

    def test_local_views_exist_only_for_tensors_stored_alike(self):
        """Bridge's local-view transfer (refit) may only skip conversion for tensors stored the same way."""
        registry = ernie55_mapping_registry(dict(_PROXY_CONFIG, mqa_latent_rope_adjacent_pairing=False))
        plain = registry.megatron_to_hf_lookup("decoder.layers.0.inner_layer.self_attention.linear_q_down_proj.weight")
        assert [spec.name for spec in plain.local_hf_param_specs()] == [
            "model.layers.0.self_attn.linear_q_down_proj.weight"
        ]
        for name in (
            "decoder.layers.0.hyper_connection.mapping_proj.weight",
            "decoder.hc_head_fn",
            "decoder.layers.2.inner_layer.self_attention.k_b_proj",
            "decoder.layers.2.inner_layer.self_attention.v_b_proj",
            "decoder.layers.2.inner_layer.self_attention.linear_q_up_proj.weight",
            "decoder.layers.2.inner_layer.self_attention.linear_kv_down_proj.weight",
            # fc1 rows that Megatron-Core may keep GLU-interleaved, depending on the model config.
            "decoder.layers.3.inner_layer.mlp.experts.linear_fc1.weight5",
            "decoder.layers.3.inner_layer.mlp.shared_experts.linear_fc1.weight",
        ):
            assert registry.megatron_to_hf_lookup(name).local_hf_param_specs() == (), name
        # [gate; up] fc1 rows of sequential experts and dense MLPs are gate and up views.
        expected_views = {
            "decoder.layers.3.inner_layer.mlp.experts.local_experts.5.linear_fc1.weight": [
                "model.layers.1.block_sparse_moe.experts.5.w1.weight",
                "model.layers.1.block_sparse_moe.experts.5.w3.weight",
            ],
            "decoder.layers.1.inner_layer.mlp.linear_fc1.weight": [
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            ],
        }
        for name, views in expected_views.items():
            assert [spec.name for spec in registry.megatron_to_hf_lookup(name).local_hf_param_specs()] == views, name

    def test_f32_checkpoint_tensors_need_fp32_parameters(self):
        module = torch.nn.Module()
        module.weight_1 = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
        mapping = mappings._Ernie55TensorMapping(
            "router.weight_1", "gate.weight_1", hf_dtype=torch.float32, keep_fp32=True
        )
        with pytest.raises(TypeError, match="keeps in fp32"):
            mapping.hf_to_megatron(torch.zeros(4, 3), module)
        with pytest.raises(ValueError, match="keep-in-fp32 tensor is F32"):
            mappings._Ernie55TensorMapping("router.weight_1", "gate.weight_1", hf_dtype=torch.bfloat16, keep_fp32=True)

    def test_fp32_checkpoint_tensors_outside_the_contract_load_into_the_model_dtype(self, pg_collection):
        """A checkpoint saved in fp32 loads into a bf16 model; only the keep-in-fp32 tensors need fp32."""
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
        mapping = mappings._Ernie55TensorMapping(
            "embedding.word_embeddings.weight", "model.embed_tokens.weight", hf_dtype=torch.float32
        )
        hf_weight = torch.randn(4, 3)
        assert torch.equal(mapping.hf_to_megatron(hf_weight, module), hf_weight)

    def test_bf16_checkpoints_widen_the_keep_in_fp32_tensors(self, pg_collection):
        """The released bf16 checkpoints store the keep-in-fp32 tensors in bf16; they load into fp32 exactly."""
        module = torch.nn.Module()
        module.weight_1 = torch.nn.Parameter(torch.zeros(4, 3))
        router = mappings._Ernie55TensorMapping("router.weight_1", "gate.weight_1", hf_dtype=_FP32, keep_fp32=True)
        hf_weight = torch.randn(4, 3, dtype=_BF16)
        assert torch.equal(router.hf_to_megatron(hf_weight, module).float(), hf_weight.float())

    def test_import_requires_the_checkpoint_dtype(self, pg_collection):
        """A checkpoint tensor in another dtype than the ERNIE 5.5 checkpoint's fails instead of being cast."""
        module = torch.nn.Module()
        module.weight_1 = torch.nn.Parameter(torch.zeros(4, 3))
        router = mappings._Ernie55TensorMapping("router.weight_1", "gate.weight_1", hf_dtype=_FP32, keep_fp32=True)
        with pytest.raises(TypeError, match=r"gate\.weight_1 is torch\.int32 in the checkpoint"):
            router.hf_to_megatron(torch.zeros(4, 3, dtype=torch.int32), module)
        latent_norm = mappings._Ernie55TensorMapping("fc2_norm.weight", "latent_norm.weight", hf_dtype=_BF16)
        with pytest.raises(TypeError, match=r"latent_norm\.weight is torch\.float32 in the checkpoint"):
            latent_norm.hf_to_megatron(torch.zeros(4), module)
        fc1 = mappings._Ernie55GatedMLPMapping(
            "mlp.experts.linear_fc1.weight0", "experts.0.w1.weight", "experts.0.w3.weight", hf_dtype=_BF16
        )
        with pytest.raises(TypeError, match=r"experts\.0\.w3\.weight is torch\.float32 in the checkpoint"):
            fc1.hf_to_megatron({"gate": torch.zeros(4, 3, dtype=_BF16), "up": torch.zeros(4, 3)}, module)
        fc2 = mappings._Ernie55ExpertFc2Mapping(
            "mlp.experts.linear_fc2.weight0", "experts.0.w2.weight", hf_dtype=_BF16
        )
        with pytest.raises(TypeError, match=r"experts\.0\.w2\.weight is torch\.float32 in the checkpoint"):
            fc2.hf_to_megatron(torch.zeros(3, 4), module)

    def test_kv_b_proj_export_rejects_a_sharded_v_b_proj(self, pg_collection):
        """The k mapping reads v_b_proj from the module; a Megatron-FSDP or GTP shard of it is rejected."""
        kv_b_proj = torch.randn(_DIMS.num_heads * 6, _DIMS.kv_lora_rank)
        k_b_proj, v_b_proj = split_kv_b_proj(kv_b_proj, _DIMS)
        module = torch.nn.Module()
        module.k_b_proj, module.v_b_proj = torch.nn.Parameter(k_b_proj), torch.nn.Parameter(v_b_proj)
        mapping = mappings._Ernie55KVUpMapping(
            "attention.k_b_proj", "kv_b_proj.weight", hf_dtype=_FP32, part="k", dims=_DIMS
        )
        assert torch.equal(mapping.megatron_to_hf(module.k_b_proj.detach(), module)["kv_b_proj.weight"], kv_b_proj)
        module.v_b_proj.is_gtp_weight_remat = True
        with pytest.raises(NotImplementedError, match="sharded v_b_proj"):
            mapping.megatron_to_hf(module.k_b_proj.detach(), module)

    def test_export_writes_the_checkpoint_dtype(self, pg_collection):
        module = torch.nn.Module()
        module.attn_sink = torch.nn.Parameter(torch.tensor([0.1, -2.5, 3.0], dtype=torch.float32))
        mapping = mappings._Ernie55TensorMapping("attn_sink", "softmax_offset", hf_dtype=torch.bfloat16)
        exported = mapping.megatron_to_hf(module.attn_sink.detach(), module)["softmax_offset"]
        assert exported.dtype == torch.bfloat16
        assert torch.equal(exported, module.attn_sink.detach().to(torch.bfloat16))


# ---------------------------------------------------------------------------------------------
# A toy model through Bridge's conversion
# ---------------------------------------------------------------------------------------------


class TestToyModel:
    def test_every_megatron_tensor_is_mapped_once(self, toy):
        registry = ernie55_mapping_registry(toy.config)
        tensors = _megatron_tensors(toy.model)
        unmapped = [name for name in tensors if registry.megatron_to_hf_lookup(name) is None]
        assert not unmapped, unmapped
        # The SD-MTP body is the decoder's last layers: its parameters appear once, under decoder names.
        assert check_ernie55_mtp_aliases(toy.model) > 0
        assert not [name for name in tensors if ".mtp_model_layer." in name and not name.endswith("expert_bias")]
        # Each HF tensor feeds one Megatron tensor, except kv_b_proj, which feeds k_b_proj and v_b_proj.
        consumers: dict[str, list[str]] = {}
        for name in tensors:
            hf_param = registry.megatron_to_hf_lookup(name).hf_param
            for hf_name in [hf_param] if isinstance(hf_param, str) else hf_param.values():
                consumers.setdefault(hf_name, []).append(name)
        shared = {hf_name: names for hf_name, names in consumers.items() if len(names) > 1}
        assert all(hf_name.endswith("kv_b_proj.weight") for hf_name in shared), shared
        assert all(
            sorted(name.rpartition(".")[2] for name in names) == ["k_b_proj", "v_b_proj"] for names in shared.values()
        )
        assert set(consumers) == set(toy.tensors)

    def test_load_overwrites_every_tensor(self, toy):
        loaded = _megatron_tensors(toy.model)
        assert loaded.keys() == toy.before.keys()
        unchanged = [name for name, tensor in loaded.items() if torch.equal(tensor, toy.before[name])]
        assert not unchanged, unchanged

    def test_load_places_every_tensor(self, toy):
        model, hf = toy.model, toy.tensors
        layout = Ernie55HFLayout.from_hf_config(toy.config)
        get = model.get_parameter

        def same(megatron: torch.Tensor, expected: torch.Tensor) -> None:
            assert megatron.dtype in (expected.dtype, torch.float32)
            assert torch.equal(megatron.detach(), expected.to(megatron.dtype))

        same(get("embedding.word_embeddings.weight"), hf["model.embed_tokens.weight"])
        same(get("output_layer.weight"), hf["lm_head.weight"])
        same(get("output_layer.multimax_ranges"), hf["model.lm_head.multimax_ranges"])
        same(get("output_layer.multimax_ts"), hf["model.lm_head.multimax_ts"])
        same(get("decoder.final_norm.weight"), hf["model.norm.weight"])
        same(get("decoder.hc_head_fn"), hf["model.mhc_contract.hc_head_fn"].T)
        same(get("decoder.hc_head_base"), hf["model.mhc_contract.hc_head_base"])
        for layer in range(layout.num_hidden_layers):
            src, attn, mlp = f"model.layers.{layer}", f"decoder.layers.{2 * layer}", f"decoder.layers.{2 * layer + 1}"
            for module, kind in ((attn, "self_attention_hyper_connection"), (mlp, "mlp_hyper_connection")):
                same(get(f"{module}.hyper_connection.mapping_proj.weight"), hf[f"{src}.{kind}.mapping_proj.weight"].T)
                for name in ("bias", "alpha_pre", "alpha_post", "alpha_res"):
                    same(get(f"{module}.hyper_connection.{name}"), hf[f"{src}.{kind}.{name}"])
            same(get(f"{attn}.inner_layer.input_layernorm.weight"), hf[f"{src}.input_layernorm.weight"])
            same(get(f"{mlp}.inner_layer.pre_mlp_layernorm.weight"), hf[f"{src}.post_attention_layernorm.weight"])
            sa, hf_attn = f"{attn}.inner_layer.self_attention", f"{src}.self_attn"
            core = f"{sa}.core_attention"
            same(get(f"{core}.vha_postmix_u"), hf[f"{hf_attn}.vha_postmix_U"])
            same(get(f"{core}.vha_postmix_v"), hf[f"{hf_attn}.vha_postmix_V"])
            same(get(f"{sa}.linear_proj.weight"), hf[f"{hf_attn}.o_proj.weight"])
            if layout.attention[layer] == "H":
                same(get(f"{sa}.linear_output_gate.weight"), hf[f"{hf_attn}.gate_proj.weight"])
                same(get(f"{sa}.linear_o_group_proj"), hf[f"{hf_attn}.linear_o_group_proj"])
                same(get(f"{core}.attn_sink"), hf[f"{hf_attn}.core_attention.attn_sink"])
                same(get(f"{core}.compressor.ape"), hf[f"{hf_attn}.core_attention.compressor.ape"])
                same(get(f"{core}.compressor.norm.weight"), hf[f"{hf_attn}.core_attention.compressor.norm.weight"])
            else:
                self._check_d_layer(model, sa, hf_attn, hf, toy.config)
            if layout.is_moe(layer):
                self._check_moe(model, f"{mlp}.inner_layer.mlp", f"{src}.block_sparse_moe", hf, toy.config)
            else:
                fc1 = torch.cat([hf[f"{src}.mlp.gate_proj.weight"], hf[f"{src}.mlp.up_proj.weight"]])
                same(get(f"{mlp}.inner_layer.mlp.linear_fc1.weight"), fc1)
                same(get(f"{mlp}.inner_layer.mlp.linear_fc2.weight"), hf[f"{src}.mlp.down_proj.weight"])
        mtp, src = "mtp.layers.0", f"model.layers.{layout.num_hidden_layers}"
        for name, hf_name in (("enorm", "enorm"), ("hnorm", "hnorm"), ("e_proj", "e_proj"), ("h_proj", "h_proj")):
            same(get(f"{mtp}.{name}.weight"), hf[f"{src}.{hf_name}.weight"])
        same(get(f"{mtp}.final_layernorm.weight"), hf[f"{src}.shared_head.norm.weight"])
        same(get(f"{mtp}.hc_head_fn"), hf[f"{src}.hc_head_fn"].T)
        body_router = model.get_submodule(f"{mtp}.mtp_model_layer.layers.1.inner_layer.mlp.router")
        decoder_router = model.get_submodule("decoder.layers.7.inner_layer.mlp.router")
        same(body_router.expert_bias, hf[f"{src}.block_sparse_moe.e_score_correction_bias"])
        same(decoder_router.expert_bias, hf["model.layers.3.block_sparse_moe.e_score_correction_bias"])
        assert body_router.weight is decoder_router.weight  # SD-MTP: shared parameters, own buffers

    @staticmethod
    def _check_d_layer(model, sa: str, hf_attn: str, hf: dict, config: dict) -> None:
        get = model.get_parameter
        heads, latent = config["hybrid_mla_num_attention_heads"], config["hybrid_mla_kv_lora_rank"]
        nope, rope, value = (
            config["hybrid_mla_qk_nope_head_dim"],
            config["hybrid_mla_qk_rope_head_dim"],
            config["hybrid_mla_v_head_dim"],
        )
        kv_b_proj = hf[f"{hf_attn}.kv_b_proj.weight"]
        k_b_proj, v_b_proj = get(f"{sa}.k_b_proj").detach(), get(f"{sa}.v_b_proj").detach()
        for head in range(heads):
            rows = kv_b_proj[head * (nope + value) : (head + 1) * (nope + value)]
            assert torch.equal(k_b_proj[head * latent : (head + 1) * latent], rows[:nope].T)
            assert torch.equal(v_b_proj[head * value : (head + 1) * value], rows[nope:])
        q_up, kv_down = hf[f"{hf_attn}.q_b_proj.weight"].clone(), hf[f"{hf_attn}.kv_a_proj_with_mqa.weight"].clone()
        if not config["mqa_latent_rope_adjacent_pairing"]:
            # megatron[2i] = hf[i], megatron[2i + 1] = hf[R/2 + i] inside each RoPE block.
            order = [index for pair in zip(range(rope // 2), range(rope // 2, rope)) for index in pair]
            for head in range(heads):
                start = head * (nope + rope) + nope
                q_up[start : start + rope] = hf[f"{hf_attn}.q_b_proj.weight"][[start + index for index in order]]
            kv_down[latent:] = hf[f"{hf_attn}.kv_a_proj_with_mqa.weight"][[latent + index for index in order]]
        assert torch.equal(get(f"{sa}.linear_q_up_proj.weight").detach(), q_up)
        assert torch.equal(get(f"{sa}.linear_kv_down_proj.weight").detach(), kv_down)
        assert torch.equal(get(f"{sa}.linear_q_down_proj.weight").detach(), hf[f"{hf_attn}.q_a_proj.weight"])
        assert torch.equal(get(f"{sa}.q_layernorm.weight").detach(), hf[f"{hf_attn}.q_a_layernorm.weight"])
        assert torch.equal(get(f"{sa}.kv_layernorm.weight").detach(), hf[f"{hf_attn}.kv_a_layernorm.weight"])
        core = f"{sa}.core_attention"
        assert torch.equal(get(f"{core}.linear_output_gate.weight").detach(), hf[f"{hf_attn}.gate_proj.weight"])
        sink = get(f"{core}.attn_sink").detach()
        assert sink.dtype == torch.float32
        assert torch.equal(sink, hf[f"{hf_attn}.core_attention.softmax_offset"].float())
        indexer, hf_indexer = f"{core}.indexer", f"{hf_attn}.core_attention.indexer"
        for name, hf_name in (
            ("linear_wq_b.weight", "wq_b.weight"),
            ("linear_wk.weight", "wk.weight"),
            ("linear_weights_proj.weight", "weights_proj.weight"),
            ("k_norm.weight", "k_norm.weight"),
            ("k_norm.bias", "k_norm.bias"),
        ):
            assert torch.equal(get(f"{indexer}.{name}").detach(), hf[f"{hf_indexer}.{hf_name}"])

    @staticmethod
    def _check_moe(model, mlp: str, hf_moe: str, hf: dict, config: dict) -> None:
        get = model.get_parameter
        for name, hf_name in (
            ("router.weight", "gate.weight"),
            ("router.weight_1", "gate.weight_1"),
            ("router.routed_scaling_factor_param", "gate.routed_scaling_factor_param"),
        ):
            assert get(f"{mlp}.{name}").dtype == torch.float32
            assert torch.equal(get(f"{mlp}.{name}").detach(), hf[f"{hf_moe}.{hf_name}"])
        assert torch.equal(model.get_buffer(f"{mlp}.router.expert_bias"), hf[f"{hf_moe}.e_score_correction_bias"])
        for expert in range(config["n_routed_experts"]):
            source = f"{hf_moe}.experts.{expert}"
            fc1 = torch.cat([hf[f"{source}.w1.weight"], hf[f"{source}.w3.weight"]])
            assert torch.equal(get(f"{mlp}.experts.local_experts.{expert}.linear_fc1.weight").detach(), fc1)
            assert torch.equal(
                get(f"{mlp}.experts.local_experts.{expert}.linear_fc2.weight").detach(), hf[f"{source}.w2.weight"]
            )
        shared_fc1 = torch.cat([hf[f"{hf_moe}.shared_experts.w1.weight"], hf[f"{hf_moe}.shared_experts.w3.weight"]])
        assert torch.equal(get(f"{mlp}.shared_experts.linear_fc1.weight").detach(), shared_fc1)
        assert torch.equal(
            get(f"{mlp}.shared_experts.linear_fc2.weight").detach(), hf[f"{hf_moe}.shared_experts.w2.weight"]
        )
        if config.get("moe_latent_size"):
            for name in ("fc1_latent_proj.weight", "fc2_latent_proj.weight"):
                assert get(f"{mlp}.{name}").dtype == torch.float32
                assert torch.equal(get(f"{mlp}.{name}").detach(), hf[f"{hf_moe}.{name}"])
            assert torch.equal(get(f"{mlp}.fc2_norm.weight").detach(), hf[f"{hf_moe}.latent_norm.weight"])

    def test_export_is_bitwise_equal_to_the_source(self, toy):
        _assert_bitwise_equal(toy.exported, toy.tensors)

    def test_reload_of_the_export_is_bitwise(self, toy, pg_collection):
        """A second model loads the export and exports it again unchanged (import -> export -> import -> export)."""
        _, megatron_overrides = _VARIANTS[toy.name]
        latent = contextlib.nullcontext() if not megatron_overrides else _latent_moe_without_te()
        with latent:
            model = _build_model(pg_collection, pattern=_PATTERN, seed=5, **_megatron_overrides(**megatron_overrides))
        bridge = _load(model, toy.config, toy.exported)
        _assert_bitwise_equal(_export(bridge, model, toy.config, toy.exported), toy.tensors)
        for name, tensor in _megatron_tensors(model).items():
            assert torch.equal(tensor, _megatron_tensors(toy.model)[name]), name


class TestImportChecks:
    """An import through :class:`Ernie55WeightConversionMixin` rejects a checkpoint that breaks the contract."""

    @staticmethod
    def _model(pg_collection):
        return _build_model(pg_collection, pattern=_PATTERN, seed=0, **_megatron_overrides())

    def test_rejects_a_tensor_no_mapping_reads(self, pg_collection):
        config = _toy_hf_config()
        tensors = dict(_toy_hf_tensors(config, seed=3))
        tensors["model.layers.0.mlp.shared_experts.w2.weight"] = torch.zeros(_HIDDEN, _MOE_FFN, dtype=_BF16)
        with pytest.raises(ValueError, match=r"1 that no mapping reads \['model\.layers\.0\.mlp\.shared_experts"):
            _load(self._model(pg_collection), config, tensors)

    def test_streaming_import_checks_the_checkpoint_too(self, pg_collection):
        config = _toy_hf_config()
        tensors = dict(_toy_hf_tensors(config, seed=3))
        tensors["model.layers.0.mlp.shared_experts.w2.weight"] = torch.zeros(_HIDDEN, _MOE_FFN, dtype=_BF16)
        bridge, model = _Ernie55TestBridge(), self._model(pg_collection)
        with pytest.raises(ValueError, match="that no mapping reads"):
            bridge.stream_weights_hf_to_megatron(_hf_pretrained(config, tensors), [model])
        # An external state dict has no checkpoint behind it to check.
        bridge.stream_weights_hf_to_megatron(_hf_pretrained(config, tensors), [model], hf_state_dict=tensors)

    def test_rejects_shared_experts_named_as_only_the_reference_reads_them(self, pg_collection):
        """Without the check the model would have no shared experts and their tensors would be skipped."""
        tensors = _toy_hf_tensors(_toy_hf_config(), seed=3)
        config = dict(_toy_hf_config(), num_shared_experts=1)
        del config["n_shared_experts"]
        with pytest.raises(ValueError, match="which the reference implementation reads"):
            _load(self._model(pg_collection), config, tensors)

    @pytest.mark.parametrize(
        "name, dtype",
        [
            ("model.layers.1.block_sparse_moe.gate.weight", torch.float64),  # keep-in-fp32 tensor stored in F64
            ("model.layers.3.self_attn.q_b_proj.weight", _FP32),  # checkpoint-dtype tensor stored in F32
            ("model.layers.3.block_sparse_moe.experts.5.w2.weight", _FP32),
            ("model.layers.3.block_sparse_moe.experts.6.w3.weight", _FP32),
        ],
    )
    def test_rejects_a_tensor_in_another_dtype(self, pg_collection, name, dtype):
        config = _toy_hf_config()
        tensors = dict(_toy_hf_tensors(config, seed=3))
        tensors[name] = tensors[name].to(dtype)
        with pytest.raises(TypeError, match=rf"{re.escape(name)} is {dtype} in the checkpoint"):
            _load(self._model(pg_collection), config, tensors)

    def test_bf16_keep_in_fp32_tensors_load_widened(self, pg_collection):
        """The released bf16 checkpoints store the keep-in-fp32 tensors in bf16 too; they load into fp32 exactly."""
        config = _toy_hf_config()
        tensors = dict(_toy_hf_tensors(config, seed=3))
        keep_fp32 = {name for name, dtype in ernie55_hf_param_dtypes(config).items() if dtype == _FP32}
        assert keep_fp32
        for name in keep_fp32:
            tensors[name] = tensors[name].to(_BF16)
        model = self._model(pg_collection)
        exported = _export(_load(model, config, tensors), model, config, tensors)
        for name in keep_fp32:
            assert exported[name].dtype == _FP32 and torch.equal(exported[name], tensors[name].float()), name

    def test_fp32_checkpoint_loads_into_the_bf16_model(self, pg_collection):
        """With ``dtype: float32`` every tensor is F32: the keep-in-fp32 ones load exactly, the others round once."""
        config = _toy_hf_config(dtype="float32")
        tensors = {}
        for name, (shape, _) in _toy_hf_spec(config).items():
            generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            tensors[name] = torch.randn(shape, generator=generator, dtype=torch.float64).float()
        model = self._model(pg_collection)
        exported = _export(_load(model, config, tensors), model, config, tensors)
        assert sorted(exported) == sorted(tensors)
        keep_fp32 = {name for name, dtype in ernie55_hf_param_dtypes(_toy_hf_config()).items() if dtype == _FP32}
        rounded = 0
        for name, source in tensors.items():
            assert exported[name].dtype == _FP32, name
            if name in keep_fp32:
                assert torch.equal(exported[name], source), name
            elif not torch.equal(exported[name], source):  # fp32 Megatron tensors (the D-layer sink) stay exact
                assert torch.equal(exported[name], source.to(_BF16).float()), name
                rounded += 1
        assert rounded > 0


# ---------------------------------------------------------------------------------------------
# Expert and pipeline parallelism on gloo ranks
# ---------------------------------------------------------------------------------------------


def _setup_rank(world_size: int, **parallelism):
    """A spawned rank: CPU stand-ins (as in ``test_ernie55_cp_model.py``) and Megatron-Core model parallelism."""
    from tests.unit_tests.models.ernie.test_ernie55_cp_model import setup_rank

    rank = setup_rank(world_size)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, **parallelism)
    return rank


def setup_ep_rank(world_size: int):
    """A spawned rank with EP = world size."""
    return _setup_rank(world_size, pipeline_model_parallel_size=1, expert_model_parallel_size=world_size)


def setup_pp_rank(world_size: int):
    """A spawned rank with PP = world size.

    At PP > 1 Megatron-Core moves the embedding copies of the first and the MTP stage to CUDA before it
    all-reduces them (``LanguageModule.setup_embeddings_and_output_layer``); on these CPU ranks
    ``Tensor.cuda`` keeps the tensor where it is.
    """
    torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
    return _setup_rank(world_size, pipeline_model_parallel_size=world_size, expert_model_parallel_size=1)


def case_expert_parallel_round_trip(rank) -> None:
    """Each EP rank loads its global experts; the export reassembles every expert on every rank."""
    config = _toy_hf_config()
    tensors = _toy_hf_tensors(config, seed=7)
    model = _build_model(
        ProcessGroupCollection.use_mpu_process_groups(), pattern=_PATTERN, seed=0, **_megatron_overrides()
    )
    bridge = _load(model, config, tensors)
    local = _EXPERTS // rank.world_size
    for layer, hf_layer in ((3, 1), (5, 2), (7, 3)):
        experts = model.get_submodule(f"decoder.layers.{layer}.inner_layer.mlp.experts.local_experts")
        assert len(experts) == local
        for index, expert in enumerate(experts):
            source = f"model.layers.{hf_layer}.block_sparse_moe.experts.{rank.rank * local + index}"
            fc1 = torch.cat([tensors[f"{source}.w1.weight"], tensors[f"{source}.w3.weight"]])
            assert torch.equal(expert.linear_fc1.weight.detach(), fc1), (layer, index)
            assert torch.equal(expert.linear_fc2.weight.detach(), tensors[f"{source}.w2.weight"]), (layer, index)
    _assert_bitwise_equal(_export(bridge, model, config, tensors), tensors)


def case_pipeline_parallel_round_trip(rank) -> None:
    """Two pipeline stages split at an ERNIE-layer boundary; MTP, the output layer and the final norm on the last.

    Every stage loads only its own layers (the embedding on both: the MTP stage keeps a copy), and the
    export broadcasts each tensor from its stage, the D layers' ``kv_b_proj`` from both K/V tensors there.
    """
    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55HybridModel
    from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import _config, _cpu_spec

    first, last = rank.rank == 0, rank.rank == rank.world_size - 1
    config = _toy_hf_config()
    tensors = _toy_hf_tensors(config, seed=11)
    megatron_config = _config(
        _PATTERN, pipeline_model_parallel_size=rank.world_size, pipeline_dtype=torch.float32, **_megatron_overrides()
    )
    model = Ernie55HybridModel(
        config=megatron_config,
        hybrid_stack_spec=_cpu_spec(),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=_PATTERN.replace("H-DE", "H-DE|", 1),
        position_embedding_type="none",
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        pre_process=first,
        post_process=last,
    )
    local_layers = [layer.layer_number - 1 for layer in model.decoder.layers]
    assert local_layers == ([0, 1, 2, 3] if first else [4, 5, 6, 7])
    assert hasattr(model, "output_layer") is last and getattr(model, "mtp_process", False) is last
    bridge = _load(model, config, tensors)
    # The second stage's first D layer is hybrid layer 6 (Hugging Face layer 3), local layer 2.
    if last:
        attention = model.decoder.layers[2].inner_layer.self_attention
        dims = Ernie55HFLayout.from_hf_config(config).d_dims
        k_b_proj, v_b_proj = split_kv_b_proj(tensors["model.layers.3.self_attn.kv_b_proj.weight"], dims)
        assert torch.equal(attention.k_b_proj.detach(), k_b_proj) and torch.equal(
            attention.v_b_proj.detach(), v_b_proj
        )
    _assert_bitwise_equal(_export(bridge, model, config, tensors), tensors)


_EP_CASES = ["case_expert_parallel_round_trip"]
_PP_CASES = ["case_pipeline_parallel_round_trip"]


def _assert_cases_pass(results: dict, cases: list[str], world_size: int) -> None:
    for case in cases:
        failures = {rank: result["error"] for rank, result in results[case].items() if not result["ok"]}
        assert len(results[case]) == world_size
        assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())


@pytest.mark.parametrize("world_size", [2, 4], ids=["ep2", "ep4"])
def test_expert_parallel_round_trip(world_size, tmp_path):
    results = run_cases(
        world_size=world_size,
        module_name=__name__,
        setup_name="setup_ep_rank",
        case_names=_EP_CASES,
        workdir=tmp_path / f"ep{world_size}",
    )
    _assert_cases_pass(results, _EP_CASES, world_size)


def test_pipeline_parallel_round_trip(tmp_path):
    results = run_cases(
        world_size=2, module_name=__name__, setup_name="setup_pp_rank", case_names=_PP_CASES, workdir=tmp_path / "pp2"
    )
    _assert_cases_pass(results, _PP_CASES, 2)
