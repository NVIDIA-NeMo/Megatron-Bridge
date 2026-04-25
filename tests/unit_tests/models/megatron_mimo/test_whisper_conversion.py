# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for the Whisper HF -> Megatron conversion script.

Covers the pure helpers (`_build_qkv_interleave_indices`, `_get_tp_concat_dim`)
and a round-trip of `convert_hf_whisper_to_megatron` against a mocked HF model
to exercise the QKV interleave + TP-shard pipeline without downloading weights.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# Loaded directly from the file: the example's `whisper/__init__.py` pulls in
# `whisper_model` / `whisper_layer_specs`, which require megatron.core extensions
# that aren't available in a CPU-only test environment.
CONVERTER_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "models"
    / "megatron_mimo"
    / "whisper"
    / "convert_hf_whisper_to_megatron.py"
)


@pytest.fixture(scope="module")
def converter():
    # `from transformers import WhisperModel` runs at import time; stub it.
    sys.modules.setdefault("transformers", MagicMock())
    spec = importlib.util.spec_from_file_location("whisper_converter_under_test", CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_hf_whisper_state_dict(num_layers, hidden_dim, ffn_dim, num_mel_bins, max_pos):
    """Build a state dict mimicking HF Whisper's encoder layout."""
    sd = {
        "encoder.conv1.weight": torch.randn(hidden_dim, num_mel_bins, 3),
        "encoder.conv1.bias": torch.randn(hidden_dim),
        "encoder.conv2.weight": torch.randn(hidden_dim, hidden_dim, 3),
        "encoder.conv2.bias": torch.randn(hidden_dim),
        "encoder.embed_positions.weight": torch.randn(max_pos, hidden_dim),
        "encoder.layer_norm.weight": torch.randn(hidden_dim),
        "encoder.layer_norm.bias": torch.randn(hidden_dim),
    }
    for i in range(num_layers):
        b = f"encoder.layers.{i}"
        sd[f"{b}.self_attn.q_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
        sd[f"{b}.self_attn.k_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
        sd[f"{b}.self_attn.v_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
        sd[f"{b}.self_attn.q_proj.bias"] = torch.randn(hidden_dim)
        # k_proj.bias intentionally absent (HF Whisper hardcodes bias=False)
        sd[f"{b}.self_attn.v_proj.bias"] = torch.randn(hidden_dim)
        sd[f"{b}.self_attn.out_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
        sd[f"{b}.self_attn.out_proj.bias"] = torch.randn(hidden_dim)
        sd[f"{b}.self_attn_layer_norm.weight"] = torch.randn(hidden_dim)
        sd[f"{b}.self_attn_layer_norm.bias"] = torch.randn(hidden_dim)
        sd[f"{b}.fc1.weight"] = torch.randn(ffn_dim, hidden_dim)
        sd[f"{b}.fc1.bias"] = torch.randn(ffn_dim)
        sd[f"{b}.fc2.weight"] = torch.randn(hidden_dim, ffn_dim)
        sd[f"{b}.fc2.bias"] = torch.randn(hidden_dim)
        sd[f"{b}.final_layer_norm.weight"] = torch.randn(hidden_dim)
        sd[f"{b}.final_layer_norm.bias"] = torch.randn(hidden_dim)
    # A decoder weight to verify the encoder-only filter skips it.
    sd["decoder.layers.0.self_attn.q_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
    return sd


def _make_mock_hf_model(state_dict, *, hidden_dim, num_heads, ffn_dim, num_layers, num_mel_bins, max_pos):
    config = SimpleNamespace(
        d_model=hidden_dim,
        encoder_attention_heads=num_heads,
        encoder_ffn_dim=ffn_dim,
        encoder_layers=num_layers,
        num_mel_bins=num_mel_bins,
        max_source_positions=max_pos,
    )
    mock = MagicMock()
    mock.state_dict.return_value = state_dict
    mock.config = config
    return mock


@pytest.mark.unit
class TestQKVInterleaveIndices:
    def test_indices_have_correct_length(self, converter):
        idx = converter._build_qkv_interleave_indices(hidden_dim=8, num_heads=4)
        assert idx.shape == (3 * 8,)

    def test_indices_are_a_permutation_of_3_hidden(self, converter):
        idx = converter._build_qkv_interleave_indices(hidden_dim=12, num_heads=3)
        assert torch.equal(torch.sort(idx).values, torch.arange(3 * 12))

    def test_layout_groups_qkv_per_head(self, converter):
        """Fused tensor must be [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]."""
        hidden_dim, num_heads = 16, 4
        head_dim = hidden_dim // num_heads
        # Use distinguishable Q/K/V so we can read off the layout.
        q = torch.arange(hidden_dim).float().unsqueeze(1).expand(-1, hidden_dim).contiguous()
        k = q + 1000
        v = q + 2000
        idx = converter._build_qkv_interleave_indices(hidden_dim, num_heads)
        fused = torch.cat([q, k, v], dim=0)[idx]

        for h in range(num_heads):
            base = h * 3 * head_dim
            assert torch.equal(fused[base : base + head_dim], q[h * head_dim : (h + 1) * head_dim])
            assert torch.equal(fused[base + head_dim : base + 2 * head_dim], k[h * head_dim : (h + 1) * head_dim])
            assert torch.equal(fused[base + 2 * head_dim : base + 3 * head_dim], v[h * head_dim : (h + 1) * head_dim])

    def test_inverse_permutation_recovers_original(self, converter):
        hidden_dim, num_heads = 24, 6
        q, k, v = torch.randn(hidden_dim, 4), torch.randn(hidden_dim, 4), torch.randn(hidden_dim, 4)
        idx = converter._build_qkv_interleave_indices(hidden_dim, num_heads)
        fused = torch.cat([q, k, v], dim=0)[idx]

        inverse = torch.empty_like(idx)
        inverse[idx] = torch.arange(idx.numel())
        assert torch.equal(fused[inverse], torch.cat([q, k, v], dim=0))


@pytest.mark.unit
class TestGetTpConcatDim:
    @pytest.mark.parametrize(
        "name,expected",
        [
            # Column-parallel (chunk on output dim)
            ("decoder.layers.0.self_attention.linear_qkv.weight", 0),
            ("decoder.layers.5.self_attention.linear_qkv.bias", 0),
            ("decoder.layers.0.mlp.linear_fc1.weight", 0),
            ("decoder.layers.0.mlp.linear_fc1.bias", 0),
            # Row-parallel (chunk on input dim)
            ("decoder.layers.0.self_attention.linear_proj.weight", 1),
            ("decoder.layers.0.mlp.linear_fc2.weight", 1),
            # Replicated tensors
            ("decoder.layers.0.self_attention.linear_proj.bias", None),
            ("decoder.layers.0.mlp.linear_fc2.bias", None),
            ("decoder.layers.0.self_attention.linear_qkv.layer_norm_weight", None),
            ("decoder.layers.0.mlp.linear_fc1.layer_norm_bias", None),
            ("ln_post.weight", None),
            ("ln_post.bias", None),
            ("conv1.weight", None),
            ("conv2.bias", None),
            ("position_embeddings.weight", None),
        ],
    )
    def test_returns_expected_dim(self, converter, name, expected):
        assert converter._get_tp_concat_dim(name) == expected


@pytest.mark.unit
class TestEndToEndConversion:
    """Run the converter against a mocked HF model and inspect the saved checkpoint."""

    NUM_LAYERS = 2
    HIDDEN_DIM = 16
    FFN_DIM = 32
    NUM_HEADS = 4
    NUM_MEL_BINS = 8
    MAX_POS = 32

    def _convert(self, converter, output_path, *, tp_size=1, use_te=True, mutate_state_dict=None):
        torch.manual_seed(0)
        sd = _make_hf_whisper_state_dict(
            self.NUM_LAYERS, self.HIDDEN_DIM, self.FFN_DIM, self.NUM_MEL_BINS, self.MAX_POS
        )
        if mutate_state_dict is not None:
            mutate_state_dict(sd)
        mock_hf = _make_mock_hf_model(
            sd,
            hidden_dim=self.HIDDEN_DIM,
            num_heads=self.NUM_HEADS,
            ffn_dim=self.FFN_DIM,
            num_layers=self.NUM_LAYERS,
            num_mel_bins=self.NUM_MEL_BINS,
            max_pos=self.MAX_POS,
        )
        with patch.object(converter, "WhisperModel") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_hf
            converter.convert_hf_whisper_to_megatron(
                hf_model_name="dummy",
                output_path=str(output_path),
                tensor_parallel_size=tp_size,
                use_te=use_te,
            )
        return sd

    def _load_shard(self, output_path, tp_rank):
        saved = torch.load(
            Path(output_path) / f"tp_rank_{tp_rank:02d}" / "model_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        return {k: v for k, v in saved["model"].items() if v is not None}

    def test_qkv_weight_round_trip(self, converter, tmp_path):
        """Each per-head slice of the fused QKV weight matches the corresponding HF Q/K/V."""
        sd = self._convert(converter, tmp_path)
        out = self._load_shard(tmp_path, 0)
        head_dim = self.HIDDEN_DIM // self.NUM_HEADS

        for layer in range(self.NUM_LAYERS):
            qkv = out[f"decoder.layers.{layer}.self_attention.linear_qkv.weight"]
            assert qkv.shape == (3 * self.HIDDEN_DIM, self.HIDDEN_DIM)
            q_in = sd[f"encoder.layers.{layer}.self_attn.q_proj.weight"]
            k_in = sd[f"encoder.layers.{layer}.self_attn.k_proj.weight"]
            v_in = sd[f"encoder.layers.{layer}.self_attn.v_proj.weight"]
            for h in range(self.NUM_HEADS):
                base = h * 3 * head_dim
                hs = slice(h * head_dim, (h + 1) * head_dim)
                assert torch.allclose(qkv[base : base + head_dim], q_in[hs])
                assert torch.allclose(qkv[base + head_dim : base + 2 * head_dim], k_in[hs])
                assert torch.allclose(qkv[base + 2 * head_dim : base + 3 * head_dim], v_in[hs])

    def test_qkv_bias_k_portion_is_zero(self, converter, tmp_path):
        """HF Whisper has no k_proj.bias, so the K slice of the fused bias must be zero."""
        self._convert(converter, tmp_path)
        out = self._load_shard(tmp_path, 0)
        head_dim = self.HIDDEN_DIM // self.NUM_HEADS

        for layer in range(self.NUM_LAYERS):
            qkv_bias = out[f"decoder.layers.{layer}.self_attention.linear_qkv.bias"]
            assert qkv_bias.shape == (3 * self.HIDDEN_DIM,)
            for h in range(self.NUM_HEADS):
                base = h * 3 * head_dim
                k_slice = qkv_bias[base + head_dim : base + 2 * head_dim]
                assert torch.all(k_slice == 0), f"K bias for head {h} should be zero"

    def test_decoder_keys_are_skipped(self, converter, tmp_path):
        self._convert(converter, tmp_path)
        out = self._load_shard(tmp_path, 0)
        # The encoder-only filter strips anything not under encoder.*; nothing should map to
        # decoder.layers.0 from a non-encoder source. (All converter outputs are *renamed*
        # under decoder.layers.* — that's Megatron's encoder block name — so we just verify
        # the converter produced exactly one set of layers, not an extra one from the seeded
        # decoder weight.)
        per_layer_qkv = [k for k in out if k.endswith("self_attention.linear_qkv.weight")]
        assert len(per_layer_qkv) == self.NUM_LAYERS

    def test_tp_sharding_splits_column_parallel_on_dim_0(self, converter, tmp_path):
        tp_size = 2
        self._convert(converter, tmp_path, tp_size=tp_size)
        out0 = self._load_shard(tmp_path, 0)
        out1 = self._load_shard(tmp_path, 1)

        for layer in range(self.NUM_LAYERS):
            qkv_key = f"decoder.layers.{layer}.self_attention.linear_qkv.weight"
            assert out0[qkv_key].shape == (3 * self.HIDDEN_DIM // tp_size, self.HIDDEN_DIM)
            assert out1[qkv_key].shape == (3 * self.HIDDEN_DIM // tp_size, self.HIDDEN_DIM)
            # Concatenating shards on dim 0 reconstructs the full fused tensor.
            full = torch.cat([out0[qkv_key], out1[qkv_key]], dim=0)
            assert full.shape == (3 * self.HIDDEN_DIM, self.HIDDEN_DIM)

            fc2_key = f"decoder.layers.{layer}.mlp.linear_fc2.weight"
            assert out0[fc2_key].shape == (self.HIDDEN_DIM, self.FFN_DIM // tp_size)
            assert out1[fc2_key].shape == (self.HIDDEN_DIM, self.FFN_DIM // tp_size)

    def test_replicated_tensors_match_across_ranks(self, converter, tmp_path):
        tp_size = 2
        self._convert(converter, tmp_path, tp_size=tp_size)
        out0 = self._load_shard(tmp_path, 0)
        out1 = self._load_shard(tmp_path, 1)

        for key in ("conv1.weight", "ln_post.bias", "position_embeddings.weight"):
            assert torch.equal(out0[key], out1[key])

    def test_te_extra_state_placeholders_present(self, converter, tmp_path):
        """TE specs need _extra_state keys (with None values) for FP8 compatibility."""
        self._convert(converter, tmp_path, use_te=True)
        saved = torch.load(tmp_path / "tp_rank_00" / "model_weights.pt", map_location="cpu", weights_only=True)
        all_keys = set(saved["model"].keys())
        for layer in range(self.NUM_LAYERS):
            for sub in ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"):
                expected = (
                    f"decoder.layers.{layer}.self_attention.{sub}._extra_state"
                    if sub in ("linear_qkv", "linear_proj")
                    else f"decoder.layers.{layer}.mlp.{sub}._extra_state"
                )
                assert expected in all_keys, f"missing TE _extra_state placeholder: {expected}"

    def test_layernorm_naming_changes_with_use_te_flag(self, converter, tmp_path):
        # use_te=True fuses the layernorm into linear_qkv / linear_fc1
        self._convert(converter, tmp_path / "te", use_te=True)
        te_keys = set(self._load_shard(tmp_path / "te", 0).keys())
        assert "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight" in te_keys
        assert "decoder.layers.0.input_layernorm.weight" not in te_keys

        # use_te=False uses standalone input_layernorm / pre_mlp_layernorm
        self._convert(converter, tmp_path / "local", use_te=False)
        local_keys = set(self._load_shard(tmp_path / "local", 0).keys())
        assert "decoder.layers.0.input_layernorm.weight" in local_keys
        assert "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight" not in local_keys

    def test_unexpected_k_proj_bias_triggers_assertion(self, converter, tmp_path):
        """A future Whisper variant adding k_proj.bias must fail loudly, not get silently zeroed."""

        def add_k_bias(sd):
            sd["encoder.layers.0.self_attn.k_proj.bias"] = torch.randn(self.HIDDEN_DIM)

        with pytest.raises(AssertionError, match="Unexpected k_proj bias"):
            self._convert(converter, tmp_path, mutate_state_dict=add_k_bias)
