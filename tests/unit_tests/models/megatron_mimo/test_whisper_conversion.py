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

    def test_tp1_baseline_full_shapes(self, converter, tmp_path):
        """TP=1 must store unsharded shapes for all per-layer linear tensors."""
        self._convert(converter, tmp_path, tp_size=1)
        out = self._load_shard(tmp_path, 0)
        for layer in range(self.NUM_LAYERS):
            b = f"decoder.layers.{layer}"
            assert out[f"{b}.self_attention.linear_qkv.weight"].shape == (3 * self.HIDDEN_DIM, self.HIDDEN_DIM)
            assert out[f"{b}.self_attention.linear_qkv.bias"].shape == (3 * self.HIDDEN_DIM,)
            assert out[f"{b}.self_attention.linear_proj.weight"].shape == (self.HIDDEN_DIM, self.HIDDEN_DIM)
            assert out[f"{b}.mlp.linear_fc1.weight"].shape == (self.FFN_DIM, self.HIDDEN_DIM)
            assert out[f"{b}.mlp.linear_fc2.weight"].shape == (self.HIDDEN_DIM, self.FFN_DIM)

    def test_tp4_reassembly_recovers_tp1_reference(self, converter, tmp_path):
        """Concatenating all 4 shards on the chunk dim recovers the TP=1 unsharded ref."""
        ref_dir = tmp_path / "ref"
        tp4_dir = tmp_path / "tp4"
        # _convert reseeds torch with manual_seed(0) before each call, so the
        # mocked HF state dict is identical across both invocations.
        self._convert(converter, ref_dir, tp_size=1)
        self._convert(converter, tp4_dir, tp_size=4)
        ref = self._load_shard(ref_dir, 0)
        shards = [self._load_shard(tp4_dir, r) for r in range(4)]

        for layer in range(self.NUM_LAYERS):
            b = f"decoder.layers.{layer}"
            for key, dim in [
                (f"{b}.self_attention.linear_qkv.weight", 0),
                (f"{b}.self_attention.linear_qkv.bias", 0),
                (f"{b}.mlp.linear_fc1.weight", 0),
                (f"{b}.mlp.linear_fc1.bias", 0),
                (f"{b}.self_attention.linear_proj.weight", 1),
                (f"{b}.mlp.linear_fc2.weight", 1),
            ]:
                merged = torch.cat([s[key] for s in shards], dim=dim)
                assert torch.equal(merged, ref[key]), f"TP=4 reassembly mismatch for {key}"

    def test_replicated_tensor_round_trip(self, converter, tmp_path):
        """conv1/conv2/position_embeddings/ln_post values match the HF source verbatim (post-fp32)."""
        sd = self._convert(converter, tmp_path)
        out = self._load_shard(tmp_path, 0)
        assert torch.equal(out["conv1.weight"], sd["encoder.conv1.weight"].float())
        assert torch.equal(out["conv1.bias"], sd["encoder.conv1.bias"].float())
        assert torch.equal(out["conv2.weight"], sd["encoder.conv2.weight"].float())
        assert torch.equal(out["conv2.bias"], sd["encoder.conv2.bias"].float())
        assert torch.equal(out["position_embeddings.weight"], sd["encoder.embed_positions.weight"].float())
        assert torch.equal(out["ln_post.weight"], sd["encoder.layer_norm.weight"].float())
        assert torch.equal(out["ln_post.bias"], sd["encoder.layer_norm.bias"].float())

    def test_per_layer_non_qkv_round_trip_te(self, converter, tmp_path):
        """out_proj/fc1/fc2 and TE-fused layernorms preserve HF values bit-for-bit at TP=1."""
        sd = self._convert(converter, tmp_path, use_te=True)
        out = self._load_shard(tmp_path, 0)
        for layer in range(self.NUM_LAYERS):
            b = f"decoder.layers.{layer}"
            hf = f"encoder.layers.{layer}"
            assert torch.equal(
                out[f"{b}.self_attention.linear_proj.weight"], sd[f"{hf}.self_attn.out_proj.weight"].float()
            )
            assert torch.equal(
                out[f"{b}.self_attention.linear_proj.bias"], sd[f"{hf}.self_attn.out_proj.bias"].float()
            )
            assert torch.equal(out[f"{b}.mlp.linear_fc1.weight"], sd[f"{hf}.fc1.weight"].float())
            assert torch.equal(out[f"{b}.mlp.linear_fc1.bias"], sd[f"{hf}.fc1.bias"].float())
            assert torch.equal(out[f"{b}.mlp.linear_fc2.weight"], sd[f"{hf}.fc2.weight"].float())
            assert torch.equal(out[f"{b}.mlp.linear_fc2.bias"], sd[f"{hf}.fc2.bias"].float())
            assert torch.equal(
                out[f"{b}.self_attention.linear_qkv.layer_norm_weight"],
                sd[f"{hf}.self_attn_layer_norm.weight"].float(),
            )
            assert torch.equal(
                out[f"{b}.self_attention.linear_qkv.layer_norm_bias"],
                sd[f"{hf}.self_attn_layer_norm.bias"].float(),
            )
            assert torch.equal(
                out[f"{b}.mlp.linear_fc1.layer_norm_weight"],
                sd[f"{hf}.final_layer_norm.weight"].float(),
            )
            assert torch.equal(
                out[f"{b}.mlp.linear_fc1.layer_norm_bias"],
                sd[f"{hf}.final_layer_norm.bias"].float(),
            )

    def test_layernorm_round_trip_use_te_false(self, converter, tmp_path):
        """With use_te=False the standalone input_layernorm / pre_mlp_layernorm preserve HF values."""
        sd = self._convert(converter, tmp_path, use_te=False)
        out = self._load_shard(tmp_path, 0)
        for layer in range(self.NUM_LAYERS):
            b = f"decoder.layers.{layer}"
            hf = f"encoder.layers.{layer}"
            assert torch.equal(out[f"{b}.input_layernorm.weight"], sd[f"{hf}.self_attn_layer_norm.weight"].float())
            assert torch.equal(out[f"{b}.input_layernorm.bias"], sd[f"{hf}.self_attn_layer_norm.bias"].float())
            assert torch.equal(out[f"{b}.pre_mlp_layernorm.weight"], sd[f"{hf}.final_layer_norm.weight"].float())
            assert torch.equal(out[f"{b}.pre_mlp_layernorm.bias"], sd[f"{hf}.final_layer_norm.bias"].float())

    def test_output_dir_auto_created_for_nested_path(self, converter, tmp_path):
        nested = tmp_path / "missing" / "nested" / "out"
        self._convert(converter, nested, tp_size=2)
        assert (nested / "tp_rank_00" / "model_weights.pt").is_file()
        assert (nested / "tp_rank_01" / "model_weights.pt").is_file()

    def test_determinism_two_runs_produce_identical_shards(self, converter, tmp_path):
        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        self._convert(converter, a_dir, tp_size=2)
        self._convert(converter, b_dir, tp_size=2)
        for rank in (0, 1):
            a = self._load_shard(a_dir, rank)
            b = self._load_shard(b_dir, rank)
            assert a.keys() == b.keys()
            for k in a:
                assert torch.equal(a[k], b[k]), f"rank {rank} key {k} differs between runs"

    def test_bf16_input_state_dict_saved_as_fp32(self, converter, tmp_path):
        """The converter casts every saved tensor to fp32 regardless of HF dtype."""

        def to_bf16(sd):
            for k in list(sd):
                sd[k] = sd[k].to(torch.bfloat16)

        self._convert(converter, tmp_path, mutate_state_dict=to_bf16)
        out = self._load_shard(tmp_path, 0)
        for key, tensor in out.items():
            assert tensor.dtype == torch.float32, f"{key} dtype is {tensor.dtype}, expected float32"

    def test_decoder_filter_skips_multiple_decoder_keys(self, converter, tmp_path):
        """Anything not under encoder.* must not leak into the converted checkpoint."""

        def add_decoder_garbage(sd):
            sd["decoder.embed_tokens.weight"] = torch.randn(64, self.HIDDEN_DIM)
            sd["decoder.layer_norm.weight"] = torch.randn(self.HIDDEN_DIM)
            sd["decoder.layers.5.fc1.weight"] = torch.randn(self.FFN_DIM, self.HIDDEN_DIM)
            sd["proj_out.weight"] = torch.randn(50, self.HIDDEN_DIM)

        self._convert(converter, tmp_path, mutate_state_dict=add_decoder_garbage)
        out = self._load_shard(tmp_path, 0)
        # Per-layer keys cap at NUM_LAYERS — the seeded decoder.layers.5 must not leak through.
        per_layer = {int(k.split(".")[2]) for k in out if k.startswith("decoder.layers.")}
        assert per_layer == set(range(self.NUM_LAYERS))
        assert "embed_tokens.weight" not in out
        assert "proj_out.weight" not in out

    def test_layer_count_matches_num_layers(self, converter, tmp_path):
        """Exactly NUM_LAYERS of each per-layer key — no off-by-one in the layer loop."""
        self._convert(converter, tmp_path, tp_size=1)
        out = self._load_shard(tmp_path, 0)
        for tail in (
            "self_attention.linear_qkv.weight",
            "self_attention.linear_qkv.bias",
            "self_attention.linear_proj.weight",
            "self_attention.linear_proj.bias",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc2.weight",
        ):
            keys = [k for k in out if k.endswith(tail)]
            assert len(keys) == self.NUM_LAYERS, f"{tail}: got {len(keys)}, want {self.NUM_LAYERS}"

    def test_linear_qkv_bias_shards_along_dim_0(self, converter, tmp_path):
        tp_size = 2
        self._convert(converter, tmp_path, tp_size=tp_size)
        out0 = self._load_shard(tmp_path, 0)
        out1 = self._load_shard(tmp_path, 1)
        for layer in range(self.NUM_LAYERS):
            key = f"decoder.layers.{layer}.self_attention.linear_qkv.bias"
            assert out0[key].shape == (3 * self.HIDDEN_DIM // tp_size,)
            assert out1[key].shape == (3 * self.HIDDEN_DIM // tp_size,)
            assert torch.cat([out0[key], out1[key]], dim=0).shape == (3 * self.HIDDEN_DIM,)

    def test_row_parallel_biases_are_replicated(self, converter, tmp_path):
        """linear_proj.bias and linear_fc2.bias are not sharded (replicated across ranks)."""
        tp_size = 2
        self._convert(converter, tmp_path, tp_size=tp_size)
        out0 = self._load_shard(tmp_path, 0)
        out1 = self._load_shard(tmp_path, 1)
        for layer in range(self.NUM_LAYERS):
            for key in (
                f"decoder.layers.{layer}.self_attention.linear_proj.bias",
                f"decoder.layers.{layer}.mlp.linear_fc2.bias",
            ):
                assert out0[key].shape == (self.HIDDEN_DIM,)
                assert torch.equal(out0[key], out1[key])


@pytest.mark.unit
class TestGetTpConcatDimExtras:
    """Cases not covered by the original parametrize block."""

    @pytest.mark.parametrize(
        "name",
        [
            "decoder.layers.0.self_attention.linear_qkv._extra_state",
            "decoder.layers.0.self_attention.linear_proj._extra_state",
            "decoder.layers.0.mlp.linear_fc1._extra_state",
            "decoder.layers.0.mlp.linear_fc2._extra_state",
        ],
    )
    def test_extra_state_keys_are_replicated(self, converter, name):
        assert converter._get_tp_concat_dim(name) is None

    @pytest.mark.parametrize(
        "name",
        [
            "totally.unrelated.tensor",
            "model.bias",
            "decoder.layers.0.unknown.weight",
            "",
        ],
    )
    def test_unrelated_names_are_replicated(self, converter, name):
        assert converter._get_tp_concat_dim(name) is None


# ---------------------------------------------------------------------------
# Loader tests: load_megatron_whisper_weights is the inverse of the converter.
# A real Megatron WhisperEncoder requires megatron.core extensions that aren't
# always available in CPU-only test envs, so we use a stand-in that records
# load_state_dict() calls.
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Stand-in for nn.Module that records load_state_dict() and reports configurable keys."""

    def __init__(self, *, missing=None, unexpected=None):
        self._missing = list(missing) if missing else []
        self._unexpected = list(unexpected) if unexpected else []
        self.received = None

    def load_state_dict(self, state_dict, strict=True):
        self.received = state_dict
        return SimpleNamespace(missing_keys=self._missing, unexpected_keys=self._unexpected)


def _run_conversion(converter, output_path, *, tp_size=1, use_te=True):
    """Module-level wrapper sharing config with TestEndToEndConversion."""
    return TestEndToEndConversion()._convert(converter, output_path, tp_size=tp_size, use_te=use_te)


@pytest.mark.unit
class TestLoadMegatronWhisperWeights:
    def test_matching_tp_loads_single_rank_and_filters_none_values(self, converter, tmp_path):
        _run_conversion(converter, tmp_path, tp_size=2)
        model = _RecordingModel()
        converter.load_megatron_whisper_weights(model, str(tmp_path), tp_rank=1, tp_size=2)
        assert model.received is not None
        # _extra_state placeholders are saved as None and must be stripped before load_state_dict.
        assert not any(v is None for v in model.received.values())
        # Rank 1 holds the second half of column-parallel tensors.
        qkv = model.received["decoder.layers.0.self_attention.linear_qkv.weight"]
        assert qkv.shape == (
            3 * TestEndToEndConversion.HIDDEN_DIM // 2,
            TestEndToEndConversion.HIDDEN_DIM,
        )

    def test_reconcile_ckpt_tp2_into_model_tp1_concatenates(self, converter, tmp_path):
        """Loading a TP=2 checkpoint into a TP=1 model must reconstruct the unsharded tensors."""
        ref_dir = tmp_path / "ref"
        ckpt_dir = tmp_path / "ckpt"
        _run_conversion(converter, ref_dir, tp_size=1)
        _run_conversion(converter, ckpt_dir, tp_size=2)

        ref = torch.load(ref_dir / "tp_rank_00" / "model_weights.pt", map_location="cpu", weights_only=True)
        ref_full = {k: v for k, v in ref["model"].items() if v is not None}

        model = _RecordingModel()
        converter.load_megatron_whisper_weights(model, str(ckpt_dir), tp_rank=0, tp_size=1)
        for key, ref_tensor in ref_full.items():
            assert torch.equal(model.received[key], ref_tensor), f"merged {key} differs from TP=1 ref"

    def test_reconcile_ckpt_tp4_into_model_tp2_resplits(self, converter, tmp_path):
        ref_dir = tmp_path / "ref_tp2"
        ckpt_dir = tmp_path / "ckpt_tp4"
        _run_conversion(converter, ref_dir, tp_size=2)
        _run_conversion(converter, ckpt_dir, tp_size=4)

        for tp_rank in range(2):
            model = _RecordingModel()
            converter.load_megatron_whisper_weights(model, str(ckpt_dir), tp_rank=tp_rank, tp_size=2)
            ref = torch.load(
                ref_dir / f"tp_rank_{tp_rank:02d}" / "model_weights.pt",
                map_location="cpu",
                weights_only=True,
            )
            ref_dict = {k: v for k, v in ref["model"].items() if v is not None}
            for key, ref_tensor in ref_dict.items():
                assert torch.equal(model.received[key], ref_tensor), (
                    f"reconciled rank {tp_rank} {key} differs from native TP=2"
                )

    def test_empty_ckpt_dir_raises(self, converter, tmp_path):
        with pytest.raises(FileNotFoundError):
            converter.load_megatron_whisper_weights(_RecordingModel(), str(tmp_path), tp_rank=0, tp_size=1)

    def test_unexpected_extra_state_keys_are_tolerated(self, converter, tmp_path):
        _run_conversion(converter, tmp_path, tp_size=1)
        model = _RecordingModel(
            unexpected=["decoder.layers.0.self_attention.linear_qkv._extra_state"],
        )
        converter.load_megatron_whisper_weights(model, str(tmp_path), tp_rank=0, tp_size=1)

    def test_other_unexpected_keys_raise(self, converter, tmp_path):
        _run_conversion(converter, tmp_path, tp_size=1)
        model = _RecordingModel(unexpected=["totally.unrelated.tensor"])
        with pytest.raises(RuntimeError, match="State dict mismatch"):
            converter.load_megatron_whisper_weights(model, str(tmp_path), tp_rank=0, tp_size=1)

    def test_missing_keys_raise(self, converter, tmp_path):
        _run_conversion(converter, tmp_path, tp_size=1)
        model = _RecordingModel(missing=["decoder.layers.0.self_attention.linear_qkv.weight"])
        with pytest.raises(RuntimeError, match="State dict mismatch"):
            converter.load_megatron_whisper_weights(model, str(tmp_path), tp_rank=0, tp_size=1)

    def test_missing_only_extra_state_is_tolerated(self, converter, tmp_path):
        _run_conversion(converter, tmp_path, tp_size=1)
        model = _RecordingModel(missing=["foo._extra_state", "bar._extra_state"])
        converter.load_megatron_whisper_weights(model, str(tmp_path), tp_rank=0, tp_size=1)


@pytest.mark.unit
class TestVerifyConversionShapeChecker:
    """Tests for verify_conversion(), the in-module shape sanity-checker."""

    def _convert_with_mock(self, converter, output, *, tp_size, use_te=True):
        """Convert and return the same mocked HF model so verify_conversion sees matching config."""
        torch.manual_seed(0)
        sd = _make_hf_whisper_state_dict(
            TestEndToEndConversion.NUM_LAYERS,
            TestEndToEndConversion.HIDDEN_DIM,
            TestEndToEndConversion.FFN_DIM,
            TestEndToEndConversion.NUM_MEL_BINS,
            TestEndToEndConversion.MAX_POS,
        )
        mock_hf = _make_mock_hf_model(
            sd,
            hidden_dim=TestEndToEndConversion.HIDDEN_DIM,
            num_heads=TestEndToEndConversion.NUM_HEADS,
            ffn_dim=TestEndToEndConversion.FFN_DIM,
            num_layers=TestEndToEndConversion.NUM_LAYERS,
            num_mel_bins=TestEndToEndConversion.NUM_MEL_BINS,
            max_pos=TestEndToEndConversion.MAX_POS,
        )
        with patch.object(converter, "WhisperModel") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_hf
            converter.convert_hf_whisper_to_megatron(
                hf_model_name="dummy",
                output_path=str(output),
                tensor_parallel_size=tp_size,
                use_te=use_te,
            )
        return mock_hf

    def test_healthy_checkpoint_passes(self, converter, tmp_path):
        mock_hf = self._convert_with_mock(converter, tmp_path, tp_size=2)
        with patch.object(converter, "WhisperModel") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_hf
            assert converter.verify_conversion(str(tmp_path), tensor_parallel_size=2) is True

    def test_corrupt_shape_fails(self, converter, tmp_path):
        mock_hf = self._convert_with_mock(converter, tmp_path, tp_size=1)
        path = tmp_path / "tp_rank_00" / "model_weights.pt"
        bad = torch.load(path, map_location="cpu", weights_only=True)
        bad["model"]["decoder.layers.0.self_attention.linear_qkv.weight"] = torch.zeros(7, 7)
        torch.save(bad, path)
        with patch.object(converter, "WhisperModel") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_hf
            assert converter.verify_conversion(str(tmp_path), tensor_parallel_size=1) is False

    def test_missing_rank_file_fails(self, converter, tmp_path):
        mock_hf = self._convert_with_mock(converter, tmp_path, tp_size=2)
        (tmp_path / "tp_rank_01" / "model_weights.pt").unlink()
        with patch.object(converter, "WhisperModel") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_hf
            assert converter.verify_conversion(str(tmp_path), tensor_parallel_size=2) is False
