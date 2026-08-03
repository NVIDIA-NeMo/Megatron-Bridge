from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge
from megatron.bridge.models.nemotronh_puzzle.nemotron_h_puzzle_bridge import (
    NemotronHPuzzleBridge,
    _block_to_override,
    _blocks_to_pattern,
)


def _puzzle_hf_config_dict() -> dict:
    # 4-layer toy puzzle backbone (mamba/moe/mamba/moe) with a 1-depth, 2-position
    # MTP block (attention/moe). MoE varies per position — exactly what the sparse
    # per_layer_config_overrides / mtp_per_layer_config_overrides are supposed to
    # carry through provider_bridge and back through megatron_to_hf_config.
    return {
        "architectures": ["NemotronHPuzzleForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_nemotron_h_puzzle.NemotronHPuzzleConfig",
            "AutoModelForCausalLM": "modeling_nemotron_h_puzzle.NemotronHPuzzleForCausalLM",
        },
        "block_configs": [
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 512, "num_experts_per_tok": 4},
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 768, "num_experts_per_tok": 8},
        ],
        "bos_token_id": 1,
        "chunk_size": 128,
        "conv_kernel": 4,
        "eos_token_id": 2,
        "expand": 2,
        "head_dim": 64,
        "hidden_act": "relu2",
        "hidden_dropout": 0.0,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "layer_norm_epsilon": 1e-5,
        "mamba_head_dim": 32,
        "mamba_hidden_act": "silu",
        "mamba_num_heads": 8,
        "mamba_proj_bias": False,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "mlp_hidden_act": "relu2",
        "model_type": "nemotron_h_puzzle",
        "moe_latent_size": 128,
        "moe_shared_expert_intermediate_size": 1024,
        "moe_shared_expert_overlap": True,
        "mtp_block_configs": [
            {"block_type": "attention"},
            {"block_type": "moe", "moe_intermediate_size": 2688, "num_experts_per_tok": 22},
        ],
        "n_group": 1,
        "n_groups": 4,
        "n_routed_experts": 16,
        "n_shared_experts": 1,
        "norm_eps": 1e-5,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "num_logits_to_keep": 1,
        "num_nextn_predict_layers": 1,
        "pad_token_id": 0,
        "rescale_prenorm_residual": True,
        "residual_in_fp32": False,
        "rms_norm_eps": 1e-5,
        "ssm_state_size": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_bias": False,
        "use_cache": True,
        "use_conv_bias": True,
        "use_mamba_kernels": True,
        "vocab_size": 1024,
    }


class TestNemotronHPuzzleBridgeRegistration:
    def test_bridge_is_subclass_of_nemotronh(self):
        assert issubclass(NemotronHPuzzleBridge, NemotronHBridge)
        assert issubclass(NemotronHPuzzleBridge, MegatronModelBridge)

    def test_hf_prefix_is_model(self):
        # Puzzle HF checkpoints put the backbone under `model.*`, not `backbone.*`.
        assert NemotronHPuzzleBridge.HF_PREFIX == "model"


class TestHfMtpConfig:
    def test_from_block_configs(self):
        cfg = Mock(spec=[])
        cfg.num_nextn_predict_layers = 1
        cfg.mtp_block_configs = [{"block_type": "attention"}, {"block_type": "moe"}]
        num, pattern = NemotronHPuzzleBridge._hf_mtp_config(cfg)
        assert num == 1
        assert pattern == "*E"

    def test_zero_layers_returns_none_pattern(self):
        cfg = Mock(spec=[])
        cfg.num_nextn_predict_layers = 0
        cfg.mtp_block_configs = []
        num, pattern = NemotronHPuzzleBridge._hf_mtp_config(cfg)
        assert num == 0
        assert pattern is None

    def test_falls_back_to_hybrid_override_pattern(self):
        cfg = Mock(spec=[])
        cfg.num_nextn_predict_layers = 1
        cfg.mtp_block_configs = []
        cfg.mtp_hybrid_override_pattern = "*-"
        num, pattern = NemotronHPuzzleBridge._hf_mtp_config(cfg)
        assert num == 1
        assert pattern == "*-"

    def test_raises_when_positive_and_no_source(self):
        cfg = Mock(spec=[])
        cfg.num_nextn_predict_layers = 1
        cfg.mtp_block_configs = []
        cfg.mtp_hybrid_override_pattern = None
        with pytest.raises(ValueError, match="mtp_block_configs"):
            NemotronHPuzzleBridge._hf_mtp_config(cfg)

    def test_raises_when_negative(self):
        cfg = Mock(spec=[])
        cfg.num_nextn_predict_layers = -1
        with pytest.raises(ValueError, match="non-negative"):
            NemotronHPuzzleBridge._hf_mtp_config(cfg)


class TestProviderBridge:
    @pytest.fixture
    def hf_config_dict(self):
        return _puzzle_hf_config_dict()

    @pytest.fixture
    def mock_hf_config(self, hf_config_dict):
        # Mock(spec=[]) so undefined attrs raise AttributeError — getattr(..., default)
        # then correctly returns the default. Without spec=[], Mock returns Mock objects
        # for arbitrary attrs, which would incorrectly slip into provider fields.
        cfg = Mock(spec=[])
        for k, v in hf_config_dict.items():
            setattr(cfg, k, v)
        return cfg

    @pytest.fixture
    def mock_pretrained(self, mock_hf_config):
        m = Mock(spec=PreTrainedCausalLM)
        m.config = mock_hf_config
        return m

    def test_returns_hybrid_provider(self, mock_pretrained):
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert isinstance(provider, HybridModelProvider)

    def test_sets_hybrid_layer_pattern_from_block_configs(self, mock_pretrained):
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert provider.hybrid_layer_pattern == "MEME"
        assert provider.hybrid_override_pattern is None
        assert provider.num_layers == 4

    def test_sets_mtp_pattern_from_mtp_block_configs(self, mock_pretrained):
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert provider.mtp_hybrid_override_pattern == "*E"
        assert provider.mtp_num_layers == 1
        assert provider.mtp_pattern_length == 2

    def test_sparse_per_layer_config_overrides(self, mock_pretrained):
        # Non-MoE positions -> None; MoE positions -> only the varying keys.
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert provider.per_layer_config_overrides == [
            None,
            {"moe_ffn_hidden_size": 512, "moe_router_topk": 4},
            None,
            {"moe_ffn_hidden_size": 768, "moe_router_topk": 8},
        ]

    def test_sparse_mtp_per_layer_config_overrides(self, mock_pretrained):
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert provider.mtp_per_layer_config_overrides == [
            None,
            {"moe_ffn_hidden_size": 2688, "moe_router_topk": 22},
        ]

    def test_sets_heterogeneous_block_specs_flag(self, mock_pretrained):
        # __post_init__ ran on the provider before per_layer_config_overrides was
        # set, so the validator didn't flip the flag — the bridge must do it.
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert provider.heterogeneous_block_specs is True

    def test_sets_global_moe_fields(self, mock_pretrained):
        # Puzzle keeps num_moe_experts and moe_shared_expert_intermediate_size
        # constant across MoE layers — they stay on the provider, not the
        # per-layer overrides.
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert provider.num_moe_experts == 16
        assert provider.moe_shared_expert_intermediate_size == 1024

    def test_raises_on_missing_block_configs(self, mock_pretrained):
        mock_pretrained.config.block_configs = []
        with pytest.raises(ValueError, match="block_configs"):
            NemotronHPuzzleBridge().provider_bridge(mock_pretrained)

    def test_raises_on_incomplete_moe_block(self, mock_pretrained):
        mock_pretrained.config.block_configs = [
            {"block_type": "mamba"},
            {"block_type": "moe"},  # missing moe_intermediate_size + num_experts_per_tok
        ]
        with pytest.raises(ValueError, match="missing required keys"):
            NemotronHPuzzleBridge().provider_bridge(mock_pretrained)

    def test_no_mtp_when_num_nextn_predict_layers_zero(self, mock_pretrained):
        mock_pretrained.config.num_nextn_predict_layers = 0
        mock_pretrained.config.mtp_block_configs = []
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert provider.mtp_hybrid_override_pattern is None
        assert provider.mtp_per_layer_config_overrides is None
        assert provider.mtp_pattern_length is None
        assert provider.mtp_num_layers == 0

    def test_dtype_bfloat16(self, mock_pretrained):
        # Sanity: parent MegatronModelBridge honors torch_dtype from the HF config.
        provider = NemotronHPuzzleBridge().provider_bridge(mock_pretrained)
        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True
        assert provider.fp16 is False


class TestMegatronToHfConfig:
    """Reverse conversion: sparse overrides -> HF block_configs / mtp_block_configs."""

    @staticmethod
    def _stub_parent_hf_cfg(**overrides) -> dict:
        # Emulate what NemotronHBridge.megatron_to_hf_config returns before the
        # Puzzle child rewrites block_configs and swaps the auto_map. Keys the
        # child pops or rewrites are all here so behavior is exercised.
        base = {
            "hybrid_override_pattern": "MEME",
            "mtp_hybrid_override_pattern": "*E",
            "moe_intermediate_size": 999,
            "num_experts_per_tok": 99,
            "auto_map": {"AutoConfig": "configuration_nemotron_h.NemotronHConfig"},
            "model_type": "nemotron_h",
        }
        base.update(overrides)
        return base

    def test_rebuilds_block_configs_from_sparse_overrides(self):
        provider = SimpleNamespace(
            per_layer_config_overrides=[
                None,
                {"moe_ffn_hidden_size": 512, "moe_router_topk": 4},
                None,
                {"moe_ffn_hidden_size": 768, "moe_router_topk": 8},
            ],
            mtp_per_layer_config_overrides=None,
        )
        with patch.object(
            NemotronHBridge, "megatron_to_hf_config", return_value=self._stub_parent_hf_cfg()
        ):
            hf_cfg = NemotronHPuzzleBridge.megatron_to_hf_config(provider)

        assert hf_cfg["block_configs"] == [
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 512, "num_experts_per_tok": 4},
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 768, "num_experts_per_tok": 8},
        ]

    def test_rebuilds_mtp_block_configs_from_sparse_overrides(self):
        provider = SimpleNamespace(
            per_layer_config_overrides=[None, {"moe_ffn_hidden_size": 512, "moe_router_topk": 4}],
            mtp_per_layer_config_overrides=[
                None,
                {"moe_ffn_hidden_size": 2688, "moe_router_topk": 22},
            ],
        )
        parent = self._stub_parent_hf_cfg(hybrid_override_pattern="ME")
        with patch.object(NemotronHBridge, "megatron_to_hf_config", return_value=parent):
            hf_cfg = NemotronHPuzzleBridge.megatron_to_hf_config(provider)

        assert hf_cfg["mtp_block_configs"] == [
            {"block_type": "attention"},
            {"block_type": "moe", "moe_intermediate_size": 2688, "num_experts_per_tok": 22},
        ]

    def test_sets_puzzle_auto_map_and_model_type(self):
        provider = SimpleNamespace(
            per_layer_config_overrides=[None],
            mtp_per_layer_config_overrides=None,
        )
        parent = self._stub_parent_hf_cfg(hybrid_override_pattern="M")
        with patch.object(NemotronHBridge, "megatron_to_hf_config", return_value=parent):
            hf_cfg = NemotronHPuzzleBridge.megatron_to_hf_config(provider)

        assert hf_cfg["model_type"] == "nemotron_h_puzzle"
        assert hf_cfg["auto_map"] == {
            "AutoConfig": "configuration_nemotron_h_puzzle.NemotronHPuzzleConfig",
            "AutoModelForCausalLM": "modeling_nemotron_h_puzzle.NemotronHPuzzleForCausalLM",
        }

    def test_strips_blockwise_global_scalars(self):
        # NemotronHPuzzleConfig strips moe_intermediate_size / num_experts_per_tok
        # from the top level (they're strictly per-block members). Emit them at
        # top level from the parent and the child must pop them so the exported
        # config round-trips through HF loading.
        provider = SimpleNamespace(
            per_layer_config_overrides=[None],
            mtp_per_layer_config_overrides=None,
        )
        parent = self._stub_parent_hf_cfg(hybrid_override_pattern="M")
        with patch.object(NemotronHBridge, "megatron_to_hf_config", return_value=parent):
            hf_cfg = NemotronHPuzzleBridge.megatron_to_hf_config(provider)

        assert "moe_intermediate_size" not in hf_cfg
        assert "num_experts_per_tok" not in hf_cfg

    def test_emits_layers_block_type_from_block_configs(self):
        provider = SimpleNamespace(
            per_layer_config_overrides=[
                None,
                {"moe_ffn_hidden_size": 512, "moe_router_topk": 4},
            ],
            mtp_per_layer_config_overrides=None,
        )
        parent = self._stub_parent_hf_cfg(hybrid_override_pattern="ME", mtp_hybrid_override_pattern=None)
        with patch.object(NemotronHBridge, "megatron_to_hf_config", return_value=parent):
            hf_cfg = NemotronHPuzzleBridge.megatron_to_hf_config(provider)

        assert hf_cfg["layers_block_type"] == ["mamba", "moe"]
        assert "mtp_layers_block_type" not in hf_cfg

    def test_emits_mtp_layers_block_type_when_mtp_present(self):
        provider = SimpleNamespace(
            per_layer_config_overrides=[None],
            mtp_per_layer_config_overrides=[
                None,
                {"moe_ffn_hidden_size": 2688, "moe_router_topk": 22},
            ],
        )
        parent = self._stub_parent_hf_cfg(hybrid_override_pattern="M")
        with patch.object(NemotronHBridge, "megatron_to_hf_config", return_value=parent):
            hf_cfg = NemotronHPuzzleBridge.megatron_to_hf_config(provider)

        assert hf_cfg["mtp_layers_block_type"] == ["attention", "moe"]


class TestHelpers:
    def test_blocks_to_pattern_covers_all_types(self):
        blocks = [
            {"block_type": "mamba"},
            {"block_type": "attention"},
            {"block_type": "moe"},
            {"block_type": "mlp"},
        ]
        assert _blocks_to_pattern(blocks) == "M*E-"

    def test_blocks_to_pattern_accepts_dataclass_style_entries(self):
        # `_block_type` and `_block_attr` support attr access as well as dict
        # get — HF configs sometimes load block entries as dataclass instances.
        blocks = [SimpleNamespace(block_type="mamba"), SimpleNamespace(block_type="moe")]
        assert _blocks_to_pattern(blocks) == "ME"

    def test_block_to_override_returns_none_for_non_moe(self):
        assert _block_to_override({"block_type": "mamba"}) is None
        assert _block_to_override({"block_type": "attention"}) is None
        assert _block_to_override({"block_type": "mlp"}) is None

    def test_block_to_override_extracts_moe_keys(self):
        got = _block_to_override(
            {"block_type": "moe", "moe_intermediate_size": 512, "num_experts_per_tok": 4}
        )
        assert got == {"moe_ffn_hidden_size": 512, "moe_router_topk": 4}

    def test_block_to_override_raises_on_missing_moe_keys(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _block_to_override({"block_type": "moe", "moe_intermediate_size": 512})
        with pytest.raises(ValueError, match="missing required keys"):
            _block_to_override({"block_type": "moe", "num_experts_per_tok": 4})
