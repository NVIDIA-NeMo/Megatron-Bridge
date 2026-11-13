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

"""CPU tests of the ERNIE 5.5 Hugging Face config and its mapping to :class:`Ernie55ModelConfig`.

Every expected value is written from the ERNIE 5.5 ``config.json`` semantics, not read back from the bridge:
the layer layout (``compress_ratios``, ``first_k_dense_replace``, the MTP depths), the H-layer stack geometry,
the D-layer ``hybrid_mla_*`` geometry, the indexer, MoE, hyper-connections and SD-MTP. The generic
``CONFIG_MAPPING`` entries that misread ERNIE keys (``num_key_value_heads``, ``head_dim``,
``max_position_embeddings``) are pinned too.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, PretrainedConfig

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# Bridge's conversion stack imports its LoRA layers, which need Transformer Engine; without TE, import it with a
# placeholder for them.
allow_conversion_imports_without_te()
pytest.importorskip("megatron.bridge.models.conversion.model_bridge", reason="needs Bridge's conversion stack")

from megatron.bridge.models.conversion import model_bridge
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.ernie.configuration_ernie55 import (
    ERNIE55_ARCHITECTURE,
    ERNIE55_LEGACY_MODEL_TYPE,
    ERNIE55_MODEL_TYPE,
    Ernie55Config,
    Ernie55LegacyConfig,
)
from megatron.bridge.models.ernie.ernie55_bridge import ERNIE55_DEFAULT_VHA_RANK, Ernie55Bridge
from megatron.bridge.models.ernie.ernie55_config import Ernie55ModelConfig, Ernie55TransformerConfig
from megatron.bridge.models.ernie.ernie55_mappings import Ernie55HFLayout, ernie55_hf_param_dtypes
from tests.unit_tests.models.ernie.ernie55_hf_configs import (
    FULL_SIZE_PATTERN,
    PROXY_CONFIG,
    PROXY_PATTERN,
    full_size_config,
    proxy_config,
)
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import needs_hook


pytestmark = pytest.mark.unit


def _hf(config: dict) -> Ernie55Config:
    return Ernie55Config(**{key: value for key, value in config.items() if key != "model_type"})


def _mapped(hf_config: Ernie55Config) -> Ernie55ModelConfig:
    bridge = Ernie55Bridge()
    bridge.hf_config = hf_config
    return bridge.hf_config_to_model_config(hf_config)


def _model_config(config: dict) -> Ernie55ModelConfig:
    return _mapped(_hf(config))


def _finalized(config: dict) -> Ernie55ModelConfig:
    model_config = _model_config(config)
    # The fused DSA backend needs cuDNN frontend and a GPU at finalize time.
    model_config.dsa_kernel_backend = "none"
    model_config.finalize()
    return model_config


def _checkpoint_export_config(
    tmp_path: Path, source: Ernie55Config, *, reference: Ernie55Config | None = None
) -> PretrainedConfig:
    """The Hugging Face config an export rebuilds for a Megatron checkpoint imported from ``source``.

    ``convert.sh export`` takes it from ``AutoBridge.from_auto_config``: the checkpoint's ``run_config.yaml`` model
    config through ``megatron_to_hf_config``, conformed to the reference model's config (``source`` by default).
    """
    training_config = pytest.importorskip("megatron.bridge.training.config", reason="needs Bridge's training stack")
    source_dir, megatron_dir, reference_dir = tmp_path / "hf", tmp_path / "megatron", tmp_path / "reference"
    source.save_pretrained(source_dir)
    (reference or source).save_pretrained(reference_dir)
    model_config = AutoBridge.from_hf_config(AutoConfig.from_pretrained(source_dir)).get_model_config()
    # The run config of an imported checkpoint, written the way save_megatron_model writes it.
    iteration = megatron_dir / "iter_0000000"
    iteration.mkdir(parents=True)
    training_config.ConfigContainer(
        model=model_config,
        train=None,
        optimizer=training_config.OptimizerConfig(use_distributed_optimizer=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        logger=training_config.LoggerConfig(),
        tokenizer=None,
        checkpoint=training_config.CheckpointConfig(save=str(megatron_dir), save_optim=False, save_rng=False),
        dist=None,
    ).to_yaml(str(iteration / "run_config.yaml"))
    return AutoBridge.from_auto_config(str(megatron_dir), str(reference_dir)).hf_pretrained


class TestHuggingFaceConfig:
    def test_auto_config_resolves_the_model_type(self):
        config = AutoConfig.for_model("ernie5_5_moe", hidden_size=64)
        assert type(config) is Ernie55Config
        assert config.model_type == ERNIE55_MODEL_TYPE == "ernie5_5_moe"
        assert config.hidden_size == 64
        assert config.architectures == [ERNIE55_ARCHITECTURE]

    def test_auto_config_still_resolves_the_legacy_model_type(self):
        config = AutoConfig.for_model("ernie5_5", hidden_size=64)
        assert type(config) is Ernie55LegacyConfig and isinstance(config, Ernie55Config)
        assert config.model_type == ERNIE55_LEGACY_MODEL_TYPE == "ernie5_5"
        current = AutoConfig.for_model("ernie5_5_moe", hidden_size=64).to_dict()
        legacy = config.to_dict()
        assert {k: v for k, v in legacy.items() if k != "model_type"} == {
            k: v for k, v in current.items() if k != "model_type"
        }

    @pytest.mark.parametrize("model_type", ["ernie5_5_moe", "ernie5_5"])
    def test_both_model_types_build_the_same_model(self, tmp_path, model_type):
        (tmp_path / "config.json").write_text(json.dumps({**PROXY_CONFIG, "model_type": model_type}))
        config = AutoConfig.from_pretrained(tmp_path)
        assert config.model_type == model_type
        bridge = Ernie55Bridge()
        bridge.hf_config = config
        assert bridge.hf_config_to_model_config(config).as_dict() == _model_config(PROXY_CONFIG).as_dict()

    def test_export_writes_the_current_model_type(self):
        legacy = {**PROXY_CONFIG, "model_type": "ernie5_5"}
        assert Ernie55Bridge.megatron_to_hf_config(_model_config(legacy))["model_type"] == "ernie5_5_moe"

    def test_from_pretrained_reads_a_checkpoint_config(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(PROXY_CONFIG))
        config = AutoConfig.from_pretrained(tmp_path)
        assert isinstance(config, Ernie55Config)
        assert config.compress_ratios == [128, -2, 128, -2, -2]
        assert config.rope_theta == 8e6
        assert config.mtp_shared_last_layer is True
        assert config.tie_word_embeddings is False

    def test_rope_parameters_carry_the_rope_base(self, tmp_path):
        """transformers 5 keeps ``rope_scaling`` as ``rope_parameters``; its readers take the RoPE base from there."""
        # A released config.json: rope_theta at the top level, the H layers' YaRN settings in rope_scaling.
        released = proxy_config(
            drop=("rotary_scaling_factor", "original_max_position_embeddings"),
            hca_rope_type="yarn",
            rope_theta=8000000,
            rope_scaling={"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536},
        )
        (tmp_path / "config.json").write_text(json.dumps(released))
        config = AutoConfig.from_pretrained(tmp_path)
        assert config.rope_parameters["rope_theta"] == config.rope_theta == 8000000
        assert config.rope_parameters["factor"] == 16

        config.save_pretrained(tmp_path / "saved")
        saved = json.loads((tmp_path / "saved" / "config.json").read_text())
        assert saved["rope_parameters"]["rope_theta"] == saved["rope_theta"] == 8000000
        reloaded = AutoConfig.from_pretrained(tmp_path / "saved")
        assert reloaded.rope_parameters == config.rope_parameters
        assert _mapped(reloaded).as_dict() == _mapped(config).as_dict()

    def test_rope_parameters_follow_the_top_level_rope_base(self):
        rope_parameters = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536}
        config = _hf(proxy_config(rope_theta=8e6, rope_parameters={**rope_parameters, "rope_theta": 10000.0}))
        assert config.rope_parameters["rope_theta"] == config.rope_theta == 8e6

    def test_config_without_rope_scaling_gets_no_rope_parameters(self):
        assert "rope_parameters" not in _hf(PROXY_CONFIG).to_dict()

    def test_optional_keys_default_to_the_reference_semantics(self):
        config = Ernie55Config()
        assert config.hca_rope_type == "rope"
        assert config.csa_dense_mode is True
        assert config.hybrid_mla_attention == "mha"
        assert config.mqa_latent_rope_adjacent_pairing is False
        assert (config.index_n_heads, config.index_head_dim, config.index_topk) == (64, 128, 2048)
        assert config.activation_situ_beta == 1.0 and config.activation_situ_linear_beta is None
        assert config.mtp_shared_last_layer is False and config.latent_moe_use_norm is False
        assert config.vha_rank is None
        # Architecture keys have no default: a config must spell them out.
        assert config.hidden_size is None and config.compress_ratios is None


class TestProxyMapping:
    def test_layout(self):
        config = _model_config(PROXY_CONFIG)
        assert isinstance(config, Ernie55ModelConfig)
        assert isinstance(config.transformer, Ernie55TransformerConfig)
        assert config.hybrid_layer_pattern == PROXY_PATTERN
        assert config.num_layers == 8
        assert config.mtp_num_layers == 1
        assert config.ernie_mtp_shared_last_layer is True
        assert config.is_hybrid_model is True
        assert config.experimental_attention_variant == "dsv4_hybrid"
        # 128 for every H layer, 0 for D, dense and MoE layers, decoder then the MTP depth.
        assert config.csa_compress_ratios == [128, 0, 0, 0, 128, 0, 0, 0, 0, 0]

    def test_h_layer_geometry(self):
        config = _model_config(PROXY_CONFIG)
        assert config.num_attention_heads == 64
        assert config.q_lora_rank == 1024
        assert config.v_head_dim == 512
        assert config.qk_pos_emb_head_dim == 64
        assert config.qk_head_dim == 448
        assert (config.output_projection_groups, config.output_projection_lora_rank) == (8, 1024)
        assert config.csa_window_size == 128
        assert config.csa_compress_rotary_base == 160000.0
        assert config.rope_type == "rope"
        assert (config.rotary_scaling_factor, config.mscale, config.mscale_all_dim) == (1.0, 1.0, 0.0)
        assert config.csa_dense_mode is True

    def test_d_layer_geometry(self):
        config = _model_config(PROXY_CONFIG)
        assert config.ernie_d_num_attention_heads == 64
        assert config.ernie_d_q_lora_rank == 1024
        assert config.ernie_d_kv_lora_rank == 512
        assert config.ernie_d_qk_head_dim == 448
        assert config.ernie_d_qk_pos_emb_head_dim == 64
        assert config.ernie_d_v_head_dim == 256
        assert config.ernie_d_rotary_base == 8e6
        # The checkpoint's RoPE channel layout, kept for weight conversion (the model rotates adjacent pairs).
        assert config.ernie_mqa_latent_rope_adjacent_pairing is True
        assert (config.dsa_indexer_n_heads, config.dsa_indexer_head_dim, config.dsa_indexer_topk) == (64, 128, 2048)
        assert config.dsa_indexer_rope_interleaved is False
        assert config.dsa_indexer_rotate_activation is True
        assert config.dsa_indexer_scoring_relu is True
        assert config.dsa_indexer_k_norm_fp32 is True
        assert config.dsa_indexer_weights_proj_output_dtype == "bf16"

    def test_moe(self):
        config = _model_config(PROXY_CONFIG)
        assert config.num_moe_experts == 32
        assert config.moe_router_topk == 4
        assert config.moe_ffn_hidden_size == 2048
        assert config.moe_shared_expert_intermediate_size == 2048
        assert config.ffn_hidden_size == 8192
        assert config.moe_router_score_function == "sigmoid"
        assert config.moe_router_enable_expert_bias is True
        assert config.moe_router_load_balancing_type == "none"
        assert config.moe_aux_loss_coeff == 0.0
        assert config.moe_router_topk_scaling_factor == 1.0
        assert config.ernie_norm_topk_prob is True
        assert config.moe_router_dtype == "fp32"
        # The checkpoint config has no update rate; the expert bias stays frozen unless a recipe sets one.
        assert config.moe_router_bias_update_rate == 0.0
        assert config.moe_latent_size is None and config.moe_use_norm_before_up_proj is False
        assert config.activation_func is F.silu and config.gated_linear_unit is True
        assert config.activation_func_tanh_clamp_scale is None
        assert config.bias_activation_fusion is False

    def test_shared_settings(self):
        config = _model_config(PROXY_CONFIG)
        assert config.vocab_size == 200027
        assert config.should_pad_vocab is False
        assert config.share_embeddings_and_output_weights is False
        assert config.position_embedding_type == "none"
        assert config.normalization == "RMSNorm" and config.layernorm_epsilon == 1e-6
        assert config.qk_layernorm is True
        assert (config.add_bias_linear, config.add_qkv_bias) == (False, False)
        assert (config.hidden_dropout, config.attention_dropout) == (0.0, 0.0)
        assert config.params_dtype == torch.bfloat16 and config.bf16 is True
        assert config.init_method_std == 0.02
        assert config.enable_mhc_connections is True
        assert config.mhc_num_residual_streams == 4
        assert config.mhc_sinkhorn_iterations == 20
        assert config.use_fused_mhc is False
        assert config.apply_rope_fusion is False
        assert config.ernie_vha_rank == ERNIE55_DEFAULT_VHA_RANK == 16
        assert config.ernie_output_gate_fp32 is False

    def test_generic_mapping_misreads_are_dropped(self):
        config = _model_config(PROXY_CONFIG)
        # num_key_value_heads=1 (the shared latent head) and head_dim=512 (the latent width) are not GQA settings.
        assert config.num_query_groups is None
        assert config.kv_channels is None
        # max_position_embeddings is the model's limit, not the training length (recipes set seq_length).
        assert config.seq_length == Ernie55ModelConfig.__dataclass_fields__["seq_length"].default

    @needs_hook
    def test_finalize(self):
        config = _finalized(PROXY_CONFIG)
        assert config.num_layers == 8
        assert config.hybrid_layer_pattern == PROXY_PATTERN
        assert config.dsa_indexer_metric_num_layers() == 5


class TestFullSizeMapping:
    def test_layout(self):
        config = _model_config(full_size_config())
        assert config.hybrid_layer_pattern == FULL_SIZE_PATTERN
        assert config.num_layers == 86
        assert config.mtp_num_layers == 1
        assert config.hybrid_layer_pattern.split("/")[0].count("H") == 21
        assert config.hybrid_layer_pattern.split("/")[0].count("D") == 22
        assert config.csa_compress_ratios == [128, 0, 0, 0] * 21 + [0, 0] + [0, 0]
        assert config.num_moe_experts == 256 and config.moe_router_topk == 6
        assert config.moe_router_topk_scaling_factor == 1.5

    @needs_hook
    def test_finalize(self):
        config = _finalized(full_size_config())
        assert config.num_layers == 86
        assert config.dsa_indexer_metric_num_layers() == 44


class TestOptionalKeys:
    def test_situ(self):
        config = _model_config(
            proxy_config(hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=2.0)
        )
        assert config.activation_func is F.silu
        assert config.activation_func_tanh_clamp_scale == 4.0
        assert config.activation_func_tanh_clamp_scale_linear == 2.0

    def test_situ_defaults(self):
        config = _model_config(proxy_config(hidden_act="situ"))
        assert config.activation_func_tanh_clamp_scale == 1.0
        assert config.activation_func_tanh_clamp_scale_linear is None

    def test_latent_moe(self):
        config = _model_config(proxy_config(moe_latent_size=1024, latent_moe_use_norm=True))
        assert config.moe_latent_size == 1024
        assert config.moe_use_norm_before_up_proj is True

    def test_vha_rank_key(self):
        assert _model_config(proxy_config(vha_rank=8)).ernie_vha_rank == 8

    def test_output_gate_fp32(self):
        assert _model_config(proxy_config(sigmoid_gate_fusion=True)).ernie_output_gate_fp32 is True

    def test_no_shared_expert(self):
        assert _model_config(proxy_config(n_shared_experts=0)).moe_shared_expert_intermediate_size is None

    def test_unnormalized_top_k_gates(self):
        assert _model_config(proxy_config(norm_topk_prob=False)).ernie_norm_topk_prob is False

    def test_half_split_rope_checkpoint(self):
        assert (
            _model_config(proxy_config(mqa_latent_rope_adjacent_pairing=False)).ernie_mqa_latent_rope_adjacent_pairing
            is False
        )
        # A config.json without the key has half-split pairs, the reference's default.
        assert (
            _model_config(
                proxy_config(drop=("mqa_latent_rope_adjacent_pairing",))
            ).ernie_mqa_latent_rope_adjacent_pairing
            is False
        )
        # A released config.json names the key rope_interleave.
        assert (
            _model_config(
                proxy_config(drop=("mqa_latent_rope_adjacent_pairing",), rope_interleave=True)
            ).ernie_mqa_latent_rope_adjacent_pairing
            is True
        )

    def test_yarn_h_layers(self):
        config = _model_config(
            proxy_config(
                hca_rope_type="yarn",
                rotary_scaling_factor=16,
                original_max_position_embeddings=65536,
                rope_scaling={
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 65536,
                    "beta_fast": 24,
                },
            )
        )
        assert config.ernie_hca_rope_type == "yarn"
        assert config.rope_type == "rope"
        assert (config.rotary_scaling_factor, config.original_max_position_embeddings) == (16.0, 65536)
        assert (config.beta_fast, config.beta_slow) == (24.0, 1.0)
        assert (config.mscale, config.mscale_all_dim) == (1.0, 0.0)

    def test_yarn_keys_fall_back_to_rope_scaling(self):
        rope_scaling = {"type": "yarn", "factor": 8, "original_max_position_embeddings": 65536, "beta_slow": 2}
        config = _model_config(proxy_config(hca_rope_type="yarn", rope_scaling=rope_scaling))
        # The top-level keys win when the config sets them.
        assert (config.rotary_scaling_factor, config.original_max_position_embeddings) == (1.0, 4096)
        assert (config.beta_fast, config.beta_slow) == (32.0, 2.0)
        # A released config.json carries only rope_scaling.
        config = _model_config(
            proxy_config(
                drop=("rotary_scaling_factor", "original_max_position_embeddings"),
                hca_rope_type="yarn",
                rope_scaling=rope_scaling,
            )
        )
        assert (config.rotary_scaling_factor, config.original_max_position_embeddings) == (8.0, 65536)

    def test_no_mtp(self):
        config = _model_config(
            proxy_config(num_nextn_predict_layers=0, compress_ratios=[128, -2, 128, -2], mtp_shared_last_layer=False)
        )
        assert config.hybrid_layer_pattern == "H-DEHEDE"
        assert config.mtp_num_layers is None
        assert config.ernie_mtp_shared_last_layer is False


@pytest.mark.parametrize(
    "overrides, drop, message",
    [
        (dict(hca_rope_type="linear"), (), "hca_rope_type"),
        (dict(hybrid_mla_attention="mha"), (), "hybrid_mla_attention"),
        ({}, ("hybrid_mla_attention",), "hybrid_mla_attention"),
        (dict(use_qk_norm=False), (), "use_qk_norm"),
        ({}, ("use_qk_norm",), "use_qk_norm"),
        (dict(qk_nope_head_dim=512), (), "qk_nope_head_dim"),
        ({}, ("qk_nope_head_dim",), "qk_nope_head_dim"),
        (dict(head_dim=448), (), "head_dim"),
        (dict(num_key_value_heads=4), (), "num_key_value_heads"),
        (dict(tie_word_embeddings=True), (), "tie_word_embeddings"),
        (dict(attention_bias=True), (), "attention_bias"),
        (dict(mlp_bias=True), (), "mlp_bias"),
        (dict(attention_dropout=0.1), (), "attention_dropout"),
        (dict(scoring_func="softmax"), (), "scoring_func"),
        (dict(hidden_act="gelu"), (), "hidden_act"),
        (dict(hc_eps=1e-5), (), "hc_eps"),
        (dict(mla_dense_layers=[1]), (), "mla_dense_layers"),
        (dict(compress_ratios=[128, -2, 4, -2, -2]), (), "compress_ratios"),
        (dict(compress_ratios=[128, -2, 128, -2]), (), "compress_ratios"),
        (dict(num_nextn_predict_layers=2, compress_ratios=[128, -2, 128, -2, -2, 128]), (), "MTP depths"),
        (dict(num_nextn_predict_layers=0, compress_ratios=[128, -2, 128, -2]), (), "mtp_shared_last_layer"),
        ({}, ("hidden_size",), "hidden_size"),
        ({}, ("hybrid_mla_v_head_dim",), "hybrid_mla_v_head_dim"),
    ],
)
def test_unsupported_configs_fail_at_mapping(overrides, drop, message):
    with pytest.raises(ValueError, match=message):
        _model_config(proxy_config(drop=drop, **overrides))


# Keys the reference implementation also reads under another name, with a value it would read and Bridge would not.
_REFERENCE_ONLY_SPELLINGS = [
    # Without the check: no shared experts.
    ("num_shared_experts", 1, "n_shared_experts"),
    ("moe_num_shared_experts", 1, "n_shared_experts"),
    # Without the check: no leading dense layer.
    ("moe_layer_start_index", 1, "first_k_dense_replace"),
    ("moe_num_experts", 32, "n_routed_experts"),
    # The reference reads num_experts before n_routed_experts: 64 experts there, 32 in Bridge.
    ("num_experts", 64, None),
]


class TestReferenceKeySpellings:
    @pytest.mark.parametrize("spelling, value, dropped", _REFERENCE_ONLY_SPELLINGS)
    def test_config_rejects_them(self, tmp_path, spelling, value, dropped):
        """Ernie55Config checks the keys as given, before its defaults (0 shared experts, 0 dense layers) hide them."""
        config = proxy_config(drop=(dropped,) if dropped else (), **{spelling: value})
        with pytest.raises(ValueError, match="which the reference implementation reads"):
            _hf(config)
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="which the reference implementation reads"):
            AutoConfig.from_pretrained(tmp_path)

    @pytest.mark.parametrize("spelling, value, dropped", _REFERENCE_ONLY_SPELLINGS)
    def test_mapping_rejects_them_on_other_config_objects(self, spelling, value, dropped):
        config = proxy_config(drop=(dropped,) if dropped else (), **{spelling: value})
        with pytest.raises(ValueError, match="which the reference implementation reads"):
            Ernie55Bridge().hf_config_to_model_config(PretrainedConfig(**config))

    def test_spellings_that_agree_are_accepted(self):
        # num_experts agrees with n_routed_experts; the reference reads the others only when the key is absent.
        config = proxy_config(num_experts=32, num_shared_experts=5, moe_layer_start_index=7)
        assert _model_config(config).as_dict() == _model_config(PROXY_CONFIG).as_dict()
        assert Ernie55Config().n_shared_experts == 0 and Ernie55Config().first_k_dense_replace == 0


class TestExport:
    @pytest.mark.parametrize("source", [PROXY_CONFIG, full_size_config()], ids=["proxy", "full-size"])
    def test_round_trip(self, source):
        exported = Ernie55Bridge.megatron_to_hf_config(_model_config(source))
        for key, value in exported.items():
            assert key in source, f"export added {key!r}"
            assert value == source[key], f"{key}: exported {value!r}, source {source[key]!r}"
        ignored = {"max_position_embeddings", "rotary_scaling_factor", "original_max_position_embeddings"}
        ignored |= {"bos_token_id", "eos_token_id", "pad_token_id"}
        assert set(source) - set(exported) == ignored

    def test_round_trip_of_the_optional_keys(self):
        source = proxy_config(
            hidden_act="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=2.0,
            moe_latent_size=1024,
            latent_moe_use_norm=True,
            vha_rank=8,
            sigmoid_gate_fusion=True,
            norm_topk_prob=False,
            mqa_latent_rope_adjacent_pairing=False,
        )
        exported = Ernie55Bridge.megatron_to_hf_config(_model_config(source))
        for key in ("hidden_act", "activation_situ_beta", "activation_situ_linear_beta", "moe_latent_size"):
            assert exported[key] == source[key]
        assert exported["latent_moe_use_norm"] is True
        assert exported["vha_rank"] == 8
        assert exported["sigmoid_gate_fusion"] is True
        assert exported["norm_topk_prob"] is False
        # The half-split RoPE layout of the source checkpoint survives the round trip (the export writes it back).
        assert exported["mqa_latent_rope_adjacent_pairing"] is False
        assert (
            _model_config(exported | {"max_position_embeddings": 262144}).as_dict() == _model_config(source).as_dict()
        )

    def test_round_trip_of_yarn_h_layers(self):
        yarn = {"type": "yarn", "factor": 16.0, "original_max_position_embeddings": 65536, "beta_fast": 32.0}
        source = proxy_config(
            hca_rope_type="yarn",
            rotary_scaling_factor=16.0,
            original_max_position_embeddings=65536,
            rope_scaling=yarn | {"beta_slow": 1.0},
        )
        exported = Ernie55Bridge.megatron_to_hf_config(_model_config(source))
        assert exported["hca_rope_type"] == "yarn"
        assert (exported["rotary_scaling_factor"], exported["original_max_position_embeddings"]) == (16.0, 65536)
        assert exported["rope_scaling"] == yarn | {"beta_slow": 1.0}
        assert (
            _model_config(exported | {"max_position_embeddings": 262144}).as_dict() == _model_config(source).as_dict()
        )

    def test_exported_config_maps_back_identically(self):
        first = _model_config(full_size_config())
        second = _model_config(Ernie55Bridge.megatron_to_hf_config(first) | {"max_position_embeddings": 262144})
        assert second.as_dict() == first.as_dict()

    @pytest.mark.parametrize("adjacent_pairing", [False, True])
    def test_checkpoint_export_config_keeps_the_weight_layout(self, tmp_path, adjacent_pairing):
        """The config an export rebuilds from a Megatron checkpoint maps the weights like the source's config.

        ``convert.sh export`` takes its Hugging Face config from ``AutoBridge.from_auto_config``: the checkpoint's
        ``run_config.yaml`` model config through ``megatron_to_hf_config``. The weight mappings depend on that config,
        so it must keep the source's tensor layout, the RoPE-row order of a checkpoint without adjacent pairing
        included, or the export would not reproduce the source.
        """
        source = _hf(proxy_config(mqa_latent_rope_adjacent_pairing=adjacent_pairing))
        exported = _checkpoint_export_config(tmp_path, source)
        assert isinstance(exported, Ernie55Config)
        assert exported.mqa_latent_rope_adjacent_pairing is adjacent_pairing
        assert Ernie55HFLayout.from_hf_config(exported) == Ernie55HFLayout.from_hf_config(source)

    def test_checkpoint_export_config_carries_the_rope_base(self, tmp_path):
        """The ``config.json`` an export writes has the D-layer RoPE base in ``rope_parameters`` too.

        transformers 5 keeps ``rope_scaling`` as ``rope_parameters``, and readers written against it take the base
        from ``rope_parameters["rope_theta"]``. The export takes ``rope_parameters`` from the reference config, so
        the base in it must be the checkpoint's, like the top-level ``rope_theta``.
        """

        def hf(rope_theta: float) -> Ernie55Config:
            yarn = {"type": "yarn", "factor": 16.0, "original_max_position_embeddings": 65536}
            return _hf(
                proxy_config(
                    hca_rope_type="yarn",
                    rotary_scaling_factor=16.0,
                    original_max_position_embeddings=65536,
                    rope_scaling=yarn,
                    rope_theta=rope_theta,
                )
            )

        exported = _checkpoint_export_config(tmp_path, hf(8e6), reference=hf(10000.0))
        assert exported.rope_theta == 8e6
        assert exported.rope_parameters["rope_theta"] == 8e6

        exported.save_pretrained(tmp_path / "export")
        saved = json.loads((tmp_path / "export" / "config.json").read_text())
        assert saved["rope_parameters"]["rope_theta"] == saved["rope_theta"] == 8e6
        assert (saved["rope_parameters"]["factor"], saved["hca_rope_type"]) == (16.0, "yarn")
        assert _mapped(AutoConfig.from_pretrained(tmp_path / "export")).as_dict() == _mapped(hf(8e6)).as_dict()


class TestBridgeRegistration:
    def test_dispatch_selects_the_ernie55_bridge(self):
        bridge = model_bridge.get_model_bridge(ERNIE55_ARCHITECTURE, hf_config=_hf(PROXY_CONFIG))
        assert isinstance(bridge, Ernie55Bridge)
        assert Ernie55Bridge.MODEL_CONFIG_CLASS is Ernie55ModelConfig
        assert Ernie55Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_auto_bridge_builds_the_model_config(self):
        model_config = AutoBridge.from_hf_config(_hf(PROXY_CONFIG)).get_model_config()
        assert isinstance(model_config, Ernie55ModelConfig)
        assert model_config.hybrid_layer_pattern == PROXY_PATTERN
        assert model_config.get_builder_cls().__name__ == "Ernie55ModelBuilder"

    def test_there_is_no_provider_path(self):
        with pytest.raises(NotImplementedError, match="builder-backed"):
            Ernie55Bridge().provider_bridge(_hf(PROXY_CONFIG))

    def test_weight_conversion_covers_the_proxy_checkpoint(self):
        """The bridge's registry reads exactly the proxy checkpoint's 439 tensors, in their safetensors dtypes."""
        bridge = Ernie55Bridge()
        with pytest.raises(ValueError, match="Hugging Face config is not set"):
            bridge.mapping_registry()
        bridge.hf_config = _hf(PROXY_CONFIG)
        registry = bridge.mapping_registry()
        assert registry.megatron_to_hf_lookup("decoder.layers.2.inner_layer.self_attention.k_b_proj").hf_param == (
            "model.layers.1.self_attn.kv_b_proj.weight"
        )
        fixture = json.loads((Path(__file__).parent / "fixtures" / "ernie55_proxy_checkpoint_dtypes.json").read_text())
        names = {"F32": torch.float32, "BF16": torch.bfloat16}
        assert ernie55_hf_param_dtypes(bridge.hf_config) == {
            name: names[dtype] for name, dtype in sorted(fixture["tensors"].items())
        }
