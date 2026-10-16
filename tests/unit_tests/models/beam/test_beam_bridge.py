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

"""Beam config, weight mapping, adapter geometry and output-layer tests."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.beam.beam_bridge import (
    _DENSE_EXPORT_DROPS,
    BeamBridge,
    _beam_settings,
    _moe_layer_pattern,
    _on_attention_layers,
    _rope_settings,
)
from megatron.bridge.models.beam.beam_builder import BeamModelBuilder
from megatron.bridge.models.beam.beam_config import BeamModelConfig, BeamTransformerConfig, beam_pipeline_layer_pattern
from megatron.bridge.models.beam.beam_mappings import (
    BeamQKVGateMapping,
    _geometry,
    merge_qkv_gate_weights,
    split_qkv_gate_weights,
)
from megatron.bridge.models.beam.modeling_beam import BeamOutputLayer, beam_logit_softcapping
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVGMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def _cfg(num_heads=8, num_query_groups=2, head_size=4, hidden=16):
    return SimpleNamespace(
        num_attention_heads=num_heads,
        num_query_groups=num_query_groups,
        kv_channels=head_size,
        hidden_size=hidden,
        attention_output_gate=True,
    )


def _model_parallel_context():
    """Initialize a 1x1 model-parallel state (NCCL under torchrun on GPU, single-process gloo on CPU)."""
    from megatron.core import parallel_state

    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend, init_method="tcp://127.0.0.1:29521", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    return parallel_state.destroy_model_parallel


def test_geometry():
    cfg = _cfg()
    head_num, groups, per_group, head_size = _geometry(cfg)
    assert (head_num, groups, per_group, head_size) == (8, 2, 4, 4)


@pytest.mark.parametrize("heads,groups,head_size,hidden", [(8, 2, 4, 16), (4, 4, 2, 8)], ids=["gqa", "mha"])
def test_qkv_gate_round_trip(heads, groups, head_size, hidden):
    """merge -> split must be the identity for GQA and MHA with a full-width gate."""
    cfg = _cfg(num_heads=heads, num_query_groups=groups, head_size=head_size, hidden=hidden)
    torch.manual_seed(0)

    q = torch.randn(heads * head_size, hidden)
    k = torch.randn(groups * head_size, hidden)
    v = torch.randn(groups * head_size, hidden)
    g = torch.randn(heads * head_size, hidden)

    merged = merge_qkv_gate_weights(cfg, q, k, v, g)

    expected_rows = (2 * heads + 2 * groups) * head_size
    assert merged.shape == (expected_rows, hidden)

    for original, restored in zip((q, k, v, g), split_qkv_gate_weights(cfg, merged)):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)


@pytest.mark.parametrize("mapping_cls", [QKVGMapping, BeamQKVGateMapping])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qkvg_mapping_round_trip(mapping_cls, dtype):
    config = _cfg()
    heads, groups, _, head_size = _geometry(config)
    gate_rows = heads * head_size if mapping_cls is BeamQKVGateMapping else heads
    weights = {
        key: torch.randn(rows, config.hidden_size, dtype=dtype)
        for key, rows in zip("qkvg", (heads * head_size, groups * head_size, groups * head_size, gate_rows))
    }
    module = torch.nn.Linear(config.hidden_size, (2 * heads + 2 * groups) * head_size, bias=False)
    module.config = config
    module.tensor_model_parallel, module.partition_dim = True, 0
    mapping = mapping_cls("weight", "q", "k", "v", "g").resolve(())
    assert type(mapping) is mapping_cls
    packed = mapping.hf_to_megatron(weights, module)
    restored = mapping.megatron_to_hf(packed, module)
    for key in weights:
        torch.testing.assert_close(restored[key], weights[key], rtol=0, atol=0)


def test_gate_must_be_full_width():
    """A per-head scalar gate is QKVGMapping's shape, not this one's."""
    cfg = _cfg()
    head_num, groups, _, head_size = _geometry(cfg)
    hidden = cfg.hidden_size
    q = torch.randn(head_num * head_size, hidden)
    k = torch.randn(groups * head_size, hidden)
    v = torch.randn(groups * head_size, hidden)
    scalar_gate = torch.randn(head_num, hidden)

    with pytest.raises(ValueError, match="num_heads\\*head_dim"):
        merge_qkv_gate_weights(cfg, q, k, v, scalar_gate)


ROPE = {"rope_type": "default", "rope_theta": 10000.0}
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]


def _hf(**overrides):
    base = dict(
        num_hidden_layers=4,
        num_experts=8,
        layer_types=list(LAYER_TYPES),
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        rope_parameters={"sliding_attention": dict(ROPE), "full_attention": None},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_rope_settings_nope_per_layer_type():
    """A layer type whose rope_parameters entry is None runs NoPE (+SSMax); Megatron flags it per layer."""
    assert _rope_settings(_hf()) == ([0, 0, 1, 0], 10000.0)

    all_rope = _hf(rope_parameters={"sliding_attention": dict(ROPE), "full_attention": dict(ROPE)})
    assert _rope_settings(all_rope) == (None, 10000.0)  # core default: no per-layer list

    all_nope = _hf(layer_types=["full_attention"] * 4, rope_parameters={"full_attention": None})
    assert _rope_settings(all_nope) == ([1, 1, 1, 1], None)


def test_rope_settings_rejects_unrepresentable():
    with pytest.raises(ValueError, match="different rope_theta"):
        _rope_settings(
            _hf(rope_parameters={"sliding_attention": dict(ROPE), "full_attention": {**ROPE, "rope_theta": 5000.0}})
        )
    with pytest.raises(ValueError, match="rope_type"):
        _rope_settings(
            _hf(rope_parameters={"sliding_attention": {**ROPE, "rope_type": "yarn"}, "full_attention": None})
        )
    with pytest.raises(ValueError, match="keyed by layer type"):
        _rope_settings(_hf(rope_parameters=dict(ROPE)))
    with pytest.raises(ValueError, match="keyed by layer type"):
        _rope_settings(_hf(rope_parameters={"sliding_attention": dict(ROPE)}))  # full_attention entry missing


def test_moe_layer_pattern():
    """mlp_layer_types decides dense vs MoE per layer, as in BeamDecoderLayer."""
    assert _moe_layer_pattern(_hf()) == [0, 1, 1, 1]
    assert _moe_layer_pattern(_hf(mlp_layer_types=["dense"] * 4)) == [0, 0, 0, 0]
    with pytest.raises(ValueError, match="mlp_layer_types"):
        _moe_layer_pattern(_hf(mlp_layer_types=["dense", "moe", "sparse", "sparse"]))
    with pytest.raises(ValueError, match="4 layers"):
        _moe_layer_pattern(_hf(mlp_layer_types=["dense", "sparse"]))
    config = _hf()
    del config.mlp_layer_types
    with pytest.raises(ValueError, match="mlp_layer_types"):
        _moe_layer_pattern(config)


BEAM_CFG = {
    "architectures": ["BeamForCausalLM"],
    "model_type": "beam",
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "initializer_range": 0.02,
    "num_experts": 8,
    "num_experts_per_tok": 4,
    "n_group": 4,
    "topk_group": 2,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": True,
    "sliding_window": 512,
    "layer_types": list(LAYER_TYPES),
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    # a non-default theta proves rotary_base is read from the nested dict, not from default_theta
    "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 5000.0}, "full_attention": None},
    "default_theta": 10000.0,  # BeamConfig class attribute the generic rope helper falls back to
    "final_logit_softcapping": 30.0,
}
NON_BEAM_KEYS = ("rope_theta", "partial_rotary_factor", "num_local_experts", "n_routed_experts")


@pytest.fixture
def hf_config():
    # spec keeps Mock from inventing attributes the bridge probes with getattr()
    cfg = Mock(spec=list(BEAM_CFG.keys()))
    for k, v in BEAM_CFG.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def mock_pretrained(hf_config):
    m = Mock(spec=PreTrainedCausalLM)
    m.config = hf_config
    m.generation_config = Mock()
    return m


@pytest.fixture
def converted_config(mock_pretrained):
    cfg = BeamBridge().hf_config_to_model_config(mock_pretrained.config)
    assert isinstance(cfg, BeamModelConfig) and isinstance(cfg.transformer, BeamTransformerConfig)
    assert cfg.hybrid_stack_spec is None  # BeamModelBuilder supplies the Beam stack spec
    return cfg


def test_registered_for_beam_only():
    assert BeamBridge.SOURCE_NAME == "BeamForCausalLM"
    assert BeamBridge.MODEL_TYPE == "beam"


def test_beam_settings_target_declared_fields(hf_config):
    """Every setting the bridge emits exists on the target config (strict overrides reject unknown names)."""
    fields = set(BeamTransformerConfig.__dataclass_fields__) | set(BeamModelConfig.__dataclass_fields__)
    assert set(_beam_settings(hf_config)) <= fields


@pytest.mark.unit
def test_provider_matches_builder_config(mock_pretrained):
    from megatron.bridge.models.beam.beam_provider import BeamModelProvider

    bridge = BeamBridge()
    with pytest.warns(FutureWarning, match="legacy compatibility"):
        provider = bridge.provider_bridge(mock_pretrained)
    assert isinstance(provider, BeamModelProvider)
    config = bridge.hf_config_to_model_config(mock_pretrained.config)
    for key in bridge.hf_config_to_model_config_kwargs(mock_pretrained.config):
        assert getattr(provider, key) == getattr(config, key), key
    assert bridge.megatron_to_hf_config(provider) == bridge.megatron_to_hf_config(config)


@pytest.mark.unit
def test_provider_pipeline_split_and_validation(mock_pretrained):
    with pytest.warns(FutureWarning):
        provider = BeamBridge().provider_bridge(mock_pretrained)
    provider.pipeline_model_parallel_size = 2
    provider.finalize()
    assert provider.hybrid_layer_pattern == "*-*E|*E*E"
    provider.pipeline_model_parallel_size = 1
    provider.finalize()
    assert provider.hybrid_layer_pattern == "*-*E*E*E"
    provider.moe_router_dtype = None
    with pytest.raises(ValueError, match="moe_router_dtype"):
        provider.finalize()


@pytest.mark.unit
def test_provider_rejects_virtual_pipeline(mock_pretrained):
    with pytest.warns(FutureWarning):
        provider = BeamBridge().provider_bridge(mock_pretrained)
    provider.virtual_pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="virtual pipeline"):
        provider.finalize()


@pytest.mark.unit
def test_provider_dataclass_round_trip(mock_pretrained):
    from megatron.bridge.models.beam.beam_provider import BeamModelProvider

    with pytest.warns(FutureWarning):
        provider = BeamBridge().provider_bridge(mock_pretrained)
    serialized = provider.asdict()
    restored = BeamModelProvider(**{f.name: serialized[f.name] for f in fields(provider) if f.init})
    assert type(restored) is BeamModelProvider
    assert BeamBridge.megatron_to_hf_config(restored) == BeamBridge.megatron_to_hf_config(provider)
    assert restored.moe_router_fp32_params and restored.ssmax


@pytest.mark.unit
def test_provider_preserves_beam_stack_spec(mock_pretrained):
    with pytest.warns(FutureWarning):
        provider = BeamBridge().provider_bridge(mock_pretrained)
    spec = provider._resolve_hybrid_stack_spec()
    assert spec.submodules.attention_layer.module.__name__ == "BeamTransformerLayer"
    assert spec.submodules.moe_layer.module.__name__ == "BeamMoETransformerLayer"
    provider.hybrid_stack_spec = spec
    with pytest.raises(ValueError, match="custom specs"):
        provider._resolve_hybrid_stack_spec()


@pytest.mark.unit
@pytest.mark.parametrize("last_stage,cap", [(True, 30.0), (False, 30.0), (True, None)])
def test_provider_applies_softcap_only_on_output_stage(mock_pretrained, last_stage, cap):
    from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider

    with pytest.warns(FutureWarning):
        provider = BeamBridge().provider_bridge(mock_pretrained)
    provider.final_logit_softcapping = cap
    model = SimpleNamespace(post_process=last_stage, output_layer=object())
    with (
        patch.object(HybridModelProvider, "provide", return_value=model) as build,
        patch("megatron.bridge.models.beam.beam_provider.extend_instance") as extend,
    ):
        assert provider.provide(pre_process=True, post_process=last_stage) is model
        build.assert_called_once_with(pre_process=True, post_process=last_stage, vp_stage=None)
        if last_stage and cap:
            extend.assert_called_once_with(model.output_layer, BeamOutputLayer)
        else:
            extend.assert_not_called()


def test_per_layer_flags_land_on_attention_layers():
    assert _on_attention_layers([1, 0, 1]) == [1, 0, 0, 0, 1, 0]


@pytest.mark.parametrize(
    "pattern,stages,expected",
    [
        ("*-*E*E*E", 1, "*-*E*E*E"),
        ("*-*E*E*E", 2, "*-*E|*E*E"),
        ("*-*E*E*E", 3, "*-*E|*E|*E"),  # earlier stages take the remainder; pairs stay together
        ("*-*E|*E*E", 4, "*-|*E|*E|*E"),  # stale separators are re-split
        ("*-*E|*E*E", 1, "*-*E*E*E"),
    ],
)
def test_pipeline_layer_pattern(pattern, stages, expected):
    assert beam_pipeline_layer_pattern(pattern, stages) == expected


def test_pipeline_layer_pattern_rejects_bad_inputs():
    with pytest.raises(ValueError, match="pair every attention layer"):
        beam_pipeline_layer_pattern("*-*", 1)
    with pytest.raises(ValueError, match="cannot fill"):
        beam_pipeline_layer_pattern("*-*E", 3)


def test_all_rope_model_does_not_enable_ssmax(hf_config):
    hf_config.rope_parameters = {layer_type: dict(ROPE) for layer_type in LAYER_TYPES}
    settings = _beam_settings(hf_config)
    assert "ssmax" not in settings and "no_rope_freq" not in settings


class TestConfigConversion:
    def test_config_fields(self, converted_config):
        expected = dict(
            num_layers=8,  # two hybrid layers per Beam layer
            hybrid_layer_pattern="*-*E*E*E",
            is_hybrid_model=True,
            hidden_size=128,
            kv_channels=16,
            num_query_groups=2,
            vocab_size=512,
            seq_length=4096,
            position_embedding_type="rope",
            rotary_base=5000.0,  # nested rope_parameters, not default_theta
            rotary_percent=1.0,
            share_embeddings_and_output_weights=False,
            attention_output_gate=True,
            attention_output_gate_bias=True,
            add_qkv_bias=False,
            add_bias_linear=False,
            qk_layernorm=True,
            normalization="RMSNorm",
            gated_linear_unit=True,
            hidden_dropout=0.0,
            attention_softmax_in_fp32=True,
            post_self_attn_layernorm=True,
            post_mlp_layernorm=True,
            depth_scaled_residual=True,
            window_size=(511, 0),
            window_attn_skip_freq=[1, 0, 1, 0, 0, 0, 1, 0],
            no_rope_freq=[0, 0, 0, 0, 1, 0, 0, 0],
            ssmax=True,
            num_moe_experts=8,
            moe_ffn_hidden_size=64,
            moe_router_topk=4,
            moe_router_num_groups=4,
            moe_router_group_topk=2,
            moe_router_topk_scaling_factor=2.5,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=False,
            moe_router_dtype="fp32",
            moe_router_fp32_params=True,
            moe_router_load_balancing_type="none",
            moe_router_bias_update_rate=0.0,
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            final_logit_softcapping=30.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
        )
        for name, value in expected.items():
            assert getattr(converted_config, name) == value, name

    def test_rejects_unnormalised_topk(self, mock_pretrained):
        mock_pretrained.config.norm_topk_prob = False
        with pytest.raises(ValueError, match="norm_topk_prob"):
            BeamBridge().hf_config_to_model_config(mock_pretrained.config)

    def test_all_dense_model_has_no_moe(self, mock_pretrained):
        """num_experts in the config must not create MoE layers when no layer is typed 'sparse'."""
        mock_pretrained.config.mlp_layer_types = ["dense"] * 4
        cfg = BeamBridge().hf_config_to_model_config(mock_pretrained.config)
        assert cfg.transformer.moe_router_num_groups is None and cfg.transformer.moe_router_group_topk is None
        assert cfg.transformer.moe_router_fp32_params is False
        cfg.finalize()
        assert cfg.transformer.num_moe_experts is None and cfg.transformer.moe_ffn_hidden_size is None

    def test_rejects_attention_bias(self, mock_pretrained):
        """The QKV+gate fusion carries no bias; refuse instead of silently dropping the HF biases."""
        mock_pretrained.config.attention_bias = True
        with pytest.raises(ValueError, match="attention_bias=True"):
            BeamBridge().hf_config_to_model_config(mock_pretrained.config)

    def test_rejects_sparse_layers_without_experts(self, mock_pretrained):
        mock_pretrained.config.num_experts = 0
        with pytest.raises(ValueError, match="num_experts"):
            BeamBridge().hf_config_to_model_config(mock_pretrained.config)

    def test_rejects_sliding_layers_without_window(self, mock_pretrained):
        mock_pretrained.config.sliding_window = None
        with pytest.raises(ValueError, match="sliding_window"):
            BeamBridge().hf_config_to_model_config(mock_pretrained.config)

    def test_config_round_trip(self, converted_config):
        hf = BeamBridge.megatron_to_hf_config(converted_config)
        for key in (
            "architectures",
            "model_type",
            "vocab_size",
            "num_hidden_layers",
            "head_dim",
            "num_experts",
            "sliding_window",
            "layer_types",
            "mlp_layer_types",
            "rope_parameters",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
            "norm_topk_prob",
            "attention_bias",
            "final_logit_softcapping",
        ):
            assert hf[key] == BEAM_CFG[key], key
        for key in NON_BEAM_KEYS:
            assert key not in hf, key

    def test_export_rejects_mixed_rope_within_layer_type(self, mock_pretrained):
        cfg = BeamBridge().hf_config_to_model_config(mock_pretrained.config)
        cfg.transformer.no_rope_freq = [1, 0, 0, 0, 1, 0, 0, 0]  # layer 0 (sliding) NoPE, layer 1 (sliding) RoPE
        with pytest.raises(ValueError, match="mix RoPE and NoPE"):
            BeamBridge.megatron_to_hf_config(cfg)

    @pytest.mark.parametrize("pattern", ["*-*E*E", "-**E*E*E", "*M*E*E*E"])
    def test_export_rejects_non_beam_patterns(self, converted_config, pattern):
        converted_config.hybrid_layer_pattern = pattern
        with pytest.raises(ValueError, match="alternating attention"):
            BeamBridge.megatron_to_hf_config(converted_config)

    def test_export_ignores_pipeline_separators(self, converted_config):
        converted_config.hybrid_layer_pattern = "*-*E|*E*E"
        hf = BeamBridge.megatron_to_hf_config(converted_config)
        assert hf["num_hidden_layers"] == 4 and hf["mlp_layer_types"] == BEAM_CFG["mlp_layer_types"]


class TestMappingRegistry:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA model construction")
    @pytest.mark.parametrize("dense", [False, True])
    @pytest.mark.parametrize("legacy_provider", [False, True])
    def test_built_model_conversion_tasks(self, hf_config, dense, legacy_provider):
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from transformers import PretrainedConfig

        teardown = _model_parallel_context()
        try:
            model_parallel_cuda_manual_seed(42)
            if dense:
                hf_config.mlp_layer_types = ["dense"] * hf_config.num_hidden_layers
            bridge = BeamBridge()
            if legacy_provider:
                with pytest.warns(FutureWarning):
                    cfg = bridge.provider_bridge(SimpleNamespace(config=hf_config))
            else:
                cfg = bridge.hf_config_to_model_config(hf_config)
            # Mapping coverage does not require the optional APEX wgrad extension.
            cfg.gradient_accumulation_fusion = False
            cfg.finalize()
            groups = ProcessGroupCollection.use_mpu_process_groups()
            if legacy_provider:
                cfg._pg_collection = groups
                model = cfg.provide()
            else:
                model = BeamModelBuilder(cfg).build_model(groups)
            # The registry lays out the hybrid layers from the HF config handed to the conversion.
            # Generic HF RoPE validation rejects Beam's None-valued NoPE entries.
            # This fixture only supplies metadata to the mapping registry.
            pretrained = PretrainedConfig()
            pretrained.update({k: getattr(hf_config, k) for k in BEAM_CFG})
            tasks = bridge.build_conversion_tasks(pretrained, [model])
            hf_names = [
                name
                for task in tasks
                for name in (
                    [task.mapping.hf_param]
                    if isinstance(task.mapping.hf_param, str)
                    else task.mapping.hf_param.values()
                )
            ]
            assert len(hf_names) == len(set(hf_names))
            # Every auto-mapped parameter's module class resolves to a TP layout (Beam subclasses included).
            for task in tasks:
                if isinstance(task.mapping, AutoMapping) and task.megatron_module is not None:
                    assert task.mapping._detect_parallelism_type(task.megatron_module) in {
                        "column",
                        "row",
                        "replicated",
                    }
            assert "model.layers.0.mlp.gate_proj.weight" in hf_names
            assert ("model.layers.1.mlp.experts.0.gate_proj.weight" in hf_names) is not dense
            assert "model.layers.0.post_attention_layernorm.weight" in hf_names
            assert "model.layers.0.post_feedforward_layernorm.weight" in hf_names
            assert "model.norm.weight" in hf_names
            # Every Megatron parameter of the split layers has a mapping.
            assert None not in tasks
            assert {name for name, _ in model.named_parameters()} <= {task.global_param_name for task in tasks}
        finally:
            teardown()

    def test_needs_the_hf_config(self):
        with pytest.raises(RuntimeError, match="needs hf_config"):
            BeamBridge().mapping_registry()

    def test_split_layer_indices(self, hf_config):
        """Beam layer i: attention at decoder.layers.{2i}, dense MLP / MoE at decoder.layers.{2i+1}."""
        bridge = BeamBridge()
        bridge.hf_config = hf_config
        registry = bridge.mapping_registry()

        assert registry.megatron_to_hf_lookup("decoder.layers.1.mlp.linear_fc1.weight").hf_param == {
            "gate": "model.layers.0.mlp.gate_proj.weight",
            "up": "model.layers.0.mlp.up_proj.weight",
        }
        assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.router.weight").hf_param == (
            "model.layers.1.mlp.gate.weight"
        )
        assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.router.expert_bias") is not None
        experts = registry.megatron_to_hf_lookup("decoder.layers.7.mlp.experts.linear_fc2.weight5")
        assert experts.hf_param == "model.layers.3.mlp.experts.5.down_proj.weight"
        # Layer kinds do not leak: no MoE on Beam layer 0, no dense MLP on Beam layer 1, no MLP on attention layers.
        assert registry.megatron_to_hf_lookup("decoder.layers.1.mlp.router.weight") is None
        assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.linear_fc1.weight") is None
        assert registry.megatron_to_hf_lookup("decoder.layers.2.mlp.linear_fc1.weight") is None
        assert registry.megatron_to_hf_lookup("decoder.layers.8.self_attention.linear_proj.weight") is None

        qkv = registry.megatron_to_hf_lookup("decoder.layers.4.self_attention.linear_qkv.weight")
        assert set(qkv.hf_param.values()) == {
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.2.self_attn.v_proj.weight",
            "model.layers.2.self_attn.gate_proj.weight",
        }
        bias = registry.megatron_to_hf_lookup("decoder.layers.4.self_attention.output_gate_bias.bias")
        assert bias.hf_param == "model.layers.2.self_attn.gate_bias"
        assert registry.megatron_to_hf_lookup("decoder.layers.4.self_attention.linear_qkv.bias") is None
        assert registry.megatron_to_hf_lookup("decoder.final_norm.weight").hf_param == "model.norm.weight"
        assert registry.hf_to_megatron_lookup("model.layers.2.self_attn.o_proj.weight").megatron_param == (
            "decoder.layers.4.self_attention.linear_proj.weight"
        )

    def test_mappings_validate_only_existing_parameters(self, hf_config):
        bridge = BeamBridge()
        bridge.hf_config = hf_config
        names = [
            "decoder.layers.1.mlp.linear_fc1.weight",
            "decoder.layers.3.mlp.experts.linear_fc1.weight0",
        ]
        hf_keys = {
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            "model.layers.1.mlp.experts.0.up_proj.weight",
        }
        mappings = bridge._validate_conversion_mappings(bridge.mapping_registry(), names, hf_keys)
        assert set(mappings) == set(names)

    def test_norms_use_gemma2_names(self, hf_config):
        """Beam names its four norms like Gemma-2; post_attention_layernorm is the *sandwich* norm."""
        bridge = BeamBridge()
        bridge.hf_config = hf_config
        registry = bridge.mapping_registry()

        def hf_name(megatron_param):
            return registry.megatron_to_hf_lookup(megatron_param).hf_param

        assert hf_name("decoder.layers.2.self_attention.linear_qkv.layer_norm_weight") == (
            "model.layers.1.input_layernorm.weight"
        )
        assert hf_name("decoder.layers.2.post_self_attn_layernorm.weight") == (
            "model.layers.1.post_attention_layernorm.weight"
        )
        # pre-MLP norm: standalone on MoE layers, fused into linear_fc1 on dense layers
        assert (
            hf_name("decoder.layers.3.pre_mlp_layernorm.weight") == "model.layers.1.pre_feedforward_layernorm.weight"
        )
        assert hf_name("decoder.layers.1.mlp.linear_fc1.layer_norm_weight") == (
            "model.layers.0.pre_feedforward_layernorm.weight"
        )
        assert hf_name("decoder.layers.3.post_mlp_layernorm.weight") == (
            "model.layers.1.post_feedforward_layernorm.weight"
        )


class TestBuilderBackedConfig:
    def test_class_switches(self):
        assert BeamBridge.MODEL_CONFIG_CLASS is BeamModelConfig
        assert BeamModelConfig.transformer_config_class is BeamTransformerConfig
        assert BeamBridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_builder_resolution_and_type_guard(self, hf_config):
        cfg = BeamBridge().hf_config_to_model_config(hf_config)
        assert cfg.get_builder_cls() is BeamModelBuilder
        builder = BeamModelBuilder(cfg)
        assert builder._model_config is cfg
        with pytest.raises(TypeError, match="BeamModelConfig"):
            BeamModelBuilder(object())

    def test_finalize_runs_core_validation(self, hf_config):
        cfg = BeamBridge().hf_config_to_model_config(hf_config)
        cfg.finalize()  # deferred MCore __post_init__: SSMax / sandwich / MoE validation
        assert cfg.hybrid_layer_pattern == "*-*E*E*E" and cfg.transformer.num_layers == 8

    def test_finalize_splits_the_pattern_for_pipeline_parallelism(self, hf_config):
        cfg = BeamBridge().hf_config_to_model_config(hf_config)
        cfg.pipeline_model_parallel_size = 2
        cfg.finalize()
        assert cfg.hybrid_layer_pattern == "*-*E|*E*E"

    def test_serialization_round_trip(self, hf_config):
        cfg = BeamBridge().hf_config_to_model_config(hf_config)
        data = cfg.as_dict()
        assert data["_builder_"] == BeamModelConfig.builder
        assert data["transformer"]["final_logit_softcapping"] == 30.0
        restored = BeamModelConfig.from_dict(data)
        assert isinstance(restored, BeamModelConfig)
        assert isinstance(restored.transformer, BeamTransformerConfig)
        assert restored.transformer.final_logit_softcapping == 30.0
        assert restored.hybrid_layer_pattern == "*-*E*E*E"
        assert tuple(restored.transformer.window_size) == (511, 0)
        assert restored.transformer.attention_output_gate_bias is True
        assert restored.get_builder_cls() is BeamModelBuilder

    def test_export_loads_as_beam_config(self, hf_config):
        """The exported dict must be a valid BeamConfig that builds a working BeamForCausalLM."""
        beam = pytest.importorskip("transformers.models.beam", reason="needs the Beam-enabled transformers")
        cfg = BeamBridge().hf_config_to_model_config(hf_config)
        hf = BeamBridge.megatron_to_hf_config(cfg)

        beam_cfg = beam.BeamConfig(**hf)
        assert beam_cfg.model_type == "beam"
        assert beam_cfg.layer_types == BEAM_CFG["layer_types"]
        assert beam_cfg.mlp_layer_types == BEAM_CFG["mlp_layer_types"]
        assert beam_cfg.rope_parameters["sliding_attention"]["rope_theta"] == 5000.0
        assert beam_cfg.rope_parameters["full_attention"] is None
        assert beam_cfg.num_experts == 8 and beam_cfg.sliding_window == 512

        model = beam.BeamForCausalLM(beam_cfg).eval()
        prefix = "model.layers.0."
        names = {n[len(prefix) :] for n, _ in model.named_parameters() if n.startswith(prefix)}
        assert {
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        } <= names
        with torch.no_grad():
            logits = model(input_ids=torch.arange(6)[None], use_cache=False).logits
        assert logits.shape == (1, 6, 512) and bool(torch.isfinite(logits).all())


class TestFp32RouterParameters:
    """Beam keeps ``mlp.gate.weight`` fp32 inside bf16 checkpoints; so does ``BeamTopKRouter``."""

    def test_enabled_on_moe_models_only(self, hf_config):
        settings = _beam_settings(hf_config)
        assert settings["moe_router_fp32_params"] is True and settings["moe_router_dtype"] == "fp32"
        hf_config.mlp_layer_types = ["dense"] * 4
        assert "moe_router_fp32_params" not in _beam_settings(hf_config)

    def test_export_weight_dtype_semantics(self):
        """``weight_dtype=None`` keeps the mixed-dtype state (fp32 routers); an explicit dtype casts everything."""
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

        weights = {
            "model.layers.1.mlp.gate.weight": torch.randn(8, 128),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.zeros(8),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(128, 128).bfloat16(),
            "ids": torch.arange(4),
        }
        faithful = MegatronModelBridge._cast_export_weight_dtype(weights, None)
        assert {k: v.dtype for k, v in faithful.items()} == {k: v.dtype for k, v in weights.items()}
        lossy = MegatronModelBridge._cast_export_weight_dtype(weights, torch.bfloat16)
        assert lossy["model.layers.1.mlp.gate.weight"].dtype == torch.bfloat16
        assert lossy["model.layers.1.self_attn.q_proj.weight"].dtype == torch.bfloat16
        assert lossy["ids"].dtype == torch.int64


class TestDenseExport:
    """An all-dense Beam export must be a valid ``BeamConfig`` (its MoE fields are typed int / float)."""

    def _dense(self, hf_config):
        hf_config.mlp_layer_types = ["dense"] * 4
        return BeamBridge().hf_config_to_model_config(hf_config)

    def test_moe_keys_dropped_for_dense_models_only(self, hf_config, mock_pretrained):
        moe = BeamBridge.megatron_to_hf_config(BeamBridge().hf_config_to_model_config(hf_config))
        assert moe["num_experts"] == 8 and moe["moe_intermediate_size"] == 64 and moe["n_group"] == 4
        dense_cfg = self._dense(hf_config)  # mock_pretrained.config is the same object
        for exported in (BeamBridge.megatron_to_hf_config(dense_cfg),):
            assert exported["mlp_layer_types"] == ["dense"] * 4
            for key in _DENSE_EXPORT_DROPS:
                assert key not in exported, key
            for key in NON_BEAM_KEYS:
                assert key not in exported, key
            assert exported["layer_types"] == BEAM_CFG["layer_types"]

    def test_dense_export_builds_beam_config_and_round_trips(self, hf_config):
        """Mirror ``AutoBridge.from_auto_config()``: synthesize the HF config from the Megatron config, conform it
        to a reference (MoE) BeamConfig, construct BeamConfig, and get a dense model config back from the bridge."""
        beam = pytest.importorskip("transformers.models.beam", reason="needs the Beam-enabled transformers")
        from megatron.bridge import AutoBridge
        from megatron.bridge.models.conversion.auto_bridge import _drop_readonly_config_properties
        from megatron.bridge.models.conversion.utils import conform_config_to_reference

        dense_cfg = self._dense(hf_config)
        exported = BeamBridge.megatron_to_hf_config(dense_cfg)
        beam_cfg = beam.BeamConfig(**exported)
        assert beam_cfg.mlp_layer_types == ["dense"] * 4
        model = beam.BeamForCausalLM(beam_cfg)
        assert not any(".experts." in n or ".mlp.gate." in n for n, _ in model.named_parameters())
        reference = beam.BeamConfig(**{k: v for k, v in BEAM_CFG.items() if k != "default_theta"})
        synthesized = conform_config_to_reference(exported, reference.to_dict())
        synthesized = _drop_readonly_config_properties(synthesized, type(reference))
        synthesized = type(reference)(**synthesized)
        assert synthesized.mlp_layer_types == ["dense"] * 4 and synthesized.num_experts == 8  # reference default
        bridge = AutoBridge.from_hf_config(synthesized)
        round_tripped = bridge.get_model_config()
        assert isinstance(round_tripped, BeamModelConfig)
        t = round_tripped.transformer
        assert t.num_moe_experts is None and t.moe_ffn_hidden_size is None and t.moe_router_num_groups is None
        assert round_tripped.hybrid_layer_pattern == dense_cfg.hybrid_layer_pattern == "*-*-*-*-"
        assert t.window_attn_skip_freq == [1, 0, 1, 0, 0, 0, 1, 0] and t.no_rope_freq == [0, 0, 0, 0, 1, 0, 0, 0]
        round_tripped.finalize()


class TestBeamOutputLayer:
    def test_softcap_matches_hf_formula_in_fp32(self):
        """Beam caps ``logits.float()``; near saturation bf16 would return exactly the cap with a zero gradient."""
        x = torch.tensor([120.0, 0.5, -3.0, -200.0], dtype=torch.bfloat16, requires_grad=True)
        y = beam_logit_softcapping(x, 30.0)
        assert y.dtype == torch.float32
        torch.testing.assert_close(y, torch.tanh(x.float() / 30.0) * 30.0, rtol=0, atol=0)
        assert float(y[0].detach()) < 30.0
        y.sum().backward()
        assert float(x.grad[0]) > 0

    def test_disabled_cap_is_identity(self):
        x = torch.randn(3, dtype=torch.bfloat16)
        assert beam_logit_softcapping(x, None) is x and beam_logit_softcapping(x, 0) is x

    def test_registered_as_column_parallel_for_conversion(self):
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        assert "BeamOutputLayer" in AutoMapping._MODULE_TYPE_REGISTRY["column"]

    def test_mixin_on_column_parallel_linear(self):
        """``extend_instance`` keeps the class name (conversion looks it up) and caps the projection in fp32."""
        from megatron.core.tensor_parallel import ColumnParallelLinear

        from megatron.bridge.models.gemma.modules import extend_instance

        teardown = _model_parallel_context()
        try:
            cfg = BeamTransformerConfig(
                num_layers=1,
                hidden_size=8,
                num_attention_heads=1,
                use_cpu_initialization=True,
                final_logit_softcapping=30.0,
            )
            cfg.finalize()
            layer = ColumnParallelLinear(
                8, 16, config=cfg, init_method=cfg.init_method, bias=False, gather_output=False, skip_bias_add=False
            )
            gen = torch.Generator().manual_seed(0)
            with torch.no_grad():  # weights ~ N(0, 20^2): pre-cap logits ~ N(0, 57^2), many past the cap
                layer.weight.copy_(torch.randn(16, 8, generator=gen) * 20.0)
            extend_instance(layer, BeamOutputLayer)
            assert isinstance(layer, BeamOutputLayer) and type(layer).__name__ == "ColumnParallelLinear"
            x = torch.randn(3, 2, 8, generator=gen)
            out, bias = layer(x)
            assert bias is None and out.dtype == torch.float32
            torch.testing.assert_close(out, torch.tanh((x @ layer.weight.t()) / 30.0) * 30.0)
            assert float((x @ layer.weight.t()).abs().max()) > 60.0
            assert 29.0 < float(out.abs().max()) < 30.0
        finally:
            teardown()


class TestTPGreaterThanKVHeads:
    """8 heads / 2 KV heads at TP4 and TP8 (core's KV-replication regime).

    Core column-shards the fused ``linear_qkv`` weight, all-gathers the per-rank projections, slices KV
    group ``rank // (TP / groups)`` and then the ``num_heads / TP`` query and gate heads at index
    ``rank % (TP / groups)``. This checks the packing layout on CPU, not distributed collectives.
    """

    def _weights(self, cfg):
        head_num, groups, _, head_size = _geometry(cfg)
        gen = torch.Generator().manual_seed(0)
        return (
            torch.randn(head_num * head_size, cfg.hidden_size, generator=gen),
            torch.randn(groups * head_size, cfg.hidden_size, generator=gen),
            torch.randn(groups * head_size, cfg.hidden_size, generator=gen),
            torch.randn(head_num * head_size, cfg.hidden_size, generator=gen),
        )

    @pytest.mark.parametrize("tp", [4, 8])
    def test_per_rank_gather_and_slice_reproduce_hf_heads(self, tp):
        cfg = _cfg()
        head_num, groups, per_group, hd = _geometry(cfg)
        q, k, v, g = self._weights(cfg)
        shards = torch.chunk(merge_qkv_gate_weights(cfg, q, k, v, g), tp, dim=0)
        x = torch.randn(5, 2, cfg.hidden_size, generator=torch.Generator().manual_seed(1))
        gathered = torch.cat([x @ s.t() for s in shards], dim=-1)  # all_gather_last_dim of the projections
        ranks_per_group = tp // groups
        for rank in range(tp):
            group, i, n = rank // ranks_per_group, rank % ranks_per_group, head_num // tp
            size = gathered.shape[-1] // groups
            local = gathered[..., group * size : (group + 1) * size].reshape(5, 2, 2 * per_group + 2, hd)
            q_l, g_l, k_l, v_l = torch.split(local, [per_group, per_group, 1, 1], dim=2)
            q_l, g_l = q_l[:, :, i * n : (i + 1) * n], g_l[:, :, i * n : (i + 1) * n]
            heads = list(range(group * per_group + i * n, group * per_group + (i + 1) * n))
            assert len(heads) == n
            torch.testing.assert_close(q_l, (x @ q.t()).reshape(5, 2, head_num, hd)[:, :, heads])
            torch.testing.assert_close(g_l, (x @ g.t()).reshape(5, 2, head_num, hd)[:, :, heads])
            torch.testing.assert_close(k_l, (x @ k.t()).reshape(5, 2, groups, hd)[:, :, group : group + 1])
            torch.testing.assert_close(v_l, (x @ v.t()).reshape(5, 2, groups, hd)[:, :, group : group + 1])
