"""DiffusionGemma config dispatch and canonical text checkpoint mappings."""

from types import SimpleNamespace

import pytest
import torch
from transformers import DiffusionGemmaConfig, DiffusionGemmaTextConfig

from megatron.bridge.diffusion.conversion.diffusion_gemma import DiffusionGemmaBridge
from megatron.bridge.diffusion.conversion.diffusion_gemma.diffusion_gemma_bridge import _PreTrainedDiffusionGemma
from megatron.bridge.diffusion.models.diffusion_gemma.provider import DiffusionGemmaModelProvider
from megatron.bridge.models.conversion.auto_bridge import AutoBridge, _resolve_pretrained_wrapper_cls


@pytest.fixture
def config():
    text = DiffusionGemmaTextConfig(
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        global_head_dim=8,
        num_global_key_value_heads=1,
        num_experts=2,
        top_k_experts=1,
        vocab_size=128,
        layer_types=["sliding_attention", "full_attention"],
    )
    return DiffusionGemmaConfig(text_config=text, architectures=["DiffusionGemmaForBlockDiffusion"])


@pytest.mark.unit
def test_config_only_selects_native_diffusion_provider(config):
    auto = AutoBridge.from_hf_config(config)
    provider = auto.to_megatron_provider(load_weights=False)
    assert isinstance(provider, DiffusionGemmaModelProvider)
    assert (provider.hidden_size, provider.num_layers, provider.num_moe_experts) == (16, 2, 2)
    assert (provider.kv_channels, provider.global_head_dim, provider.num_global_key_value_heads) == (4, 8, 1)
    assert provider.attention_k_eq_v and provider.share_embeddings_and_output_weights
    assert provider.moe_aux_loss_coeff == 0 and provider.moe_router_load_balancing_type == "none"
    assert provider.params_dtype == torch.bfloat16
    assert _resolve_pretrained_wrapper_cls(config) is _PreTrainedDiffusionGemma
    wrapper = _PreTrainedDiffusionGemma()
    wrapper.config = config
    assert AutoBridge(wrapper)._pretrained_wrapper_cls is _PreTrainedDiffusionGemma


@pytest.mark.unit
def test_canonical_mapping_and_distinct_scalars(config):
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = config
    registry = bridge.mapping_registry()
    expected = {
        "embedding.word_embeddings.weight": "model.decoder.embed_tokens.weight",
        "decoder.layers.0.encoder_layer_scalar": "model.encoder.language_model.layers.0.layer_scalar",
        "decoder.layers.0.layer_scalar": "model.decoder.layers.0.layer_scalar",
        "self_conditioning.pre_norm.weight": "model.decoder.self_conditioning.pre_norm.weight",
        "self_conditioning.down_proj.weight": "model.decoder.self_conditioning.down_proj.weight",
    }
    for native, hf in expected.items():
        assert registry.megatron_to_hf_lookup(native).hf_param == hf
    assert registry.hf_to_megatron_lookup("model.encoder.vision_tower.foo.weight") is None
    assert registry.hf_to_megatron_lookup("lm_head.weight") is None
    for native in ("self_conditioning.gate_proj.weight", "self_conditioning.up_proj.weight"):
        assert registry.megatron_to_hf_lookup(native) is not None


@pytest.mark.unit
def test_global_value_synthesis_and_export_omission(config):
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = config
    names = {role: f"model.decoder.layers.1.self_attn.{role}_proj.weight" for role in "qkv"}
    state = {names["q"]: torch.randn(32, 16), names["k"]: torch.randn(8, 16)}
    loaded = bridge.maybe_modify_loaded_hf_weight(names, state)
    assert torch.equal(loaded["v"], state[names["k"]])
    exported = bridge.maybe_modify_converted_hf_weight(None, {names["v"]: loaded["v"]}, None)
    assert exported == {}


@pytest.mark.unit
@pytest.mark.parametrize("field,value", [("quantization_config", {"quant_method": "fp8"}), ("dtype", "float8_e4m3fn")])
def test_rejects_quantized_config(config, field, value):
    config = DiffusionGemmaConfig.from_dict({**config.to_dict(), field: value})
    with pytest.raises(ValueError, match="BF16|quantized"):
        DiffusionGemmaBridge().provider_bridge(SimpleNamespace(config=config))


@pytest.mark.unit
def test_rejects_unequal_shared_encoder_alias(config):
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = config
    name = "model.decoder.layers.0.router.scale"
    state = {name: torch.ones(1), name.replace("model.decoder.", "model.encoder.language_model."): torch.zeros(1)}
    with pytest.raises(ValueError, match="shared|tied"):
        bridge.maybe_modify_loaded_hf_weight(name, state)


@pytest.mark.unit
def test_rejects_clipped_text_config(config):
    config.text_config.use_clipped_linears = True
    with pytest.raises(ValueError, match="clipped"):
        DiffusionGemmaBridge().provider_bridge(SimpleNamespace(config=config))


@pytest.mark.unit
def test_shared_aliases_must_match_but_scalars_remain_independent(config):
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = config
    canonical = "model.decoder.norm.weight"
    state = {canonical: torch.ones(16), "model.encoder.language_model.norm.weight": torch.ones(16)}
    assert torch.equal(bridge.maybe_modify_loaded_hf_weight(canonical, state), state[canonical])
    scalar = "model.decoder.layers.0.layer_scalar"
    state = {scalar: torch.ones(1), "model.encoder.language_model.layers.0.layer_scalar": torch.zeros(1)}
    assert torch.equal(bridge.maybe_modify_loaded_hf_weight(scalar, state), state[scalar])


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_rejects_raw_quantized_tensor(config, dtype):
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = config
    name = "model.decoder.norm.weight"
    state = {name: torch.ones(16).to(dtype)}
    with pytest.raises(ValueError, match="quantized|floating"):
        bridge.maybe_modify_loaded_hf_weight(name, state)


@pytest.mark.unit
@pytest.mark.parametrize("layer,missing_role", [(0, "v"), (1, "k"), (1, "q")])
def test_incomplete_attention_weights_are_rejected(config, layer, missing_role):
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = config
    names = {role: f"model.decoder.layers.{layer}.self_attn.{role}_proj.weight" for role in "qkv"}
    state = {name: torch.ones(4, 16) for role, name in names.items() if role != missing_role}
    with pytest.raises(ValueError, match="missing|required"):
        bridge.maybe_modify_loaded_hf_weight(names, state)


@pytest.mark.unit
def test_conflicting_global_value_weight_is_rejected(config):
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = config
    names = {role: f"model.decoder.layers.1.self_attn.{role}_proj.weight" for role in "qkv"}
    state = {name: torch.ones(8, 16) for name in names.values()}
    state[names["v"]] = torch.zeros(8, 16)
    with pytest.raises(ValueError, match="K=V|tied"):
        bridge.maybe_modify_loaded_hf_weight(names, state)


@pytest.mark.unit
@pytest.mark.parametrize("layer,expert", [(0, 1), (1, 0)])
@pytest.mark.parametrize("projection,hf_projection", [("linear_fc1", "gate_up_proj"), ("linear_fc2", "down_proj")])
def test_sequential_experts_share_fused_hf_array(config, layer, expert, projection, hf_projection):
    from megatron.bridge.models.conversion.param_mapping import FusedExpertMapping, FusedGatedExpertMapping
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    bridge = DiffusionGemmaBridge()
    bridge.hf_config = config
    native = f"decoder.layers.{layer}.mlp.experts.local_experts.{expert}.{projection}.weight"
    mapping = bridge.mapping_registry().megatron_to_hf_lookup(native)
    expected_type = FusedGatedExpertMapping if projection == "linear_fc1" else FusedExpertMapping
    assert isinstance(mapping, expected_type)
    assert mapping.hf_param == f"model.decoder.layers.{layer}.experts.{hf_projection}"
    assert mapping.megatron_param == native
    assert mapping.is_expert and mapping.is_grouped_export
    assert extract_expert_number_from_param(mapping.megatron_param) == expert
