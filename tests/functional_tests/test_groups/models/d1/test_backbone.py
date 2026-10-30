"""Full tiny native backbone mapping coverage and masking contracts."""

import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import functional as F
from transformers import Lfm2Config
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.d1.d1_mappings import backbone_mappings
from megatron.bridge.models.d1.modeling_d1.backbone import D1Backbone


@pytest.mark.parametrize("causal", [True, False])
def test_backbone_roundtrip_and_reference(causal: bool, native_process_groups) -> None:
    torch.manual_seed(27)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=False,
    )
    hf_config = Lfm2Config(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        layer_types=["conv", "full_attention"],
        block_auto_adjust_ff_dim=False,
        conv_bias=False,
        norm_eps=1e-5,
        rope_theta=1000000.0,
    )
    hf_config._attn_implementation = "sdpa"
    if causal:
        reference_class = get_class_from_dynamic_module(
            "lfm2_vl.Lfm2ForCausalLM",
            "LiquidAI/d1-3B",
            revision="051bcc464b01b9f92942b364d9586b0ef5912432",
            code_revision="051bcc464b01b9f92942b364d9586b0ef5912432",
        )
        reference = reference_class(hf_config).model.cuda()
    else:
        reference_class = get_class_from_dynamic_module(
            "encoder.Trunk",
            "LiquidAI/d1-omni-600M",
            revision="02b55d7076f15129e59ab3f94783f32c4b088674",
            code_revision="02b55d7076f15129e59ab3f94783f32c4b088674",
        )
        reference = reference_class(hf_config.to_dict()).cuda()
    native = D1Backbone(
        config, vocab_size=32, layer_types=("conv", "full_attention"), causal=causal, rotary_base=1000000.0
    )
    registry = MegatronMappingRegistry(*backbone_mappings(megatron_prefix="native", hf_prefix="reference"))
    original = {f"reference.{name}": parameter.detach().clone() for name, parameter in reference.named_parameters()}
    consumed = set()
    exported = {}
    for name, parameter in native.named_parameters():
        mapping = registry.megatron_to_hf_lookup(f"native.{name}")
        assert mapping is not None, name
        if isinstance(mapping.hf_param, str):
            source = original[mapping.hf_param]
            consumed.add(mapping.hf_param)
        else:
            source = {kind: original[published] for kind, published in mapping.hf_param.items()}
            consumed.update(mapping.hf_param.values())
        owner = native.get_submodule(name.rsplit(".", 1)[0])
        converted = mapping.hf_to_megatron(source, owner)
        assert converted.shape == parameter.shape, name
        with torch.no_grad():
            parameter.copy_(converted)
        exported.update(mapping.megatron_to_hf(parameter.detach(), owner))
    assert set(original) == consumed == set(exported)
    for name in original:
        assert torch.equal(original[name], exported[name]), name

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 0, 0]], device="cuda")
    valid = ids != 0
    prefixes = torch.tensor([0, 0] if causal else [2, 3], device="cuda")
    values = native.embed_tokens(ids).detach().requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    actual = native(values, valid_tokens=valid, media_lengths=prefixes)
    if causal:
        expected = reference(inputs_embeds=reference_values, attention_mask=valid, use_cache=False).last_hidden_state
    else:
        expected = reference(reference_values, valid, prefixes)
    # Compare real positions; padding outputs are never read by the head.
    torch.testing.assert_close(actual[valid], expected[valid], atol=2e-5, rtol=2e-4)
    probe = torch.randn_like(actual)
    (actual[valid] * probe[valid]).sum().backward()
    (expected[valid] * probe[valid]).sum().backward()
    torch.testing.assert_close(values.grad, reference_values.grad, atol=2e-5, rtol=2e-4)
    gradients = {}
    for name, parameter in native.named_parameters():
        if name == "embed_tokens.weight":
            assert parameter.grad is None
            continue
        assert parameter.grad is not None, name
        mapping = registry.megatron_to_hf_lookup(f"native.{name}")
        owner = native.get_submodule(name.rsplit(".", 1)[0])
        gradients.update(mapping.megatron_to_hf(parameter.grad, owner))
    for name, parameter in reference.named_parameters():
        if name == "embed_tokens.weight":
            assert parameter.grad is None
            continue
        torch.testing.assert_close(gradients[f"reference.{name}"], parameter.grad, atol=2e-5, rtol=2e-4)

    changed = values.detach().clone()
    changed[0, 4:] += 10
    changed[1, 5:] += 10
    result = native(changed, valid_tokens=valid, media_lengths=prefixes)
    unchanged_length = 4 if causal else 2
    torch.testing.assert_close(actual[0, :unchanged_length], result[0, :unchanged_length], atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual[1, :5], result[1, :5], atol=2e-5, rtol=2e-4)
