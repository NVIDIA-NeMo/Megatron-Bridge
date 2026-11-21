"""Native Omni head weight round trip and PyTorch reference parity."""

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.d1.d1_mappings import decision_head_mappings
from megatron.bridge.models.d1.modeling_d1.decision_head import D1DecisionHead


def test_head_mapping_forward_and_backward(native_process_groups) -> None:
    torch.manual_seed(17)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=1,
        ffn_hidden_size=256,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=False,
        attention_softmax_in_fp32=True,
    )
    native = D1DecisionHead(config, layers=2)
    reference = nn.Module()
    reference.type_emb = nn.Embedding(3, 64)
    reference.head = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(64, 1, 256, 0.0, batch_first=True, norm_first=True),
        num_layers=2,
        enable_nested_tensor=False,
    )
    reference.scorer = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    reference = reference.cuda()
    registry = MegatronMappingRegistry(*decision_head_mappings(megatron_prefix="native", hf_prefix="reference"))
    original = {f"reference.{name}": parameter.detach().clone() for name, parameter in reference.named_parameters()}
    consumed = set()
    exported = {}
    for name, parameter in native.named_parameters():
        mapping = registry.megatron_to_hf_lookup(f"native.{name}")
        assert mapping is not None, name
        assert isinstance(mapping.hf_param, str)
        assert mapping.hf_param not in consumed
        consumed.add(mapping.hf_param)
        owner = native.get_submodule(name.rsplit(".", 1)[0])
        converted = mapping.hf_to_megatron(original[mapping.hf_param], owner)
        assert converted.shape == parameter.shape, name
        with torch.no_grad():
            parameter.copy_(converted)
        exported.update(mapping.megatron_to_hf(parameter.detach(), owner))
    assert consumed == set(original)
    assert set(exported) == set(original)
    assert len(original) == 31
    for name in original:
        assert torch.equal(exported[name], original[name]), name

    values = torch.randn(3, 7, 64, device="cuda", requires_grad=True)
    reference_values = values.detach().clone().requires_grad_()
    valid = torch.tensor([[True] * 7, [True] * 5 + [False] * 2, [True] * 6 + [False]], device="cuda")
    markers = torch.tensor([[2, 4, 6], [1, 3, 0], [2, 4, 0]], device="cuda")
    marker_mask = torch.tensor([[True, True, True], [True, True, False], [True, True, False]], device="cuda")
    types = torch.tensor([0, 1, 2], device="cuda")
    actual = native(
        values, valid_tokens=valid, marker_positions=markers, marker_mask=marker_mask, question_types=types
    )
    expected_states = reference_values + reference.type_emb(types)[:, None, :]
    for layer in reference.head.layers:
        expected_states = layer(expected_states, src_key_padding_mask=~valid)
    expected = reference.scorer(expected_states.gather(1, markers[..., None].expand(-1, -1, 64)))
    expected = expected.squeeze(-1).float().masked_fill(~marker_mask, -1e4)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1), atol=2e-6, rtol=2e-4)
    labels = torch.tensor([2, 0, 1], device="cuda")
    nn.functional.cross_entropy(actual, labels).backward()
    nn.functional.cross_entropy(expected, labels).backward()
    assert values.grad is not None and torch.count_nonzero(values.grad) > 0
    torch.testing.assert_close(values.grad, reference_values.grad, atol=2e-6, rtol=2e-4)
    exported_gradients = {}
    for name, parameter in native.named_parameters():
        assert parameter.grad is not None, name
        mapping = registry.megatron_to_hf_lookup(f"native.{name}")
        owner = native.get_submodule(name.rsplit(".", 1)[0])
        exported_gradients.update(mapping.megatron_to_hf(parameter.grad, owner))
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(exported_gradients[f"reference.{name}"], parameter.grad, atol=2e-6, rtol=2e-4)
