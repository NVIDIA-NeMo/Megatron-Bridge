"""GPU checks for the native TE projections and D1 convolution boundaries."""

import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import functional as F

from megatron.bridge.models.d1.modeling_d1.short_conv import D1ShortConv


@pytest.mark.parametrize("causal", [True, False])
def test_short_conv_reference_and_gradients(causal: bool, native_process_groups) -> None:
    torch.manual_seed(7)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
    )
    model = D1ShortConv(config, causal=causal)
    states = torch.randn(7, 2, 64, device="cuda", requires_grad=True)
    valid = torch.tensor([[True] * 7, [True] * 5 + [False] * 2], device="cuda")
    prefixes = torch.tensor([2, 3], device="cuda")
    actual = model(states, valid_tokens=valid, media_lengths=prefixes)

    # Separate prefix and text convolutions provide an independent boundary
    # oracle, without reproducing the production tap masking.
    projected = F.linear(states * valid.T.unsqueeze(-1), model.in_proj.weight)
    input_gate, output_gate, values = projected.permute(1, 2, 0).chunk(3, dim=1)
    gated = input_gate * values
    mixed_rows = []
    for row, prefix in enumerate(prefixes.tolist()):
        row_values = gated[row : row + 1]
        if causal:
            mixed = F.conv1d(F.pad(row_values, (2, 0)), model.conv.weight, groups=64)
        else:
            mixed = F.conv1d(F.pad(row_values, (1, 1)), model.conv.weight, groups=64)
            prefix_mix = F.conv1d(F.pad(row_values[..., :prefix], (1, 1)), model.conv.weight, groups=64)
            mixed = torch.cat((prefix_mix, mixed[..., prefix:]), dim=-1)
        mixed_rows.append(mixed)
    reference = F.linear((output_gate * torch.cat(mixed_rows)).permute(2, 0, 1), model.out_proj.weight)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    probe = torch.randn_like(actual)
    parameters = [states, model.in_proj.weight, model.conv.weight, model.out_proj.weight]
    actual_grads = torch.autograd.grad((actual * probe).sum(), parameters, retain_graph=True)
    reference_grads = torch.autograd.grad((reference * probe).sum(), parameters)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert torch.count_nonzero(actual_grad) > 0
        torch.testing.assert_close(actual_grad, reference_grad, atol=1e-6, rtol=1e-4)

    changed = states.detach().clone()
    changed[3:, 0] += 10
    changed[5:, 1] += 10
    result = model(changed, valid_tokens=valid, media_lengths=prefixes)
    if causal:
        torch.testing.assert_close(actual[:3, 0], result[:3, 0])
    else:
        torch.testing.assert_close(actual[:2, 0], result[:2, 0])
    torch.testing.assert_close(actual[:, 1], result[:, 1])
