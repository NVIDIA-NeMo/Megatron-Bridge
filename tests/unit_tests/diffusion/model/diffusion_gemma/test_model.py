"""Numerical and gradient contracts for the native shared-stack diffusion model."""

import pytest
import torch

from megatron.bridge.diffusion.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaSelfConditioning,
    build_attention_masks,
    prefix_canvas_attention,
)


pytestmark = pytest.mark.unit


def test_masks_exclude_padding_and_canvas_target_and_anchor_window_to_start():
    positions = torch.tensor([[0, 1, 2, 3, 4, 5]])
    canvas = torch.tensor([[3, 4]])
    valid = torch.tensor([[True, True, True, True, True, False]])
    encoder, decoder = build_attention_masks(positions, canvas, valid, torch.tensor([3]), 2)
    assert encoder["full_attention"][0, 0, 2].tolist() == [True, True, True, False, False, False]
    assert decoder["full_attention"][0, 0, 0].tolist() == [True, True, True, False, False, False, True, True]
    assert decoder["sliding_attention"][0, 0, 0].tolist() == [False, False, True, False, False, False, True, True]
    assert torch.equal(decoder["sliding_attention"][..., 0, :], decoder["sliding_attention"][..., 1, :])


def test_sdpa_prefix_gradients_and_detached_preview():
    torch.manual_seed(3)
    query = torch.randn(2, 1, 2, 4, requires_grad=True)
    key = torch.randn(2, 1, 1, 4, requires_grad=True)
    value = torch.randn(2, 1, 1, 4, requires_grad=True)
    prefix_key = torch.randn(3, 1, 1, 4, requires_grad=True)
    prefix_value = torch.randn(3, 1, 1, 4, requires_grad=True)
    mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)
    with torch.no_grad():
        preview = prefix_canvas_attention(query, key, value, mask, (prefix_key, prefix_value))
    assert not preview.requires_grad
    result = prefix_canvas_attention(query, key, value, mask, (prefix_key, prefix_value))
    q = query.permute(1, 2, 0, 3)
    k = torch.cat((prefix_key, key)).permute(1, 2, 0, 3).repeat_interleave(2, dim=1)
    v = torch.cat((prefix_value, value)).permute(1, 2, 0, 3).repeat_interleave(2, dim=1)
    expected = ((q @ k.transpose(-1, -2)).softmax(-1) @ v).permute(2, 0, 1, 3).flatten(2)
    torch.testing.assert_close(result, expected)
    result.square().sum().backward()
    assert prefix_key.grad.abs().sum() > 0
    assert prefix_value.grad.abs().sum() > 0


def test_self_conditioning_matches_hf_analogbits():
    from types import SimpleNamespace

    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaSelfConditioning as HFSelfConditioning,
    )

    module = DiffusionGemmaSelfConditioning(4, 8, 1e-6, torch.float32)
    reference = HFSelfConditioning(
        SimpleNamespace(hidden_size=4, intermediate_size=8, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh")
    )
    reference.load_state_dict(module.state_dict())
    noise = torch.randn(2, 3, 4, requires_grad=True)
    soft = torch.randn_like(noise)
    torch.testing.assert_close(module(noise, soft), reference(noise, soft))
    module(noise, soft).sum().backward()
    assert module.down_proj.weight.grad is not None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size"),
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size"),
        ({"context_parallel_size": 2}, "context_parallel_size"),
        ({"sequence_parallel": True}, "sequence_parallel"),
        ({"recompute_granularity": "full"}, "activation recompute"),
        ({"moe_aux_loss_coeff": 0.01}, "auxiliary loss"),
        ({"share_embeddings_and_output_weights": False}, "tied embeddings"),
        ({"attention_dropout": 0.1}, "dropout"),
        ({"apply_query_key_layer_scaling": True}, "apply_query_key_layer_scaling"),
        ({"qk_clip": True}, "qk_clip"),
        ({"log_max_attention_logit": True}, "log_max_attention_logit"),
        ({"use_inference_optimized_layers": True}, "use_inference_optimized_layers"),
        ({"flash_decode": True}, "flash_decode"),
        ({"fp8": "hybrid"}, "FP8/FP4"),
        ({"fp4": "nvfp4"}, "FP8/FP4"),
    ],
)
def test_provider_rejects_unsupported_modes(overrides, message):
    from megatron.bridge.diffusion.models.diffusion_gemma.provider import DiffusionGemmaModelProvider

    with pytest.raises((ValueError, AssertionError), match=message):
        provider = DiffusionGemmaModelProvider(**overrides)
        provider.validate_diffusion_config()


@pytest.mark.parametrize("case", ["left_padding", "position_offset", "canvas_boundary"])
def test_attention_masks_reject_unsupported_position_contract(case):
    positions = torch.arange(5)[None]
    valid = torch.ones(1, 5, dtype=torch.bool)
    prefix = torch.tensor([2])
    canvas = torch.tensor([[2, 3]])
    if case == "left_padding":
        valid[0, 0] = False
    elif case == "position_offset":
        positions = positions + 1
    else:
        prefix = torch.tensor([1])
    with pytest.raises(ValueError):
        build_attention_masks(positions, canvas, valid, prefix, 3)
