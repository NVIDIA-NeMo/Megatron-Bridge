"""Non-unit affine normalization parity against pinned publisher code."""

import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.d1.modeling_d1.backbone import D1RMSNorm
from megatron.bridge.models.d1.modeling_d1.omni_model import OMNI_REPOSITORY, OMNI_REVISION


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nonunit_rmsnorm_forward_and_backward(dtype, native_process_groups):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        params_dtype=dtype,
        use_cpu_initialization=False,
    )
    reference_class = get_class_from_dynamic_module(
        "encoder.RMSNorm", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
    )
    reference = reference_class(64, 1e-5).cuda().to(dtype)
    native = D1RMSNorm(config, 64, 1e-5)
    with torch.no_grad():
        gamma = torch.linspace(0.7, 1.3, 64, device="cuda", dtype=dtype)
        reference.weight.copy_(gamma)
        native.weight.copy_(gamma)
    generator = torch.Generator(device="cuda").manual_seed(19)
    values = torch.randn(3, 7, 64, device="cuda", dtype=dtype, generator=generator).requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    actual = native(values)
    expected = reference(reference_values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn(actual.shape, device="cuda", dtype=dtype, generator=generator)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(values.grad, reference_values.grad, atol=0, rtol=0)
    torch.testing.assert_close(native.weight.grad, reference.weight.grad, atol=0, rtol=0)
