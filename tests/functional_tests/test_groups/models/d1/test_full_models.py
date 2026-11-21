"""Complete native D1 wrappers against pinned publisher models and real media."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import Lfm2VlConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge import AutoBridge
from megatron.bridge.models.d1.modeling_d1.omni_model import OMNI_REPOSITORY, OMNI_REVISION


def _native_with_weights(reference):
    # All-ones affine norms conceal order-of-rounding mistakes. Exercise the
    # released parameter contract before both conversion and backward parity.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if parameter.ndim == 1 and "norm" in name.lower() and name.endswith("weight"):
                parameter.copy_(torch.linspace(0.7, 1.3, parameter.numel(), device=parameter.device))
    bridge = AutoBridge.from_hf_config(reference.config)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.gradient_accumulation_fusion = False
    provider.params_dtype = torch.float32
    provider.bf16 = False
    provider.fp16 = False
    provider.finalize()
    native = provider.provide().eval()
    conversion_names = set(bridge._model_bridge._megatron_global_param_names_all_pp_ranks([native]))
    assert set(dict(native.named_parameters())).issubset(conversion_names)
    registry = bridge._model_bridge.mapping_registry()
    original = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
    consumed, exported = set(), {}
    for name, parameter in native.named_parameters():
        mapping = registry.megatron_to_hf_lookup(name)
        assert mapping is not None, name
        if isinstance(mapping.hf_param, str):
            source = original[mapping.hf_param]
            consumed.add(mapping.hf_param)
        else:
            source = {kind: original[key] for kind, key in mapping.hf_param.items()}
            consumed.update(mapping.hf_param.values())
        owner = native.get_submodule(name.rsplit(".", 1)[0])
        converted = mapping.hf_to_megatron(source, owner)
        assert converted.shape == parameter.shape, name
        with torch.no_grad():
            parameter.copy_(converted)
        exported.update(mapping.megatron_to_hf(parameter.detach(), owner))
    assert set(original) == consumed == set(exported)
    for name, value in original.items():
        assert torch.equal(value, exported[name]), name
    return native, registry


def _compare_gradients(native, reference, registry, *, atol, rtol):
    gradients = {}
    for name, parameter in native.named_parameters():
        mapping = registry.megatron_to_hf_lookup(name)
        owner = native.get_submodule(name.rsplit(".", 1)[0])
        if parameter.grad is None:
            names = [mapping.hf_param] if isinstance(mapping.hf_param, str) else mapping.hf_param.values()
            gradients.update({key: None for key in names})
        else:
            gradients.update(mapping.megatron_to_hf(parameter.grad, owner))
    for name, parameter in reference.named_parameters():
        assert (gradients[name] is None) == (parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                gradients[name], parameter.grad, atol=atol, rtol=rtol, msg=lambda message: name + ": " + message
            )


@pytest.mark.parametrize("modality", ["text", "image", "audio"])
def test_omni_complete_model_public_probabilities_and_backward(modality, native_process_groups):
    torch.manual_seed(7)
    cls = get_class_from_dynamic_module(
        "modeling_d1.D1OmniModel", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
    )
    config = cls.config_class(
        architectures=["D1OmniModel"],
        dtype="float32",
        tie_word_embeddings=False,
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "conv_bias": False,
            "block_auto_adjust_ff_dim": True,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 16,
            "rope_theta": 1e6,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "patch_size": 16,
            "num_patches": 256,
            "vision_use_head": False,
        },
        audio_config={
            "feat_in": 128,
            "n_layers": 1,
            "d_model": 64,
            "subsampling_conv_channels": 8,
            "ff_expansion_factor": 2,
            "n_heads": 1,
            "conv_kernel_size": 9,
            "residual_width": 32,
        },
        projector_hidden_size=64,
        head_layers=2,
        max_length=128,
        temperatures={"choice:3-5": 1.4, "score:3-5": 1.7, "noul:2": 1.5},
    )
    reference = cls(config).cuda().eval()
    native, registry = _native_with_weights(reference)
    question_class = get_class_from_dynamic_module(
        "prompt.Question", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
    )
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device="cuda")
    valid = torch.ones_like(ids, dtype=torch.bool)
    if modality == "image":
        image = Image.new("RGB", (32, 32), color=(100, 50, 200))
        reference_prefix, prefix = reference.vision([image]), native.encode_images([image])
    elif modality == "audio":
        waveform = np.sin(np.arange(8000, dtype=np.float32) * 0.1) * 0.1
        reference_prefix, prefix = reference.audio(waveform), native.encode_audio(waveform)
    else:
        reference_prefix = prefix = None
    if prefix is not None:
        torch.testing.assert_close(prefix, reference_prefix, atol=0, rtol=0)
        assert prefix.shape[1] > 0
    prefix_length = 0 if prefix is None else prefix.shape[1]
    lengths = None if prefix is None else torch.tensor([prefix_length], device="cuda")
    questions = [
        (question_class("choice", "Choose", {"A": "first", "B": "second", "C": "third"}), [1, 3, 5], 0, "choice:3-5"),
        (
            question_class("score", "Risk", ["none", "low", "medium", "high", "critical"]),
            [1, 2, 3, 4, 5],
            1,
            "score:3-5",
        ),
        (question_class("noul", "Suspicious?"), [1, 3], 2, "noul:2"),
    ]
    for question, positions, type_id, temperature in questions:
        markers = torch.tensor([positions], device="cuda")
        marker_mask = torch.ones_like(markers, dtype=torch.bool)
        types = torch.tensor([type_id], device="cuda")
        result = native(
            ids,
            attention_mask=valid,
            marker_positions=markers,
            marker_mask=marker_mask,
            question_types=types,
            media_prefix=prefix,
            media_lengths=lengths,
        )
        logits = result.logits / config.temperatures[temperature] if modality == "text" else result.logits
        probabilities = logits.softmax(-1)
        if question.type == "noul":
            probabilities = probabilities.flip(-1)
        expected = torch.tensor(
            reference._forward([(reference_prefix, ids[0].tolist(), positions, question, modality == "text")]),
            device="cuda",
        )
        torch.testing.assert_close(probabilities, expected, atol=2e-5, rtol=2e-4)

    reference_values = reference.encoder.embed_tokens(ids)
    if reference_prefix is not None:
        reference_values = torch.cat((reference_prefix, reference_values), dim=1)
    all_valid = torch.ones(reference_values.shape[:2], device="cuda", dtype=torch.bool)
    encoded = reference.encoder(reference_values, all_valid, torch.tensor([prefix_length], device="cuda"))
    reference_logits = reference.head(encoded[:, prefix_length:], valid, markers, marker_mask, types)
    labels = torch.tensor([1], device="cuda")
    training = native(
        ids,
        attention_mask=valid,
        marker_positions=markers,
        marker_mask=marker_mask,
        question_types=types,
        media_prefix=prefix,
        media_lengths=lengths,
        labels=labels,
    )
    expected_loss = F.cross_entropy(reference_logits.float(), labels)
    torch.testing.assert_close(training.loss, expected_loss, atol=2e-5, rtol=2e-4)
    training.loss.backward()
    expected_loss.backward()
    assert native.encoder.embed_tokens.weight.grad.abs().sum() > 0
    assert native.head.scorer_output.weight.grad.abs().sum() > 0
    _compare_gradients(native, reference, registry, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("modality", ["text", "image"])
def test_vl_complete_model_tied_weights_media_logits_and_backward(modality, native_process_groups):
    torch.manual_seed(7)
    revision = "051bcc464b01b9f92942b364d9586b0ef5912432"
    cls = get_class_from_dynamic_module(
        "lfm2_vl.Lfm2VlForConditionalGeneration", "LiquidAI/d1-3B", revision=revision, code_revision=revision
    )
    config = Lfm2VlConfig(
        architectures=["Lfm2VlForConditionalGeneration"],
        image_token_id=31,
        tie_word_embeddings=True,
        projector_hidden_size=64,
        projector_use_layernorm=False,
        projector_bias=True,
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "conv_bias": False,
            "block_auto_adjust_ff_dim": False,
            "pad_token_id": 0,
            "use_cache": False,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "patch_size": 16,
            "num_patches": 16,
            "vision_use_head": False,
        },
    )
    reference = cls(config).cuda().eval()
    reference.tie_weights()
    native, registry = _native_with_weights(reference)
    assert native.lm_head.weight is native.get_input_embeddings().weight
    if modality == "image":
        ids = torch.tensor([[1, 31, 31, 31, 31, 2, 3]], device="cuda")
        media = {
            "pixel_values": torch.randn(1, 16, 768, device="cuda"),
            "spatial_shapes": torch.tensor([[4, 4]], device="cuda"),
            "pixel_attention_mask": torch.ones(1, 16, device="cuda", dtype=torch.int32),
        }
    else:
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device="cuda")
        media = {}
    expected = reference(ids, use_cache=False, **media).logits
    actual = native(ids, **media).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    probe = torch.randn_like(actual)
    (actual * probe).mean().backward()
    (expected * probe).mean().backward()
    _compare_gradients(native, reference, registry, atol=2e-5, rtol=2e-4)
