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

"""Compare D1-3B's tiny native Bridge backbone with its pinned release."""

import argparse
import json
import logging
from typing import TYPE_CHECKING

import torch
from d1_bridge_validation import build_native, export_gradients, standalone_cuda
from d1_publisher_fixtures import initialize_nonunit_rmsnorm


if TYPE_CHECKING:
    from transformers import Lfm2VlConfig


def tiny_config() -> "Lfm2VlConfig":
    """Build the non-unit-norm text/image random checkpoint fixture."""
    from transformers import Lfm2VlConfig

    return Lfm2VlConfig(
        architectures=["Lfm2VlForConditionalGeneration"],
        image_token_id=31,
        tie_word_embeddings=True,
        projector_hidden_size=64,
        projector_use_layernorm=False,
        projector_bias=True,
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 96,
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


logger = logging.getLogger(__name__)


def validate(device: torch.device) -> None:
    """Compare complete tiny checkpoints; execute pinned publisher code explicitly."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.set_num_threads(1)
    torch.manual_seed(7)
    revision = "051bcc464b01b9f92942b364d9586b0ef5912432"
    reference_class = get_class_from_dynamic_module(
        "lfm2_vl.Lfm2VlForConditionalGeneration",
        "LiquidAI/d1-3B",
        revision=revision,
        code_revision=revision,
    )
    config = tiny_config()
    reference = reference_class(config).eval().to(device)
    # HF's released from_pretrained path reties after replacing the language stack.
    reference.tie_weights()
    nonunit_rmsnorms = initialize_nonunit_rmsnorm(reference)
    native, registry = build_native(config, reference)
    assert reference.lm_head.weight is reference.get_input_embeddings().weight
    assert native.lm_head.weight is native.get_input_embeddings().weight
    histories = {"reference": {}, "native": {}}

    def capture(side, name):
        """Record intermediate random-fixture activations for failure diagnosis."""

        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                histories[side][name] = value.detach().float().cpu()

        return hook

    reference_modules = dict(reference.named_modules())
    native_modules = dict(native.named_modules())
    for name in sorted(set(reference_modules) & set(native_modules)):
        if name.endswith(
            (
                "operator_norm",
                "ffn_norm",
                "q_layernorm",
                "k_layernorm",
                "self_attn",
                "conv.in_proj",
                "conv.out_proj",
                "feed_forward",
                "embedding_norm",
            )
        ):
            reference_modules[name].register_forward_hook(capture("reference", name))
            native_modules[name].register_forward_hook(capture("native", name))
    checks = []
    for dtype in (torch.float32, torch.bfloat16):
        reference.to(dtype)
        native.to(dtype)
        for modality in ("text", "image"):
            histories["reference"].clear()
            histories["native"].clear()
            if modality == "image":
                ids = torch.tensor([[1, 31, 31, 31, 31, 2, 3]], device=device)
                vision = {
                    "pixel_values": torch.randn(1, 16, 768, dtype=dtype, device=device),
                    "spatial_shapes": torch.tensor([[4, 4]], device=device),
                    "pixel_attention_mask": torch.ones(1, 16, dtype=torch.int32, device=device),
                }
            else:
                ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device=device)
                vision = {}
            expected = reference(ids, use_cache=False, **vision).logits
            actual = native(ids, **vision).logits
            tolerance = 2e-5 if dtype == torch.float32 else 0.01
            try:
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            except AssertionError:
                differences = []
                for name in sorted(set(histories["reference"]) & set(histories["native"])):
                    wanted, found = histories["reference"][name], histories["native"][name]
                    if wanted.shape != found.shape and wanted.shape == found.transpose(0, 1).shape:
                        found = found.transpose(0, 1)
                    differences.append(
                        {
                            "module": name,
                            "reference_shape": list(wanted.shape),
                            "native_shape": list(found.shape),
                            "max_error": float((found - wanted).abs().max()) if found.shape == wanted.shape else None,
                        }
                    )
                logger.info(
                    json.dumps(
                        {
                            "diagnostic_dtype": str(dtype),
                            "modality": modality,
                            "intermediate_differences": differences,
                        }
                    )
                )
                raise
            cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
            assert cosine > 0.9999
            native.zero_grad()
            reference.zero_grad()
            probe = torch.randn_like(actual, dtype=torch.float32)
            (actual.float() * probe).mean().backward()
            (expected.float() * probe).mean().backward()
            max_gradient_error = 0.0
            parameters_checked = 0
            native_gradients = export_gradients(native, registry)
            for name, parameter in reference.named_parameters():
                if parameter.grad is not None:
                    comparison = native_gradients[name]
                    try:
                        torch.testing.assert_close(comparison, parameter.grad, atol=tolerance, rtol=tolerance)
                    except AssertionError:
                        logger.info(
                            json.dumps(
                                {
                                    "diagnostic_dtype": str(dtype),
                                    "modality": modality,
                                    "parameter": name,
                                    "max_gradient_error": float((comparison - parameter.grad).abs().max()),
                                    "max_reference_gradient": float(parameter.grad.abs().max()),
                                    "max_native_gradient": float(comparison.abs().max()),
                                    "max_logit_abs_error": float((actual - expected).abs().max()),
                                }
                            )
                        )
                        raise
                    max_gradient_error = max(max_gradient_error, float((comparison - parameter.grad).abs().max()))
                    parameters_checked += 1
            checks.append(
                {
                    "modality": modality,
                    "dtype": str(dtype),
                    "max_logit_abs_error": float((actual - expected).abs().max().detach()),
                    "logit_cosine": float(cosine.detach()),
                    "max_parameter_gradient_error": max_gradient_error,
                    "parameters_with_gradients_checked": parameters_checked,
                }
            )
    logger.info(
        json.dumps(
            {
                "scope": "Bridge tiny D1-3B image/text backbone",
                "source_revision": revision,
                "checkpoint_keys": len(reference.state_dict()),
                "strict_full_weight_load": True,
                "nonunit_rmsnorms": nonunit_rmsnorms,
                "native_weight_round_trip": True,
                "tied_identity": "preserved",
                "torch_version": torch.__version__,
                "device": str(device),
                "checks": checks,
            }
        )
    )


def main() -> None:
    """Run only when explicitly invoked in the supported one-GPU environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with standalone_cuda() as device:
        validate(device)


if __name__ == "__main__":
    main()
