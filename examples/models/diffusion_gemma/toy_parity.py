#!/usr/bin/env python3
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

"""Tiny DiffusionGemma HF <-> Megatron conversion and encoder parity checks."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.transformer.enums import AttnBackend
from transformers.models.diffusion_gemma import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

from megatron.bridge import AutoBridge


def toy_config() -> DiffusionGemmaConfig:
    """Return a tiny, two-layer version of the public multimodal architecture."""
    return DiffusionGemmaConfig(
        architectures=["DiffusionGemmaForBlockDiffusion"],
        image_token_id=2000,
        boi_token_id=2001,
        eoi_token_id=2002,
        canvas_length=16,
        text_config={
            "vocab_size": 2048,
            "hidden_size": 256,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "global_head_dim": 128,
            "num_global_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "sliding_window": 16,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_bidirectional_attention": "vision",
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 64,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
            },
            "final_logit_softcapping": 30.0,
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "patch_size": 4,
            "pooling_kernel_size": 2,
            "position_embedding_size": 256,
            "standardize": True,
        },
    )


def text_case() -> dict[str, torch.Tensor]:
    """Build a deterministic text-only encoder input."""
    input_ids = torch.tensor([[2, 17, 44, 91, 105, 333, 777, 1024, 1500, 7, 8, 9]])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1])[None],
        "mm_token_type_ids": torch.zeros_like(input_ids),
    }


def image_case(config: DiffusionGemmaConfig) -> dict[str, torch.Tensor]:
    """Build a deterministic image/text input with the model's patch layout."""
    patch_size = config.vision_config.patch_size
    pooling = config.vision_config.pooling_kernel_size
    image = torch.linspace(-1.0, 1.0, 3 * 16 * 16).reshape(3, 16, 16)
    patches_height = image.shape[1] // patch_size
    patches_width = image.shape[2] // patch_size
    pixel_values = image.reshape(3, patches_height, patch_size, patches_width, patch_size)
    pixel_values = pixel_values.permute(1, 3, 2, 4, 0).reshape(1, patches_height * patches_width, -1)
    grid_x, grid_y = torch.meshgrid(torch.arange(patches_width), torch.arange(patches_height), indexing="xy")
    image_position_ids = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
    soft_tokens = pixel_values.shape[1] // pooling**2
    input_ids = torch.tensor(
        [[2, 17, config.boi_token_id, *([config.image_token_id] * soft_tokens), config.eoi_token_id, 44, 91, 105]]
    )
    mm_token_type_ids = (input_ids == config.image_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1])[None],
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": pixel_values,
        "image_position_ids": image_position_ids,
    }


def native_hf_encoder_inputs(model: DiffusionGemmaForBlockDiffusion, inputs: dict[str, torch.Tensor]) -> dict:
    """Pass HF's vision-aware mask explicitly; 5.12.1 drops it for 2D masks."""
    encoder = model.model.encoder
    masks = encoder.create_masks_for_generate(
        config=model.config,
        inputs_embeds=torch.empty((*inputs["input_ids"].shape, 0), device=inputs["input_ids"].device),
        attention_mask=inputs["attention_mask"].bool(),
        past_key_values=None,
        position_ids=inputs["position_ids"],
        mm_token_type_ids=inputs["mm_token_type_ids"],
    )
    return {**inputs, "attention_mask": masks}


def main() -> None:
    """Run exact weight roundtrip and multimodal forward comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--attention-backend", choices=[b.name for b in AttnBackend], default="unfused")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "diffusiongemma-toy"
        config = toy_config()
        config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        hf_model = DiffusionGemmaForBlockDiffusion(config).to(torch.float32).eval()
        hf_model.save_pretrained(model_dir, safe_serialization=True)

        bridge = AutoBridge.from_hf_pretrained(model_dir, torch_dtype=torch.float32)
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = 1
        provider.expert_tensor_parallel_size = 1
        provider.sequence_parallel = False
        provider.params_dtype = torch.float32
        provider.pipeline_dtype = torch.float32
        provider.bf16 = False
        provider.fp16 = False
        provider.attention_backend = AttnBackend[args.attention_backend]
        provider.finalize()
        provider.initialize_model_parallel(seed=0)
        megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)
        megatron_model = [module.cuda().eval() for module in megatron_model]

        exported = 0
        mismatches = []
        for name, tensor in bridge.export_hf_weights(megatron_model, show_progress=False):
            original = bridge.hf_pretrained.state[name].to(tensor.device)
            exported += 1
            if tensor.shape != original.shape or not torch.equal(tensor, original):
                mismatches.append(name)
        expected = len(list(bridge.hf_pretrained.state.keys()))
        if exported != expected or mismatches:
            raise AssertionError(
                f"Export mismatch: exported={exported}, expected={expected}, mismatches={mismatches[:10]}"
            )

        hf_model = hf_model.cuda()
        cases = {"text": text_case(), "image": image_case(config)}
        results = {}
        for name, raw_inputs in cases.items():
            inputs = {key: value.cuda() for key, value in raw_inputs.items()}
            with torch.no_grad():
                hf_hidden = hf_model.model.encoder(
                    **native_hf_encoder_inputs(hf_model, inputs), use_cache=False
                ).last_hidden_state
                megatron_hidden = (
                    megatron_model[0]
                    .encode(
                        input_ids=inputs["input_ids"],
                        position_ids=inputs["position_ids"],
                        pixel_values=inputs.get("pixel_values"),
                        image_position_ids=inputs.get("image_position_ids"),
                        mm_token_type_ids=inputs.get("mm_token_type_ids"),
                    )
                    .last_hidden_state
                )
            results[name] = {
                "encoder_shape": list(hf_hidden.shape),
                "image_tokens": int((inputs["input_ids"] == config.image_token_id).sum()),
                "encoder_max_abs_diff": float((hf_hidden - megatron_hidden).abs().max()),
            }
        decoder_results = {}
        decoder_ids = torch.tensor(
            [[5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 102, 303, 707]], device="cuda"
        )
        for name, raw_inputs in cases.items():
            inputs = {key: value.cuda() for key, value in raw_inputs.items()}
            for self_conditioning in (False, True):
                sc_logits = None
                if self_conditioning:
                    sc_logits = torch.linspace(
                        -2, 2, decoder_ids.numel() * config.text_config.vocab_size, device="cuda"
                    )
                    sc_logits = sc_logits.reshape(1, decoder_ids.shape[1], config.text_config.vocab_size)
                with torch.no_grad():
                    hf_decoder = (
                        hf_model(
                            **{
                                key: inputs[key]
                                for key in (
                                    "input_ids",
                                    "position_ids",
                                    "mm_token_type_ids",
                                    "pixel_values",
                                    "image_position_ids",
                                )
                                if key in inputs
                            },
                            attention_mask=native_hf_encoder_inputs(hf_model, inputs)["attention_mask"],
                            decoder_input_ids=decoder_ids,
                            self_conditioning_logits=sc_logits,
                            decoder_attention_mask=torch.ones(
                                (1, inputs["input_ids"].shape[1] + decoder_ids.shape[1]),
                                device="cuda",
                                dtype=torch.long,
                            ),
                            decoder_position_ids=torch.arange(
                                inputs["input_ids"].shape[1],
                                inputs["input_ids"].shape[1] + decoder_ids.shape[1],
                                device="cuda",
                            )[None],
                        )
                        .logits.float()
                        .cpu()
                    )
                    megatron_decoder = (
                        megatron_model[0](
                            input_ids=inputs["input_ids"],
                            decoder_input_ids=decoder_ids,
                            position_ids=inputs["position_ids"],
                            pixel_values=inputs.get("pixel_values"),
                            image_position_ids=inputs.get("image_position_ids"),
                            mm_token_type_ids=inputs.get("mm_token_type_ids"),
                            encoder_attention_mask=inputs["attention_mask"],
                            self_conditioning_logits=sc_logits,
                        )
                        .logits.float()
                        .cpu()
                    )
                diff = hf_decoder - megatron_decoder
                decoder_results[f"{name}_self_conditioning_{self_conditioning}"] = {
                    "shape": list(hf_decoder.shape),
                    "max_abs_diff": float(diff.abs().max()),
                    "relative_l2": float(diff.norm() / hf_decoder.norm().clamp_min(1e-12)),
                }
        result = {
            "exported_tensors": exported,
            "weight_mismatches": 0,
            "cases": results,
            "decoder": decoder_results,
            "atol": args.atol,
            "attention_backend": args.attention_backend,
            "tf32_disabled": True,
            "passed": all(case["encoder_max_abs_diff"] <= args.atol for case in results.values())
            and all(case["relative_l2"] <= 0.001 for case in decoder_results.values()),
        }
        print(json.dumps(result, indent=2), flush=True)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2))
        if not result["passed"]:
            raise AssertionError(f"Encoder parity failed: {results}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
