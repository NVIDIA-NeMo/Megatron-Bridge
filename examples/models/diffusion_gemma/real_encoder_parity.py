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

"""Real DiffusionGemma HF vs Megatron TP1/EP1 encoder parity on synthetic inputs."""

import argparse
import copy
import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import unwrap_model
from PIL import Image
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from megatron.bridge import AutoBridge


def synthetic_image() -> Image.Image:
    """Render a deterministic synthetic patterned image."""
    image = Image.new("RGB", (768, 512), "white")
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            if (x // 32 + y // 32) % 2 == 0:
                pixels[x, y] = (230, 238, 250)
            if 80 <= y <= 110 and 60 <= x <= 700:
                pixels[x, y] = (20, 20, 20)
            if 180 <= y <= 200 and 80 <= x <= 520:
                pixels[x, y] = (90, 90, 90)
    return image


def encode_cases(processor):
    """Prepare synthetic text and image prompts with the public processor."""
    text_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Answer with a short JSON object describing a blue square."}],
        }
    ]
    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": synthetic_image()},
                {"type": "text", "text": "Describe the shapes in this synthetic image as JSON."},
            ],
        }
    ]
    cases = {}
    for name, messages in (("text", text_messages), ("image", image_messages)):
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = dict(batch)
        inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1])[None]
        cases[name] = inputs
    return cases


def device_inputs(inputs, dtype=torch.bfloat16):
    """Move processed input tensors to CUDA, casting only floating payloads."""
    output = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            output[key] = value.cuda()
            if output[key].is_floating_point():
                output[key] = output[key].to(dtype)
        else:
            output[key] = value
    return output


def native_hf_masks(model, inputs):
    """Build explicit HF encoder masks including image bidirectionality."""
    ids = inputs["input_ids"]
    return model.model.encoder.create_masks_for_generate(
        config=model.config,
        inputs_embeds=torch.empty((*ids.shape, 0), device=ids.device, dtype=model.dtype),
        attention_mask=inputs["attention_mask"].bool(),
        past_key_values=None,
        position_ids=inputs["position_ids"],
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )


def compare(reference, candidate):
    """Measure hidden-state differences and per-token cosine similarity."""
    reference = reference.float()
    candidate = candidate.float()
    difference = reference - candidate
    token_cosine = torch.nn.functional.cosine_similarity(reference[0], candidate[0], dim=-1)
    return {
        "shape": list(reference.shape),
        "max_abs_diff": float(difference.abs().max()),
        "mean_abs_diff": float(difference.abs().mean()),
        "relative_l2": float(difference.norm() / reference.norm().clamp_min(1e-12)),
        "min_token_cosine": float(token_cosine.min()),
        "mean_token_cosine": float(token_cosine.mean()),
    }


def layer_outputs(model, inputs):
    """Collect Megatron encoder states at each transformer layer."""
    captured = []
    hooks = []
    for layer in model.language_model.decoder.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output: captured.append(
                    (output[0] if isinstance(output, tuple) else output).detach().transpose(0, 1).float().cpu()
                )
            )
        )
    with torch.no_grad():
        hidden = model.encode(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_position_ids=inputs.get("image_position_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        ).last_hidden_state.cpu()
    for hook in hooks:
        hook.remove()
    return hidden, captured


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _flat(value, sequence_first=False):
    value = value.detach().float().cpu()
    if sequence_first and value.ndim == 3:
        value = value.transpose(0, 1)
    return value.reshape(-1, value.shape[-1])


def _relative(reference, candidate):
    reference = reference.reshape(-1, reference.shape[-1]).float()
    candidate = candidate.reshape(-1, candidate.shape[-1]).float()
    return float((reference - candidate).norm() / reference.norm().clamp_min(1e-12))


def hf_layer0_trace(model, inputs, layer_index=0):
    """Trace intermediate reference activations in a selected layer."""
    layer = model.model.encoder.language_model.layers[layer_index]
    trace = {}
    hooks = []

    def output_hook(name):
        return lambda _module, _inputs, output: trace.__setitem__(name, _flat(_first_tensor(output)))

    def input_hook(name):
        return lambda _module, inputs: trace.__setitem__(name, _flat(inputs[0]))

    hooks.extend(
        [
            layer.register_forward_pre_hook(input_hook("layer_input")),
            layer.post_attention_layernorm.register_forward_hook(output_hook("attention_output")),
            layer.pre_feedforward_layernorm.register_forward_pre_hook(input_hook("mlp_residual")),
            layer.pre_feedforward_layernorm.register_forward_hook(output_hook("shared_input")),
            layer.mlp.register_forward_hook(output_hook("shared_raw")),
            layer.post_feedforward_layernorm_1.register_forward_hook(output_hook("shared_normed")),
            layer.pre_feedforward_layernorm_2.register_forward_hook(output_hook("expert_input")),
            layer.experts.register_forward_hook(output_hook("routed_raw")),
            layer.post_feedforward_layernorm_2.register_forward_hook(output_hook("routed_normed")),
            layer.post_feedforward_layernorm.register_forward_hook(output_hook("ffn_normed")),
            layer.register_forward_hook(output_hook("layer_output")),
        ]
    )

    def router_hook(_module, _inputs, output):
        probabilities, weights, indices = output
        dense = torch.zeros_like(probabilities, dtype=torch.float32)
        dense.scatter_(1, indices, weights.float())
        trace["router_probabilities"] = probabilities.detach().float().cpu()
        trace["router_weights"] = dense.detach().cpu()
        trace["router_indices"] = indices.detach().cpu()

    hooks.append(layer.router.register_forward_hook(router_hook))
    with torch.no_grad():
        model.model.encoder(
            **{
                key: inputs[key]
                for key in ("input_ids", "position_ids", "mm_token_type_ids", "pixel_values", "image_position_ids")
                if key in inputs
            },
            attention_mask=native_hf_masks(model, inputs),
            use_cache=False,
        )
    for hook in hooks:
        hook.remove()
    return trace


def megatron_layer0_trace(model, inputs, layer_index=0):
    """Trace the matching Megatron layer's intermediate activations."""
    layer = model.language_model.decoder.layers[layer_index]
    trace = {}
    hooks = []

    def output_hook(name):
        return lambda _module, _inputs, output: trace.__setitem__(name, _flat(_first_tensor(output), True))

    def input_hook(name):
        def hook(_module, inputs, kwargs):
            value = inputs[0] if inputs else kwargs["hidden_states"]
            trace[name] = _flat(value, True)

        return hook

    hooks.extend(
        [
            layer.register_forward_pre_hook(input_hook("layer_input"), with_kwargs=True),
            layer.self_attention.register_forward_hook(output_hook("attention_output")),
            layer.mlp.shared_experts.register_forward_hook(output_hook("shared_raw")),
            layer.mlp.post_shared_expert_layernorm.register_forward_hook(output_hook("shared_normed")),
            layer.mlp.post_moe_layernorm.register_forward_pre_hook(
                lambda _module, inputs: trace.__setitem__("routed_raw", _flat(inputs[0], True))
            ),
            layer.mlp.post_moe_layernorm.register_forward_hook(output_hook("routed_normed")),
            layer.post_ffn_layernorm.register_forward_hook(output_hook("ffn_normed")),
            layer.register_forward_hook(output_hook("layer_output")),
        ]
    )

    def router_hook(_module, _inputs, output):
        probabilities, routing_map = output
        trace["router_weights"] = probabilities.detach().float().cpu().reshape(-1, probabilities.shape[-1])
        trace["router_indices"] = torch.topk(trace["router_weights"], k=model.config.moe_router_topk, dim=-1).indices

    hooks.append(layer.mlp.router.register_forward_hook(router_hook))
    original = layer.mlp.forward_with_separate_inputs

    def wrapped(expert_input, shared_input, router_input, padding_mask=None):
        trace["expert_input"] = _flat(expert_input, True)
        trace["shared_input"] = _flat(shared_input, True)
        trace["mlp_residual"] = _flat(router_input, True)
        return original(expert_input, shared_input, router_input, padding_mask=padding_mask)

    layer.mlp.forward_with_separate_inputs = wrapped
    try:
        with torch.no_grad():
            model.encode(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_position_ids=inputs.get("image_position_ids"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
    finally:
        layer.mlp.forward_with_separate_inputs = original
        for hook in hooks:
            hook.remove()
    return trace


def compare_layer0(reference, candidate):
    """Compare corresponding intermediate tensors from the selected layer."""
    names = [
        "layer_input",
        "attention_output",
        "mlp_residual",
        "shared_input",
        "shared_raw",
        "shared_normed",
        "expert_input",
        "router_weights",
        "routed_raw",
        "routed_normed",
        "ffn_normed",
        "layer_output",
    ]
    result = {
        name: _relative(reference[name], candidate[name]) for name in names if name in reference and name in candidate
    }
    hf_indices = torch.sort(reference["router_indices"], dim=-1).values
    megatron_indices = torch.sort(candidate["router_indices"], dim=-1).values
    exact = (hf_indices == megatron_indices).all(dim=-1)
    result["router_topk_exact_token_fraction"] = float(exact.float().mean())
    if "router_probabilities" in reference:
        k = reference["router_indices"].shape[-1]
        sorted_probabilities = torch.sort(reference["router_probabilities"], dim=-1, descending=True).values
        margins = sorted_probabilities[:, k - 1] - sorted_probabilities[:, k]
        result["router_topk_margin_min"] = float(margins.min())
        result["router_topk_margin_median"] = float(margins.median())
        mismatched = torch.nonzero(~exact, as_tuple=False).flatten()
        result["router_mismatch_tokens"] = mismatched.tolist()
        result["router_mismatch_margins"] = margins[mismatched].tolist()
    return result


def compare_state(reference_module, candidate_module):
    """Audit matching perception module state tensors exactly."""
    reference = reference_module.state_dict()
    candidate = candidate_module.state_dict()
    missing = sorted(set(reference) - set(candidate))
    unexpected = sorted(set(candidate) - set(reference))
    mismatches = []
    for name in sorted(set(reference) & set(candidate)):
        expected = reference[name].detach().cpu()
        actual = candidate[name].detach().cpu()
        if expected.shape != actual.shape:
            mismatches.append(
                {"name": name, "reason": "shape", "expected": list(expected.shape), "actual": list(actual.shape)}
            )
            continue
        diff = (expected.float() - actual.float()).abs().max().item() if expected.numel() else 0.0
        if expected.dtype != actual.dtype or diff:
            mismatches.append(
                {
                    "name": name,
                    "expected_dtype": str(expected.dtype),
                    "actual_dtype": str(actual.dtype),
                    "max_abs_diff": float(diff),
                }
            )
    return {
        "tensors": len(reference),
        "missing": missing[:20],
        "unexpected": unexpected[:20],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
    }


def vision_layer_outputs(vision_tower, inputs):
    """Collect perception tower intermediate states for the fixed image."""
    captured = {}
    hooks = [
        vision_tower.patch_embedder.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("patch_embedder", output.detach().float().cpu())
        )
    ]
    for index, layer in enumerate(vision_tower.encoder.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: captured.__setitem__(
                    f"layer_{index}", (output[0] if isinstance(output, tuple) else output).detach().float().cpu()
                )
            )
        )
    with torch.no_grad():
        output = (
            vision_tower(
                pixel_values=inputs["pixel_values"],
                pixel_position_ids=inputs["image_position_ids"],
            )
            .last_hidden_state.detach()
            .float()
            .cpu()
        )
    for hook in hooks:
        hook.remove()
    captured["vision_output"] = output
    return captured


def main() -> None:
    """Audit the real multimodal encoder and isolate first divergences."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-token-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    cases = encode_cases(processor)
    hf = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    hf_vision = copy.deepcopy(hf.model.encoder.vision_tower).cpu()
    hf_embed_vision = copy.deepcopy(hf.model.encoder.embed_vision).cpu()
    references = {}
    with torch.no_grad():
        for name, raw_inputs in cases.items():
            inputs = device_inputs(raw_inputs)
            model_inputs = {
                key: inputs[key]
                for key in ("input_ids", "position_ids", "mm_token_type_ids", "pixel_values", "image_position_ids")
                if key in inputs
            }
            hf_output = hf.model.encoder(
                **model_inputs,
                attention_mask=native_hf_masks(hf, inputs),
                use_cache=False,
                output_hidden_states=True,
            )
            references[name] = {
                "last_hidden_state": hf_output.last_hidden_state.cpu(),
                "layers": [hidden.cpu() for hidden in hf_output.hidden_states[1:]],
                "layer0_trace": hf_layer0_trace(hf, inputs),
            }
            if "pixel_values" in inputs:
                references[name]["image_features"] = hf.model.encoder.get_image_features(
                    inputs["pixel_values"], inputs["image_position_ids"], return_dict=True
                ).pooler_output.cpu()
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    hf_fp32 = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    with torch.no_grad():
        for name, raw_inputs in cases.items():
            inputs = device_inputs(raw_inputs, dtype=torch.float32)
            model_inputs = {
                key: inputs[key]
                for key in ("input_ids", "position_ids", "mm_token_type_ids", "pixel_values", "image_position_ids")
                if key in inputs
            }
            references[name]["fp32_last_hidden_state"] = hf_fp32.model.encoder(
                **model_inputs,
                attention_mask=native_hf_masks(hf_fp32, inputs),
                use_cache=False,
            ).last_hidden_state.cpu()
            references[name]["hf_bf16_vs_fp32"] = compare(
                references[name]["fp32_last_hidden_state"], references[name]["last_hidden_state"]
            )
    del hf_fp32
    gc.collect()
    torch.cuda.empty_cache()

    bridge = AutoBridge.from_hf_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.params_dtype = torch.bfloat16
    provider.pipeline_dtype = torch.bfloat16
    provider.bf16 = True
    provider.fp16 = False
    provider.attention_backend = AttnBackend.unfused
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    megatron = unwrap_model(provider.provide_distributed_model(wrap_with_ddp=False)[0]).cuda().eval()
    for module in megatron.vision_tower.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
    hf_vision = hf_vision.cuda().eval()
    hf_embed_vision = hf_embed_vision.cuda().eval()
    vision_diagnostics = {
        "vision_tower_state": compare_state(hf_vision, megatron.vision_tower),
        "embed_vision_state": compare_state(hf_embed_vision, megatron.embed_vision),
    }

    results = {}
    with torch.no_grad():
        for name, raw_inputs in cases.items():
            inputs = device_inputs(raw_inputs)
            candidate, candidate_layers = layer_outputs(megatron, inputs)
            references_case = references[name]
            results[name] = compare(references_case["last_hidden_state"], candidate)
            results[name]["megatron_bf16_vs_hf_fp32"] = compare(references_case["fp32_last_hidden_state"], candidate)
            results[name]["hf_bf16_vs_hf_fp32"] = references_case["hf_bf16_vs_fp32"]
            layer_diffs = [
                compare(reference_layer, candidate_layer)["relative_l2"]
                for reference_layer, candidate_layer in zip(references_case["layers"], candidate_layers, strict=True)
            ]
            results[name]["layer_relative_l2_first_10"] = layer_diffs[:10]
            results[name]["layer_relative_l2_max"] = max(layer_diffs)
            results[name]["layer0_components"] = compare_layer0(
                references_case["layer0_trace"], megatron_layer0_trace(megatron, inputs)
            )
            if "image_features" in references_case:
                hf_vision_outputs = vision_layer_outputs(hf_vision, inputs)
                megatron_vision_outputs = vision_layer_outputs(megatron.vision_tower, inputs)
                vision_diagnostics["layer_outputs"] = {
                    name: compare(
                        hf_vision_outputs[name].reshape(1, -1, hf_vision_outputs[name].shape[-1]),
                        megatron_vision_outputs[name].reshape(1, -1, megatron_vision_outputs[name].shape[-1]),
                    )["relative_l2"]
                    for name in hf_vision_outputs
                }
                candidate_features = megatron.get_image_features(
                    inputs["pixel_values"], image_position_ids=inputs["image_position_ids"]
                ).cpu()
                results[name]["image_features"] = compare(
                    references_case["image_features"].reshape(1, -1, references_case["image_features"].shape[-1]),
                    candidate_features.reshape(1, -1, candidate_features.shape[-1]),
                )
            results[name]["candidate_layer_count"] = len(candidate_layers)
            results[name]["input_tokens"] = int(inputs["input_ids"].shape[1])
            results[name]["image_tokens"] = int((inputs["input_ids"] == provider.image_token_id).sum())

    passed = all(
        result["min_token_cosine"] >= args.min_token_cosine and result["relative_l2"] <= args.max_relative_l2
        for result in results.values()
    )
    report = {
        "model_path": args.model_path,
        "tp": 1,
        "ep": 1,
        "dtype": "bfloat16",
        "attention_backend": "unfused",
        "tf32_disabled": True,
        "thresholds": {"min_token_cosine": args.min_token_cosine, "max_relative_l2": args.max_relative_l2},
        "cases": results,
        "vision_diagnostics": vision_diagnostics,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise AssertionError("Real DiffusionGemma encoder parity failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
