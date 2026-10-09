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

"""Real DiffusionGemma FP32 HF vs Megatron encoder and decoder parity."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import unwrap_model
from real_encoder_parity import compare, device_inputs, encode_cases, native_hf_masks
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from megatron.bridge import AutoBridge


def hf_inputs(inputs):
    """Project processed inputs onto the HF forward signature."""
    return {
        key: inputs[key]
        for key in ("input_ids", "position_ids", "mm_token_type_ids", "pixel_values", "image_position_ids")
        if key in inputs
    }


def capture_hf_routes(model, call):
    """Capture reference router weights for encoder and decoder passes."""
    routes = []
    hooks = []
    routers = [layer.router for layer in model.model.encoder.language_model.layers]
    routers.extend(layer.router for layer in model.model.decoder.layers)
    for router in routers:

        def hook(_module, _inputs, output):
            probabilities, weights, indices = output
            dense = torch.zeros_like(probabilities, dtype=torch.float32)
            dense.scatter_(1, indices, weights.float())
            routes.append(dense.detach().cpu())

        hooks.append(router.register_forward_hook(hook))
    try:
        output = call()
    finally:
        for hook in hooks:
            hook.remove()
    return output, routes


def replay_routes(model, routes, call):
    """Replay fixed routing to isolate decoder arithmetic from top-k flips."""
    hooks = []
    index = 0
    for layer in model.language_model.decoder.layers:

        def hook(_module, _inputs, output):
            nonlocal index
            probabilities, routing_map = output
            weights = (
                routes[index].to(device=probabilities.device, dtype=probabilities.dtype).reshape_as(probabilities)
            )
            index += 1
            return weights, (weights != 0).reshape_as(routing_map)

        hooks.append(layer.mlp.router.register_forward_hook(hook))
    try:
        output = call()
    finally:
        for hook in hooks:
            hook.remove()
    if index != len(routes):
        raise RuntimeError(f"Consumed {index} HF routes, expected {len(routes)}")
    return output


def main() -> None:
    """Compare real-checkpoint FP32 text and image denoising outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-l2", type=float, default=1e-3)
    parser.add_argument("--min-token-cosine", type=float, default=0.9999)
    parser.add_argument(
        "--image-encoder-max-relative-l2",
        type=float,
        default=0.02,
        help="Diagnostic bound for image encoder hidden states; HF eager vs SDPA differs by ~9.9%%.",
    )
    parser.add_argument("--image-encoder-min-token-cosine", type=float, default=0.98)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    cases = encode_cases(processor)
    decoder_ids = torch.arange(256, dtype=torch.long)[None] % processor.tokenizer.vocab_size
    references = {}
    hf = DiffusionGemmaForBlockDiffusion.from_pretrained(
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
            masks = native_hf_masks(hf, inputs)
            canvas = decoder_ids.cuda()
            encoder = hf.model.encoder(**hf_inputs(inputs), attention_mask=masks, use_cache=False).last_hidden_state
            decoder_output, routes = capture_hf_routes(
                hf,
                lambda hf=hf: hf(
                    **hf_inputs(inputs),
                    attention_mask=masks,
                    decoder_input_ids=canvas,
                    decoder_attention_mask=torch.ones(
                        (1, inputs["input_ids"].shape[1] + canvas.shape[1]), device="cuda", dtype=torch.long
                    ),
                    decoder_position_ids=torch.arange(
                        inputs["input_ids"].shape[1], inputs["input_ids"].shape[1] + canvas.shape[1], device="cuda"
                    )[None],
                ),
            )
            references[name] = {
                "encoder": encoder.cpu(),
                "decoder": decoder_output.logits.float().cpu(),
                "routes": routes,
            }
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    bridge = AutoBridge.from_hf_pretrained(args.model_path, torch_dtype=torch.float32)
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
    provider.attention_backend = AttnBackend.unfused
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    megatron = unwrap_model(provider.provide_distributed_model(wrap_with_ddp=False)[0]).cuda().eval()

    results = {}
    with torch.no_grad():
        for name, raw_inputs in cases.items():
            inputs = device_inputs(raw_inputs, dtype=torch.float32)

            def megatron_call():
                return megatron(
                    input_ids=inputs["input_ids"],
                    decoder_input_ids=decoder_ids.cuda(),
                    position_ids=inputs["position_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    image_position_ids=inputs.get("image_position_ids"),
                    mm_token_type_ids=inputs.get("mm_token_type_ids"),
                    encoder_attention_mask=inputs["attention_mask"],
                )

            output = replay_routes(megatron, references[name]["routes"], megatron_call)
            results[name] = {
                "encoder": compare(references[name]["encoder"], output.encoder_last_hidden_state.cpu()),
                "decoder_logits": compare(references[name]["decoder"], output.logits.cpu()),
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "image_tokens": int((inputs["input_ids"] == provider.image_token_id).sum()),
            }

    def within(metrics, max_relative_l2, min_token_cosine):
        return metrics["relative_l2"] <= max_relative_l2 and metrics["min_token_cosine"] >= min_token_cosine

    passed = all(
        within(result["decoder_logits"], args.max_relative_l2, args.min_token_cosine) for result in results.values()
    )
    for result in results.values():
        if result["image_tokens"]:
            passed &= within(
                result["encoder"], args.image_encoder_max_relative_l2, args.image_encoder_min_token_cosine
            )
        else:
            passed &= within(result["encoder"], args.max_relative_l2, args.min_token_cosine)
    report = {
        "dtype": "float32",
        "hf_routing_replayed": True,
        "tp": 1,
        "ep": 1,
        "thresholds": {
            "outputs_and_text_encoder": {
                "max_relative_l2": args.max_relative_l2,
                "min_token_cosine": args.min_token_cosine,
            },
            "image_encoder_hidden_diagnostic": {
                "max_relative_l2": args.image_encoder_max_relative_l2,
                "min_token_cosine": args.image_encoder_min_token_cosine,
            },
        },
        "cases": results,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise AssertionError("Real DiffusionGemma FP32 parity failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
