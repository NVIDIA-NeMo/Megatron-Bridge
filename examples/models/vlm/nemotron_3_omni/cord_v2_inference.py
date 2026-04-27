# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CORD-V2 inference for a Nemotron Omni Megatron checkpoint.

Loads N samples from the CORD-V2 HF dataset, runs the Megatron model on each
(image + "Describe this image." prompt), and writes a JSON file containing the
input prompt, the gold response (parsed receipt JSON as a token string), and
the model's generated prediction for each sample. Image bytes are also saved
to disk so the outputs can be eyeballed later.

Vision backbone: dynamic resolution, temporal_patch_dim=1, separate_video_embedder=True
(matches `nemotron_omni_cord_v2_sft_config` with the updated dynamic-resolution training).

Usage:
  uv run torchrun --nproc-per-node=8 examples/models/vlm/nemotron_3_omni/cord_v2_inference.py \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path ${WORKSPACE}/results/nemotron_omni_cord_v2_sft_config_sft/checkpoints \
    --tp 4 --ep 2 \
    --max_samples 100 \
    --output ${WORKSPACE}/results/cord_v2_eval.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets.hf_dataset_makers import make_cord_v2_dataset
from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0

_VISION_PATCH_DIM = 16


def _build_vision_packed_seq_params(imgs_sizes: Optional[torch.Tensor]) -> Optional[PackedSeqParams]:
    """PackedSeqParams from per-image (H, W) sizes — mirrors nemotron_omni_step."""
    if imgs_sizes is None or imgs_sizes.numel() == 0:
        return None
    sizes = imgs_sizes.tolist() if torch.is_tensor(imgs_sizes) else list(imgs_sizes)
    seq_lens = [(int(h) // _VISION_PATCH_DIM) * (int(w) // _VISION_PATCH_DIM) for h, w in sizes]
    cu = [0]
    for sl in seq_lens:
        cu.append(cu[-1] + sl)
    device = imgs_sizes.device if torch.is_tensor(imgs_sizes) else torch.device("cpu")
    cu_tensor = torch.tensor(cu, dtype=torch.int32, device=device)
    max_len = torch.tensor(max(seq_lens) if seq_lens else 0, dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_tensor,
        cu_seqlens_kv=cu_tensor,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
    )


class SingleBatchIterator:
    def __init__(self, input_ids, position_ids, attention_mask, **kwargs):
        self.batch = dict(tokens=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        for key in ("images", "imgs_sizes", "num_frames", "vision_packed_seq_params"):
            if kwargs.get(key) is not None:
                self.batch[key] = kwargs[key]
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def vlm_forward_step(data_iterator, model, **_):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }
    if "images" in batch:
        forward_args["images"] = batch["images"]
    else:
        forward_args["images"] = torch.tensor(
            [], dtype=torch.bfloat16, device=batch["tokens"].device
        ).reshape(0, 0, 0)
    for key in ("imgs_sizes", "num_frames", "vision_packed_seq_params"):
        if key in batch:
            forward_args[key] = batch[key]

    def loss_func(x, **_):
        return x

    output = model(**forward_args)
    if isinstance(output, tuple):
        output = output[0]
    return output, loss_func


def prepare_image_sample(tokenizer, processor, image, prompt, system_prompt=None):
    """Build input_ids and dynamic-resolution image tensors for a single image + text prompt."""
    text_content = f"<image>\n{prompt}"
    messages = [{"role": "user", "content": text_content}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    input_ids = inputs.input_ids

    # Adjust image tokens: collapse <img>...<image>...</img> to single <image> per tile.
    img_start_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_id = tokenizer.convert_tokens_to_ids("</img>")
    num_patches = torch.ones(1, dtype=torch.long)  # 1 tile per image
    if img_start_id != tokenizer.unk_token_id and (input_ids == img_start_id).any():
        input_ids = adjust_image_tokens(input_ids, num_patches, img_start_id, img_end_id)

    # Patchify [1, 3, H, W] → [1, num_patches, 3*P*P] (mirrors collate.py dynamic-res path).
    pv = inputs.pixel_values  # [1, 3, H, W], float32, already normalized
    P = _VISION_PATCH_DIM
    _, C, H, W = pv.shape
    py, px = H // P, W // P
    pv_patched = (
        pv[0]
        .reshape(3, py, P, px, P)
        .permute(1, 3, 0, 2, 4)
        .reshape(py * px, 3 * P * P)
        .unsqueeze(0)
        .contiguous()
        .bfloat16()
    )
    imgs_sizes = torch.tensor([[H, W]], dtype=torch.long)
    num_frames = torch.tensor([1], dtype=torch.long)

    return input_ids, pv_patched, imgs_sizes, num_frames


@torch.no_grad()
def generate(model, tokenizer, input_ids, images, imgs_sizes, num_frames, max_new_tokens=200):
    prompt_len = input_ids.size(1)
    input_ids = input_ids.cuda()
    images = images.cuda()
    imgs_sizes = imgs_sizes.cuda()
    num_frames = num_frames.cuda()

    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    generated_ids = input_ids.clone()
    stop_tokens = {tokenizer.eos_token_id}

    fwd_bwd = get_forward_backward_func()
    for _ in range(max_new_tokens):
        # Rebuild each iteration: RADIO mutates cu_seqlens_q in-place when inserting class tokens,
        # so reusing the same object would cause cu_seqlens to grow by class_token_len each step.
        vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes)
        iterator = SingleBatchIterator(
            input_ids, position_ids, attention_mask,
            images=images, imgs_sizes=imgs_sizes, num_frames=num_frames,
            vision_packed_seq_params=vision_packed_seq_params,
        )
        output = fwd_bwd(
            forward_step_func=vlm_forward_step,
            data_iterator=iterator,
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=input_ids.size(1),
            micro_batch_size=1,
            collect_non_loss_data=True,
        )
        if isinstance(output, list) and len(output) > 0:
            output = output[0]
            if isinstance(output, tuple):
                output = output[0]

        if parallel_state.is_pipeline_last_stage():
            world_size = parallel_state.get_tensor_model_parallel_world_size()
            gathered = [torch.zeros_like(output) for _ in range(world_size)]
            dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
            full = torch.cat(gathered, dim=2)
            next_token_ids = torch.argmax(full[:, -1], dim=-1, keepdim=True)
        else:
            next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

        dist.broadcast(next_token_ids, get_last_rank())
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        input_ids = generated_ids
        position_ids = (
            torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if int(next_token_ids.item()) in stop_tokens:
            break

    cleaned = tokenizer.decode(generated_ids[0, prompt_len:].tolist(), skip_special_tokens=True).strip()
    full_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
    return cleaned, full_text


def extract_gt_text_from_conversation(conv):
    """Extract the ground-truth assistant text from a CORD-V2 conversation."""
    for turn in conv:
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        return item.get("text", "")
            elif isinstance(content, str):
                return content
    return ""


def main():
    parser = argparse.ArgumentParser(description="CORD-V2 inference for Nemotron Omni")
    parser.add_argument("--hf_model_path", type=str, required=True)
    parser.add_argument("--megatron_model_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path (images saved next to it under cord_v2_images/)")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    args = parser.parse_args()

    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)
    model_provider = bridge.to_megatron_provider(load_weights=(args.megatron_model_path is None))
    model_provider.tensor_model_parallel_size = args.tp
    model_provider.pipeline_model_parallel_size = args.pp
    model_provider.expert_model_parallel_size = args.ep
    model_provider.expert_tensor_parallel_size = args.etp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.dynamic_resolution = True
    model_provider.temporal_patch_dim = 1
    model_provider.separate_video_embedder = True
    model_provider.temporal_ckpt_compat = True
    model_provider.vision_class_token_len = 10
    model_provider.initialize_model_parallel(seed=0)

    if args.megatron_model_path:
        print_rank_0(f"Loading Megatron checkpoint from {args.megatron_model_path}")
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": args.tp,
                "pipeline_model_parallel_size": args.pp,
                "expert_model_parallel_size": args.ep,
                "expert_tensor_parallel_size": args.etp,
                "pipeline_dtype": torch.bfloat16,
                "dynamic_resolution": True,
                "temporal_patch_dim": 1,
                "separate_video_embedder": True,
                "temporal_ckpt_compat": True,
                "vision_class_token_len": 10,
            },
            wrap_with_ddp=False,
        )
        model = [m.cuda().eval() for m in model]
        for m in model:
            inner = m.module if hasattr(m, "module") else m
            if hasattr(inner, "config"):
                inner.config.grad_scale_func = None
            if hasattr(inner, "llava_model") and hasattr(inner.llava_model, "config"):
                inner.llava_model.config.grad_scale_func = None
    else:
        print_rank_0(f"Converting HF from {args.hf_model_path} on the fly")
        model_provider.finalize()
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)
        model = [m.cuda().bfloat16().eval() for m in model]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_rank_0(f"Loading CORD-V2 split={args.split} ...")
    examples = make_cord_v2_dataset(split=args.split)
    n = min(args.max_samples, len(examples))
    print_rank_0(f"Running inference on {n}/{len(examples)} samples")

    output_path = Path(args.output)
    images_dir = output_path.parent / "cord_v2_images"
    if dist.get_rank() == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(n):
        ex = examples[i]
        conv = ex["conversation"]
        user_content = conv[0]["content"]
        image = None
        for item in user_content:
            if item.get("type") == "image":
                image = item.get("image")
        gt_text = extract_gt_text_from_conversation(conv)

        if dist.get_rank() == 0:
            img_path = images_dir / f"sample_{i:03d}.png"
            try:
                image.save(img_path)
            except Exception as e:
                print_rank_0(f"WARN: could not save sample image {i}: {e}")

        input_ids, pv_patched, imgs_sizes, num_frames = prepare_image_sample(
            tokenizer, processor, image, args.prompt
        )

        cleaned, prediction_full = generate(
            model, tokenizer, input_ids, pv_patched, imgs_sizes, num_frames,
            max_new_tokens=args.max_new_tokens,
        )

        record = {
            "sample_index": i,
            "image_path": str(images_dir / f"sample_{i:03d}.png"),
            "prompt": args.prompt,
            "ground_truth": gt_text,
            "prediction": cleaned,
            "prediction_full_decode": prediction_full,
        }
        results.append(record)
        print_rank_0(f"[{i + 1}/{n}] done (gt_len={len(gt_text)}, pred_len={len(cleaned)})")

    if dist.get_rank() == 0:
        with open(output_path, "w") as f:
            json.dump(
                {
                    "checkpoint": args.megatron_model_path,
                    "hf_model": args.hf_model_path,
                    "split": args.split,
                    "num_samples": len(results),
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )
        print_rank_0(f"Saved results to {output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
