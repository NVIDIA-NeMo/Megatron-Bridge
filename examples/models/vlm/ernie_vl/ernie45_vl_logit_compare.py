# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Logit comparison between HuggingFace and Megatron ERNIE 4.5 VL MoE models.

This script compares 1-step forward-pass logits between the HuggingFace ERNIE 4.5
VL MoE model and its Megatron-Core conversion via AutoBridge. It is designed for
validating weight conversion correctness with the real 28B model.

Strategy (sequential, same GPU):
    1. Load HF model on rank 0 GPU -> forward pass -> save logits to CPU
    2. Delete HF model, free GPU memory
    3. Load Megatron model (via AutoBridge) across GPUs -> forward pass -> gather logits
    4. Compare logits: cosine similarity, top-k token match, absolute diff

Launch:
    # Text-only comparison (TP=2, EP=2 -> 4 GPUs):
    torchrun --nproc_per_node=4 ernie45_vl_logit_compare.py \
        --hf-model-path /path/to/ERNIE-4.5-VL-28B-A3B-Thinking \
        --prompt "Hello, how are you?" \
        --tp 2 --ep 2

    # Single-GPU comparison:
    torchrun --nproc_per_node=1 ernie45_vl_logit_compare.py \
        --hf-model-path /path/to/ERNIE-4.5-VL-28B-A3B-Thinking \
        --prompt "Hello"

    # With image (VL inference):
    torchrun --nproc_per_node=1 ernie45_vl_logit_compare.py \
        --hf-model-path /path/to/ERNIE-4.5-VL-28B-A3B-Thinking \
        --prompt "Describe this image." \
        --image-path /path/to/image.jpg

Exit code 0 = PASS (cosine similarity >= threshold), non-zero = FAIL.
"""

import argparse
import gc
import os
import sys

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import disable_mtp_for_inference

SIMILARITY_THRESHOLD = 0.98


def _is_rank_0() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def print_rank_0(msg: str):
    if _is_rank_0():
        print(msg, flush=True)


# ========================================================================== #
# Image+Text Preprocessing
# ========================================================================== #

def preprocess_image_text(hf_model_path: str, prompt: str, image_path: str):
    """Use the HF processor to preprocess an image+text prompt.

    Builds the chat template with image placeholder, runs the processor to get
    input_ids, position_ids, pixel patches, grid_thw, and token_type_ids.

    Returns a dict with all tensors needed for both HF and Megatron forward.
    """
    from transformers import AutoProcessor
    from PIL import Image

    processor = AutoProcessor.from_pretrained(
        hf_model_path, trust_remote_code=True
    )

    # Build chat messages with image
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_path}},
        ]
    }]

    # Apply chat template
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print_rank_0(f"  Chat template text (first 200 chars): {text[:200]}...")

    # Process vision info (loads and resizes images)
    image_inputs, video_inputs = processor.process_vision_info(messages)
    print_rank_0(f"  Images loaded: {len(image_inputs) if image_inputs else 0}")

    # Run the processor to get all inputs
    proc_out = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    print_rank_0(f"  Processor output keys: {list(proc_out.keys())}")
    print_rank_0(f"  input_ids shape: {proc_out['input_ids'].shape}")
    if "images" in proc_out:
        print_rank_0(f"  images (pixel patches) shape: {proc_out['images'].shape}")
    if "grid_thw" in proc_out:
        print_rank_0(f"  grid_thw: {proc_out['grid_thw']}")
    if "position_ids" in proc_out:
        print_rank_0(f"  position_ids shape: {proc_out['position_ids'].shape}")
    if "token_type_ids" in proc_out:
        print_rank_0(f"  token_type_ids shape: {proc_out['token_type_ids'].shape}")

    return proc_out, processor


# ========================================================================== #
# Phase 1: HF Model Forward
# ========================================================================== #

def run_hf_forward(
    hf_model_path: str,
    input_ids: torch.Tensor,
    tokenizer,
    processor_output=None,
    processor=None,
) -> torch.Tensor:
    """Run HF model forward pass on rank 0 and return last-token logits (on CPU).

    If processor_output is provided, uses image+text inputs.
    Otherwise, runs text-only with simple 3D M-RoPE position_ids.

    Returns None on non-rank-0 processes.
    """
    if not _is_rank_0():
        return None

    print_rank_0("=== Phase 1: Loading HF model ===")
    from transformers import AutoModelForCausalLM

    # Load HF model directly onto GPU 0.  Using device_map={"": device} avoids
    # accelerate's AlignDevicesHook / meta-device dispatch, which breaks for
    # data-dependent ops (torch.nonzero) in the ERNIE VL MoE routing.
    # The 28B model fits in ~56 GB bf16 on a single 80 GB GPU.
    hf_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{hf_local_rank}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    ).eval()

    print_rank_0(f"  HF model loaded: {type(hf_model).__name__} on {device}")

    # Remove accelerate dispatch hooks.  Even with device_map={"": device},
    # accelerate may attach AlignDevicesHook to some modules.  These hooks
    # route tensors through meta-device shape inference, which breaks for
    # data-dependent ops (torch.nonzero) used in ERNIE VL MoE routing.
    from accelerate.hooks import remove_hook_from_module
    for _name, _module in hf_model.named_modules():
        remove_hook_from_module(_module)
    print_rank_0("  Removed accelerate dispatch hooks from all modules.")

    # Disable use_correction_bias on all MoE layers.  This flag controls
    # accumulation of expert usage statistics (expert_num_local) for the
    # auxiliary-loss correction bias -- a training-only feature.  During
    # inference the code path is unnecessary and triggers a torch.nonzero()
    # error when experts_type_mask boolean tensors are used for fancy
    # indexing on meta-dispatched tensors.
    _fixed_moe = 0
    for _name, _module in hf_model.named_modules():
        if hasattr(_module, 'use_correction_bias') and _module.use_correction_bias:
            _module.use_correction_bias = False
            _fixed_moe += 1
    if _fixed_moe:
        print_rank_0(f"  Disabled use_correction_bias on {_fixed_moe} MoE layers (inference-only).")

    # Safety: fix inv_freq if stuck on meta device (only happens with
    # device_map="auto").  With device_map={"": device} this is a no-op.
    for name, module in hf_model.named_modules():
        if hasattr(module, 'inv_freq') and isinstance(module.inv_freq, torch.Tensor):
            if module.inv_freq.device.type == 'meta':
                dim = module.inv_freq.shape[0] * 2  # inv_freq has shape [dim//2]
                theta = 10000.0
                module.inv_freq = 1.0 / (theta ** (
                    torch.arange(0, dim, 2, dtype=torch.float32) / dim
                ))
                print_rank_0(f"  Fixed meta inv_freq in {name} -> CPU, shape={module.inv_freq.shape}")

    if processor_output is not None:
        # Image+text VL forward
        # Register image preprocessor for GPU-side pixel normalization
        if processor is not None and hasattr(hf_model, 'add_image_preprocess'):
            hf_model.add_image_preprocess(processor)
            print_rank_0("  Image preprocessor registered on HF model")

        print_rank_0("  Running HF forward pass (image+text)...")
        forward_kwargs = {
            "input_ids": processor_output["input_ids"].to(device),
        }

        # Map processor output keys to HF model forward parameter names
        if "images" in processor_output:
            forward_kwargs["images"] = processor_output["images"].to(device)
        if "grid_thw" in processor_output:
            forward_kwargs["grid_thw"] = processor_output["grid_thw"].to(device)
        if "position_ids" in processor_output:
            forward_kwargs["position_ids"] = processor_output["position_ids"].to(device)
        if "token_type_ids" in processor_output:
            # HF forward() expects token_type_ids with shape [bsz, seq_len+1].
            # The extra trailing element is a "next-token type" lookahead
            # (normally appended by prepare_inputs_for_generation, but we call
            # forward() directly).  Append a zero (TokenType.text) to match.
            tti = processor_output["token_type_ids"]
            if tti.shape[1] == processor_output["input_ids"].shape[1]:
                tti = torch.cat([tti, torch.zeros(tti.shape[0], 1, dtype=tti.dtype)], dim=1)
            forward_kwargs["token_type_ids"] = tti.to(device)
        if "image_type_ids" in processor_output:
            forward_kwargs["image_type_ids"] = processor_output["image_type_ids"].to(device)

        with torch.no_grad():
            hf_output = hf_model(**forward_kwargs)
    else:
        # Text-only forward
        print_rank_0("  Running HF forward pass (text-only)...")
        seq_len = input_ids.size(1)
        # 3D M-RoPE position_ids: [batch, seq_len, 3] -- for text-only, all 3 dims identical
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)   # [1, seq_len]
            .unsqueeze(-1)  # [1, seq_len, 1]
            .expand(1, seq_len, 3)  # [1, seq_len, 3]
            .clone()
        )
        with torch.no_grad():
            hf_output = hf_model(
                input_ids=input_ids.to(device),
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool, device=device),
                position_ids=position_ids,
            )

    hf_logits = hf_output.logits[0, -1, :].float().cpu()  # [vocab_size]

    hf_next_token = torch.argmax(hf_logits)
    top5_vals, top5_ids = torch.topk(hf_logits, 5)
    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]

    print_rank_0(f"  HF logits shape: {hf_output.logits.shape}")
    print_rank_0(f"  HF logits stats: mean={hf_logits.mean():.4f}, std={hf_logits.std():.4f}")
    print_rank_0(f"  HF next token: {hf_next_token.item()} ('{tokenizer.decode([hf_next_token.item()])}')")
    print_rank_0(f"  HF Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")

    # Free HF model
    del hf_model, hf_output
    gc.collect()
    torch.cuda.empty_cache()
    print_rank_0("  HF model freed.")

    return hf_logits


# ========================================================================== #
# Phase 2: Megatron Model Forward
# ========================================================================== #

class SingleBatchIterator:
    def __init__(self, batch):
        self.batch = batch
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def ernie_vl_forward_step(data_iterator, model, **kwargs):
    """Forward step for ERNIE 4.5 VL model (text or image+text, no loss)."""
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "mm_token_type_ids": batch["mm_token_type_ids"],
        "attention_mask": None,
    }
    if "pixel_values" in batch and batch["pixel_values"] is not None:
        forward_args["pixel_values"] = batch["pixel_values"]
    if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
        forward_args["image_grid_thw"] = batch["image_grid_thw"]
    # Pass moe_mm_token_type_ids for dual-pool MoE routing (text vs vision experts).
    # Without this, all tokens are routed to text_moe_layer only, which causes
    # significant logit divergence for image+text inputs.
    if "moe_mm_token_type_ids" in batch and batch["moe_mm_token_type_ids"] is not None:
        forward_args["moe_mm_token_type_ids"] = batch["moe_mm_token_type_ids"]

    output = model(**forward_args)
    if isinstance(output, tuple):
        output = output[0]

    def loss_func(x, **kwargs):
        return x

    return output, loss_func


def run_megatron_forward(
    hf_model_path: str,
    input_ids: torch.Tensor,
    tokenizer,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    pixel_values=None,
    image_grid_thw=None,
    mm_token_type_ids=None,
) -> torch.Tensor:
    """Load Megatron model via AutoBridge, run forward, return last-token logits (on CPU).

    Returns logits only on last pipeline stage + TP rank 0 + EP rank 0.
    Returns None on other ranks.
    """
    print_rank_0("=== Phase 2: Loading Megatron model via AutoBridge ===")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=42)

    megatron_models = model_provider.provide_distributed_model(wrap_with_ddp=False)

    for m in megatron_models:
        disable_mtp_for_inference(m)
        m.eval()
        if hasattr(m, "config"):
            m.config.deallocate_pipeline_outputs = False

    print_rank_0(f"  Megatron model built. {len(megatron_models)} component(s).")

    # Prepare input
    seq_len = input_ids.size(1)
    input_ids_cuda = input_ids.cuda()

    if mm_token_type_ids is None:
        mm_token_type_ids = torch.zeros(1, seq_len, dtype=torch.int32, device="cuda")
    else:
        mm_token_type_ids = mm_token_type_ids.to(dtype=torch.int32, device="cuda")

    batch = {
        "tokens": input_ids_cuda,
        "mm_token_type_ids": mm_token_type_ids,
        # moe_mm_token_type_ids drives dual-pool MoE routing: text tokens (0)
        # go to text_moe_layer, vision tokens (>=1) go to vision_moe_layer.
        # For the logit comparison, it uses the same values as mm_token_type_ids.
        "moe_mm_token_type_ids": mm_token_type_ids.clone(),
    }
    if pixel_values is not None:
        batch["pixel_values"] = pixel_values.cuda()
        print_rank_0(f"  pixel_values shape: {batch['pixel_values'].shape}")
    if image_grid_thw is not None:
        batch["image_grid_thw"] = image_grid_thw.cuda()
        print_rank_0(f"  image_grid_thw: {batch['image_grid_thw']}")

    # Forward
    print_rank_0("  Running Megatron forward pass...")
    with torch.no_grad():
        fwd_bwd_func = get_forward_backward_func()
        iterator = SingleBatchIterator(batch)

        output = fwd_bwd_func(
            forward_step_func=ernie_vl_forward_step,
            data_iterator=iterator,
            model=megatron_models,
            num_microbatches=1,
            forward_only=True,
            seq_length=seq_len,
            micro_batch_size=1,
            collect_non_loss_data=True,
        )

    # Process output on last pipeline stage
    is_last_stage = (
        not dist.is_initialized()
        or parallel_state.is_pipeline_last_stage()
    )

    megatron_logits_cpu = None

    if is_last_stage:
        if isinstance(output, list) and len(output) > 0:
            output = output[0]

        # Gather TP shards
        if dist.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
            world_size = parallel_state.get_tensor_model_parallel_world_size()
            gathered = [torch.zeros_like(output) for _ in range(world_size)]
            dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
            output = torch.cat(gathered, dim=2)

        megatron_logits = output[0, -1, :].float()  # [padded_vocab_size]

        is_primary = (
            not dist.is_initialized()
            or (
                parallel_state.get_tensor_model_parallel_rank() == 0
                and parallel_state.get_expert_model_parallel_rank() == 0
            )
        )

        if is_primary:
            megatron_next_token = torch.argmax(megatron_logits)
            top5_vals, top5_ids = torch.topk(megatron_logits, 5)
            top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]

            print_rank_0(f"  Megatron output shape: {output.shape}")
            print_rank_0(f"  Megatron logits stats: mean={megatron_logits.mean():.4f}, std={megatron_logits.std():.4f}")
            print_rank_0(f"  Megatron next token: {megatron_next_token.item()} ('{tokenizer.decode([megatron_next_token.item()])}')")
            print_rank_0(f"  Megatron Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")

            megatron_logits_cpu = megatron_logits.cpu()

    return megatron_logits_cpu


# ========================================================================== #
# Phase 3: Comparison
# ========================================================================== #

def compare_logits(hf_logits: torch.Tensor, megatron_logits: torch.Tensor, tokenizer, threshold: float = SIMILARITY_THRESHOLD):
    """Compare HF and Megatron logits. Returns True if pass."""
    print_rank_0("\n=== Phase 3: Comparing Logits ===")

    # Truncate Megatron logits to HF vocab size (Megatron may pad vocab)
    hf_vocab_size = hf_logits.shape[0]
    megatron_logits_cmp = megatron_logits[:hf_vocab_size]

    # Token match
    hf_next = torch.argmax(hf_logits)
    mg_next = torch.argmax(megatron_logits_cmp)
    token_match = hf_next.item() == mg_next.item()

    hf_decoded = tokenizer.decode([hf_next.item()])
    mg_decoded = tokenizer.decode([mg_next.item()])

    print_rank_0(f"  HF next token:      {hf_next.item()} ('{hf_decoded}')")
    print_rank_0(f"  Megatron next token: {mg_next.item()} ('{mg_decoded}')")
    print_rank_0(f"  Token match: {token_match}")

    # Cosine similarity
    cosine_sim = torch.cosine_similarity(
        hf_logits.unsqueeze(0).float(),
        megatron_logits_cmp.unsqueeze(0).float(),
    ).item()
    print_rank_0(f"  Cosine similarity: {cosine_sim:.6f} ({cosine_sim * 100:.2f}%)")

    # Absolute diff
    diff = (hf_logits.float() - megatron_logits_cmp.float()).abs()
    print_rank_0(f"  Logits diff: max={diff.max():.6f}, mean={diff.mean():.6f}, median={diff.median():.6f}")

    # Top-5 overlap
    hf_top5 = set(torch.topk(hf_logits, 5).indices.tolist())
    mg_top5 = set(torch.topk(megatron_logits_cmp, 5).indices.tolist())
    overlap = len(hf_top5 & mg_top5)
    print_rank_0(f"  Top-5 overlap: {overlap}/5")

    passed = cosine_sim >= threshold
    status = "PASS" if passed else "FAIL"
    within = "within" if passed else "outside"
    print_rank_0(f"\n  Result: {status} (cosine {cosine_sim:.4f} {within} threshold {threshold})")

    return passed


# ========================================================================== #
# Main
# ========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="ERNIE 4.5 VL MoE logit comparison (HF vs Megatron)")
    parser.add_argument("--hf-model-path", required=True, help="Path to HF model directory")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Text prompt for comparison")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Cosine similarity threshold")
    parser.add_argument("--image-path", type=str, default=None, help="Path to image file for VL inference")
    args = parser.parse_args()

    print_rank_0("=== ERNIE 4.5 VL Logit Comparison ===")
    print_rank_0(f"  Model: {args.hf_model_path}")
    print_rank_0(f"  Prompt: '{args.prompt}'")
    print_rank_0(f"  Image: {args.image_path or '(none, text-only)'}")
    print_rank_0(f"  TP={args.tp}, PP={args.pp}, EP={args.ep}")
    print_rank_0(f"  Threshold: {args.threshold}")

    # Setup distributed (must happen before Phase 1 so _is_rank_0() works)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare inputs depending on mode (text-only vs image+text)
    processor_output = None
    processor = None
    pixel_values = None
    image_grid_thw = None
    mm_token_type_ids = None

    if args.image_path:
        # ============================================================
        # Image+Text mode: use processor to prepare all inputs
        # ============================================================
        print_rank_0("\n=== Preprocessing: Image+Text ===")
        processor_output, processor = preprocess_image_text(
            args.hf_model_path, args.prompt, args.image_path
        )
        input_ids = processor_output["input_ids"]

        # Extract vision tensors for Megatron side
        if "images" in processor_output:
            pixel_values = processor_output["images"]  # [N_patches, 588]
        if "grid_thw" in processor_output:
            image_grid_thw = processor_output["grid_thw"]  # [num_images, 3]

        # Build mm_token_type_ids for Megatron from processor's token_type_ids
        # The HF processor outputs token_type_ids with shape [bsz, seq_len+1]
        # (extra token for shifted labels). The Megatron model expects
        # mm_token_type_ids with shape [bsz, seq_len].
        if "token_type_ids" in processor_output:
            hf_token_type_ids = processor_output["token_type_ids"]
            # Take the first seq_len values (drop the extra trailing token)
            mm_token_type_ids = hf_token_type_ids[:, :input_ids.size(1)].to(torch.int32)
            num_img_tokens = (mm_token_type_ids == 1).sum().item()
            print_rank_0(f"  mm_token_type_ids: {mm_token_type_ids.shape}, "
                         f"image tokens: {num_img_tokens}/{input_ids.size(1)}")
    else:
        # ============================================================
        # Text-only mode: simple tokenization
        # ============================================================
        inputs = tokenizer(args.prompt, return_tensors="pt")
        input_ids = inputs.input_ids

    # Pad sequence length to be divisible by TP size (needed for sequence parallel)
    tp_size = args.tp
    seq_len = input_ids.size(1)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        padding = torch.full(
            (input_ids.shape[0], pad_len),
            tokenizer.pad_token_id or 0,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, padding], dim=1)
        # Also pad mm_token_type_ids if present
        if mm_token_type_ids is not None:
            mm_padding = torch.zeros(
                mm_token_type_ids.shape[0], pad_len,
                dtype=mm_token_type_ids.dtype,
            )
            mm_token_type_ids = torch.cat([mm_token_type_ids, mm_padding], dim=1)

    print_rank_0(f"  Input IDs shape: {input_ids.shape} (original seq_len={seq_len})")

    # Phase 1: HF forward (rank 0 only)
    hf_logits = run_hf_forward(
        args.hf_model_path, input_ids, tokenizer,
        processor_output=processor_output,
        processor=processor,
    )

    # Synchronize all ranks before Phase 2
    dist.barrier()

    # Destroy the process group before Phase 2 since AutoBridge's
    # initialize_model_parallel() will create its own process groups.
    dist.destroy_process_group()

    # Phase 2: Megatron forward (all ranks)
    megatron_logits = run_megatron_forward(
        args.hf_model_path,
        input_ids,
        tokenizer,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )

    # Phase 3: Compare (rank 0 only, which has both logit tensors)
    if _is_rank_0() and megatron_logits is not None and hf_logits is not None:
        passed = compare_logits(hf_logits, megatron_logits, tokenizer, args.threshold)
    else:
        passed = True  # Non-primary ranks don't compare

    # Broadcast pass/fail to all ranks
    if dist.is_initialized():
        passed_tensor = torch.tensor([1 if passed else 0], device="cuda")
        dist.broadcast(passed_tensor, 0)
        passed = passed_tensor.item() == 1
        dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
