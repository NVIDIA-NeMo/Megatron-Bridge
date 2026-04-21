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

"""
Forward-pass comparison between HuggingFace and Megatron Nemotron Omni models.

Supports text-only, image-only, audio-only, and image+audio comparisons.
Supports both single-token logit comparison (default) and multi-token greedy
generation comparison (``--max_new_tokens N`` with N > 1).

For text/image, the HF model's ``forward()`` can be used directly for
single-token comparison.  For audio (or multi-token mode), HF ``generate()``
is used since HF ``forward()`` does NOT process sound.

Examples:
  # Audio-only, single-token logit comparison:
  python examples/conversion/compare_nemotron_omni.py \
    --hf_model_path=/path/to/omni_checkpoint \
    --audio_path=/path/to/audio.wav \
    --prompt="Transcribe the audio."

  # Audio-only, 30-token generation comparison:
  torchrun --nproc_per_node=2 examples/conversion/compare_nemotron_omni.py \
    --hf_model_path=/path/to/omni_checkpoint \
    --audio_path=/path/to/audio.wav \
    --prompt="Transcribe the audio." --tp 2 --max_new_tokens 30

  # Image + audio comparison:
  torchrun --nproc_per_node=2 examples/conversion/compare_nemotron_omni.py \
    --hf_model_path=/path/to/omni_checkpoint \
    --image_path=/path/to/image.png \
    --audio_path=/path/to/audio.wav \
    --prompt="Describe what you see and hear." --tp 2

  # Image-only comparison:
  python examples/conversion/compare_nemotron_omni.py \
    --hf_model_path=/path/to/omni_checkpoint \
    --image_path="https://example.com/image.png" \
    --prompt="Describe this image."
"""

import argparse
import os
from typing import Optional

import numpy as np
import requests
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import (
    compute_mel_features,
    load_audio,
)
from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


def _is_rank_0() -> bool:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    elif "LOCAL_RANK" in os.environ:
        return int(os.environ.get("LOCAL_RANK", 0)) == 0
    return True


# ---------------------------------------------------------------------------
# Input processing (duplicated from generation script -- no __init__.py)
# ---------------------------------------------------------------------------

class SingleBatchIterator:
    """Single-batch iterator for Megatron forward pass."""

    def __init__(self, input_ids, position_ids, attention_mask, **kwargs):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        if kwargs.get("images") is not None:
            self.batch["images"] = kwargs["images"]
        elif kwargs.get("pixel_values") is not None:
            self.batch["pixel_values"] = kwargs["pixel_values"]
        if kwargs.get("sound_clips") is not None:
            self.batch["sound_clips"] = kwargs["sound_clips"]
            self.batch["sound_length"] = kwargs["sound_length"]
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def vlm_forward_step(data_iterator, model, **kwargs):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask"),
    }
    if "images" in batch:
        forward_args["images"] = batch["images"]
    elif "pixel_values" in batch:
        forward_args["pixel_values"] = batch["pixel_values"]
    else:
        forward_args["images"] = torch.tensor([], dtype=torch.bfloat16, device=batch["tokens"].device)
    if "sound_clips" in batch:
        forward_args["sound_clips"] = batch["sound_clips"]
        forward_args["sound_length"] = batch["sound_length"]

    def loss_func(x, **kwargs):
        return x

    output = model(**forward_args)
    if isinstance(output, tuple):
        output = output[0]
    return output, loss_func


def load_image(image_path: str) -> Image.Image:
    if image_path.startswith(("http://", "https://")):
        return Image.open(requests.get(image_path, stream=True).raw)
    return Image.open(image_path)


def process_image_inputs(processor, image_path, prompt, system_prompt=None):
    if "," in image_path:
        image_paths = image_path.split(",")
        content = []
        for i, path in enumerate(image_paths):
            content.append({"type": "text", "text": f"{'\n' if i > 0 else ''}Image-{i + 1}: "})
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": "\n" + prompt})
    else:
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]
    messages = [{"role": "user", "content": content}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=processor.tokenizer.pad_token is not None,
        return_tensors="pt",
    )
    return inputs.input_ids, inputs.pixel_values, inputs.num_patches


# ---------------------------------------------------------------------------
# HF inference helpers
# ---------------------------------------------------------------------------

def _load_hf_model(hf_model_path):
    """Load HF model on rank 0 using trust_remote_code (for omni model class)."""
    if not _is_rank_0():
        return None

    from transformers import AutoModelForCausalLM

    print_rank_0("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    hf_model = hf_model.eval()
    print_rank_0("HF model loaded.")
    return hf_model


def _hf_forward_text_or_image(hf_model, input_ids, pixel_values=None):
    """Run HF forward() -- works for text-only and image-only (no audio)."""
    if hf_model is None:
        return None
    with torch.no_grad():
        hf_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        }
        if pixel_values is not None:
            hf_inputs["pixel_values"] = pixel_values
        output = hf_model(**hf_inputs)
        return output.logits[0, -1, :]


def _hf_generate_with_audio(hf_model, input_ids, sound_clips_raw, pixel_values=None):
    """Run HF generate() with audio -- extracts first-token logits.

    The HF omni model's ``generate()`` is the only path that processes sound.
    We run with ``max_new_tokens=1, do_sample=False`` and hook into
    ``output_scores`` to get the first-token logits.
    """
    if hf_model is None:
        return None
    with torch.no_grad():
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "sound_clips": sound_clips_raw,
            "max_new_tokens": 1,
            "do_sample": False,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if pixel_values is not None:
            generate_kwargs["pixel_values"] = pixel_values
        output = hf_model.generate(**generate_kwargs)
        return output.scores[0][0]


def _hf_generate_tokens(hf_model, input_ids, max_new_tokens,
                         sound_clips_raw=None, pixel_values=None):
    """Run HF greedy generation for N tokens, return token IDs and per-step scores.

    Uses ``generate()`` which handles all modalities (text, image, audio).
    Returns ``(generated_token_ids, per_step_scores)`` where
    ``generated_token_ids`` is a 1-D long tensor of length <= max_new_tokens
    and ``per_step_scores`` is a list of (vocab_size,) float tensors.
    """
    if hf_model is None:
        return None, None
    with torch.no_grad():
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if sound_clips_raw is not None:
            generate_kwargs["sound_clips"] = sound_clips_raw
        if pixel_values is not None:
            generate_kwargs["pixel_values"] = pixel_values
        output = hf_model.generate(**generate_kwargs)
        prompt_len = input_ids.shape[1]
        generated_tokens = output.sequences[0, prompt_len:]
        scores = [s[0] for s in output.scores]
        return generated_tokens, scores


# ---------------------------------------------------------------------------
# Megatron inference
# ---------------------------------------------------------------------------

def _run_megatron_forward(megatron_model, input_ids, images=None,
                          sound_clips=None, sound_length=None):
    """Run a single Megatron forward pass, return last-position logits."""
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    fwd_bwd_function = get_forward_backward_func()
    iterator = SingleBatchIterator(
        input_ids, position_ids, attention_mask,
        images=images,
        sound_clips=sound_clips,
        sound_length=sound_length,
    )

    output = fwd_bwd_function(
        forward_step_func=vlm_forward_step,
        data_iterator=iterator,
        model=megatron_model,
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
        if world_size > 1:
            gathered = [torch.zeros_like(output) for _ in range(world_size)]
            dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
            output = torch.cat(gathered, dim=2)
        return output[0, -1, :]
    return None


def _run_megatron_generation(megatron_model, input_ids, max_new_tokens,
                              tokenizer, images=None, sound_clips=None,
                              sound_length=None):
    """Run Megatron greedy auto-regressive generation for N tokens.

    Returns ``(generated_token_ids, per_step_logits)`` where
    ``generated_token_ids`` is a 1-D long tensor and ``per_step_logits``
    is a list of (vocab_size,) float tensors (only on the last pipeline
    stage / TP rank 0; None elsewhere).
    """
    generated_tokens = []
    generated_logits = []
    current_ids = input_ids.clone()
    stop_tokens = [tokenizer.eos_token_id]

    for step in range(max_new_tokens):
        position_ids = (
            torch.arange(current_ids.size(1), dtype=torch.long, device=current_ids.device)
            .unsqueeze(0)
            .expand_as(current_ids)
        )
        attention_mask = torch.ones_like(current_ids, dtype=torch.bool)

        fwd_bwd_function = get_forward_backward_func()
        iterator = SingleBatchIterator(
            current_ids, position_ids, attention_mask,
            images=images,
            sound_clips=sound_clips,
            sound_length=sound_length,
        )

        output = fwd_bwd_function(
            forward_step_func=vlm_forward_step,
            data_iterator=iterator,
            model=megatron_model,
            num_microbatches=1,
            forward_only=True,
            seq_length=current_ids.size(1),
            micro_batch_size=1,
            collect_non_loss_data=True,
        )
        if isinstance(output, list) and len(output) > 0:
            output = output[0]
            if isinstance(output, tuple):
                output = output[0]

        if parallel_state.is_pipeline_last_stage():
            world_size = parallel_state.get_tensor_model_parallel_world_size()
            if world_size > 1:
                gathered = [torch.zeros_like(output) for _ in range(world_size)]
                dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
                output = torch.cat(gathered, dim=2)
            logits = output[0, -1, :]
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True).unsqueeze(0)
            generated_logits.append(logits)
        else:
            next_token_ids = torch.ones((1, 1), device=current_ids.device, dtype=current_ids.dtype)

        torch.distributed.broadcast(next_token_ids, get_last_rank())
        generated_tokens.append(next_token_ids.item())
        current_ids = torch.cat([current_ids, next_token_ids], dim=-1)

        if next_token_ids.item() in stop_tokens:
            break

        if step % 10 == 0:
            print_rank_0(f"  Megatron generation step {step}/{max_new_tokens}")

    token_tensor = torch.tensor(generated_tokens, dtype=torch.long, device=input_ids.device)
    return token_tensor, generated_logits if generated_logits else None


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _print_comparison(label, hf_logits, meg_logits, tokenizer, threshold):
    """Print comparison metrics between HF and Megatron logits."""
    hf_logits = hf_logits.float()
    meg_logits = meg_logits.float()

    hf_token = torch.argmax(hf_logits)
    meg_token = torch.argmax(meg_logits)

    print_rank_0(f"--- {label} ---")
    print_rank_0(f"  HF  next token: {hf_token.item()} ('{tokenizer.decode([hf_token.item()])}')")
    print_rank_0(f"  Meg next token: {meg_token.item()} ('{tokenizer.decode([meg_token.item()])}')")
    print_rank_0(f"  Token match: {hf_token.item() == meg_token.item()}")

    diff = (hf_logits - meg_logits).abs()
    print_rank_0(f"  Logit diff -- max: {diff.max():.6f}, mean: {diff.mean():.6f}")

    cosine = torch.cosine_similarity(hf_logits.unsqueeze(0), meg_logits.unsqueeze(0))
    print_rank_0(f"  Cosine similarity: {cosine.item():.6f}  (threshold >= {threshold})")

    hf_top5_vals, hf_top5_ids = torch.topk(hf_logits, 5)
    hf_top5 = [(tokenizer.decode([i]), v.item()) for i, v in zip(hf_top5_ids, hf_top5_vals)]
    meg_top5_vals, meg_top5_ids = torch.topk(meg_logits, 5)
    meg_top5 = [(tokenizer.decode([i]), v.item()) for i, v in zip(meg_top5_ids, meg_top5_vals)]
    print_rank_0(f"  HF  Top-5: {hf_top5}")
    print_rank_0(f"  Meg Top-5: {meg_top5}")

    if cosine.item() < threshold:
        print_rank_0(f"  WARNING: cosine similarity below threshold ({cosine.item():.6f} < {threshold})")
    return cosine.item()


def _print_generation_comparison(label, hf_tokens, meg_tokens, hf_scores,
                                  meg_logits, tokenizer, threshold):
    """Print multi-token generation comparison between HF and Megatron."""
    max_len = min(len(hf_tokens), len(meg_tokens))

    print_rank_0(f"--- {label}: {max_len}-token generation comparison ---")

    matches = 0
    first_mismatch = -1
    cosine_sims = []

    for i in range(max_len):
        hf_tok = hf_tokens[i].item()
        meg_tok = meg_tokens[i].item()
        match = hf_tok == meg_tok
        if match:
            matches += 1
        elif first_mismatch == -1:
            first_mismatch = i

        hf_word = tokenizer.decode([hf_tok])
        meg_word = tokenizer.decode([meg_tok])
        status = "OK" if match else "MISMATCH"

        if hf_scores is not None and meg_logits is not None and i < len(hf_scores) and i < len(meg_logits):
            cos = torch.cosine_similarity(
                hf_scores[i].unsqueeze(0).float(),
                meg_logits[i].unsqueeze(0).float(),
            ).item()
            cosine_sims.append(cos)
            print_rank_0(f"  [{i:3d}] {status:8s}  HF='{hf_word}' ({hf_tok})  Meg='{meg_word}' ({meg_tok})  cos={cos:.4f}")
        else:
            print_rank_0(f"  [{i:3d}] {status:8s}  HF='{hf_word}' ({hf_tok})  Meg='{meg_word}' ({meg_tok})")

    print_rank_0(f"  Token match: {matches}/{max_len}")
    if first_mismatch >= 0:
        print_rank_0(f"  First mismatch at position: {first_mismatch}")
    else:
        print_rank_0(f"  All {max_len} tokens match!")
    if cosine_sims:
        avg_cos = sum(cosine_sims) / len(cosine_sims)
        min_cos = min(cosine_sims)
        print_rank_0(f"  Avg cosine similarity: {avg_cos:.6f}  (min: {min_cos:.6f}, threshold >= {threshold})")

    hf_text = tokenizer.decode(hf_tokens.tolist())
    meg_text = tokenizer.decode(meg_tokens.tolist())
    print_rank_0(f"  HF  generated: {hf_text}")
    print_rank_0(f"  Meg generated: {meg_text}")

    return matches, max_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    tp, pp, ep, etp = args.tp, args.pp, args.ep, args.etp
    max_new_tokens = args.max_new_tokens
    has_audio = args.audio_path is not None
    has_image = args.image_path is not None
    multi_token = max_new_tokens > 1

    # ---- Bridge and provider setup (HF weights loaded to CPU, not GPU) ----
    print_rank_0("Loading bridge...")
    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)
    if args.megatron_model_path:
        model_provider = bridge.to_megatron_provider(load_weights=False)
    else:
        model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.initialize_model_parallel(seed=0)

    # ---- Tokenizer / processor ----
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=True)
    img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_token_id = tokenizer.convert_tokens_to_ids("</img>")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Process inputs ----
    images = None
    sound_clips = None
    sound_length = None
    sound_clips_raw = None

    if has_audio and has_image:
        content = [
            {"type": "image", "image": args.image_path},
            {"type": "audio", "audio": args.audio_path},
            {"type": "text", "text": args.prompt},
        ]
        messages = [{"role": "user", "content": content}]
        if args.system_prompt:
            messages.insert(0, {"role": "system", "content": args.system_prompt})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs_raw, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs_raw, audio=[args.audio_path],
            padding=processor.tokenizer.pad_token is not None, return_tensors="pt",
        )
        input_ids = inputs.input_ids
        images = inputs.pixel_values.bfloat16() if inputs.pixel_values is not None else None
        num_patches = inputs.num_patches if hasattr(inputs, "num_patches") else 0
        input_ids = adjust_image_tokens(input_ids, num_patches, img_start_token_id, img_end_token_id)

        sound_clips_raw = inputs["sound_clips"]
        waveform = sound_clips_raw[0] if isinstance(sound_clips_raw, list) else sound_clips_raw
        mel = compute_mel_features(waveform)
        sound_clips = mel.unsqueeze(0)
        sound_length = torch.tensor([mel.shape[0]], dtype=torch.long)

    elif has_audio:
        content = [
            {"type": "audio", "audio": args.audio_path},
            {"type": "text", "text": args.prompt},
        ]
        messages = [{"role": "user", "content": content}]
        if args.system_prompt:
            messages.insert(0, {"role": "system", "content": args.system_prompt})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], audio=[args.audio_path], return_tensors="pt")
        input_ids = inputs.input_ids

        sound_clips_raw = inputs["sound_clips"]
        waveform = sound_clips_raw[0] if isinstance(sound_clips_raw, list) else sound_clips_raw
        mel = compute_mel_features(waveform)
        sound_clips = mel.unsqueeze(0)
        sound_length = torch.tensor([mel.shape[0]], dtype=torch.long)

    elif has_image:
        input_ids, pixel_values, num_patches = process_image_inputs(
            processor, args.image_path, args.prompt, args.system_prompt,
        )
        images = pixel_values.bfloat16()
        input_ids = adjust_image_tokens(input_ids, num_patches, img_start_token_id, img_end_token_id)
    else:
        inputs = processor(text=[args.prompt], return_tensors="pt")
        input_ids = inputs.input_ids

    # Move to GPU
    input_ids = input_ids.cuda()
    if images is not None:
        images = images.cuda()
    if sound_clips is not None:
        sound_clips = sound_clips.bfloat16().cuda()
        sound_length = sound_length.cuda()

    print_rank_0(f"Input shape: {input_ids.shape}")
    print_rank_0(f"Images: {'yes' if images is not None else 'no'}")
    print_rank_0(f"Sound:  {'yes' if sound_clips is not None else 'no'}")
    print_rank_0(f"Mode:   {'multi-token (' + str(max_new_tokens) + ')' if multi_token else 'single-token logit'}")

    # ---- HF inference: load model, run, then free GPU before Megatron ----
    hf_model = _load_hf_model(args.hf_model_path)

    print_rank_0("=== HF INFERENCE ===")
    if multi_token:
        hf_tokens, hf_scores = _hf_generate_tokens(
            hf_model, input_ids, max_new_tokens,
            sound_clips_raw=sound_clips_raw if has_audio else None,
            pixel_values=images,
        )
        hf_logits = None
        if hf_tokens is not None:
            print_rank_0(f"  HF generated {len(hf_tokens)} tokens: {tokenizer.decode(hf_tokens.tolist())}")
    else:
        hf_tokens, hf_scores = None, None
        if has_audio:
            hf_logits = _hf_generate_with_audio(
                hf_model, input_ids, sound_clips_raw,
                pixel_values=images,
            )
        else:
            hf_logits = _hf_forward_text_or_image(hf_model, input_ids, pixel_values=images)

    del hf_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print_rank_0("HF model freed from GPU.")

    # Broadcast HF results to all ranks
    if torch.distributed.is_initialized():
        if multi_token:
            if hf_tokens is None:
                hf_tokens = torch.zeros(max_new_tokens, device=input_ids.device, dtype=torch.long)
            n_tokens = torch.tensor([len(hf_tokens)], device=input_ids.device, dtype=torch.long)
            torch.distributed.broadcast(n_tokens, 0)
            if len(hf_tokens) < max_new_tokens:
                pad = torch.zeros(max_new_tokens - len(hf_tokens), device=input_ids.device, dtype=torch.long)
                hf_tokens = torch.cat([hf_tokens, pad])
            torch.distributed.broadcast(hf_tokens, 0)
            hf_tokens = hf_tokens[:n_tokens.item()]
        else:
            if hf_logits is None:
                vocab_size = tokenizer.vocab_size
                hf_logits = torch.zeros(vocab_size, device=input_ids.device, dtype=torch.float32)
            else:
                hf_logits = hf_logits.float()
            torch.distributed.broadcast(hf_logits, 0)

    # ---- Create Megatron model (GPU now has space) ----
    print_rank_0("Creating Megatron model...")
    if args.megatron_model_path:
        megatron_model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
                "pipeline_dtype": torch.bfloat16,
            },
            wrap_with_ddp=False,
        )
    else:
        megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    megatron_model = [m.cuda().eval() for m in megatron_model]

    # ---- Megatron inference ----
    print_rank_0("=== MEGATRON INFERENCE ===")
    with torch.no_grad():
        if multi_token:
            meg_tokens, meg_logits_list = _run_megatron_generation(
                megatron_model, input_ids, max_new_tokens, tokenizer,
                images=images,
                sound_clips=sound_clips,
                sound_length=sound_length,
            )
            meg_logits = None
        else:
            meg_tokens, meg_logits_list = None, None
            meg_logits = _run_megatron_forward(
                megatron_model, input_ids,
                images=images,
                sound_clips=sound_clips,
                sound_length=sound_length,
            )

    # ---- Compare ----
    is_last_stage = not torch.distributed.is_initialized() or parallel_state.is_pipeline_last_stage()
    if multi_token:
        is_tp_rank_0 = (
            not torch.distributed.is_initialized()
            or parallel_state.get_tensor_model_parallel_rank() == 0
        )
        if is_last_stage and is_tp_rank_0 and meg_tokens is not None:
            modality = "audio" if has_audio else ("image" if has_image else "text")
            threshold = 0.95 if has_audio else 0.98
            print_rank_0("=== COMPARISON ===")
            _print_generation_comparison(
                modality, hf_tokens, meg_tokens, hf_scores, meg_logits_list,
                tokenizer, threshold,
            )
            print_rank_0("=== COMPARISON COMPLETE ===")
        elif is_last_stage and is_tp_rank_0:
            print_rank_0(f"  Meg generated {len(meg_tokens)} tokens: {tokenizer.decode(meg_tokens.tolist())}")
            print_rank_0("  (HF scores not available on this rank for logit comparison)")
    elif is_last_stage and meg_logits is not None:
        is_tp_rank_0 = (
            not torch.distributed.is_initialized()
            or parallel_state.get_tensor_model_parallel_rank() == 0
        )
        if is_tp_rank_0:
            modality = "audio" if has_audio else ("image" if has_image else "text")
            threshold = 0.95 if has_audio else 0.98
            print_rank_0("=== COMPARISON ===")
            _print_comparison(modality, hf_logits, meg_logits, tokenizer, threshold)
            print_rank_0("=== COMPARISON COMPLETE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare HF and Megatron Nemotron Omni models")
    parser.add_argument("--hf_model_path", type=str, required=True, help="Path to HF omni model.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt.")
    parser.add_argument("--system_prompt", type=str, default="/no_think", help="System prompt.")
    parser.add_argument("--image_path", type=str, default=None, help="Path or URL to image(s).")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to audio file (WAV/MP3/FLAC).")
    parser.add_argument("--max_new_tokens", type=int, default=1,
                        help="Number of tokens to generate for comparison (1 = logit comparison, >1 = generation comparison)")
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to Megatron checkpoint.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
