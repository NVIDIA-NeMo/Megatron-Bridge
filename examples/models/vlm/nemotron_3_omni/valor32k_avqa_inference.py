# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
VALOR32K-AVQA inference script for Nemotron Omni.

Runs audio-visual QA inference on VALOR32K-AVQA test samples using a
Megatron checkpoint. Outputs predictions and computes accuracy.

Vision backbone uses the temporal video embedder path
(dynamic_resolution=True, temporal_patch_dim=2, separate_video_embedder=True),
matching the SFT data pipeline in NemotronOmniTaskEncoder with
use_temporal_video_embedder=True. Frames are pre-patchified into a packed
[1, total_patches, 3*P*P] tensor with imgs_sizes / num_frames so RADIO ViT
exercises the trained `video_embedder`.

Usage:
  torchrun --nproc-per-node=8 examples/models/vlm/nemotron_3_omni/valor32k_avqa_inference.py \
    --hf_model_path /path/to/nemotron-3-nano-omni-ea1_v1.0 \
    --megatron_model_path /path/to/checkpoint \
    --data_root /path/to/valor32k_avqa \
    --tp 4 --ep 2 \
    --max_samples 10
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, ParakeetFeatureExtractor

from megatron.bridge import AutoBridge
from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
    adjust_image_tokens,
    maybe_path_or_url_to_data_urls,
    pil_image_from_base64,
)
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


# ---------------------------------------------------------------------------
# Temporal video embedder constants (must match nemotron_omni_provider.py and
# the SFT recipe `nemotron_omni_valor32k_sft_config`).
# ---------------------------------------------------------------------------

_VIDEO_TEMPORAL_PATCH_SIZE = 2
_VIDEO_FRAME_H = 512
_VIDEO_FRAME_W = 512
_VISION_PATCH_DIM = 16

# CLIP / RADIO normalization (mirrors NemotronOmniTaskEncoder._patchify_frame)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _patchify_frame(
    pil_img: Image.Image, target_h: int = _VIDEO_FRAME_H, target_w: int = _VIDEO_FRAME_W
) -> torch.Tensor:
    """Resize + normalize a PIL frame and pack into [num_patches, 3*P*P]."""
    from torchvision import transforms

    img = pil_img.convert("RGB").resize((target_w, target_h))
    tensor = transforms.ToTensor()(img)
    mean = torch.tensor(_CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(_CLIP_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    P = _VISION_PATCH_DIM
    py, px = target_h // P, target_w // P
    patches = tensor.reshape(3, py, P, px, P).permute(1, 3, 0, 2, 4).reshape(py * px, 3 * P * P)
    return patches


def _build_vision_packed_seq_params(imgs_sizes: Optional[torch.Tensor]) -> Optional[PackedSeqParams]:
    """Vision PackedSeqParams from pre-grouping per-frame (H, W).

    Mirrors `megatron.bridge.training.nemotron_omni_step._build_vision_packed_seq_params`.
    """
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


# ---------------------------------------------------------------------------
# Forward step (same as hf_to_megatron_generate_nemotron_omni.py)
# ---------------------------------------------------------------------------

class SingleBatchIterator:
    def __init__(self, input_ids, position_ids, attention_mask, **kwargs):
        self.batch = dict(tokens=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        if kwargs.get("images") is not None:
            self.batch["images"] = kwargs["images"]
        if kwargs.get("sound_clips") is not None:
            self.batch["sound_clips"] = kwargs["sound_clips"]
        if kwargs.get("sound_length") is not None:
            self.batch["sound_length"] = kwargs["sound_length"]
        # Temporal video embedder inputs (dynamic-resolution pre-patchified path)
        if kwargs.get("imgs_sizes") is not None:
            self.batch["imgs_sizes"] = kwargs["imgs_sizes"]
        if kwargs.get("num_frames") is not None:
            self.batch["num_frames"] = kwargs["num_frames"]
        if kwargs.get("vision_packed_seq_params") is not None:
            self.batch["vision_packed_seq_params"] = kwargs["vision_packed_seq_params"]
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
    if "images" not in forward_args:
        forward_args["images"] = torch.tensor([], dtype=torch.bfloat16, device=batch["tokens"].device).reshape(0, 0, 0)
    if "sound_clips" in batch:
        forward_args["sound_clips"] = batch["sound_clips"]
    if "sound_length" in batch:
        forward_args["sound_length"] = batch["sound_length"]
    # Temporal video embedder plumbing (matches nemotron_omni_step.forward_step)
    if "imgs_sizes" in batch:
        forward_args["imgs_sizes"] = batch["imgs_sizes"]
    if "num_frames" in batch:
        forward_args["num_frames"] = batch["num_frames"]
    if "vision_packed_seq_params" in batch:
        forward_args["vision_packed_seq_params"] = batch["vision_packed_seq_params"]

    def loss_func(x, **kw):
        return x

    output = model(**forward_args)
    if isinstance(output, tuple):
        output = output[0]
    return output, loss_func


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def build_video_id_map(videos_dir: Path) -> dict:
    """Map video_id → filename stem (files are {youtube_id}_{start}_{end}.mp4)."""
    mapping = {}
    for f in videos_dir.iterdir():
        if f.suffix == ".mp4":
            parts = f.stem.rsplit("_", 2)
            if len(parts) == 3 and parts[0] not in mapping:
                mapping[parts[0]] = f.stem
    return mapping


def process_sample(
    qa: dict,
    vid_map: dict,
    data_root: Path,
    tokenizer,
    processor,
    feature_extractor,
    video_fps: float = 1.0,
    video_nframes: int = 8,
    temporal_patch_size: int = _VIDEO_TEMPORAL_PATCH_SIZE,
):
    """Process a single VALOR32K-AVQA sample into model inputs.

    Mirrors the SFT temporal data pipeline (NemotronOmniTaskEncoder with
    use_temporal_video_embedder=True): frames are grouped in pairs in the
    prompt, all frames are pre-patchified into [1, total_patches, 3*P*P],
    and imgs_sizes / num_frames are emitted so RADIO's temporal grouping +
    video_embedder run on inference inputs.
    """
    import math

    video_id = str(qa["video_id"])
    file_stem = vid_map.get(video_id)
    if file_stem is None:
        return None

    video_path = data_root / "videos" / f"{file_stem}.mp4"
    audio_path = data_root / "audio" / f"{file_stem}.wav"
    if not video_path.exists():
        return None

    # Extract video frames
    image_urls, metadata = maybe_path_or_url_to_data_urls(
        str(video_path), fps=max(0, int(video_fps)), nframe=max(0, video_nframes), nframe_max=-1,
    )
    frames = [pil_image_from_base64(url) for url in image_urls]
    fps = metadata.fps if metadata and metadata.fps else video_fps

    # RADIO's _apply_temporal_grouping fuses `tps` consecutive frames per
    # tubelet, so trim trailing frames to a multiple of tps (matches the SFT
    # data layer which also produces tps-aligned frame counts).
    tps = temporal_patch_size
    usable = (len(frames) // tps) * tps
    if usable == 0:
        print_rank_0(f"[skip] {video_id}: only {len(frames)} frames, need >= {tps}")
        return None
    if usable != len(frames):
        frames = frames[:usable]

    # Group frames by temporal_patch_size for the prompt: one <image> per pair,
    # with the training-time timestamp format.
    paired_images = []
    video_prompt_lines = ["This is a video:"]
    for i in range(0, len(frames), tps):
        group = frames[i : i + tps]
        ts_parts = [f"frame {i + j + 1} sampled at {(i + j) / fps:.2f} seconds" for j in range(len(group))]
        video_prompt_lines.append(" and ".join(ts_parts) + ": <image>")
        paired_images.append(group[0])

    # Build question with MCQ options
    question = qa["question"]
    options = qa.get("options", [])
    if options:
        option_labels = "ABCD"
        option_text = "\n".join(f"{option_labels[i]}. {opt}" for i, opt in enumerate(options))
        question = f"{question}\n{option_text}"

    # Build prompt
    content = "\n".join(video_prompt_lines) + "\n" + question
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Run the HF processor on representative frames (one per pair) with
    # max_num_tiles=1 (or max_num_patches=1) to emit input_ids with proper
    # <img>/<image>/</img> wrappers. We only consume `input_ids` here --
    # pixel_values is replaced below with the pre-patchified all-frames tensor.
    # Different model versions use different attribute names for the tile limit.
    _tile_attr = "max_num_tiles" if hasattr(processor.image_processor, "max_num_tiles") else "max_num_patches"
    orig_tiles = getattr(processor.image_processor, _tile_attr)
    setattr(processor.image_processor, _tile_attr, 1)
    try:
        inputs = processor(text=[prompt], images=paired_images, return_tensors="pt")
    finally:
        setattr(processor.image_processor, _tile_attr, orig_tiles)

    input_ids = inputs.input_ids
    # One <image> token per tubelet after adjust_image_tokens (matches the
    # number of tubelets that RADIO's temporal grouping will produce).
    num_patches = torch.ones(len(paired_images), dtype=torch.long)

    # Adjust image tokens
    img_start = tokenizer.convert_tokens_to_ids("<img>")
    img_end = tokenizer.convert_tokens_to_ids("</img>")
    if (input_ids == img_start).any():
        input_ids = adjust_image_tokens(input_ids, num_patches, img_start, img_end)

    # Pre-patchify ALL frames into [1, total_patches, 3*P*P]: dynamic-resolution
    # input that RADIO's _apply_temporal_grouping splits per-frame and fuses
    # into tubelets via `video_embedder`.
    all_patches = [_patchify_frame(f, _VIDEO_FRAME_H, _VIDEO_FRAME_W) for f in frames]
    images = torch.cat(all_patches, dim=0).unsqueeze(0).bfloat16()
    imgs_sizes = torch.tensor([[_VIDEO_FRAME_H, _VIDEO_FRAME_W]] * len(frames), dtype=torch.long)
    num_frames = torch.tensor([len(frames)], dtype=torch.long)

    # Process audio
    sound_clips = None
    sound_length = None
    if audio_path.exists():
        import soundfile as sf
        waveform, sr = sf.read(str(audio_path), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        waveform = waveform[: int(10.0 * 16000)]  # max 10s

        audio_features = feature_extractor([waveform], sampling_rate=16000, return_tensors="pt")
        sound_clips = audio_features.input_features.bfloat16()
        sound_length = torch.tensor([sound_clips.shape[1]], dtype=torch.long)

        # Compute audio token count and insert into input_ids
        mel_len = sound_clips.shape[1]
        token_len = float(mel_len)
        for _ in range(3):
            token_len = math.floor((token_len + 2 - 3) / 2 + 1)
        n_sound_tokens = max(1, int(token_len))

        sound_id = tokenizer.convert_tokens_to_ids("<so_embedding>")
        so_start_id = tokenizer.convert_tokens_to_ids("<so_start>")
        so_end_id = tokenizer.convert_tokens_to_ids("<so_end>")
        img_end_positions = (input_ids[0] == img_end).nonzero(as_tuple=True)[0]
        insert_pos = int(img_end_positions[-1]) + 1 if len(img_end_positions) > 0 else 1

        sound_block = torch.tensor(
            [so_start_id] + [sound_id] * n_sound_tokens + [so_end_id],
            dtype=input_ids.dtype,
        ).unsqueeze(0)
        input_ids = torch.cat([input_ids[:, :insert_pos], sound_block, input_ids[:, insert_pos:]], dim=1)

    # Build answer
    correct_idx = qa.get("correct_answer_idx", 0)
    answer = options[correct_idx] if options and correct_idx < len(options) else ""

    return {
        "input_ids": input_ids,
        "images": images,
        "imgs_sizes": imgs_sizes,
        "num_frames": num_frames,
        "sound_clips": sound_clips,
        "sound_length": sound_length,
        "question": qa["question"],
        "options": options,
        "correct_answer": answer,
        "correct_idx": correct_idx,
        "video_id": video_id,
        "modality": qa.get("modality", "unknown"),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# Accept answers like "C", "C.", "C)", "C. Yellow" by mapping the leading letter
# to the option text. Reasoning-tuned models often emit just the letter, which
# a pure substring match against the full option text would reject.
_LETTER_RE = re.compile(r"^\s*([A-D])(?:[\.\)\:\-\s]|$)")


def grade_prediction(prediction: str, options: list, correct_answer: str) -> bool:
    m = _LETTER_RE.match(prediction)
    if m and options:
        idx = ord(m.group(1)) - ord("A")
        if idx < len(options) and options[idx].lower() == correct_answer.lower():
            return True
    return correct_answer.lower() in prediction.lower()


def generate(model, tokenizer, sample, max_new_tokens=50):
    """Greedy generation loop."""
    input_ids = sample["input_ids"].cuda()
    images = sample["images"].cuda() if sample["images"] is not None else None
    sound_clips = sample["sound_clips"].cuda() if sample["sound_clips"] is not None else None
    sound_length = sample["sound_length"].cuda() if sample["sound_length"] is not None else None
    imgs_sizes = sample["imgs_sizes"].cuda() if sample.get("imgs_sizes") is not None else None
    num_frames = sample["num_frames"].cuda() if sample.get("num_frames") is not None else None

    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    generated_ids = input_ids.clone()
    stop_tokens = [tokenizer.eos_token_id]

    for step in range(max_new_tokens):
        with torch.no_grad():
            # Rebuild each iteration: RADIO mutates cu_seqlens_q in-place when inserting class tokens,
            # so reusing the same object would cause cu_seqlens to grow by class_token_len each step.
            vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes)
            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(
                input_ids, position_ids, attention_mask,
                images=images, sound_clips=sound_clips, sound_length=sound_length,
                imgs_sizes=imgs_sizes, num_frames=num_frames,
                vision_packed_seq_params=vision_packed_seq_params,
            )
            output = fwd_bwd_function(
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
                output = torch.cat(gathered, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            dist.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            if next_token_ids.item() in stop_tokens:
                break

    # Decode: return both a cleaned prediction (for grading) and the full raw
    # decode (with special tokens and prompt) for inspection.
    gen_ids = generated_ids[0, sample["input_ids"].size(1):]
    cleaned = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
    full_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
    return cleaned, full_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VALOR32K-AVQA Inference")
    parser.add_argument("--hf_model_path", type=str, required=True)
    parser.add_argument("--megatron_model_path", type=str, default=None,
                        help="Megatron checkpoint path. If omitted, converts from HF on the fly.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--output", type=str, default=None, help="Save predictions to JSON file")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_name = "val" if args.split == "validation" else args.split
    qa_file = data_root / f"combined_dataset_{split_name}_flattened.json"

    # Load model. AVQA is video-only, so always enable the temporal video
    # embedder path: dynamic-resolution + tps=2 + separate_video_embedder.
    # Inputs are pre-patchified in `process_sample` to match this contract.
    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)
    model_provider = bridge.to_megatron_provider(load_weights=(args.megatron_model_path is None))
    model_provider.tensor_model_parallel_size = args.tp
    model_provider.pipeline_model_parallel_size = args.pp
    model_provider.expert_model_parallel_size = args.ep
    model_provider.expert_tensor_parallel_size = args.etp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.dynamic_resolution = True
    model_provider.temporal_patch_dim = _VIDEO_TEMPORAL_PATCH_SIZE
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
                "temporal_patch_dim": _VIDEO_TEMPORAL_PATCH_SIZE,
                "separate_video_embedder": True,
                "temporal_ckpt_compat": True,
                "vision_class_token_len": 10,
            },
            wrap_with_ddp=False,
        )
        model = [m.cuda().eval() for m in model]
        # Set grad_scale_func to None for inference (training checkpoints have optimizer config)
        for m in model:
            inner = m.module if hasattr(m, "module") else m
            if hasattr(inner, "config"):
                inner.config.grad_scale_func = None
            if hasattr(inner, "llava_model") and hasattr(inner.llava_model, "config"):
                inner.llava_model.config.grad_scale_func = None
    else:
        print_rank_0(f"Converting HF model from {args.hf_model_path} on the fly")
        model_provider.finalize()
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)
        model = [m.cuda().bfloat16().eval() for m in model]

    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=True)
    feature_extractor = ParakeetFeatureExtractor(sampling_rate=16000, feature_size=128)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load QA data
    with open(qa_file) as f:
        qa_pairs = json.load(f)
    print_rank_0(f"Loaded {len(qa_pairs)} QA pairs from {qa_file}")

    # Build video ID map
    vid_map = build_video_id_map(data_root / "videos")
    print_rank_0(f"Video ID map: {len(vid_map)} entries")

    # Run inference
    results = []
    correct = 0
    total = 0
    max_samples = min(args.max_samples, len(qa_pairs))

    for i in range(max_samples):
        qa = qa_pairs[i]
        sample = process_sample(
            qa, vid_map, data_root, tokenizer, processor, feature_extractor,
        )
        if sample is None:
            print_rank_0(f"[{i+1}/{max_samples}] Skipped: video not found for {qa['video_id']}")
            continue

        prediction, prediction_full = generate(model, tokenizer, sample, max_new_tokens=args.max_new_tokens)

        is_correct = grade_prediction(prediction, sample["options"], sample["correct_answer"])
        if is_correct:
            correct += 1
        total += 1

        result = {
            "video_id": sample["video_id"],
            "question": sample["question"],
            "options": sample["options"],
            "correct_answer": sample["correct_answer"],
            "prediction": prediction,
            "prediction_full_decode": prediction_full,
            "is_correct": is_correct,
            "modality": sample["modality"],
        }
        results.append(result)

        print_rank_0(
            f"[{i+1}/{max_samples}] Q: {sample['question'][:60]}... "
            f"| GT: {sample['correct_answer']} | Pred: {prediction[:60]} "
            f"| {'OK' if is_correct else 'WRONG'}"
        )

    # Summary
    acc = correct / total * 100 if total > 0 else 0
    print_rank_0(f"\n{'='*60}")
    print_rank_0(f"Results: {correct}/{total} correct ({acc:.1f}%)")
    print_rank_0(f"{'='*60}")

    # Save results
    if args.output and dist.get_rank() == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": total, "results": results}, f, indent=2)
        print_rank_0(f"Results saved to {output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
