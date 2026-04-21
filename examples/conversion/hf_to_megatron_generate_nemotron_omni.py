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
Nemotron Omni generation script (vision + audio + text).

Examples:
  # Audio-only:
  python examples/conversion/hf_to_megatron_generate_nemotron_omni.py \
    --hf_model_path=/path/to/omni_checkpoint \
    --audio_path=/path/to/audio.wav \
    --prompt="Transcribe the audio." \
    --max_new_tokens 300

  # Image + audio:
  python examples/conversion/hf_to_megatron_generate_nemotron_omni.py \
    --hf_model_path=/path/to/omni_checkpoint \
    --image_path="https://example.com/image.png" \
    --audio_path=/path/to/audio.wav \
    --prompt="Describe what you see and hear." \
    --max_new_tokens 300

  # Image-only (same as VL script):
  python examples/conversion/hf_to_megatron_generate_nemotron_omni.py \
    --hf_model_path=/path/to/omni_checkpoint \
    --image_path="https://example.com/image.png" \
    --prompt="Describe this image." \
    --max_new_tokens 300

  # Video:
  python examples/conversion/hf_to_megatron_generate_nemotron_omni.py \
    --hf_model_path=/path/to/omni_checkpoint \
    --video_path=/path/to/video.mp4 \
    --prompt="Describe what you see." \
    --max_new_tokens 300
"""

import argparse
from typing import Optional

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


class SingleBatchIterator:
    """Iterator that yields a single batch of data for generation.

    Supports vision inputs (images) and/or sound inputs (sound_clips,
    sound_length) alongside text tokens.
    """

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


def vlm_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for multimodal generation.

    Passes vision and/or sound tensors alongside text tokens.
    """
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


# ---------------------------------------------------------------------------
# Input processing helpers (duplicated from VL script because
# examples/conversion/ is not a Python package)
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> Image.Image:
    """Load an image from URL or file path."""
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path)
        response.raise_for_status()
        return Image.open(requests.get(image_path, stream=True).raw)
    else:
        return Image.open(image_path)


def process_image_inputs(processor, image_path: Optional[str], prompt: str, system_prompt: Optional[str] = None):
    """Process image inputs for vision-language model.

    Returns:
        Tuple of (input_ids, pixel_values, num_patches)
    """
    if image_path:
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
    else:
        inputs = processor(text=[prompt], return_tensors="pt")
        return inputs.input_ids, None, 0


def process_video_inputs(processor, video_path: Optional[str], prompt: str, system_prompt: Optional[str] = None):
    """Process video inputs for vision-language model."""
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )

    video_fps = -1
    video_nframe = 10
    video_nframe_max = -1

    image_urls, metadata = maybe_path_or_url_to_data_urls(
        video_path,
        fps=max(0, int(video_fps)),
        nframe=max(0, int(video_nframe)),
        nframe_max=int(video_nframe_max),
    )
    frames = [pil_image_from_base64(image_url) for image_url in image_urls]

    print(f"Video Metadata: {metadata}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{video_path}"},
                {"type": "text", "text": "\n" + prompt},
            ],
        }
    ]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if metadata:
        inputs = processor(
            text=[prompt],
            videos=frames,
            videos_kwargs={"video_metadata": metadata},
            return_tensors="pt",
        )
    else:
        inputs = processor(text=[prompt], videos=frames, return_tensors="pt")
    return inputs.input_ids, inputs.pixel_values_videos, inputs.num_patches


def process_audio_inputs(processor, audio_path: str, prompt: str, system_prompt: Optional[str] = None):
    """Process audio inputs for the omni model.

    The HF processor tokenizes and expands ``<audio>`` into the correct number
    of ``<so_embedding>`` tokens and returns raw waveforms.  This function
    additionally converts waveforms to mel spectrograms (required by
    ``BridgeSoundEncoder`` / ``FastConformerModel``).

    Returns:
        Tuple of (input_ids, sound_clips, sound_length) where sound_clips is
        a float tensor of shape ``(1, frames, mel_bins)`` and sound_length is
        a long tensor of shape ``(1,)`` with the mel frame count.
    """
    content = [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], audio=[audio_path], return_tensors="pt")
    input_ids = inputs.input_ids

    waveforms = inputs["sound_clips"]
    waveform = waveforms[0] if isinstance(waveforms, list) else waveforms

    mel = compute_mel_features(waveform)
    sound_clips = mel.unsqueeze(0)
    sound_length = torch.tensor([mel.shape[0]], dtype=torch.long)

    return input_ids, sound_clips, sound_length


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    """Main function for multimodal generation (vision + audio + text)."""
    tp = args.tp
    pp = args.pp
    ep = args.ep
    etp = args.etp

    if args.megatron_model_path:
        print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.initialize_model_parallel(seed=0)
        model = bridge.load_megatron_model(
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
        print_rank_0(f"Loading HuggingFace model from: {args.hf_model_path}")
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.initialize_model_parallel(seed=0)
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=True)
    img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_token_id = tokenizer.convert_tokens_to_ids("</img>")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Process inputs depending on which modalities are provided
    # ------------------------------------------------------------------
    images = None
    pixel_values = None
    sound_clips = None
    sound_length = None

    if args.audio_path and args.image_path:
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
            text=[text],
            images=image_inputs_raw,
            audio=[args.audio_path],
            padding=processor.tokenizer.pad_token is not None,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids
        images = inputs.pixel_values.bfloat16() if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None else None
        num_patches = inputs.num_patches if hasattr(inputs, "num_patches") else 0
        input_ids = adjust_image_tokens(input_ids, num_patches, img_start_token_id, img_end_token_id)

        waveforms = inputs["sound_clips"]
        waveform = waveforms[0] if isinstance(waveforms, list) else waveforms
        mel = compute_mel_features(waveform)
        sound_clips = mel.unsqueeze(0)
        sound_length = torch.tensor([mel.shape[0]], dtype=torch.long)

    elif args.audio_path:
        input_ids, sound_clips, sound_length = process_audio_inputs(
            processor, args.audio_path, args.prompt, args.system_prompt
        )
    elif args.video_path:
        input_ids, pixel_values, num_patches = process_video_inputs(
            processor, args.video_path, args.prompt, args.system_prompt
        )
        images = pixel_values.bfloat16()
        input_ids = adjust_image_tokens(input_ids, num_patches, img_start_token_id, img_end_token_id)
        video_token_id = tokenizer.convert_tokens_to_ids("<video>")
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        input_ids = torch.where(input_ids == video_token_id, image_token_id, input_ids)
    elif args.image_path:
        input_ids, pixel_values, num_patches = process_image_inputs(
            processor, args.image_path, args.prompt, args.system_prompt
        )
        images = pixel_values.bfloat16()
        input_ids = adjust_image_tokens(input_ids, num_patches, img_start_token_id, img_end_token_id)
    else:
        inputs = processor(text=[args.prompt], return_tensors="pt")
        input_ids = inputs.input_ids

    pixel_values = None

    # Move to GPU
    input_ids = input_ids.cuda()
    if images is not None:
        images = images.cuda()
    if sound_clips is not None:
        sound_clips = sound_clips.bfloat16().cuda()
        sound_length = sound_length.cuda()

    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    generated_ids = input_ids.clone()

    stop_tokens = [tokenizer.eos_token_id]

    # Greedy generation loop
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(
                input_ids,
                position_ids,
                attention_mask,
                pixel_values=pixel_values,
                images=images,
                sound_clips=sound_clips,
                sound_length=sound_length,
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
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                output = torch.cat(gathered_tensors, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                if step < 5:
                    print_rank_0(f"Step {step}: output shape={output.shape}, var={output.var():.4f}")
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                    print_rank_0(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print_rank_0(
                        f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            if next_token_ids.item() in stop_tokens:
                break

    generated_text = tokenizer.decode(list(generated_ids[0]))
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    if args.image_path:
        print_rank_0(f"Image: {args.image_path}")
    if args.audio_path:
        print_rank_0(f"Audio: {args.audio_path}")
    if args.video_path:
        print_rank_0(f"Video: {args.video_path}")
    print_rank_0(f"Prompt: {args.prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nemotron Omni Generation (Vision + Audio + Text)")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace omni model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe what you see and hear.",
        help="Input prompt.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="/no_think",
        help="System prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to the Megatron model checkpoint")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path or URL to image(s). Multiple paths separated by commas.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path or URL to a video.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to an audio file (WAV/MP3/FLAC).",
    )
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
