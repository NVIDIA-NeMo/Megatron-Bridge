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
Collation utilities for building VLM training batches from conversation examples.
"""

import warnings

import torch
import torch.nn.functional as F
from PIL import Image  # noqa: F401  # may be used downstream by processors

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs, Qwen2_5_VLVisualInputs


# Local message used when optional qwen_vl_utils dependency is missing
MISSING_QWEN_VL_UTILS_MSG = (
    "qwen_vl_utils is required for Qwen2.5 VL processing. Please `pip install qwen-vl-utils` or"
    " provide compatible vision preprocessing."
)

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False


def _gather_assistant_text_segments(example: dict) -> list[str]:
    """Extract assistant text segments from the structured conversation example.

    The example schema is expected to be {"conversation": [{"role": ..., "content": [...]} ...]} where
    content is a list of items like {"type": "text"|"image"|..., "text": "..."}.
    Returns a list of concatenated text strings, one per assistant turn.
    """
    texts: list[str] = []
    for turn in example.get("conversation", []):
        if turn.get("role") != "assistant":
            continue
        parts = turn.get("content", [])
        buf = []
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    buf.append(p["text"])
        elif isinstance(parts, str):
            buf.append(parts)
        if buf:
            texts.append("".join(buf))
    return texts


def create_multiturn_loss_mask_by_search(
    example: dict, input_ids, processor, skipped_tokens: torch.Tensor
) -> list[int]:
    """Tokenizer-agnostic masking via substring search of assistant texts.

    - Tokenize full conversation with processor already done -> input_ids
    - Extract assistant text strings from the structured example
    - For each assistant text, tokenize without special tokens and search sequentially
    - On success, unmask that span; otherwise leave masked
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = input_ids.tolist()
    mask = [0] * len(ids)

    def try_mark(span_text: str, start_from: int) -> int:
        """Tokenize a span and mark its occurrence if found. Returns new search start index."""
        variants = [span_text, span_text + "\n", span_text.strip(), span_text.strip() + "\n"]
        for text in variants:
            span_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not span_tokens:
                continue
            # naive sequential search from start_from
            for i in range(start_from, len(ids) - len(span_tokens) + 1):
                if ids[i : i + len(span_tokens)] == span_tokens:
                    for j in range(i, i + len(span_tokens)):
                        mask[j] = 1
                    return i + len(span_tokens)
        return start_from

    search_start = 0
    for asst_text in _gather_assistant_text_segments(example):
        search_start = try_mark(asst_text, search_start)

    if sum(mask) == 0:
        warnings.warn("*" * 100)
        warnings.warn(f"All tokens are masked for example:\n{example}.")
        warnings.warn("*" * 100)

    # Ensure pad/skipped tokens are masked
    ids_t = torch.tensor(ids)
    for k, t in enumerate(ids_t):
        if t in skipped_tokens:
            mask[k] = 0
    return mask


def phi4_mm_collate_fn(examples, processor):
    """Collate function for Phi-4 MM model audio input"""

    # Extract conversations and audio data
    conversations = [example["conversation"] for example in examples]
    audios = [example["audio"] for example in examples]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    audio_inputs = [(audio["array"], audio["sampling_rate"]) if isinstance(audio, dict) else audio for audio in audios]
    batch = processor(
        text=texts, audios=audio_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)

    loss_masks = []
    for i, conversation in enumerate(conversations):
        input_ids = batch["input_ids"][i].tolist()

        assistant_content = conversation[1]["content"]
        assistant_tokens = processor.tokenizer(assistant_content, add_special_tokens=False)["input_ids"]

        loss_mask = [0] * len(input_ids)
        for start_idx in range(len(input_ids) - len(assistant_tokens) + 1):
            if input_ids[start_idx : start_idx + len(assistant_tokens)] == assistant_tokens:
                for j in range(len(assistant_tokens)):
                    loss_mask[start_idx + j] = 1
                break
        loss_masks.append(loss_mask)

    max_len = max(len(mask) for mask in loss_masks)
    padded_loss_masks = [mask + [0] * (max_len - len(mask)) for mask in loss_masks]
    batch["loss_mask"] = torch.tensor(padded_loss_masks, dtype=torch.float)

    labels[batch["loss_mask"] == 0] = -100
    batch["labels"] = labels

    # Remove specified batch features if present
    for key in ["input_image_embeds", "image_sizes", "image_attention_mask"]:
        if key in batch:
            del batch[key]
    return batch


def qwen2_5_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    texts = [processor.apply_chat_template(example["conversation"], tokenize=False) for example in examples]
    # Build per-example images (list) and split by presence
    per_example_images = []
    has_images = []
    for example in examples:
        imgs = process_vision_info(example["conversation"])[0]
        if imgs is None:
            imgs = []
        elif not isinstance(imgs, list):
            imgs = [imgs]
        per_example_images.append(imgs)
        has_images.append(len(imgs) > 0)

    idx_with = [i for i, h in enumerate(has_images) if h]
    idx_without = [i for i, h in enumerate(has_images) if not h]

    batch_with = None
    batch_without = None

    if idx_with:
        texts_with = [texts[i] for i in idx_with]
        images_with = [per_example_images[i] for i in idx_with]
        batch_with = processor(
            text=texts_with,
            images=images_with,
            padding=True,
            return_tensors="pt",
            min_pixels=200704,  # 256*28*28
            max_pixels=1003520,  # 1280*28*28
        )

        batch_with = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch_with.items()}

    if idx_without:
        texts_without = [texts[i] for i in idx_without]
        batch_without = processor(
            text=texts_without,
            padding=True,
            return_tensors="pt",
        )

        batch_without = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch_without.items()}

    # Merge batches back to original order
    if batch_with is not None and batch_without is None:
        batch = batch_with
    elif batch_with is None and batch_without is not None:
        batch = batch_without
    else:
        # Both exist: pad to common max length and interleave rows
        pad_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
        in_with = batch_with["input_ids"]
        in_without = batch_without["input_ids"]
        max_len = max(in_with.shape[1], in_without.shape[1])

        def pad_to(x, tgt_len):
            if x.shape[1] == tgt_len:
                return x
            pad_len = tgt_len - x.shape[1]
            return F.pad(x, (0, pad_len), value=pad_id)

        in_with = pad_to(in_with, max_len)
        in_without = pad_to(in_without, max_len)

        input_ids = torch.full((len(examples), max_len), pad_id, dtype=in_with.dtype)
        # Place rows
        for row, i in enumerate(idx_with):
            input_ids[i] = in_with[row]
        for row, i in enumerate(idx_without):
            input_ids[i] = in_without[row]

        batch = {"input_ids": input_ids}
        # Carry over vision tensors if present
        if "pixel_values" in batch_with:
            batch["pixel_values"] = batch_with["pixel_values"]
        if "image_grid_thw" in batch_with:
            batch["image_grid_thw"] = batch_with["image_grid_thw"]

    labels = batch["input_ids"].clone()[:, 1:].contiguous()
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    # Ensure position_ids exist for the model
    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
            .contiguous()
        )
    # Prefer general search-based masking using structured example content (not template-specific)
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    # Enforce label masking to match shifted loss_mask
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t
    # Build Qwen2VL visual inputs object and attach to batch; remove raw keys
    visual_inputs = Qwen2_5_VLVisualInputs(
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
    )
    if "pixel_values" in batch:
        del batch["pixel_values"]
    if "image_grid_thw" in batch:
        del batch["image_grid_thw"]
    batch["visual_inputs"] = visual_inputs
    return batch


def nemotron_nano_v2_vl_collate_fn(examples: list, processor, start_of_response_token=None) -> dict[str, torch.Tensor]:
    """Collate function for Nemotron Nano V2 VL model."""
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens

    skipped_tokens = extract_skipped_token_ids(processor)
    # this assumes the first message in conversation is the video message
    is_video = examples[0]["conversation"][0]["content"][0]["type"] == "video"
    if is_video:
        from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
            maybe_path_or_url_to_data_urls,
            pil_image_from_base64,
        )

        assert len(examples) == 1, "Nemotron Nano V2 VL processor only supports batch size == 1"
        frames = []
        video_fps = -1
        video_nframe = 10
        video_nframe_max = -1

        for example in examples:
            video_path = example["conversation"][0]["content"][0]["path"]
            image_urls, metadata = maybe_path_or_url_to_data_urls(
                video_path,
                fps=max(0, int(video_fps)),
                nframe=max(0, int(video_nframe)),
                nframe_max=int(video_nframe_max),
            )
            frames.append([pil_image_from_base64(image_url) for image_url in image_urls])

        prompt = processor.apply_chat_template([example["conversation"] for example in examples], tokenize=False)
        batch = processor(
            text=prompt,
            videos=frames,
            videos_kwargs={"video_metadata": metadata},
            return_tensors="pt",
        )
    else:
        batch = processor.apply_chat_template(
            [example["conversation"] for example in examples],
            tokenize=True,
            padding=processor.tokenizer.pad_token is not None,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )
    loss_mask = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]

    img_start_token_id = 131073  # tokenizer.convert_tokens_to_ids("<img>")
    img_end_token_id = 131074  # tokenizer.convert_tokens_to_ids("</img>")
    adjusted_batch = adjust_image_tokens(
        {
            "input_ids": batch["input_ids"],
            "loss_mask": torch.tensor(loss_mask),
        },
        batch["num_patches"],
        img_start_token_id,
        img_end_token_id,
    )

    if is_video:
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<video>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        adjusted_batch["input_ids"] = torch.where(
            adjusted_batch["input_ids"] == video_token_id, image_token_id, adjusted_batch["input_ids"]
        )

    batch["input_ids"] = adjusted_batch["input_ids"]
    loss_mask = adjusted_batch["loss_mask"]

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    key = "pixel_values_videos" if is_video else "pixel_values"
    pv = batch[key].to(torch.bfloat16)
    del batch[key]
    batch["visual_inputs"] = GenericVisualInputs(pixel_values=pv)
    # roll label by 1 and fill last token with IGNORE_INDEX
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = IGNORE_INDEX
    batch["labels"] = labels

    loss_mask_t = torch.tensor(loss_mask, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask_t
    return batch


def nemotron_omni_collate_fn(
    examples: list,
    processor,
    start_of_response_token=None,
    *,
    pack_sequences: bool = False,
) -> dict[str, torch.Tensor]:
    """Collate function for Nemotron Omni model (vision + audio + language).

    Extends nemotron_nano_v2_vl_collate_fn with audio support. Each example
    may carry an ``audio_path`` field pointing to a 16 kHz mono WAV file.
    Audio is converted to mel spectrograms and added to the batch as
    ``sound_clips`` / ``sound_length`` tensors consumed by LLaVAModel.forward().

    When ``pack_sequences=True``, samples in the microbatch are concatenated
    along the sequence dim into a single ``[1, sum(L_i)]`` batch, and
    ``cu_seqlens`` / ``cu_seqlens_unpadded`` / ``cu_seqlens_argmin`` /
    ``max_seqlen`` are emitted so TE's THD attention kernels handle per-sample
    masking without an attention mask. Requires ``mbs > 1`` to be meaningful.
    """
    from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import (
        compute_mel_features,
        load_audio,
    )
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens

    # Ensure the tokenizer has a pad_token: the processor pads only when one is set,
    # and mbs>1 needs padding to collate sequences of different lengths. Safe no-op
    # when pad_token is already set.
    if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    skipped_tokens = extract_skipped_token_ids(processor)
    first_content = examples[0]["conversation"][0]["content"]
    is_video = isinstance(first_content, list) and first_content[0].get("type") == "video"

    # --- Vision path ---
    # The Nemotron Omni chat template does not expand {"type": "image"} content
    # into <image> tokens — it stringifies the list. We must convert conversations
    # to use explicit <image> text and pass PIL images via processor(images=...).
    if is_video:
        from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
            maybe_path_or_url_to_data_urls,
            pil_image_from_base64,
        )

        assert len(examples) == 1, "Nemotron Omni processor only supports batch size == 1 for video"
        frames = []
        video_nframe = 10

        for example in examples:
            video_path = example["conversation"][0]["content"][0]["path"]
            image_urls, metadata = maybe_path_or_url_to_data_urls(
                video_path,
                fps=0,
                nframe=max(0, int(video_nframe)),
                nframe_max=-1,
            )
            frames.append([pil_image_from_base64(image_url) for image_url in image_urls])

        prompt = processor.apply_chat_template([ex["conversation"] for ex in examples], tokenize=False)
        batch = processor(text=prompt, videos=frames, videos_kwargs={"video_metadata": metadata}, return_tensors="pt")
    else:
        # Convert structured {"type": "image"} content to explicit <image> text
        all_images = []
        images_per_ex: list[list] = []
        text_conversations = []
        for example in examples:
            images_for_example = []
            text_conv = []
            for turn in example["conversation"]:
                if isinstance(turn["content"], list):
                    text_parts = []
                    for item in turn["content"]:
                        if item["type"] == "image":
                            text_parts.append("<image>")
                            images_for_example.append(item["image"])
                        elif item["type"] == "text":
                            text_parts.append(item["text"])
                    text_conv.append({"role": turn["role"], "content": "\n".join(text_parts)})
                elif isinstance(turn["content"], str):
                    text_conv.append(turn)
                else:
                    text_conv.append({"role": turn["role"], "content": str(turn["content"])})
            all_images.extend(images_for_example)
            images_per_ex.append(images_for_example)
            text_conversations.append(text_conv)

        prompts = [
            processor.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in text_conversations
        ]
        # Normalize audio tokens: replace model-agnostic <|audio_1|> with Nemotron Omni's <so_embedding>
        audio_token = getattr(processor.tokenizer, "audio_token", "<so_embedding>")
        prompts = [p.replace("<|audio_1|>", audio_token) for p in prompts]
        if all_images:
            # Older Nemotron-VL image processors use fixed 512x512 tiles and expose
            # `max_num_tiles`; the newer Nemotron-3 Omni Reasoning processor uses
            # dynamic-resolution patches (no `max_num_tiles` attr, has
            # `max_num_patches` instead). Detect which path we're on.
            is_dynamic_res_processor = not hasattr(processor.image_processor, "max_num_tiles")
            if is_dynamic_res_processor:
                # Variable per-image (H, W) makes ``return_tensors="pt"`` fail to
                # stack pixel_values across examples. Process each example
                # separately and re-combine: right-pad input_ids across examples,
                # keep pixel_values as a flat list of per-image ``[3, H_i, W_i]``
                # tensors (patchified below with per-image (py, px)).
                per_ex_batches = [
                    processor(
                        text=[prompt],
                        images=imgs if imgs else None,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    )
                    for prompt, imgs in zip(prompts, images_per_ex)
                ]
                pad_id = processor.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = processor.tokenizer.eos_token_id or 0
                ids_list = [b["input_ids"][0] for b in per_ex_batches]
                max_len = max(t.shape[0] for t in ids_list)
                padded_ids = torch.full(
                    (len(per_ex_batches), max_len), pad_id, dtype=ids_list[0].dtype
                )
                for i, ids in enumerate(ids_list):
                    padded_ids[i, : ids.shape[0]] = ids
                pv_list: list[torch.Tensor] = []
                for b in per_ex_batches:
                    if "pixel_values" in b and b["pixel_values"] is not None:
                        pv_b = b["pixel_values"]
                        if pv_b.dim() == 4:
                            for img in pv_b:
                                pv_list.append(img)
                        elif pv_b.dim() == 3:
                            pv_list.append(pv_b)
                batch = {"input_ids": padded_ids}
                if pv_list:
                    batch["pixel_values"] = pv_list  # list[Tensor[3, H_i, W_i]]
            else:
                # Static-tile path: single-tile per image to match RADIO seq_length.
                orig_tiles = processor.image_processor.max_num_tiles
                processor.image_processor.max_num_tiles = 1
                batch = processor(
                    text=prompts,
                    images=all_images,
                    padding=processor.tokenizer.pad_token is not None,
                    truncation=True,
                    return_tensors="pt",
                )
                processor.image_processor.max_num_tiles = orig_tiles
        else:
            is_dynamic_res_processor = False
            batch = processor.tokenizer(
                prompts,
                padding=processor.tokenizer.pad_token is not None,
                truncation=True,
                return_tensors="pt",
            )

    # --- Audio path ---
    # Support both audio_path (file path) and audio (raw waveform tuple from CV17-style datasets)
    has_audio = any(ex.get("audio_path") or ex.get("audio") for ex in examples)
    if has_audio:
        import numpy as np

        max_dur = examples[0].get("max_audio_duration", 30.0)
        max_samples = int(max_dur * 16000)

        mel_list = []
        mel_lengths = []
        n_audio_tokens_list = []
        for ex in examples:
            audio_path = ex.get("audio_path")
            audio_tuple = ex.get("audio")  # (array, sr) from CV17-style datasets
            if audio_path:
                waveform = load_audio(audio_path, target_sr=16000)
            elif audio_tuple is not None:
                array, sr = audio_tuple
                waveform = np.asarray(array, dtype=np.float32)
                if sr != 16000:
                    import librosa

                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            else:
                mel_list.append(torch.zeros(1, 128))
                mel_lengths.append(1)
                n_audio_tokens_list.append(0)
                continue
            waveform = waveform[:max_samples]
            mel = compute_mel_features(waveform, sampling_rate=16000)
            mel_list.append(mel)
            mel_len = mel.shape[0]
            mel_lengths.append(mel_len)
            # Compute encoder output length from mel frame count using
            # BridgeSoundEncoder._compute_output_lengths formula:
            # Conv2D subsampling: floor((L + 2*padding - kernel_size) / stride + 1)
            # applied log2(subsampling_factor)=3 times, kernel=3, stride=2, padding=1
            import math as _math

            token_len = float(mel_len)
            for _ in range(3):
                token_len = _math.floor((token_len + 2 * 1 - 3) / 2 + 1)
            n_audio_tokens_list.append(max(1, int(token_len)))

        max_mel_len = max(mel_lengths)
        padded_mels = torch.zeros(len(examples), max_mel_len, mel_list[0].shape[-1])
        for i, mel in enumerate(mel_list):
            padded_mels[i, : mel.shape[0]] = mel
        mel_lengths_t = torch.tensor(mel_lengths, dtype=torch.long)

        sound_token_id = processor.tokenizer.convert_tokens_to_ids("<so_embedding>")

        new_input_ids_list = []
        for i, ex in enumerate(examples):
            ids = batch["input_ids"][i]
            n_tokens = n_audio_tokens_list[i]
            if n_tokens > 0:
                # Find existing <so_embedding> token(s) and replace with correct count
                sound_mask = ids == sound_token_id
                existing_count = sound_mask.sum().item()
                if existing_count > 0:
                    # Remove existing sound tokens and insert correct count at same position
                    first_pos = sound_mask.nonzero(as_tuple=True)[0][0].item()
                    ids_before = ids[:first_pos]
                    ids_after = ids[first_pos + existing_count :]
                    sound_tokens = torch.full((n_tokens,), sound_token_id, dtype=ids.dtype)
                    ids = torch.cat([ids_before, sound_tokens, ids_after])
                else:
                    # No existing sound token, insert at position 1
                    sound_tokens = torch.full((n_tokens,), sound_token_id, dtype=ids.dtype)
                    ids = torch.cat([ids[:1], sound_tokens, ids[1:]])
            new_input_ids_list.append(ids)

        max_len = max(ids.shape[0] for ids in new_input_ids_list)
        pad_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
        padded_ids = torch.full((len(examples), max_len), pad_id, dtype=new_input_ids_list[0].dtype)
        for i, ids in enumerate(new_input_ids_list):
            padded_ids[i, : ids.shape[0]] = ids
        batch["input_ids"] = padded_ids
        batch["sound_clips"] = padded_mels
        batch["sound_length"] = mel_lengths_t

    # --- Loss mask (same pattern as nemotron_vl) ---
    loss_mask = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])
    ]

    # --- Image token adjustment (only when images are present) ---
    img_start_token_id = processor.tokenizer.convert_tokens_to_ids("<img>")
    img_end_token_id = processor.tokenizer.convert_tokens_to_ids("</img>")
    has_img_tokens = (batch["input_ids"] == img_start_token_id).any()
    if has_img_tokens:
        # Dynamic-res: one <image> token per image; LM-side expansion is driven
        # by per-image ``num_image_tiles`` (set below to shuffled_count_i) with
        # ``img_seq_len=1``. Static-tile path keeps the HF processor's num_patches.
        if is_dynamic_res_processor:
            key_pv = "pixel_values_videos" if is_video else "pixel_values"
            pv_ref = batch.get(key_pv)
            if pv_ref is None:
                n_imgs = 0
            elif isinstance(pv_ref, list):
                n_imgs = len(pv_ref)
            else:
                n_imgs = int(pv_ref.shape[0])
            num_tiles_for_adjust = torch.ones(n_imgs, dtype=torch.long)
        else:
            num_tiles_for_adjust = batch.get(
                "num_patches", torch.zeros(len(examples), dtype=torch.long)
            )
        adjusted_batch = adjust_image_tokens(
            {"input_ids": batch["input_ids"], "loss_mask": torch.tensor(loss_mask)},
            num_tiles_for_adjust,
            img_start_token_id,
            img_end_token_id,
        )
    else:
        adjusted_batch = {"input_ids": batch["input_ids"], "loss_mask": torch.tensor(loss_mask)}

    if is_video:
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<video>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        adjusted_batch["input_ids"] = torch.where(
            adjusted_batch["input_ids"] == video_token_id, image_token_id, adjusted_batch["input_ids"]
        )

    batch["input_ids"] = adjusted_batch["input_ids"]
    loss_mask = adjusted_batch["loss_mask"]

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    key = "pixel_values_videos" if is_video else "pixel_values"
    if key in batch:
        pv_raw = batch[key]
        del batch[key]
        # Dynamic-resolution image path (newer Nemotron-3 Omni Reasoning):
        # patchify per-image with its own (py, px) and concatenate into a
        # single [1, total_patches, 3*P*P] sequence. Emit per-image (H, W) in
        # ``imgs_sizes`` and per-image shuffled token counts in
        # ``num_image_tiles`` so the LM merge (with img_seq_len=1) can fan
        # out variable per-image token counts. Handles both the uniform-4D
        # case (mbs=1 or all images same shape) and the list-of-tensors case
        # (mbs>1 with mixed shapes).
        if (not is_video) and is_dynamic_res_processor and (
            isinstance(pv_raw, list) or (torch.is_tensor(pv_raw) and pv_raw.dim() == 4 and pv_raw.shape[0] > 0)
        ):
            P = 16  # RADIO patch_dim
            if isinstance(pv_raw, list):
                imgs_iter = [t.to(torch.bfloat16) for t in pv_raw]
            else:
                pv_t = pv_raw.to(torch.bfloat16)
                imgs_iter = [pv_t[i] for i in range(pv_t.shape[0])]
            patch_seqs: list[torch.Tensor] = []
            sizes: list[list[int]] = []
            num_tiles: list[int] = []
            for img in imgs_iter:
                assert img.dim() == 3, f"expected [3,H,W], got {tuple(img.shape)}"
                C, H, W = img.shape
                assert H % P == 0 and W % P == 0, f"Image {H}x{W} not divisible by patch_dim {P}"
                py, px = H // P, W // P
                # [3, H, W] → [py, P, px, P, 3] → [py*px, 3*P*P]
                patched = (
                    img.reshape(3, py, P, px, P)
                    .permute(1, 3, 0, 2, 4)
                    .reshape(py * px, 3 * P * P)
                    .contiguous()
                )
                patch_seqs.append(patched)
                sizes.append([H, W])
                num_tiles.append((py * px) // 4)
            pv = torch.cat(patch_seqs, dim=0).unsqueeze(0).contiguous()
            batch["imgs_sizes"] = torch.tensor(sizes, dtype=torch.long)
            batch["num_frames"] = torch.tensor([1] * len(imgs_iter), dtype=torch.long)
            # ``torch.int`` matches LLaVAModel._preprocess_data's expected dtype
            # (``image_token_mask.int().clone()`` on the destination side).
            batch["num_image_tiles"] = torch.tensor(num_tiles, dtype=torch.int)
        else:
            pv = pv_raw.to(torch.bfloat16) if torch.is_tensor(pv_raw) else pv_raw
        batch["visual_inputs"] = GenericVisualInputs(pixel_values=pv)
    else:
        batch["visual_inputs"] = None

    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = IGNORE_INDEX
    batch["labels"] = labels

    loss_mask_t = torch.tensor(loss_mask, dtype=torch.float, device=batch["input_ids"].device)
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask_t

    if pack_sequences and batch["input_ids"].shape[0] > 1:
        # Pack [B, S_padded] → [1, sum(L_i)] using actual per-sample lengths.
        # Derive per-sample length from input_ids non-pad positions.
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(processor.tokenizer, "eos_token_id", 0) or 0
        input_ids_b = batch["input_ids"]
        labels_b = batch["labels"]
        loss_mask_b = batch["loss_mask"]
        pos_b = batch["position_ids"]
        B = input_ids_b.shape[0]
        lengths = [int((input_ids_b[i] != pad_id).sum().item()) for i in range(B)]
        ids_flat = torch.cat([input_ids_b[i, : lengths[i]] for i in range(B)], dim=0).unsqueeze(0)
        labels_flat = torch.cat([labels_b[i, : lengths[i]] for i in range(B)], dim=0).unsqueeze(0)
        loss_mask_flat = torch.cat([loss_mask_b[i, : lengths[i]] for i in range(B)], dim=0).unsqueeze(0)
        pos_flat = torch.cat(
            [torch.arange(L, dtype=pos_b.dtype, device=pos_b.device) for L in lengths], dim=0
        ).unsqueeze(0)

        cu = [0]
        for L in lengths:
            cu.append(cu[-1] + L)
        cu_t = torch.tensor(cu, dtype=torch.int32)
        # argmin = len(cu): downstream `cu_seqlens_padded[: argmin]` becomes a no-op.
        argmin_t = torch.tensor(len(cu), dtype=torch.int32)
        max_t = torch.tensor(max(lengths), dtype=torch.int32)

        batch["input_ids"] = ids_flat
        batch["labels"] = labels_flat
        batch["loss_mask"] = loss_mask_flat
        batch["position_ids"] = pos_flat
        batch["cu_seqlens"] = cu_t
        batch["cu_seqlens_unpadded"] = cu_t.clone()
        batch["cu_seqlens_argmin"] = argmin_t
        batch["cu_seqlens_unpadded_argmin"] = argmin_t.clone()
        batch["max_seqlen"] = max_t
        # TE derives the causal + per-sample mask from cu_seqlens; drop attention_mask.
        batch["attention_mask"] = None

    return batch


def ministral3_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Collate function for Ministral 3 VL model."""
    skipped_tokens = extract_skipped_token_ids(processor)

    if processor.chat_template is not None:
        batch = processor.apply_chat_template(
            [example["conversation"] for example in examples],
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        texts = []
        for example in examples:
            conv_text = []
            for msg in example["conversation"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Handle multimodal content (list of items)
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                text_parts.append("[IMG]")
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = " ".join(text_parts)

                conv_text.append(f"{role.capitalize()}: {content}")
            texts.append("\n".join(conv_text))

        images = []
        for example in examples:
            ex_images = []
            for msg in example.get("conversation", []):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            if "image" in item:
                                ex_images.append(item["image"])
                            elif "path" in item:
                                ex_images.append(Image.open(item["path"]))
            images.append(ex_images if ex_images else None)
        batch = processor(
            text=texts,
            images=[img if img else [] for img in images],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    if "input_ids" in batch:
        labels = batch["input_ids"].clone()[:, 1:]
        labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
        labels[torch.isin(labels, skipped_tokens)] = -100
        batch["labels"] = labels

        # Create loss mask using search-based masking for assistant turns
        loss_masks = [
            create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
            for example, input_ids in zip(examples, batch["input_ids"])
        ]
        loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
        # Unmask the last token (EOS) so the model learns when to stop generating
        loss_mask_t[:, -1] = 1
        # Shift loss mask to align with next-token labels timeline
        loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
        # Enforce label masking to match shifted loss_mask
        batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
        batch["loss_mask"] = loss_mask_t

    if "position_ids" not in batch and "input_ids" in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1).clone()
        )

    # Wrap visual tensors in GenericVisualInputs so vlm_step.py picks them up
    visual_kwargs = {}
    for vk in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "image_sizes"):
        if vk in batch:
            visual_kwargs[vk] = batch.pop(vk)
    batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None

    return batch


def default_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Default collate function for VLM models."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1).clone()
        )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t
    # Build Qwen2VL visual inputs object and attach to batch; remove raw keys
    visual_inputs = Qwen2_5_VLVisualInputs(
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
    )
    if "pixel_values" in batch:
        del batch["pixel_values"]
    if "image_grid_thw" in batch:
        del batch["image_grid_thw"]
    batch["visual_inputs"] = visual_inputs
    return batch


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3VLProcessor": qwen2_5_collate_fn,
    "NemotronNanoVLV2Processor": nemotron_nano_v2_vl_collate_fn,
    "NemotronH_Nano_Omni_Reasoning_V3Processor": nemotron_omni_collate_fn,
    "PixtralProcessor": ministral3_collate_fn,  # Ministral3 uses PixtralProcessor
    "default": default_collate_fn,
}
