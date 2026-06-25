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

"""Nemotron Omni collator implementations."""

import torch

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.hf_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.data.sequence_batching import pad_or_pack_sequence
from megatron.bridge.data.vlm_processing import assistant_mask_boundary_config_from_markers, build_assistant_loss_mask
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


CHATML_ASSISTANT_START = "<|im_start|>assistant\n"
CHATML_TURN_END = "<|im_end|>"


def nemotron_omni_collate_fn(
    examples: list,
    processor,
    start_of_response_token=None,
    *,
    visual_keys: object = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    enable_in_batch_packing: bool = False,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for Nemotron Omni model (vision + audio + language).

    Extends nemotron_nano_v2_vl_collate_fn with audio support. Each example
    may carry an ``audio_path`` field pointing to a 16 kHz mono WAV file.
    Audio is converted to mel spectrograms and added to the batch as
    ``sound_clips`` / ``sound_length`` tensors consumed by LLaVAModel.forward().

    When ``enable_in_batch_packing=True``, samples in the microbatch are concatenated
    along the sequence dim into a single ``[1, sum(L_i)]`` batch, and
    ``cu_seqlens`` / ``cu_seqlens_argmin`` / ``max_seqlen`` are emitted so
    TE's THD attention kernels handle per-sample masking without an attention
    mask. ``cu_seqlens_unpadded`` is also emitted when per-sequence padding is
    inserted for CP/SP divisibility.
    """
    del visual_keys, min_pixels, max_pixels

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
    boundary_config = assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=CHATML_ASSISTANT_START,
        assistant_end=CHATML_TURN_END,
    )
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
                padded_ids = torch.full((len(per_ex_batches), max_len), pad_id, dtype=ids_list[0].dtype)
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
    loss_mask = torch.stack(
        [
            build_assistant_loss_mask(
                example,
                input_ids,
                processor,
                skipped_tokens,
                boundary_config=boundary_config,
            ).to(dtype=torch.int)
            for example, input_ids in zip(examples, batch["input_ids"])
        ]
    )

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
            num_tiles_for_adjust = batch.get("num_patches", torch.zeros(len(examples), dtype=torch.long))
        adjusted_batch = adjust_image_tokens(
            {"input_ids": batch["input_ids"], "loss_mask": loss_mask},
            num_tiles_for_adjust,
            img_start_token_id,
            img_end_token_id,
        )
    else:
        adjusted_batch = {"input_ids": batch["input_ids"], "loss_mask": loss_mask}

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
        if (
            (not is_video)
            and is_dynamic_res_processor
            and (isinstance(pv_raw, list) or (torch.is_tensor(pv_raw) and pv_raw.dim() == 4 and pv_raw.shape[0] > 0))
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
                patched = img.reshape(3, py, P, px, P).permute(1, 3, 0, 2, 4).reshape(py * px, 3 * P * P).contiguous()
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

    loss_mask_t = loss_mask.to(dtype=torch.float, device=batch["input_ids"].device)
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask_t

    pad_or_pack_sequence(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        ignore_index=IGNORE_INDEX,
    )

    return batch
