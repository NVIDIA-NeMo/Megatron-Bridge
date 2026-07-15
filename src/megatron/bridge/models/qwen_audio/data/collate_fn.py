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

"""Qwen audio collator implementations."""

import warnings

import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.conversation_processing import (
    chat_template_kwargs_from_example,
    gather_assistant_text_segments,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.training.utils.visual_inputs import Qwen2AudioInputs


def qwen2_audio_collate_fn(
    examples: list,
    processor,
    *,
    visual_keys: object = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2-Audio model.

    Uses HF-compatible label construction:
    - Backward search for assistant text spans (matching HF Trainer convention)
    - No skipped_tokens masking on labels (model learns to predict EOS/im_end)
    - Loss mask derived directly from active label positions

    Qwen2-Audio keeps packing in ``audio_lm_step`` because audio features must
    remain batch-aligned until model input preparation.
    """
    del visual_keys, min_pixels, max_pixels
    if enable_in_batch_packing:
        warnings.warn(
            "Qwen2-Audio defers in-batch packing to audio_lm_step; collate returns right-padded rows.",
            stacklevel=2,
        )

    texts = []
    audio_inputs = []
    audio_sampling_rates = set()
    has_audio_without_sampling_rate = False
    for example in examples:
        texts.append(
            processor.apply_chat_template(
                example["conversation"],
                tokenize=False,
                **chat_template_kwargs_from_example(example),
            )
        )
        audio = example.get("audio")
        if audio is not None:
            if isinstance(audio, tuple):
                audio_inputs.append(audio[0])
                audio_sampling_rates.add(int(audio[1]))
            elif isinstance(audio, dict):
                audio_inputs.append(audio["array"])
                sampling_rate = audio.get("sampling_rate")
                if sampling_rate is None:
                    has_audio_without_sampling_rate = True
                else:
                    audio_sampling_rates.add(int(sampling_rate))
            else:
                audio_inputs.append(audio)
                has_audio_without_sampling_rate = True

    if len(audio_sampling_rates) > 1:
        raise ValueError("Qwen2-Audio batches require a single sampling rate.")
    if audio_sampling_rates and has_audio_without_sampling_rate:
        raise ValueError("Qwen2-Audio batches cannot mix audio with known and unknown sampling rates.")
    audio_kwargs = {"sampling_rate": next(iter(audio_sampling_rates))} if audio_sampling_rates else {}

    tokenizer = getattr(processor, "tokenizer", processor)

    saved_padding_side = getattr(tokenizer, "padding_side", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
    try:
        batch = processor(
            text=texts,
            audio=audio_inputs if audio_inputs else None,
            return_tensors="pt",
            padding=True,
            **audio_kwargs,
        )
    finally:
        if tokenizer is not None and saved_padding_side is not None:
            tokenizer.padding_side = saved_padding_side

    input_ids = batch["input_ids"]
    pad_token_id = tokenizer.pad_token_id

    # --- HF-compatible label construction ---
    # Step 1: Build unshifted labels (same convention as HF Trainer)
    hf_labels = input_ids.clone()

    for i, example in enumerate(examples):
        ids = input_ids[i].tolist()
        assistant_texts = gather_assistant_text_segments(example)

        # Find assistant span using backward search (like HF's Qwen2AudioCollator)
        found = -1
        for asst_text in assistant_texts:
            asst_token_ids = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
            span_len = len(asst_token_ids)
            if span_len == 0:
                continue
            for start in range(len(ids) - span_len, -1, -1):
                if ids[start : start + span_len] == asst_token_ids:
                    found = start
                    break
            if found >= 0:
                break

        if found >= 0:
            # Mask everything before the assistant span (prompt + special tokens)
            hf_labels[i, :found] = IGNORE_INDEX
        else:
            warnings.warn(f"Could not find assistant span for example {i}, masking all labels", stacklevel=2)
            hf_labels[i, :] = IGNORE_INDEX

        # Mask padding tokens
        if pad_token_id is not None:
            hf_labels[i][input_ids[i] == pad_token_id] = IGNORE_INDEX

    # Step 2: Shift labels for Megatron (labels[j] = hf_labels[j+1])
    labels = hf_labels[:, 1:]
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    batch["labels"] = labels

    # Step 3: Derive loss_mask from active label positions
    batch["loss_mask"] = (labels != IGNORE_INDEX).float()

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
            .contiguous()
        )
    batch["audio_inputs"] = Qwen2AudioInputs(
        input_features=batch.get("input_features"),
        feature_attention_mask=batch.get("feature_attention_mask"),
    )
    for key in ("input_features", "feature_attention_mask"):
        batch.pop(key, None)

    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=False,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        pad_token_id=int(pad_token_id or 0),
        ignore_index=IGNORE_INDEX,
    )
    return batch
