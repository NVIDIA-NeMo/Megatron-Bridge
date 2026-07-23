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

"""Nemotron VL collator implementations."""

import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.conversation_processing import (
    build_assistant_loss_mask,
    infer_assistant_mask_boundary_config,
    shared_chat_template_kwargs_from_examples,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


def nemotron_nano_v2_vl_collate_fn(
    examples: list,
    processor,
    start_of_response_token=None,
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
    """Collate function for Nemotron Nano V2 VL model."""
    del visual_keys, min_pixels, max_pixels

    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens

    # The video processor only accepts one example, while in-batch packing
    # requires a multi-example microbatch.
    is_video = examples[0]["conversation"][0]["content"][0]["type"] == "video"
    if enable_in_batch_packing:
        raise ValueError(
            "Nemotron-VL collation does not support in-batch packing because image embeddings expand "
            "after packed-sequence boundaries are built."
        )

    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = infer_assistant_mask_boundary_config(processor)
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

        prompt = processor.apply_chat_template(
            [example["conversation"] for example in examples],
            tokenize=False,
            **shared_chat_template_kwargs_from_examples(examples),
        )
        batch = processor(
            text=prompt,
            videos=frames,
            videos_kwargs={"video_metadata": metadata},
            return_tensors="pt",
        )
    else:
        # Ensure a pad_token is set so padding can produce uniform-length tensors.
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        with use_processor_right_padding(processor):
            batch = processor.apply_chat_template(
                [example["conversation"] for example in examples],
                tokenize=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_dict=True,
                **shared_chat_template_kwargs_from_examples(examples),
            )
    loss_mask = torch.stack(
        [
            build_assistant_loss_mask(
                example,
                input_ids,
                processor,
                skipped_tokens,
                boundary_config=boundary_config,
            ).to(dtype=torch.int)
            for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
        ]
    )

    img_start_token_id = 131073  # tokenizer.convert_tokens_to_ids("<img>")
    img_end_token_id = 131074  # tokenizer.convert_tokens_to_ids("</img>")
    aligned_batch = {"input_ids": batch["input_ids"], "loss_mask": loss_mask}
    if "attention_mask" in batch:
        aligned_batch["attention_mask"] = batch["attention_mask"]
    adjusted_batch = adjust_image_tokens(
        aligned_batch,
        batch["num_patches"],
        img_start_token_id,
        img_end_token_id,
        padding_values={
            "input_ids": int(getattr(processor.tokenizer, "pad_token_id", 0) or 0),
            "loss_mask": 0,
            "attention_mask": 0,
        },
    )

    if is_video:
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<video>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        adjusted_batch["input_ids"] = torch.where(
            adjusted_batch["input_ids"] == video_token_id, image_token_id, adjusted_batch["input_ids"]
        )

    batch["input_ids"] = adjusted_batch["input_ids"]
    loss_mask = adjusted_batch["loss_mask"]
    if "attention_mask" in adjusted_batch:
        batch["attention_mask"] = adjusted_batch["attention_mask"]

    batch_size, seq_len = batch["input_ids"].shape
    batch["position_ids"] = torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)

    key = "pixel_values_videos" if is_video else "pixel_values"
    pv = batch[key].to(torch.bfloat16)
    batch[key] = pv
    batch["visual_inputs"] = GenericVisualInputs(pixel_values=pv)
    # roll label by 1 and fill last token with IGNORE_INDEX
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = IGNORE_INDEX
    batch["labels"] = labels

    loss_mask_t = loss_mask.to(dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask_t
    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        pad_token_id=int(getattr(processor.tokenizer, "pad_token_id", 0) or 0),
        ignore_index=IGNORE_INDEX,
    )
    return batch
