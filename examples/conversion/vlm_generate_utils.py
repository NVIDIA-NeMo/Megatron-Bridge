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

"""Shared utilities for VLM generation scripts."""

from typing import Optional

import requests
import torch
from PIL import Image


try:
    from qwen_vl_utils import process_vision_info

    _HAS_QWEN_VL_UTILS = True
except ImportError:
    _HAS_QWEN_VL_UTILS = False


def patch_kimi_vision_processor(hf_model_path: str):
    """Monkey-patch KimiK25VisionProcessor.from_dict to avoid duplicate keyword errors.

    The upstream from_dict passes both **config and **kwargs to cls(), which causes
    'got multiple values for keyword argument' when AutoProcessor injects '_from_auto'.
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        klass = get_class_from_dynamic_module(
            "kimi_k25_vision_processing.KimiK25VisionProcessor",
            hf_model_path,
        )
        if klass is None or getattr(klass, "_from_dict_patched", False):
            return

        @classmethod  # type: ignore[misc]
        def _patched_from_dict(cls, config_dict, **kwargs):
            config = config_dict.copy()
            for key in list(kwargs.keys()):
                config.pop(key, None)
            media_proc_cfg = config.pop("media_proc_cfg", {})
            return cls(media_proc_cfg=media_proc_cfg, **config, **kwargs)

        klass.from_dict = _patched_from_dict
        klass._from_dict_patched = True
    except Exception:
        pass


def pre_expand_image_tokens(input_ids, grid_thws, image_token_id, spatial_merge_size=2):
    """Pre-expand single image placeholders to N placeholders matching vision
    feature count.

    With PP > 1 the pipeline schedule needs to know the actual sequence length
    upfront.  Dynamic expansion inside the model changes seq_length during
    forward, causing send/recv shape mismatches.  Pre-expanding makes the model
    use the 1:1 replacement path (is_pre_expanded=True).
    """
    if grid_thws is None:
        return input_ids

    feature_counts = []
    for grid_thw in grid_thws:
        t, h, w = grid_thw.tolist()
        num_features = int(t * (h // spatial_merge_size) * (w // spatial_merge_size))
        feature_counts.append(num_features)

    expanded = []
    feat_idx = 0
    for token_id in input_ids[0]:
        if token_id.item() == image_token_id and feat_idx < len(feature_counts):
            expanded.extend([image_token_id] * feature_counts[feat_idx])
            feat_idx += 1
        else:
            expanded.append(token_id.item())

    return torch.tensor([expanded], dtype=input_ids.dtype, device=input_ids.device)


def pad_input_ids_to_tp_multiple(input_ids, tp_size: int, pad_token_id: int = 0):
    """Pad input_ids so sequence length is divisible by tp_size.

    Needed for sequence-parallel, which is required for MoE models using TP + EP.
    No-op when tp_size is 1 or the sequence is already aligned.
    """
    if tp_size <= 1:
        return input_ids
    seq_len = input_ids.shape[1]
    remainder = seq_len % tp_size
    if remainder == 0:
        return input_ids
    pad_len = tp_size - remainder
    padding = torch.full((input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids, padding], dim=1)


def load_image(image_path: str) -> Image.Image:
    """Load an image from URL or file path."""
    if image_path.startswith(("http://", "https://")):
        return Image.open(requests.get(image_path, stream=True).raw)
    return Image.open(image_path)


def to_cuda(x):
    """Move a tensor, list of tensors, or None to CUDA."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [t.cuda() for t in x]
    return x.cuda()


def process_image_inputs(
    processor,
    image_path: Optional[str],
    prompt: str,
    *,
    is_kimi: bool = False,
    image_token_id: Optional[int] = None,
):
    """Process image + prompt into model inputs.

    Returns:
        (input_ids, pixel_values, image_grid_thw, image_sizes, mm_token_type_ids)
    """
    if not image_path:
        inputs = processor(text=[prompt], return_tensors="pt")
        return inputs.input_ids, None, None, None, None

    if is_kimi:
        return _process_kimi_inputs(processor, image_path, prompt, image_token_id)

    if not _HAS_QWEN_VL_UTILS:
        raise ImportError(
            "qwen_vl_utils is required for non-Kimi VLM image processing. Install it with: pip install qwen-vl-utils"
        )
    return _process_default_inputs(processor, image_path, prompt)


def _process_kimi_inputs(processor, image_path, prompt, image_token_id):
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": load_image(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor(messages=messages)
    grid_thws = getattr(inputs, "grid_thws", None)
    input_ids = pre_expand_image_tokens(inputs.input_ids, grid_thws, image_token_id)
    return input_ids, inputs.pixel_values, grid_thws, None, None


def _process_default_inputs(processor, image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return (
        inputs.input_ids,
        inputs.get("pixel_values"),
        inputs.get("image_grid_thw"),
        inputs.get("image_sizes"),
        inputs.get("mm_token_type_ids"),
    )
