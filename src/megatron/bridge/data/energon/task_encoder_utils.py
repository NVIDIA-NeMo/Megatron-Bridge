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

"""Shared utilities for Energon-based VLM task encoders.

Contains helpers extracted from the Qwen-VL task encoder so they can be
reused by the generic ``HFTaskEncoder`` and any future
model-specific encoders.
"""

import inspect
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory
from webdataset.autodecode import Decoder, imagehandler

from megatron.bridge.utils.safe_pickle import safe_pickle_loads


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Mask / position-id helpers
# ---------------------------------------------------------------------------
def get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    eod_mask_loss: bool,
    reset_attention_mask: bool,
    reset_position_ids: bool,
    compute_attention_mask: bool = True,
):
    """Build masks and position ids for a left-to-right model.

    Args:
        data: Input token ids of shape ``[b, s]``.
        eod_token: End-of-document token id.
        eod_mask_loss: If True, zero out loss at EOD positions.
        reset_attention_mask: If True, block cross-document attention.
        reset_position_ids: If True, restart position ids after each EOD.
        compute_attention_mask: If False, skip attention mask computation
            and return ``None`` for the mask.

    Returns:
        Tuple of ``(attention_mask, loss_mask, position_ids)`` where:

        - **attention_mask** -- ``[att_mask_batch, 1, s, s]`` boolean mask
          (``True`` = masked / blocked) or ``None`` when
          *compute_attention_mask* is False.
        - **loss_mask** -- ``[b, s]`` float mask (1.0 = keep, 0.0 = drop).
        - **position_ids** -- ``[b, s]`` position indices.
    """
    micro_batch_size, seq_length = data.size()

    att_mask_batch = micro_batch_size if reset_attention_mask else 1
    attention_mask = None
    if compute_attention_mask:
        attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length
        )

    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(micro_batch_size):
            eod_index = position_ids[b, data[b] == eod_token]
            if reset_position_ids:
                eod_index = eod_index.clone()
            prev_index = 0
            for j in range(eod_index.size(0)):
                i = eod_index[j]
                if reset_attention_mask and attention_mask is not None:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    if compute_attention_mask and attention_mask is not None:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------
def find_pattern_indices(sequence: np.ndarray, pattern, start: int = 0):
    """Find the ``[start, end)`` indices of the first occurrence of *pattern* in *sequence*.

    Args:
        sequence: 1-D array (or list) to search in.
        pattern: Sub-sequence to look for.
        start: Index in *sequence* at which to begin searching.

    Returns:
        ``(start_idx, end_idx)`` of the first match, or ``(-1, -1)`` if not
        found.
    """
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(sequence)
    pattern = np.array(pattern, dtype=sequence.dtype)
    n, m = sequence.shape[0], pattern.shape[0]
    if m == 0 or start >= n:
        return -1, -1
    end_limit = n - m + 1
    for i in range(start, max(end_limit, start)):
        if np.array_equal(sequence[i : i + m], pattern):
            return i, i + m
    return -1, -1


# ---------------------------------------------------------------------------
# PIL conversion helpers
# ---------------------------------------------------------------------------
def _tensor_to_pil(t):
    """Convert a ``[C, H, W]`` float tensor in ``[0, 1]`` to a PIL Image (uint8 ``[0, 255]``)."""
    from PIL import Image

    img_np = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def _images_to_pil(imgs):
    """Convert WDS tensor images to PIL to match the HF flow input format.

    WDS ``imagehandler`` decodes JPEG to float tensors in ``[0, 1]``.  The HF
    flow passes PIL images (uint8 ``[0, 255]``) to the processor.  Converting
    to PIL here ensures the processor applies identical rescaling and
    normalization in both flows.

    Args:
        imgs: A single ``[C, H, W]`` tensor, a ``[N, C, H, W]`` batch tensor,
            or a list of tensors / PIL images.

    Returns:
        A list of PIL images, or the input unchanged if it is not a tensor.
    """
    if isinstance(imgs, torch.Tensor):
        if imgs.dim() == 3:
            return [_tensor_to_pil(imgs)]
        elif imgs.dim() == 4:
            return [_tensor_to_pil(img) for img in imgs]
    elif isinstance(imgs, list):
        return [_tensor_to_pil(img) if isinstance(img, torch.Tensor) else img for img in imgs]
    return imgs


def _videos_to_pil(videos):
    """Convert WDS video frame tensors to PIL to match the HF flow input format.

    Args:
        videos: A list of videos, where each video is either a list of frame
            tensors or a ``[N, C, H, W]`` batch tensor.  ``None`` is passed
            through unchanged.

    Returns:
        A nested list ``[[PIL.Image, ...], ...]`` with one sub-list per video,
        or ``None`` if *videos* is ``None``.
    """
    if videos is None:
        return None
    result = []
    for video in videos:
        if isinstance(video, list):
            result.append([_tensor_to_pil(f) if isinstance(f, torch.Tensor) else f for f in video])
        elif isinstance(video, torch.Tensor):
            if video.dim() == 4:
                result.append([_tensor_to_pil(f) for f in video])
            elif video.dim() == 3:
                result.append([_tensor_to_pil(video)])
            else:
                result.append([video])
        else:
            result.append(video)
    return result


# ---------------------------------------------------------------------------
# Sample / dataset types
# ---------------------------------------------------------------------------
@dataclass
class ChatMLSample(Sample):
    """Multi-turn samples with optional media and chat-template tools."""

    conversation: str  # JSON string of GPT-format conversations
    imgs: Optional[List[torch.Tensor]] = None
    videos: Optional[List[List[torch.Tensor]]] = None
    audio: Optional[torch.Tensor] = None  # Raw waveform tensor [num_samples] or pre-computed mel [frames, mel_bins]
    tools: Optional[List[Dict[str, Any]]] = None


class videohandler:
    """Webdataset decoder handler for video fields stored as pickled frame lists."""

    def __init__(self, imagespec):
        self.extensions = ["jpgs", "mp4s", "videos"]
        self.extensions_mapping = {"jpgs": "jpg", "mp4s": "jpg", "videos": "jpg"}
        self.image_handler = imagehandler(imagespec)

    def __call__(self, key, data):
        """Decode pickled video data into lists of image tensors."""
        extension = re.sub(r".*[.]", "", key).lower()
        if extension not in self.extensions:
            return None
        data = safe_pickle_loads(data)
        key = self.extensions_mapping[extension]
        if extension == "jpgs":
            data = [self.image_handler(key, d) for d in data]
        else:
            data = [[self.image_handler(key, d) for d in video] for video in data]
        return data


class audiohandler:
    """Webdataset decoder handler for audio fields stored as raw WAV/FLAC bytes."""

    EXTENSIONS = {"wav", "flac", "mp3", "audio"}

    def __call__(self, key, data):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.EXTENSIONS:
            return None
        try:
            import io

            import soundfile as sf

            waveform, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
            if sr != 16000:
                import librosa

                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            return torch.from_numpy(waveform.astype(np.float32))
        except Exception:
            logging.warning(f"Failed to decode audio for key {key}")
            return None


class ChatMLWebdataset(DefaultDecoderWebdatasetFactory[ChatMLSample]):
    """Webdataset factory for multi-turn ChatML samples with multimodal support.

    Extends ``DefaultDecoderWebdatasetFactory`` to decode webdataset shards into
    ``ChatMLSample`` instances, using custom handlers for image and video fields.

    Args:
        path: Root path of the webdataset shards.
        auto_decode: Whether to install custom image/video decoders.  Passed
            through to the parent class.
        image_decode_spec: Decode spec forwarded to ``imagehandler`` /
            ``videohandler`` (e.g. ``"torchrgb"``).  When ``None`` (the
            default), falls back to the parent's ``image_decode`` attribute
            for backward compatibility with callers that set it via
            ``**kwargs``, and ultimately defaults to ``"torchrgb"``.
        **kwargs: Forwarded to ``DefaultDecoderWebdatasetFactory.__init__``.
            A ``decoder`` key, if present, is silently dropped because this
            class installs its own decoder.
    """

    __sample_type__ = ChatMLSample

    def __init__(self, path: EPath, *, auto_decode: bool = True, image_decode_spec: Optional[str] = None, **kwargs):
        kwargs.pop("decoder", None)
        kwargs.pop("auto_decode", None)
        parent_parameters = inspect.signature(DefaultDecoderWebdatasetFactory.__init__).parameters
        if "decoder" in parent_parameters:
            # Energon 7.4+ accepts a decoder object directly.
            kwargs["decoder"] = None
        else:
            # Earlier 7.x releases construct their optional PyAV decoder when
            # auto_decode is true, even for image-only datasets.
            kwargs["auto_decode"] = False
        super().__init__(path, **kwargs)
        if auto_decode:
            spec = image_decode_spec if image_decode_spec is not None else getattr(self, "image_decode", "torchrgb")
            self._decoder = Decoder(
                [
                    imagehandler(spec),
                    videohandler(spec),
                    audiohandler(),
                ]
            )


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------
def _load_chatml_payload(conversation: Any) -> Any:
    if isinstance(conversation, (str, bytes)):
        return json.loads(conversation)
    return conversation


def extract_chatml_tools(conversation: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract top-level chat-template tools from a wrapped ChatML payload."""
    payload = _load_chatml_payload(conversation)
    if not isinstance(payload, dict):
        return None
    tools = payload.get("tools")
    if tools is None:
        return None
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("ChatML tools must be a list of tool-definition dictionaries.")
    return deepcopy(tools)


def cook_chatml_sample(conversation: Any) -> List[Dict]:
    """Normalize a ChatML conversation to ``[{"role": ..., "content": ...}, ...]``.

    Accepts both ``from``/``value`` (GPT-style) and ``role``/``content``
    (OpenAI-style) formats. OpenAI message fields such as ``tool_calls``,
    ``tool_call_id``, and ``name`` are preserved.

    Args:
        conversation: A JSON string, bytes, list of turn dicts, or a dict
            with a ``"conversation"``, ``"conversations"``, or ``"messages"`` key.

    Returns:
        A list of OpenAI-style message dictionaries.

    Raises:
        ValueError: If the payload is empty or does not contain a conversation list.
    """
    payload = _load_chatml_payload(conversation)
    if isinstance(payload, dict):
        payload = payload.get("conversations", payload.get("messages", payload.get("conversation")))
    if not isinstance(payload, list) or not payload:
        raise ValueError("ChatML payload must contain a non-empty conversation list.")

    converted: List[Dict] = []
    for turn_idx, turn in enumerate(payload):
        if not isinstance(turn, dict):
            raise ValueError(f"ChatML turn {turn_idx} must be a dictionary.")

        converted_turn = deepcopy(turn)
        if "from" in converted_turn:
            role = converted_turn.pop("from")
            content = converted_turn.pop("value", "")
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            converted_turn["role"] = role
            converted_turn["content"] = content
        elif "role" not in converted_turn:
            raise ValueError(f"ChatML turn {turn_idx} must contain either 'role' or 'from'.")

        converted.append(converted_turn)

    return converted
