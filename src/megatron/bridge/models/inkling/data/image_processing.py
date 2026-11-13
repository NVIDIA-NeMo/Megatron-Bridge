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

"""Inkling2 image preprocessing for training.

The ``thinkingmachines/Inkling-2-Small`` release names an ``Inkling2ImageProcessor`` in its
``processor_config.json`` but ships no implementation, and the Transformers version Bridge pins has none
(Transformers adds one in v5.23.0), so this implements the release's settings: an image keeps its native size,
is padded on the
right and bottom to a multiple of 40 pixels with a value just below black, rescaled by 1/255 and normalized
with mean and standard deviation 0.5 (to [-1, 1]), and cut into 40x40-pixel patches in row-major order.
Each patch repeats twice in time, and each one becomes one image token for the language model. The vision
encoder splits a patch into four 20-pixel patches and merges their outputs back into one embedding.

With ``max_image_tokens``, a larger image is encoded at the largest size within that token budget that keeps
its aspect ratio, resized with a Lanczos3 kernel in FP32 arithmetic that takes its sine from the C library's
single-precision ``sinf``. On smooth images it stays within one level of Pillow's ``LANCZOS`` filter.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import io
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PATCH_SIZE = 40
TEMPORAL_PATCH_SIZE = 2
PROCESSOR_CONFIG_FILE = "processor_config.json"
# The ``image_processor`` section of the release's processor_config.json, the only settings this port implements.
IMAGE_PROCESSOR_CONFIG = {
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "Inkling2ImageProcessor",
    "image_std": [0.5, 0.5, 0.5],
    "rescale_factor": 1 / 255,
    "size": {"height": PATCH_SIZE, "width": PATCH_SIZE},
}
_MEAN = np.array(IMAGE_PROCESSOR_CONFIG["image_mean"], dtype=np.float32)
_STD = np.array(IMAGE_PROCESSOR_CONFIG["image_std"], dtype=np.float32)
_LANCZOS_LOBES = np.float32(3)


@functools.cache
def _sinf() -> Callable[[float], float]:
    """Return the C library's single-precision sine.

    The Lanczos3 kernel evaluates its sine in single precision. A double-precision sine rounded to FP32 differs for
    about 1% of the kernel's arguments, enough to move some resized pixels by one level.
    """
    try:
        # The running process normally has the C math library loaded already.
        function = ctypes.CDLL(None).sinf
    except AttributeError:
        path = ctypes.util.find_library("m")
        if path is None:
            raise RuntimeError("The C library's sinf, which the Lanczos3 resize uses, is unavailable") from None
        function = ctypes.CDLL(path).sinf
    function.restype = ctypes.c_float
    function.argtypes = [ctypes.c_float]
    return function


def _sinc(value: np.float32) -> np.float32:
    if value == np.float32(0):
        return np.float32(1)
    angle = value * np.float32(math.pi)
    return np.float32(_sinf()(angle)) / angle


def _lanczos_weights(source_size: int, target_size: int, output_index: int) -> tuple[int, np.ndarray]:
    """First source index and normalized FP32 Lanczos3 weights for one output index."""
    ratio = np.float32(source_size) / np.float32(target_size)
    scale = max(np.float32(1), ratio)
    support = _LANCZOS_LOBES * scale
    center = (np.float32(output_index) + np.float32(0.5)) * ratio
    left = min(max(math.floor(center - support), 0), source_size - 1)
    right = min(max(math.ceil(center + support), left + 1), source_size)
    center -= np.float32(0.5)
    weights = np.empty(right - left, dtype=np.float32)
    total = np.float32(0)
    for index in range(left, right):
        distance = (np.float32(index) - center) / scale
        weight = np.float32(0)
        if abs(distance) < _LANCZOS_LOBES:
            weight = _sinc(distance) * _sinc(distance / _LANCZOS_LOBES)
        weights[index - left] = weight
        total += weight
    return left, weights / total


def resize_lanczos3(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize ``[H, W, C]`` uint8 pixels with a Lanczos3 kernel: rows first, then columns.

    Each output value sums its taps in order in FP32, then rounds half up and clips to uint8.
    """
    source_height, source_width, channels = pixels.shape
    vertical = np.empty((height, source_width, channels), dtype=np.float32)
    for y in range(height):
        left, weights = _lanczos_weights(source_height, height, y)
        value = np.zeros((source_width, channels), dtype=np.float32)
        for offset, weight in enumerate(weights):
            value += pixels[left + offset].astype(np.float32) * weight
        vertical[y] = value
    output = np.empty((height, width, channels), dtype=np.uint8)
    for x in range(width):
        left, weights = _lanczos_weights(source_width, width, x)
        value = np.zeros((height, channels), dtype=np.float32)
        for offset, weight in enumerate(weights):
            value += vertical[:, left + offset] * weight
        value = np.clip(value, np.float32(0), np.float32(255))
        output[:, x] = np.floor(value.astype(np.float64) + 0.5).astype(np.uint8)
    return output


@dataclass(frozen=True)
class Inkling2EncodedImage:
    """One preprocessed image.

    Attributes:
        patches: ``[rows * columns, 2, 40, 40, 3]`` BF16 patches, the model's ``pixel_values`` rows.
        grid_thw: ``[1, 3]`` grid ``(1, rows, columns)``, the model's ``image_grid_thw`` row.
        num_tokens: Image tokens the language model reserves for this image, ``rows * columns``.
    """

    patches: torch.Tensor
    grid_thw: torch.Tensor
    num_tokens: int


def _image_bytes(image: bytes | str | os.PathLike | Image.Image) -> bytes:
    """Return encoded image bytes; a PIL image is saved as PNG, as the reference does."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        mode = "RGBA" if image.has_transparency_data else "RGB"
        with image.convert(mode) as pixels:
            pixels.save(buffer, format="PNG")
        return buffer.getvalue()
    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        if "://" in path and not path.startswith("file://"):
            raise ValueError(f"Inkling2 image preprocessing reads local files only, got {path!r}")
        return Path(path.removeprefix("file://")).read_bytes()
    raise TypeError(f"Expected image bytes, a PIL image or a path, got {type(image).__name__}")


def _decode_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode to ``[H, W, 3]`` uint8, compositing transparent images onto an 8-pixel checkerboard."""
    if image_bytes.startswith(b"\xff\xd8") and not image_bytes.endswith(b"\xff\xd9"):
        # A JPEG without its end marker; the reference completes it before decoding.
        image_bytes += b"\xff\xd9"
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        if not image.has_transparency_data:
            return np.array(image.convert("RGB"), dtype=np.uint8)
        rgba = np.array(image.convert("RGBA"), dtype=np.uint16)
    height, width, _ = rgba.shape
    y, x = np.ogrid[:height, :width]
    background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153).astype(np.uint16)
    alpha = rgba[:, :, 3:4]
    return ((rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127) // 255).astype(np.uint8)


def load_processor_config(path: str | os.PathLike, **download_kwargs: Any) -> dict[str, Any]:
    """Read a checkpoint's ``processor_config.json``.

    Args:
        path: Checkpoint directory or Hugging Face repository ID.
        **download_kwargs: ``hf_hub_download`` arguments, such as ``revision``, for a repository ID.

    Returns:
        The parsed configuration.
    """
    if Path(path).is_dir():
        config_path = Path(path) / PROCESSOR_CONFIG_FILE
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(str(path), PROCESSOR_CONFIG_FILE, **download_kwargs))
    with config_path.open() as stream:
        return json.load(stream)


class Inkling2ImageProcessor:
    """Turns images into Inkling2 vision patches and image-token counts.

    Args:
        max_image_tokens: Token budget per image; a larger image is encoded at the largest size that fits,
            keeping its aspect ratio. ``None`` keeps every image at its native size, as the reference does.
    """

    patch_size = PATCH_SIZE
    temporal_patch_size = TEMPORAL_PATCH_SIZE

    def __init__(self, *, max_image_tokens: int | None = None) -> None:
        if max_image_tokens is not None and max_image_tokens < 1:
            raise ValueError(f"max_image_tokens must be positive, got {max_image_tokens}")
        self.max_image_tokens = max_image_tokens

    @classmethod
    def from_pretrained(
        cls, path: str | os.PathLike, *, max_image_tokens: int | None = None, **download_kwargs: Any
    ) -> Inkling2ImageProcessor:
        """Load from a checkpoint, checking the ``image_processor`` section of its ``processor_config.json``.

        Args:
            path: Checkpoint directory or Hugging Face repository ID.
            max_image_tokens: Token budget per image, or ``None`` for native sizes.
            **download_kwargs: ``hf_hub_download`` arguments, such as ``revision``, for a repository ID.

        Returns:
            The image processor.

        Raises:
            ValueError: If the checkpoint asks for image processing this port does not implement.
        """
        return cls.from_dict(load_processor_config(path, **download_kwargs), max_image_tokens=max_image_tokens)

    @classmethod
    def from_dict(
        cls, processor_config: dict[str, Any], *, max_image_tokens: int | None = None
    ) -> Inkling2ImageProcessor:
        """Build from a parsed ``processor_config.json``.

        Args:
            processor_config: The checkpoint's processor configuration.
            max_image_tokens: Token budget per image, or ``None`` for native sizes.

        Returns:
            The image processor.

        Raises:
            ValueError: If the configuration asks for image processing this port does not implement.
        """
        if processor_config.get("image_processor") != IMAGE_PROCESSOR_CONFIG:
            raise ValueError(
                f"Unsupported Inkling2 image processor configuration: {processor_config.get('image_processor')}"
            )
        return cls(max_image_tokens=max_image_tokens)

    @staticmethod
    def count_tokens(width: int, height: int) -> int:
        """Return the image tokens for an image of ``width`` x ``height`` pixels."""
        if width <= 0 or height <= 0:
            return 0
        return -(-width // PATCH_SIZE) * -(-height // PATCH_SIZE)

    def fitted_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the size an image is encoded at: native, or the largest same-aspect size within the budget."""
        if self.max_image_tokens is None or self.count_tokens(width, height) <= self.max_image_tokens:
            return width, height
        scale = math.sqrt(self.max_image_tokens * PATCH_SIZE * PATCH_SIZE / (width * height))
        while True:
            fitted = max(1, int(width * scale)), max(1, int(height * scale))
            if self.count_tokens(*fitted) <= self.max_image_tokens:
                return fitted
            scale *= 0.99

    def encode(self, image: bytes | str | os.PathLike | Image.Image) -> Inkling2EncodedImage:
        """Preprocess one image.

        Args:
            image: Encoded image bytes, a PIL image, or a local path.

        Returns:
            The image's patches, grid, and token count.
        """
        pixels = _decode_rgb(_image_bytes(image))
        height, width, _ = pixels.shape
        fitted_width, fitted_height = self.fitted_size(width, height)
        if (fitted_width, fitted_height) != (width, height):
            pixels = resize_lanczos3(pixels, fitted_width, fitted_height)
            width, height = fitted_width, fitted_height
        rows, columns = -(-height // PATCH_SIZE), -(-width // PATCH_SIZE)
        padded = np.full((rows * PATCH_SIZE, columns * PATCH_SIZE, 3), -1, dtype=np.float32)
        padded[:height, :width] = pixels
        # Rescaling by division matches Transformers' fused (pixel - mean / factor) / (std / factor) in BF16.
        normalized = (padded / np.float32(255) - _MEAN) / _STD
        patches = (
            torch.from_numpy(normalized)
            .reshape(rows, PATCH_SIZE, columns, PATCH_SIZE, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(rows * columns, 1, PATCH_SIZE, PATCH_SIZE, 3)
            .to(torch.bfloat16)
            .expand(-1, TEMPORAL_PATCH_SIZE, -1, -1, -1)
            .contiguous()
        )
        return Inkling2EncodedImage(
            patches=patches,
            grid_thw=torch.tensor([[1, rows, columns]], dtype=torch.int64),
            num_tokens=rows * columns,
        )


__all__ = [
    "IMAGE_PROCESSOR_CONFIG",
    "PROCESSOR_CONFIG_FILE",
    "Inkling2EncodedImage",
    "Inkling2ImageProcessor",
    "load_processor_config",
    "resize_lanczos3",
]
