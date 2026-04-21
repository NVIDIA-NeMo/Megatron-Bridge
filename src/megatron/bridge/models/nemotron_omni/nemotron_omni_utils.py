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

import math
from typing import Union

import numpy as np
import torch


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and resample to ``target_sr`` Hz.

    Supports WAV, MP3, FLAC, and other formats handled by *soundfile*
    (with *librosa* as a fallback for MP3 and other FFmpeg-decoded formats).

    Args:
        path: Path to the audio file.
        target_sr: Target sampling rate in Hz.

    Returns:
        1-D float32 numpy array of the mono waveform at ``target_sr``.
    """
    try:
        import soundfile as sf

        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa

        waveform, sr = librosa.load(path, sr=None, mono=True)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)

    if sr != target_sr:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    return waveform.astype(np.float32)


def compute_mel_features(
    waveform: Union[np.ndarray, list],
    sampling_rate: int = 16000,
    num_mel_bins: int = 128,
) -> torch.Tensor:
    """Convert a raw waveform to a mel spectrogram tensor.

    Uses HF ``ParakeetFeatureExtractor`` (from ``transformers``) to produce
    mel features compatible with ``BridgeSoundEncoder`` / ``FastConformerModel``.

    Args:
        waveform: 1-D float32 numpy array (or list) of the mono waveform.
        sampling_rate: Sampling rate of *waveform* (must match the extractor).
        num_mel_bins: Number of mel frequency bins.

    Returns:
        Float tensor of shape ``(frames, num_mel_bins)`` -- a single clip
        ready to be batched and passed as ``sound_clips`` to the model.
    """
    from transformers import ParakeetFeatureExtractor

    extractor = ParakeetFeatureExtractor(
        feature_size=num_mel_bins,
        sampling_rate=sampling_rate,
    )
    features = extractor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    mel = features["input_features"].squeeze(0)
    return mel


def compute_audio_token_count(
    waveform: Union[np.ndarray, list],
    hop_length: int = 160,
    subsampling_factor: int = 8,
) -> int:
    """Compute the expected number of audio tokens for a waveform.

    Mirrors the formula used by the HF ``NemotronNanoVLV2Processor`` to
    determine how many ``<so_embedding>`` tokens to insert.

    When using the HF processor directly, this computation is performed
    automatically.  This utility is provided for custom pipelines that
    bypass the processor (e.g. nemo-rl data loaders).

    Args:
        waveform: 1-D waveform array (only its length is used).
        hop_length: Hop length in samples for mel feature extraction.
        subsampling_factor: Subsampling factor of the conformer encoder.

    Returns:
        Number of audio tokens (at least 1).
    """
    num_frames = len(waveform) // hop_length
    n_tokens = math.ceil(num_frames / subsampling_factor)
    return max(1, n_tokens)
