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

import json

import pytest
import torch

from megatron.bridge.data.energon.metadata import sample_metadata_kwargs
from megatron.bridge.data.energon.nemotron_omni_task_encoder import (
    NemotronOmniTaskBatch,
    NemotronOmniTaskEncoder,
    NemotronOmniTaskSample,
)
from megatron.bridge.data.energon.task_encoder_utils import IGNORE_INDEX, ChatMLSample
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


pytestmark = pytest.mark.unit


SO_EMBEDDING_ID = 90
SO_START_ID = 91
SO_END_ID = 92
IMG_START_ID = 93
IMG_END_ID = 94
ANSWER_IDS = [21, 22]


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self):
        self.last_conversation = None

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        self.last_conversation = conversation
        return "rendered prompt"

    def encode(self, text, add_special_tokens=False):
        return ANSWER_IDS

    def convert_tokens_to_ids(self, token):
        mapping = {
            "<so_embedding>": SO_EMBEDDING_ID,
            "<so_start>": SO_START_ID,
            "<so_end>": SO_END_ID,
            "<img>": IMG_START_ID,
            "</img>": IMG_END_ID,
        }
        return mapping[token]


class _Processor:
    def __init__(
        self,
        *,
        input_ids: list[int],
        pixel_values: torch.Tensor | None = None,
        num_patches: list[int] | None = None,
    ):
        self.tokenizer = _Tokenizer()
        self.image_processor = type("ImageProcessor", (), {"max_num_tiles": 4})()
        self.input_ids = torch.tensor([input_ids], dtype=torch.long)
        self.pixel_values = pixel_values
        self.num_patches = num_patches
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        out = {"input_ids": self.input_ids.clone()}
        if self.pixel_values is not None:
            out["pixel_values"] = self.pixel_values.clone()
        if self.num_patches is not None:
            out["num_patches"] = torch.tensor(self.num_patches, dtype=torch.long)
        return out


def _make_chatml_sample(conversation, *, imgs=None, videos=None, audio=None, key="k1"):
    return ChatMLSample(
        **sample_metadata_kwargs(key=key, restore_key=(), subflavors={}),
        imgs=imgs,
        videos=videos,
        audio=audio,
        conversation=json.dumps(conversation),
    )


def test_encode_sample_inserts_audio_tokens_after_image_end(monkeypatch):
    import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils

    monkeypatch.setattr(
        omni_utils,
        "compute_mel_features",
        lambda waveform, sampling_rate=16000, num_mel_bins=4: torch.ones(9, num_mel_bins),
    )
    processor = _Processor(input_ids=[1, IMG_END_ID, *ANSWER_IDS, 2])
    encoder = NemotronOmniTaskEncoder(processor=processor, seq_length=32, num_mel_bins=4)
    sample = _make_chatml_sample(
        [
            {"role": "user", "content": "Describe the audio."},
            {"role": "assistant", "content": "answer"},
        ],
        audio=torch.tensor([0.0, 0.1, -0.1], dtype=torch.float32),
    )

    encoded = encoder.encode_sample(sample)

    assert encoded.input_ids.tolist() == [
        1,
        IMG_END_ID,
        SO_START_ID,
        SO_EMBEDDING_ID,
        SO_EMBEDDING_ID,
        SO_END_ID,
        21,
        22,
        2,
    ]
    assert encoded.sound_clips.shape == (9, 4)
    assert encoded.sound_length.item() == 9
    assert encoded.labels[5:7].tolist() == ANSWER_IDS
    assert encoded.loss_mask[5:7].tolist() == [1.0, 1.0]
    assert encoded.labels[0].item() == IGNORE_INDEX


def test_encode_sample_uses_mock_video_frames_for_temporal_metadata(monkeypatch):
    frames = ["frame-1", "frame-2", "frame-3"]
    processor = _Processor(input_ids=[1, *ANSWER_IDS, 2], pixel_values=torch.ones(2, 3, 4, 4))
    encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=32,
        temporal_patch_size=2,
        video_fps=2.0,
        use_temporal_video_embedder=True,
        patch_dim=16,
    )

    monkeypatch.setattr(NemotronOmniTaskEncoder, "_decode_video_bytes", staticmethod(lambda *args, **kwargs: frames))
    monkeypatch.setattr(
        NemotronOmniTaskEncoder,
        "_patchify_frame",
        lambda self, frame, target_h=512, target_w=512: torch.ones(2, 3),
    )

    sample = _make_chatml_sample(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": "sample.mp4"},
                    {"type": "text", "text": "What happens?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
        videos=b"fake mp4 bytes",
    )

    encoded = encoder.encode_sample(sample)

    user_prompt = processor.tokenizer.last_conversation[0]["content"]
    assert "frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.50 seconds: <image>" in user_prompt
    assert "frame 3 sampled at 1.00 seconds: <image>" in user_prompt
    assert len(processor.calls[0]["images"]) == 2
    assert processor.image_processor.max_num_tiles == 4
    assert encoded.visual_tensors["pixel_values"].shape == (1, 6, 3)
    assert encoded.imgs_sizes.tolist() == [[512, 512], [512, 512], [512, 512]]
    assert encoded.num_frames.tolist() == [3]
    assert encoded.num_image_tiles.tolist() == [256, 256, 256]


def test_encode_sample_keeps_shifted_labels_aligned_after_image_token_adjustment():
    processor = _Processor(
        input_ids=[1, IMG_START_ID, 95, 95, 95, IMG_END_ID, *ANSWER_IDS, 2],
        pixel_values=torch.ones(1, 3, 4, 4),
        num_patches=[1],
    )
    encoder = NemotronOmniTaskEncoder(processor=processor, seq_length=32)
    sample = _make_chatml_sample(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Describe it."}],
            },
            {"role": "assistant", "content": "answer"},
        ],
        imgs=[torch.zeros(3, 4, 4)],
    )

    encoded = encoder.encode_sample(sample)

    assert encoded.input_ids.tolist() == [1, IMG_START_ID, 95, IMG_END_ID, *ANSWER_IDS, 2]
    assert encoded.labels.tolist() == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 21, 22, IGNORE_INDEX, IGNORE_INDEX]
    assert encoded.loss_mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_batch_packs_text_sequences_then_encode_batch():
    processor = _Processor(input_ids=[1, 2])
    encoder = NemotronOmniTaskEncoder(
        processor=processor,
        num_mel_bins=4,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )
    s1 = NemotronOmniTaskSample(
        __key__="a",
        __subflavors__={},
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, IGNORE_INDEX]),
        loss_mask=torch.tensor([1.0, 0.0]),
    )
    s2 = NemotronOmniTaskSample(
        __key__="b",
        __subflavors__={"source": "mock"},
        input_ids=torch.tensor([3, 4, 5]),
        labels=torch.tensor([4, 5, IGNORE_INDEX]),
        loss_mask=torch.tensor([1.0, 1.0, 0.0]),
    )

    batch = encoder.batch([s1, s2])

    assert isinstance(batch, NemotronOmniTaskBatch)
    assert batch.input_ids.tolist() == [[1, 2, 0, 0, 3, 4, 5, 0]]
    assert batch.attention_mask is None
    assert batch.labels.tolist() == [[2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch.loss_mask.tolist() == [[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
    assert batch.position_ids.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]
    assert batch.cu_seqlens_q.tolist() == [0, 2, 5]
    assert batch.cu_seqlens_kv.tolist() == [0, 2, 5]
    assert batch.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert batch.cu_seqlens_kv_padded.tolist() == [0, 4, 8]
    assert batch.max_seqlen_q.item() == 4
    assert batch.max_seqlen_kv.item() == 4
    encoded = encoder.encode_batch(batch)
    assert encoded["tokens"] is batch.input_ids
    assert encoded["sound_clips"] is None
    assert encoded["cu_seqlens_q"] is batch.cu_seqlens_q
    assert encoded["cu_seqlens_kv"] is batch.cu_seqlens_kv
    assert encoded["cu_seqlens_q_padded"] is batch.cu_seqlens_q_padded
    assert encoded["cu_seqlens_kv_padded"] is batch.cu_seqlens_kv_padded
    assert encoded["max_seqlen_q"] is batch.max_seqlen_q
    assert encoded["max_seqlen_kv"] is batch.max_seqlen_kv
    assert "cu_seqlens" not in encoded
    assert "cu_seqlens_unpadded" not in encoded
    assert "cu_seqlens_argmin" not in encoded
    assert isinstance(encoded["visual_inputs"], GenericVisualInputs)
    assert encoded["visual_inputs"].pixel_values is None


def test_batch_rejects_multimodal_in_batch_packing():
    encoder = NemotronOmniTaskEncoder(
        processor=_Processor(input_ids=[1, 2]),
        enable_in_batch_packing=True,
    )
    sample = NemotronOmniTaskSample(
        __key__="image",
        __subflavors__={},
        input_ids=torch.tensor([1, IMG_START_ID, 95, IMG_END_ID, 2]),
        labels=torch.tensor([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 2, IGNORE_INDEX]),
        loss_mask=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        visual_tensors={"pixel_values": torch.ones(1, 4, 3)},
        num_patches=torch.tensor([1]),
    )

    with pytest.raises(ValueError, match="does not support image, video, or audio samples"):
        encoder.batch([sample])


def test_batch_rejects_audio_in_batch_packing():
    encoder = NemotronOmniTaskEncoder(
        processor=_Processor(input_ids=[1, 2]),
        num_mel_bins=4,
        enable_in_batch_packing=True,
    )
    sample = NemotronOmniTaskSample(
        __key__="audio",
        __subflavors__={},
        input_ids=torch.tensor([1, SO_START_ID, SO_EMBEDDING_ID, SO_END_ID, 2]),
        labels=torch.tensor([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 2, IGNORE_INDEX]),
        loss_mask=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        sound_clips=torch.ones(8, 4),
        sound_length=torch.tensor(8),
    )

    with pytest.raises(ValueError, match="does not support image, video, or audio samples"):
        encoder.batch([sample])


def test_batch_rejects_mixed_audio_and_no_audio_samples():
    encoder = NemotronOmniTaskEncoder(processor=_Processor(input_ids=[1, 2]), num_mel_bins=4)
    audio_sample = NemotronOmniTaskSample(
        __key__="audio",
        __subflavors__={},
        input_ids=torch.tensor([1, SO_EMBEDDING_ID, 2]),
        labels=torch.tensor([IGNORE_INDEX, 2, IGNORE_INDEX]),
        loss_mask=torch.tensor([0.0, 1.0, 0.0]),
        sound_clips=torch.ones(8, 4),
        sound_length=torch.tensor(8),
    )
    text_sample = NemotronOmniTaskSample(
        __key__="text",
        __subflavors__={},
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([2, IGNORE_INDEX]),
        loss_mask=torch.tensor([1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="does not support mixing audio and no-audio samples"):
        encoder.batch([audio_sample, text_sample])


def test_batch_emits_padded_metadata_for_aligned_cp_multiple():
    encoder = NemotronOmniTaskEncoder(
        processor=_Processor(input_ids=[1, 2]),
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )
    samples = [
        NemotronOmniTaskSample(
            __key__=key,
            __subflavors__={},
            input_ids=torch.tensor(tokens),
            labels=torch.tensor([*tokens[1:], IGNORE_INDEX]),
            loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        )
        for key, tokens in (("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8]))
    ]

    batch = encoder.batch(samples)

    assert batch.cu_seqlens_q.tolist() == [0, 4, 8]
    assert batch.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert batch.cu_seqlens_kv_padded.tolist() == [0, 4, 8]


def test_batch_nonpacked_applies_collate_sequence_padding():
    processor = _Processor(input_ids=[1, 2])
    samples = [
        NemotronOmniTaskSample(
            __key__="a",
            __subflavors__={},
            input_ids=torch.tensor([1, 2, 3]),
            labels=torch.tensor([2, 3, IGNORE_INDEX]),
            loss_mask=torch.tensor([1.0, 1.0, 0.0]),
        ),
        NemotronOmniTaskSample(
            __key__="b",
            __subflavors__={},
            input_ids=torch.tensor([4, 5, 6, 7, 8]),
            labels=torch.tensor([5, 6, 7, 8, IGNORE_INDEX]),
            loss_mask=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
        ),
    ]

    multiple_encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=16,
        num_mel_bins=4,
        pad_to_max_length=False,
        pad_to_multiple_of=4,
    )
    multiple_batch = multiple_encoder.batch(samples)
    assert multiple_batch.input_ids.shape == (2, 8)
    assert multiple_batch.input_ids[0].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    assert multiple_batch.labels[0].tolist() == [
        2,
        3,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert multiple_batch.position_ids[0].tolist() == list(range(8))

    fixed_encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=6,
        num_mel_bins=4,
        pad_to_max_length=True,
        pad_to_multiple_of=4,
    )
    fixed_batch = fixed_encoder.batch(samples)
    assert fixed_batch.input_ids.shape == (2, 6)
    assert fixed_batch.position_ids[0].tolist() == list(range(6))
