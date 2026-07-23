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

import megatron.bridge.models.nemotron_omni.data.collate_fn as omni_collate
from megatron.bridge.data.energon.hf_task_encoder import HFEnergonBatch, HFEnergonSample
from megatron.bridge.data.energon.metadata import batch_metadata_kwargs, sample_metadata_kwargs
from megatron.bridge.data.energon.nemotron_omni_task_encoder import (
    NemotronOmniTaskBatch,
    NemotronOmniTaskEncoder,
    NemotronOmniTaskSample,
)
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


pytestmark = pytest.mark.unit


PAD_AND_END_ID = 11
SO_EMBEDDING_ID = 90
SO_START_ID = 91
SO_END_ID = 92
IMG_START_ID = 93
IMG_END_ID = 94
IMAGE_TOKEN_ID = 95


class _Tokenizer:
    pad_token = "<|im_end|>"
    eos_token = "<|im_end|>"
    pad_token_id = PAD_AND_END_ID
    eos_token_id = PAD_AND_END_ID
    added_tokens_decoder = {}
    audio_token = "<so_embedding>"

    def __init__(self, rows: list[list[int]]):
        self.rows = rows
        self.conversations = []
        self.row_offset = 0

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        self.conversations.append(conversation)
        return "rendered prompt"

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, str):
            marker_tokens = {
                "<|im_start|>assistant\n": [101],
                "<|im_end|>": [PAD_AND_END_ID],
                "<|im_end|>\n": [PAD_AND_END_ID, 102],
                "<|im_start|>system\n": [103],
                "<|im_start|>developer\n": [104],
                "<|im_start|>user\n": [105],
                "<|im_start|>tool\n": [106],
            }
            return {"input_ids": marker_tokens.get(prompts, [1])}
        rows = self.rows[self.row_offset : self.row_offset + len(prompts)]
        self.row_offset += len(prompts)
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = torch.tensor(row)
            attention_mask[row_index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_tokens_to_ids(self, token):
        return {
            "<so_embedding>": SO_EMBEDDING_ID,
            "<so_start>": SO_START_ID,
            "<so_end>": SO_END_ID,
            "<img>": IMG_START_ID,
            "</img>": IMG_END_ID,
            "<image>": IMAGE_TOKEN_ID,
            "<video>": 96,
        }[token]


class _Processor:
    def __init__(self, rows: list[list[int]]):
        self.tokenizer = _Tokenizer(rows)
        self.image_processor = type("ImageProcessor", (), {"max_num_patches": 64})()
        self.processor_calls = 0

    def __call__(self, **kwargs):
        row = self.tokenizer.rows[self.processor_calls]
        self.processor_calls += 1
        output = {"input_ids": torch.tensor([row], dtype=torch.long)}
        if kwargs.get("images") is not None:
            output["pixel_values"] = torch.ones(len(kwargs["images"]), 3, 32, 32)
        return output


def _sample(conversation, *, key="sample", imgs=None, videos=None, audio=None):
    return ChatMLSample(
        **sample_metadata_kwargs(key=key, restore_key=(), subflavors={}),
        conversation=json.dumps(conversation),
        imgs=imgs,
        videos=videos,
        audio=audio,
    )


def _mask_all_tokens(example, input_ids, processor, skipped_tokens, **kwargs):  # noqa: ARG001
    return torch.ones_like(input_ids, dtype=torch.float32)


def test_encoder_uses_generic_hf_style_sample_and_batch_contract():
    assert NemotronOmniTaskSample is HFEnergonSample
    assert issubclass(NemotronOmniTaskBatch, HFEnergonBatch)

    encoder = NemotronOmniTaskEncoder(processor=_Processor([[1, 2]]), pad_to_multiple_of=1)
    encoded = encoder.encode_sample(
        _sample(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        )
    )

    assert isinstance(encoded, HFEnergonSample)
    assert encoded.example["conversation"][0]["content"] == "question"


def test_encoder_rejects_unsupported_visual_keys():
    with pytest.raises(ValueError, match=r"visual_keys must be exactly \('pixel_values',\)"):
        NemotronOmniTaskEncoder(
            processor=_Processor([[1, 2]]),
            visual_keys=("pixel_values", "image_sizes"),
        )


def test_legacy_visual_tensors_batch_contract_is_normalized_and_exposed():
    pixel_values = torch.ones(1, 3, 4, 4)
    metadata = batch_metadata_kwargs(keys=["sample"])
    batch = NemotronOmniTaskBatch(**metadata, visual_tensors={"pixel_values": pixel_values})

    assert batch.visual_inputs is not None
    assert batch.visual_inputs.pixel_values is pixel_values
    assert batch.visual_tensors["pixel_values"] is pixel_values

    replacement = torch.zeros_like(pixel_values)
    batch.visual_tensors = {"pixel_values": replacement}
    assert batch.visual_inputs is not None
    assert batch.visual_inputs.pixel_values is replacement

    item_replacement = torch.full_like(pixel_values, 2)
    batch.visual_tensors["pixel_values"] = item_replacement
    assert batch.visual_inputs.pixel_values is item_replacement
    del batch.visual_tensors["pixel_values"]
    assert batch.visual_inputs.pixel_values is None
    assert "pixel_values" not in batch.visual_tensors

    empty_batch = NemotronOmniTaskBatch(**metadata)
    empty_batch.visual_tensors["pixel_values"] = pixel_values
    assert empty_batch.visual_inputs is not None
    assert empty_batch.visual_inputs.pixel_values is pixel_values

    with pytest.raises(ValueError, match="only one of visual_inputs or legacy visual_tensors"):
        NemotronOmniTaskBatch(
            **metadata,
            visual_inputs=GenericVisualInputs(pixel_values=pixel_values),
            visual_tensors={"pixel_values": pixel_values},
        )
    with pytest.raises(ValueError, match="only one of visual_inputs or legacy visual_tensors"):
        NemotronOmniTaskBatch(**metadata, visual_inputs=GenericVisualInputs(), visual_tensors={})


def test_energon_batch_preserves_real_tokens_when_pad_id_equals_turn_end(monkeypatch):
    monkeypatch.setattr(omni_collate, "build_assistant_loss_mask", _mask_all_tokens)
    processor = _Processor(
        [
            [100, IMG_START_ID, IMAGE_TOKEN_ID, IMG_END_ID, PAD_AND_END_ID, 21, PAD_AND_END_ID],
            [200, IMG_START_ID, IMAGE_TOKEN_ID, IMG_END_ID, PAD_AND_END_ID, 31, 32, PAD_AND_END_ID],
        ]
    )
    encoder = NemotronOmniTaskEncoder(processor=processor, seq_length=16, pad_to_multiple_of=1)
    samples = [
        encoder.encode_sample(
            _sample(
                [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": key}]},
                    {"role": "assistant", "content": "answer"},
                ],
                key=key,
                imgs=[torch.ones(3, 32, 32)],
            )
        )
        for key in ("row-a", "row-b")
    ]

    batch = encoder.batch(samples)
    encoded = encoder.encode_batch(batch)

    assert batch.input_ids.tolist() == [
        [100, IMG_START_ID, IMAGE_TOKEN_ID, IMG_END_ID, PAD_AND_END_ID, 21, PAD_AND_END_ID, PAD_AND_END_ID],
        [200, IMG_START_ID, IMAGE_TOKEN_ID, IMG_END_ID, PAD_AND_END_ID, 31, 32, PAD_AND_END_ID],
    ]
    assert not bool((batch.input_ids == 0).any())
    assert encoded["input_ids"] is batch.input_ids


def test_energon_audio_uses_shared_collator_token_expansion(monkeypatch):
    monkeypatch.setattr(omni_collate, "build_assistant_loss_mask", _mask_all_tokens)
    monkeypatch.setattr(
        "megatron.bridge.models.nemotron_omni.nemotron_omni_utils.compute_mel_features",
        lambda waveform, sampling_rate=16000, num_mel_bins=4: torch.ones(9, num_mel_bins),
    )
    processor = _Processor([[1, IMG_END_ID, 21, PAD_AND_END_ID]])
    encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=32,
        num_mel_bins=4,
        pad_to_multiple_of=1,
    )
    encoded = encoder.encode_sample(
        _sample(
            [{"role": "user", "content": "audio"}, {"role": "assistant", "content": "answer"}],
            audio=torch.tensor([0.0, 0.1, -0.1]),
        )
    )

    batch = encoder.batch([encoded])

    assert batch.input_ids.tolist() == [
        [1, IMG_END_ID, SO_START_ID, SO_EMBEDDING_ID, SO_EMBEDDING_ID, SO_END_ID, 21, PAD_AND_END_ID]
    ]
    assert batch.sound_clips.shape == (1, 9, 4)
    assert batch.sound_length.tolist() == [9]


def test_energon_audio_placeholder_uses_framed_shared_expansion(monkeypatch):
    monkeypatch.setattr(omni_collate, "build_assistant_loss_mask", _mask_all_tokens)
    monkeypatch.setattr(
        "megatron.bridge.models.nemotron_omni.nemotron_omni_utils.compute_mel_features",
        lambda waveform, sampling_rate=16000, num_mel_bins=4: torch.ones(9, num_mel_bins),
    )
    processor = _Processor([[1, SO_EMBEDDING_ID, 21, PAD_AND_END_ID]])
    encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=32,
        num_mel_bins=4,
        pad_to_multiple_of=1,
    )
    encoded = encoder.encode_sample(
        _sample(
            [{"role": "user", "content": "<|audio_1|>"}, {"role": "assistant", "content": "answer"}],
            audio=torch.tensor([0.0, 0.1, -0.1]),
        )
    )

    batch = encoder.batch([encoded])

    assert batch.input_ids.tolist() == [
        [1, SO_START_ID, SO_EMBEDDING_ID, SO_EMBEDDING_ID, SO_END_ID, 21, PAD_AND_END_ID]
    ]


def test_energon_temporal_video_is_processed_in_shared_collator(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(omni_collate, "build_assistant_loss_mask", _mask_all_tokens)
    monkeypatch.setattr(
        omni_collate,
        "_patchify_frame",
        lambda frame, *, height, width, patch_dim: torch.ones(2, 3),
    )
    processor = _Processor([[1, IMG_START_ID, 97, IMG_END_ID, IMG_START_ID, 97, IMG_END_ID, 21, PAD_AND_END_ID]])
    encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=32,
        temporal_patch_size=2,
        video_fps=2.0,
        use_temporal_video_embedder=True,
        patch_dim=16,
        pad_to_multiple_of=1,
    )
    frames = [Image.new("RGB", (16, 16), color=value) for value in (0, 64, 128)]
    encoded = encoder.encode_sample(
        _sample(
            [
                {
                    "role": "user",
                    "content": [{"type": "video"}, {"type": "text", "text": "what happens?"}],
                },
                {"role": "assistant", "content": "answer"},
            ],
            videos=[frames],
        )
    )

    batch = encoder.batch([encoded])

    assert batch.visual_inputs is not None
    assert batch.visual_inputs.pixel_values.shape == (1, 6, 3)
    assert batch.imgs_sizes.tolist() == [[512, 512], [512, 512], [512, 512]]
    assert batch.num_frames.tolist() == [3]
    assert processor.processor_calls == 0
    assert processor.image_processor.max_num_patches == 64
    rendered_video = processor.tokenizer.conversations[0][0]["content"]
    assert "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.50 seconds" in rendered_video
    assert "Frame 3 sampled at 1.00 seconds" in rendered_video
    assert rendered_video.count("<img><image></img>") == 2
    assert batch.input_ids.tolist() == [
        [1, IMG_START_ID, 97, IMG_END_ID, IMG_START_ID, 97, IMG_END_ID, 21, PAD_AND_END_ID]
    ]


def test_energon_single_frame_video_uses_temporal_embedder_contract(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(omni_collate, "build_assistant_loss_mask", _mask_all_tokens)
    monkeypatch.setattr(
        omni_collate,
        "_patchify_frame",
        lambda frame, *, height, width, patch_dim: torch.ones(2, 3),
    )
    processor = _Processor([[1, IMG_START_ID, 97, IMG_END_ID, 21, PAD_AND_END_ID]])
    encoder = NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=32,
        temporal_patch_size=2,
        video_fps=2.0,
        use_temporal_video_embedder=True,
        patch_dim=16,
        pad_to_multiple_of=1,
    )
    encoded = encoder.encode_sample(
        _sample(
            [{"role": "user", "content": [{"type": "video"}]}],
            videos=[[Image.new("RGB", (16, 16), color=64)]],
        )
    )

    batch = encoder.batch([encoded])

    assert batch.visual_inputs.pixel_values.shape == (1, 4, 3)
    assert batch.imgs_sizes.tolist() == [[512, 512], [512, 512]]
    assert batch.num_frames.tolist() == [2]
    assert "Frame 1 sampled at 0.00 seconds" in processor.tokenizer.conversations[0][0]["content"]


def test_energon_raw_video_bytes_remain_one_owned_video_per_sample():
    raw_video = b"raw-mp4-payload"
    encoder = NemotronOmniTaskEncoder(processor=_Processor([[1]]), pad_to_multiple_of=1)

    encoded = encoder.encode_sample(
        _sample(
            [{"role": "user", "content": [{"type": "video"}]}],
            videos=raw_video,
        )
    )

    video_part = encoded.example["conversation"][0]["content"][0]
    assert video_part == {"type": "video", "video": raw_video}
    assert encoded.example["videos"] == [raw_video]


def test_energon_multiple_raw_video_bytes_keep_placeholder_order():
    raw_videos = [b"first-mp4", b"second-mp4"]
    encoder = NemotronOmniTaskEncoder(processor=_Processor([[1]]), pad_to_multiple_of=1)

    encoded = encoder.encode_sample(
        _sample(
            [{"role": "user", "content": [{"type": "video"}, {"type": "video"}]}],
            videos=raw_videos,
        )
    )

    content = encoded.example["conversation"][0]["content"]
    assert [part["video"] for part in content] == raw_videos
    assert encoded.example["videos"] == raw_videos


def test_raw_video_bytes_are_decoded_through_one_temporary_mp4(monkeypatch):
    raw_video = b"raw-mp4-payload"
    decoded_frames = [object(), object()]

    def _fake_decode(path, *, video_fps, video_nframes):
        with open(path, "rb") as video_file:
            assert video_file.read() == raw_video
        assert path.endswith(".mp4")
        assert video_fps == 3.0
        assert video_nframes == 5
        return decoded_frames, 2.5

    monkeypatch.setattr(omni_collate, "_decode_video_path", _fake_decode)

    frames, sampled_fps = omni_collate._video_frames(raw_video, video_fps=3.0, video_nframes=5)

    assert frames is decoded_frames
    assert sampled_fps == 2.5


def test_energon_canonical_collator_rejects_collator_side_packing():
    encoder = NemotronOmniTaskEncoder(
        processor=_Processor([[1, 2]]),
        enable_in_batch_packing=True,
        pad_to_multiple_of=1,
    )

    sample = encoder.encode_sample(_sample([{"role": "user", "content": "text"}]))
    with pytest.raises(ValueError, match="Canonical Nemotron Omni owns sequence packing"):
        encoder.batch([sample])
