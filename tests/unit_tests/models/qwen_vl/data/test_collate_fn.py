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

"""Qwen-VL collates with real HF processors, built offline from local files."""

import json
import re
from pathlib import Path

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoProcessor, PreTrainedTokenizerFast

from megatron.bridge.data.collators.registry import resolve_model_collate
from megatron.bridge.models.qwen_vl.data.collate_fn import prepare_qwen_vl_sequence


# qwen-vl-utils needs torchvision.
qwen_vl_utils = pytest.importorskip("qwen_vl_utils")
pytestmark = pytest.mark.unit

VISUAL_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
TIMESTAMP = re.compile(r"<([\d.]+) seconds>")
# ChatML, with the media placeholders of the Qwen-VL chat templates.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['content'] is string -%}{{- message['content'] -}}"
    "{%- else -%}{%- for part in message['content'] -%}"
    "{%- if part['type'] == 'image' -%}{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}"
    "{%- elif part['type'] == 'video' -%}{{- '<|vision_start|><|video_pad|><|vision_end|>' -}}"
    "{%- else -%}{{- part['text'] -}}{%- endif -%}"
    "{%- endfor -%}{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
)


def _load_processor(path: Path, *, qwen3_vl: bool):
    """Write the files of a Qwen3-VL or Qwen2.5-VL processor and load them with AutoProcessor.

    The image and video settings are those of Qwen/Qwen3-VL-8B-Instruct and Qwen/Qwen2.5-VL-7B-Instruct.
    The tokenizer is a byte-level BPE without merges, so decode gives back the exact text.
    """
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={char: index for index, char in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
    tokenizer.add_tokens([*special_tokens, "<|image_pad|>", "<|video_pad|>"], special_tokens=True)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.save_pretrained(path)

    common = {
        "processor_class": "Qwen3VLProcessor" if qwen3_vl else "Qwen2_5_VLProcessor",
        "patch_size": 16 if qwen3_vl else 14,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    if qwen3_vl:
        image_size = {"shortest_edge": 65536, "longest_edge": 16777216}
        video = {
            "video_processor_type": "Qwen3VLVideoProcessor",
            "size": {"shortest_edge": 4096, "longest_edge": 25165824},
        }
    else:
        image_size = {"shortest_edge": 3136, "longest_edge": 12845056}
        video = {"video_processor_type": "Qwen2VLVideoProcessor", "size": image_size}
    image = {"image_processor_type": "Qwen2VLImageProcessorFast", "size": image_size}
    (path / "preprocessor_config.json").write_text(json.dumps({**common, **image}))
    (path / "video_preprocessor_config.json").write_text(json.dumps({**common, **video}))
    (path / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    return AutoProcessor.from_pretrained(path)


def _example(*media: dict, answer: str = "It is red.") -> dict:
    return {
        "conversation": [
            {"role": "user", "content": [*media, {"type": "text", "text": "What do you see?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def _clip(frames: int, sample_fps: float) -> dict:
    """A video part with frames that were already sampled at ``sample_fps``."""
    return {
        "type": "video",
        "video": [Image.new("RGB", (320, 180), (index * 20, 0, 0)) for index in range(frames)],
        "sample_fps": sample_fps,
    }


def _media_examples() -> list[dict]:
    return [
        _example(
            {"type": "image", "image": Image.new("RGB", (1000, 700), "red")},
            # Smaller than the minimum of the processor: a second resize changes its size.
            {"type": "image", "image": Image.new("RGB", (200, 140), "red")},
        ),
        _example(_clip(12, sample_fps=2.0), answer="It gets brighter."),
        _example(_clip(8, sample_fps=1.0), answer="It gets brighter."),
        {"conversation": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
    ]


def _cat(values: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [value for value in values if value is not None]
    return torch.cat(present) if present else None


def _model_card_rows(examples: list[dict], processor) -> tuple[list[torch.Tensor], dict]:
    """Preprocess each example as the Qwen3-VL model card does for inference."""
    input_ids, visual = [], {key: [] for key in VISUAL_KEYS}
    for example in examples:
        images, videos, video_kwargs = qwen_vl_utils.process_vision_info(
            example["conversation"],
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        kwargs = {
            "text": [processor.apply_chat_template(example["conversation"], tokenize=False)],
            "do_resize": False,
            "return_tensors": "pt",
            **video_kwargs,
        }
        if images:
            kwargs["images"] = images
        if videos:
            frames, metadata = zip(*videos)
            kwargs["videos"], kwargs["video_metadata"] = list(frames), list(metadata)
        batch = processor(**kwargs)
        input_ids.append(batch["input_ids"][0])
        for key in VISUAL_KEYS:
            visual[key].append(batch.get(key))
    return input_ids, {key: _cat(values) for key, values in visual.items()}


def _bridge_rows(path: str, examples: list[dict], processor) -> tuple[list[torch.Tensor], dict]:
    """Preprocess the examples with one of the Bridge Qwen-VL paths, without pixel bounds."""
    if path == "prepare_qwen_vl_sequence":
        prepared = [
            prepare_qwen_vl_sequence(example, processor, min_pixels=None, max_pixels=None) for example in examples
        ]
        visual = {key: _cat([item.visual_values.get(key) for item in prepared]) for key in VISUAL_KEYS}
        return [item.row["input_ids"] for item in prepared], visual

    collate = resolve_model_collate(type(processor).__name__)
    packed = path == "in_batch_packing"
    batch = collate(examples, processor, min_pixels=None, max_pixels=None, enable_in_batch_packing=packed)
    if packed:
        tokens, bounds = batch["input_ids"].reshape(-1), batch["cu_seqlens_q"].tolist()
        input_ids = [tokens[start:end] for start, end in zip(bounds[:-1], bounds[1:])]
    else:
        input_ids = [row[mask.bool()] for row, mask in zip(batch["input_ids"], batch["attention_mask"])]
    return input_ids, {key: getattr(batch["visual_inputs"], key) for key in VISUAL_KEYS}


def _timestamps(processor, input_ids: torch.Tensor) -> list[float]:
    return [float(value) for value in TIMESTAMP.findall(processor.tokenizer.decode(input_ids))]


@pytest.mark.parametrize("path", ["padded", "in_batch_packing", "prepare_qwen_vl_sequence"])
def test_qwen3_vl_collate_matches_model_card_preprocessing(tmp_path, path):
    processor = _load_processor(tmp_path, qwen3_vl=True)
    examples = _media_examples()

    input_ids, visual = _bridge_rows(path, examples, processor)
    expected_input_ids, expected_visual = _model_card_rows(examples, processor)

    assert len(input_ids) == len(expected_input_ids)
    for row, expected_row in zip(input_ids, expected_input_ids):
        assert torch.equal(row, expected_row)
    for key in VISUAL_KEYS:
        assert torch.equal(visual[key], expected_visual[key]), key
    # One resize at factor 32: 1000x700 becomes 992x704, not 1024x704, and 200x140 stays at 192x128.
    assert visual["image_grid_thw"].tolist() == [[1, 44, 62], [1, 8, 12]]
    # Every sampled frame reaches the model (12 and 8 frames, 2 frames per temporal patch),
    # and the timestamps come from the sampled fps of each clip.
    assert visual["video_grid_thw"][:, 0].tolist() == [6, 4]
    assert _timestamps(processor, input_ids[1]) == [0.2, 1.2, 2.2, 3.2, 4.2, 5.2]
    assert _timestamps(processor, input_ids[2]) == [0.5, 2.5, 4.5, 6.5]


def test_qwen3_vl_collate_bounds_image_pixels_in_one_resize(tmp_path):
    processor = _load_processor(tmp_path, qwen3_vl=True)
    examples = [
        # Scaled up to at least QWEN_VL_MIN_PIXELS: 544x384 pixels.
        _example({"type": "image", "image": Image.new("RGB", (200, 140), "red")}),
        # Scaled down to at most QWEN_VL_MAX_PIXELS: 1184x832 pixels.
        _example({"type": "image", "image": Image.new("RGB", (2000, 1400), "red")}),
        # An image part with its own bound keeps it, as at inference: 352x256 pixels.
        _example({"type": "image", "image": Image.new("RGB", (1000, 700), "red"), "max_pixels": 100352}),
        # A bound with the value None is not set, as in qwen-vl-utils: 1184x832 pixels.
        _example({"type": "image", "image": Image.new("RGB", (2000, 1400), "red"), "max_pixels": None}),
    ]

    collate = resolve_model_collate(type(processor).__name__)
    batch = collate(examples, processor)
    # A max_pixels below the default min_pixels also lowers min_pixels: 384x288 and 352x256 pixels.
    small_budget_batch = collate(examples, processor, max_pixels=100352)

    assert batch["visual_inputs"].image_grid_thw.tolist() == [[1, 24, 34], [1, 52, 74], [1, 16, 22], [1, 52, 74]]
    assert small_budget_batch["visual_inputs"].image_grid_thw.tolist() == [
        [1, 18, 24],
        [1, 16, 22],
        [1, 16, 22],
        [1, 16, 22],
    ]


def test_energon_visual_token_budget_counts_the_collate_tokens(tmp_path):
    energon = pytest.importorskip("megatron.bridge.models.qwen_vl.data.energon")
    from megatron.energon import SkipSample

    from megatron.bridge.data.energon.metadata import sample_metadata_kwargs

    processor = _load_processor(tmp_path, qwen3_vl=True)
    sample = energon.ChatMLSample(
        **sample_metadata_kwargs(key="clip", restore_key="clip", subflavors={}),
        imgs=None,
        videos=[_clip(8, sample_fps=2.0)["video"]],
        conversation=json.dumps(
            [
                {"role": "user", "content": "<video>What happens?"},
                {"role": "assistant", "content": "It gets brighter."},
            ]
        ),
    )

    def encode(max_visual_tokens):
        encoder = energon.QwenVLTaskEncoder(
            tokenizer=processor.tokenizer, image_processor=processor, max_visual_tokens=max_visual_tokens
        )
        return encoder.encode_sample(sample)

    batch = resolve_model_collate(type(processor).__name__)([encode(None).example], processor)
    collate_tokens = int(batch["visual_inputs"].video_grid_thw.prod(dim=-1).sum()) // 4

    # The HF video processor alone counts fewer tokens for these frames than the collate makes.
    assert encode(collate_tokens).example is not None
    with pytest.raises(SkipSample):
        encode(collate_tokens - 1)


def test_qwen3_vl_collate_rejects_only_a_truncation_inside_the_visual_tokens(tmp_path):
    processor = _load_processor(tmp_path, qwen3_vl=True)
    collate = resolve_model_collate(type(processor).__name__)
    # The first answer comes before the clip, so the truncated row keeps supervised tokens.
    example = {
        "conversation": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": [_clip(8, sample_fps=2.0), {"type": "text", "text": "What happens?"}]},
            {"role": "assistant", "content": "It gets brighter."},
        ]
    }
    batch = collate([example], processor)
    video_tokens = batch["input_ids"][0] == processor.video_token_id
    first_video_token = int(video_tokens.nonzero()[0])
    length = int(batch["attention_mask"][0].sum())

    with pytest.raises(ValueError, match="removes visual tokens from rows \\[0\\]"):
        collate([example], processor, sequence_length=first_video_token + 1)
    # A truncation that cuts only the text after the clip keeps every visual token.
    truncated = collate([example], processor, sequence_length=length - 3)["input_ids"][0]
    assert int((truncated == processor.video_token_id).sum()) == int(video_tokens.sum())


def test_qwen2_5_collate_passes_sampled_fps_to_video_positions(tmp_path):
    processor = _load_processor(tmp_path, qwen3_vl=False)

    batch = resolve_model_collate(type(processor).__name__)(_media_examples(), processor)

    visual = batch["visual_inputs"]
    # second_per_grid_ts is temporal_patch_size / sampled fps: 2 / 2.0 and 2 / 1.0.
    assert visual.second_per_grid_ts.tolist() == [1.0, 2.0]
    assert visual.video_grid_thw[:, 0].tolist() == [6, 4]
    # Qwen2.5-VL images keep the patch-14 resize and the pixel bounds of the processor call.
    assert visual.image_grid_thw.tolist() == [[1, 50, 72], [1, 28, 38]]
