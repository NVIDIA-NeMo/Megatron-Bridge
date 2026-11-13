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

"""Inkling2 image-text SFT data: image preprocessing, the processor, and the collator."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoProcessor, PreTrainedTokenizerFast

from megatron.bridge.data.builders.direct_hf_sft import DirectHFSFTDatasetConfig, load_direct_hf_sft_processor
from megatron.bridge.data.collators.registry import always_use_model_collate, resolve_model_collate
from megatron.bridge.data.conversation_processing import tokenize_chat_example
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.models.inkling import Inkling2Processor
from megatron.bridge.models.inkling.data.collate_fn import Inkling2VisualInputs, inkling2_collate_fn
from megatron.bridge.models.inkling.data.image_processing import (
    IMAGE_PROCESSOR_CONFIG,
    Inkling2ImageProcessor,
    resize_lanczos3,
)
from megatron.bridge.models.inkling.data.processor import reasoning_effort_level


pytestmark = pytest.mark.unit

_GENERATION_UTILS = Path(__file__).parents[4] / "scripts" / "inference" / "vlm_generation_utils.py"

FIXTURES = Path(__file__).parent / "fixtures"
# The release's processor_config.json and chat template (thinkingmachines/Inkling-2-Small, revision 78cb6c0e).
RELEASE_PROCESSOR_CONFIG = json.loads((FIXTURES / "inkling2_small_78cb6c0e_processor_config.json").read_text())
RELEASE_TEMPLATE = (FIXTURES / "inkling2_small_78cb6c0e_chat_template.jinja").read_text()
IMAGE_TOKEN = "<|unused_200054|>"
FRAMING = (
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_image|>",
    "<|content_model_end_sampling|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_xml|>",
    "<|end_message|>",
    IMAGE_TOKEN,
)
WORDS = ("Describe", "this", "image", ".", "receipt", "total", "4", "What", "is", "2", "+", "?", "Thinking", "effort")
WORDS += ("level", ":", "0", "9", "0.9", "tool_declare", "add", "finish")
TOKENS = ("<unk>", "<pad>", *FRAMING, *WORDS)
IMAGE_TOKEN_ID = TOKENS.index(IMAGE_TOKEN)
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
# One OpenAI-style tool and one bare function whose description, parameters and type take the template's defaults.
TOOLS = [
    {
        "type": "function",
        "function": {"name": "add", "description": "Addiert für 2 + 2 .", "parameters": {"type": "object"}},
    },
    {"name": "finish"},
]


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A minimal Inkling2 checkpoint directory: config, the release's processor config, a framing-aware tokenizer."""
    tokenizer = Tokenizer(WordLevel({token: index for index, token in enumerate(TOKENS)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>")
    fast.add_special_tokens({"additional_special_tokens": list(FRAMING)})
    fast.save_pretrained(tmp_path)
    config = {
        "architectures": ["Inkling2ForConditionalGeneration"],
        "model_type": "inkling2",
        "image_token_id": IMAGE_TOKEN_ID,
        "text_config": {"vocab_size": 128, "unpadded_vocab_size": 64},
        "vision_config": {"hidden_size": 32, "num_hidden_layers": 2, "native_patch_size": 20},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "processor_config.json").write_text(json.dumps(RELEASE_PROCESSOR_CONFIG))
    return tmp_path


def _image(width: int, height: int) -> Image.Image:
    """An RGB image whose pixel at (x, y) is (x, y, x + y) modulo 256."""
    y, x = np.mgrid[:height, :width]
    return Image.fromarray(np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8))


def _normalized(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((rgb.astype(np.float32) / np.float32(255) - MEAN) / STD).to(torch.bfloat16)


def _conversation(image: Image.Image | None) -> list[dict]:
    user = [{"type": "text", "text": "Describe this image ."}]
    if image is not None:
        user.insert(0, {"type": "image", "image": image})
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": [{"type": "text", "text": "receipt total 4"}]},
    ]


def _release_template_ids(processor: Inkling2Processor, conversation: list[dict], image_tokens: list[int], **kwargs):
    """The release template's token IDs for a conversation, with each image placeholder repeated per patch."""
    text = processor.tokenizer.apply_chat_template(
        conversation, chat_template=RELEASE_TEMPLATE, tokenize=False, **kwargs
    )
    sizes = iter(image_tokens)
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    return [t for token in ids for t in ([token] * next(sizes) if token == IMAGE_TOKEN_ID else [token])]


def _assistant_turns(tokens: list[str]) -> list[int]:
    """1 for every token of an assistant turn, from its first ``<|message_model|>`` through end of sampling."""
    mask, inside = [], False
    for token in tokens:
        inside = inside or token == "<|message_model|>"
        mask.append(int(inside))
        inside = inside and token != "<|content_model_end_sampling|>"
    return mask


def _generation_utils():
    """Import ``scripts/inference/vlm_generation_utils.py``, which is not part of a package."""
    spec = importlib.util.spec_from_file_location("vlm_generation_utils_under_test", _GENERATION_UTILS)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


class TestImageProcessing:
    def test_patches_cover_the_image_in_row_major_order(self) -> None:
        encoded = Inkling2ImageProcessor().encode(_image(81, 41))

        assert encoded.num_tokens == 6
        assert encoded.grid_thw.tolist() == [[1, 2, 3]]
        assert encoded.patches.shape == (6, 2, 40, 40, 3) and encoded.patches.dtype == torch.bfloat16
        assert torch.equal(encoded.patches[:, 0], encoded.patches[:, 1])
        pixels = np.asarray(_image(81, 41))
        # Patch 1 is the first row's second column: pixels (y 0-39, x 40-79).
        assert torch.equal(encoded.patches[1, 0], _normalized(pixels[:40, 40:80]))
        # Patch 5 holds x 80 of the last row (y 40) and padding everywhere else.
        assert torch.equal(encoded.patches[5, 0, 0, 0], _normalized(pixels[40, 80]))
        padding = _normalized(np.full(3, -1))
        assert torch.equal(encoded.patches[5, 0, 1, 1], padding)
        assert torch.equal(encoded.patches[2, 0, 0, 39], padding)

    def test_transparent_pixels_show_a_checkerboard(self) -> None:
        encoded = Inkling2ImageProcessor().encode(Image.new("RGBA", (16, 8), (255, 0, 0, 0)))

        assert torch.equal(encoded.patches[0, 0, 0, 0], _normalized(np.full(3, 204)))
        assert torch.equal(encoded.patches[0, 0, 0, 8], _normalized(np.full(3, 153)))

    def test_token_count_matches_the_patch_grid(self) -> None:
        assert Inkling2ImageProcessor.count_tokens(41, 39) == 2
        assert Inkling2ImageProcessor.count_tokens(40, 40) == 1
        assert Inkling2ImageProcessor.count_tokens(0, 40) == 0

    def test_the_release_processor_config_is_the_implemented_one(self) -> None:
        assert RELEASE_PROCESSOR_CONFIG["image_processor"] == IMAGE_PROCESSOR_CONFIG
        assert RELEASE_PROCESSOR_CONFIG["image_token"] == IMAGE_TOKEN

    @pytest.mark.parametrize(
        "change", [{"size": {"height": 14, "width": 14}}, {"image_mean": [0.48145466, 0.4578275, 0.40821073]}]
    )
    def test_other_image_processor_configurations_are_rejected(self, checkpoint: Path, change: dict) -> None:
        image_processor = {**RELEASE_PROCESSOR_CONFIG["image_processor"], **change}
        (checkpoint / "processor_config.json").write_text(
            json.dumps({**RELEASE_PROCESSOR_CONFIG, "image_processor": image_processor})
        )
        with pytest.raises(ValueError, match="Unsupported Inkling2 image processor configuration"):
            Inkling2ImageProcessor.from_pretrained(checkpoint)

    def test_pixels_are_scaled_to_minus_one_to_one(self) -> None:
        encoded = Inkling2ImageProcessor().encode(Image.new("RGB", (40, 40), (0, 255, 128)))
        assert encoded.patches[0, 0, 0, 0].tolist() == [-1.0, 1.0, torch.tensor(128 / 127.5 - 1).bfloat16().item()]


def _smooth(width: int, height: int) -> np.ndarray:
    """A smooth RGB pattern: resampling filters of the same kind agree on it to within rounding."""
    y, x = np.mgrid[:height, :width]
    channels = [127 + 127 * np.sin(x / 7 + channel) * np.cos(y / 11) for channel in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


class TestImageBudget:
    @pytest.mark.parametrize(
        ("size", "target"),
        [((300, 200), (97, 131)), ((40, 30), (123, 77)), ((1200, 900), (311, 233)), ((23, 17), (11, 9))],
    )
    def test_resize_is_within_one_level_of_pillow_lanczos_on_smooth_images(
        self, size: tuple[int, int], target: tuple[int, int]
    ) -> None:
        pixels = _smooth(*size)
        expected = np.asarray(Image.fromarray(pixels).resize(target, Image.Resampling.LANCZOS)).astype(np.int16)
        assert np.abs(resize_lanczos3(pixels, *target).astype(np.int16) - expected).max() <= 1

    def test_resize_keeps_constant_images_and_same_size_images(self) -> None:
        constant = np.full((17, 23, 3), 173, dtype=np.uint8)
        assert np.array_equal(resize_lanczos3(constant, 11, 9), np.full((9, 11, 3), 173, dtype=np.uint8))
        pixels = np.asarray(_image(23, 17))
        assert np.array_equal(resize_lanczos3(pixels, 23, 17), pixels)

    def test_large_images_shrink_into_the_budget_and_small_ones_do_not(self) -> None:
        processor = Inkling2ImageProcessor(max_image_tokens=12)
        assert processor.fitted_size(120, 80) == (120, 80)
        width, height = processor.fitted_size(400, 300)
        assert Inkling2ImageProcessor.count_tokens(width, height) <= 12
        assert abs(width / height - 400 / 300) < 0.02
        encoded = processor.encode(_image(400, 300))
        assert encoded.num_tokens <= 12
        assert encoded.grid_thw.tolist() == [[1, -(-height // 40), -(-width // 40)]]
        assert Inkling2ImageProcessor().encode(_image(400, 300)).num_tokens == 80

    def test_autoprocessor_passes_the_budget_through(self, checkpoint: Path) -> None:
        processor = AutoProcessor.from_pretrained(checkpoint, max_image_tokens=12)
        assert processor.image_processor.max_image_tokens == 12
        assert AutoProcessor.from_pretrained(checkpoint).image_processor.max_image_tokens is None


class TestProcessor:
    def test_autoprocessor_and_the_dataset_loader_resolve_it(self, checkpoint: Path) -> None:
        processor = AutoProcessor.from_pretrained(checkpoint)
        assert isinstance(processor, Inkling2Processor)
        assert processor.image_token_id == IMAGE_TOKEN_ID

        config = DirectHFSFTDatasetConfig(
            seq_length=256, source=HFDatasetSourceConfig(dataset_name="cord_v2"), hf_processor_path=str(checkpoint)
        )
        assert isinstance(load_direct_hf_sft_processor(config, None), Inkling2Processor)
        assert always_use_model_collate("Inkling2Processor")
        assert resolve_model_collate("Inkling2Processor") is inkling2_collate_fn

    def test_image_turn_renders_the_release_framing(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        ids = processor.tokenizer.convert_tokens_to_ids
        input_ids, mask, images = processor.render(_conversation(_image(81, 41)))

        effort = processor.tokenizer.encode("Thinking effort level : 0.9", add_special_tokens=False)
        text = processor.tokenizer.encode("Describe this image .", add_special_tokens=False)
        answer = processor.tokenizer.encode("receipt total 4", add_special_tokens=False)
        system = [ids("<|message_system|>"), ids("<|content_text|>"), *effort, ids("<|end_message|>")]
        user = [ids("<|message_user|>"), ids("<|content_image|>"), *[IMAGE_TOKEN_ID] * 6, ids("<|end_message|>")]
        user += [ids("<|message_user|>"), ids("<|content_text|>"), *text, ids("<|end_message|>")]
        model = [ids("<|message_model|>"), ids("<|content_text|>"), *answer, ids("<|end_message|>")]
        model.append(ids("<|content_model_end_sampling|>"))
        assert input_ids == system + user + model
        assert mask == [0] * (len(system) + len(user)) + [1] * len(model)
        assert [image.num_tokens for image in images] == [6]

    @pytest.mark.parametrize("effort", [None, "none", 0.2])
    def test_conversations_match_the_release_template_with_one_placeholder_per_patch(
        self, checkpoint: Path, effort
    ) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint, reasoning_effort=effort)
        call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": {"k": "v"}}}
        conversation = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "image", "image": _image(81, 41)}, "What is 2 + 2 ?"]},
            {"role": "assistant", "content": "", "reasoning_content": "2 + 2 is 4", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c", "content": [{"type": "image", "image": _image(40, 40)}]},
            {"role": "assistant", "content": "4"},
        ]
        input_ids, mask, images = processor.render(conversation)
        kwargs = {} if effort is None else {"reasoning_effort": effort}
        reference = processor.tokenizer.encode(
            processor.tokenizer.apply_chat_template(
                conversation, chat_template=RELEASE_TEMPLATE, tokenize=False, **kwargs
            ),
            add_special_tokens=False,
        )
        sizes = iter(image.num_tokens for image in images)
        expanded = [t for token in reference for t in ([token] * next(sizes) if token == IMAGE_TOKEN_ID else [token])]

        assert input_ids == expanded
        assert [image.num_tokens for image in images] == [6, 1]
        assert reasoning_effort_level(effort) in processor.tokenizer.decode(input_ids).replace(" ", "")

    def test_an_image_token_the_model_does_not_replace_is_rejected(self, checkpoint: Path) -> None:
        config = json.loads((checkpoint / "config.json").read_text())
        (checkpoint / "config.json").write_text(json.dumps({**config, "image_token_id": IMAGE_TOKEN_ID - 1}))
        with pytest.raises(ValueError, match="but the model replaces token"):
            Inkling2Processor.from_pretrained(checkpoint)

    def test_text_turns_match_the_text_sft_pipeline(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        conversation = [
            {"role": "system", "content": "What is 2 + 2 ?"},
            {"role": "user", "content": "What is 2 + 2 ?"},
            {"role": "assistant", "content": "4", "reasoning_content": "2 + 2 is 4"},
        ]
        input_ids, mask, _ = processor.render(conversation)
        # Text SFT renders with the release's template and finds the assistant turn from its markers.
        processor.tokenizer.chat_template = RELEASE_TEMPLATE
        reference = tokenize_chat_example({"messages": conversation}, processor.tokenizer)

        assert input_ids == reference.input_ids.tolist()
        assert mask == [int(value) for value in reference.assistant_mask.tolist()]

    def test_rendered_text_and_images_match_the_conversation_path(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        processor.tokenizer.chat_template = RELEASE_TEMPLATE
        image = _image(81, 41)
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image ."}]}]
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[text], images=[image], return_tensors="pt")
        prompt[0]["content"][0]["image"] = image
        input_ids, _, images = processor.render(prompt)

        assert processor.chat_template == RELEASE_TEMPLATE
        assert text.count(IMAGE_TOKEN) == 1
        assert batch.input_ids[0].tolist() == input_ids
        assert batch["attention_mask"].tolist() == [[1] * len(input_ids)]
        assert torch.equal(batch.get("pixel_values"), images[0].patches)
        assert torch.equal(batch.get("image_grid_thw"), images[0].grid_thw)

    def test_rendered_text_needs_one_image_per_placeholder(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        with pytest.raises(ValueError, match="more image placeholders than images"):
            processor(text=["<|message_user|><|content_image|>" + IMAGE_TOKEN + "<|end_message|>"])
        with pytest.raises(ValueError, match="More images than image placeholders"):
            processor(text=["Describe this image ."], images=[_image(40, 40)])
        plain = processor(text="Describe this image .")

        assert plain.input_ids.shape == (1, 4) and plain.get("pixel_values") is None

    def test_rendered_text_takes_one_image_as_a_path_or_bytes(self, checkpoint: Path, tmp_path: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        path = tmp_path / "image.png"
        _image(81, 41).save(path)
        text = "<|message_user|><|content_image|>" + IMAGE_TOKEN + "<|end_message|>"
        expected = processor(text=[text], images=[Image.open(path)])

        # Strings and bytes are sequences, but each is one image here.
        for image in (str(path), path, path.read_bytes()):
            batch = processor(text=[text], images=image)
            assert torch.equal(batch.input_ids, expected.input_ids)
            assert torch.equal(batch["pixel_values"], expected["pixel_values"])
            assert batch["image_grid_thw"].tolist() == [[1, 2, 3]]

    def test_generation_helper_formats_inkling2_prompts_at_native_size(self, checkpoint: Path, tmp_path: Path) -> None:
        utils = _generation_utils()
        processor = Inkling2Processor.from_pretrained(checkpoint)
        processor.tokenizer.chat_template = RELEASE_TEMPLATE
        _image(81, 41).save(tmp_path / "image.png")

        input_ids, pixel_values, grid, *rest = utils.process_image_inputs(
            processor, str(tmp_path / "image.png"), "Describe this image .", is_inkling2=True
        )

        # 81 x 41 pixels at native size: 3 x 2 patches of 40 pixels, one placeholder each.
        assert grid.tolist() == [[1, 2, 3]]
        assert int((input_ids == IMAGE_TOKEN_ID).sum()) == 6 == pixel_values.shape[0]
        assert rest == [None, None, None]

        # Without an image the prompt is text only.
        input_ids, pixel_values, grid, *rest = utils.process_image_inputs(
            processor, None, "Describe a cat .", is_inkling2=True
        )

        rendered = processor.apply_chat_template([{"role": "user", "content": "Describe a cat ."}])
        assert "<|message_user|><|content_text|>Describe a cat .<|end_message|>" in rendered
        assert input_ids.tolist() == [processor.tokenizer.encode(rendered, add_special_tokens=False)]
        assert pixel_values is None and grid is None and rest == [None, None, None]

    def test_generation_helper_leaves_transparency_to_the_processor(self, checkpoint: Path, tmp_path: Path) -> None:
        utils = _generation_utils()
        processor = Inkling2Processor.from_pretrained(checkpoint)
        processor.tokenizer.chat_template = RELEASE_TEMPLATE
        path = tmp_path / "clear.png"
        # Fully transparent, with red under the transparency.
        Image.new("RGBA", (40, 40), (255, 0, 0, 0)).save(path)

        _, pixel_values, *_ = utils.process_image_inputs(
            processor, str(path), "Describe this image .", is_inkling2=True
        )

        # The processor's checkerboard, as in training, not the hidden red.
        assert torch.equal(pixel_values, processor.image_processor.encode(path).patches)

    def test_batches_pad_right_and_keep_images_in_order(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        batch = processor([_conversation(_image(40, 40)), _conversation(None), _conversation(_image(81, 41))])

        assert batch["input_ids"].shape[0] == 3
        assert batch["attention_mask"].sum(dim=1).tolist()[1] < batch["input_ids"].shape[1]
        assert batch["image_grid_thw"].tolist() == [[1, 1, 1], [1, 2, 3]]
        assert batch["pixel_values"].shape == (7, 2, 40, 40, 3)
        assert int((batch["input_ids"] == IMAGE_TOKEN_ID).sum()) == 7

    def test_text_is_cut_to_fit_but_images_never_are(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        assert processor([_conversation(None)], max_length=5)["input_ids"].shape == (1, 5)
        with pytest.raises(ValueError, match="image placeholders do not fit"):
            processor([_conversation(_image(81, 41))], max_length=5)

    @pytest.mark.parametrize("role", ["assistant", "system"])
    def test_images_outside_user_or_tool_messages_are_rejected(self, checkpoint: Path, role: str) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        conversation = [{"role": role, "content": [{"type": "image", "image": _image(40, 40)}]}]
        with pytest.raises(ValueError, match="only supported for user and tool messages"):
            processor.render(conversation)


class TestCollator:
    def test_labels_follow_the_assistant_turn_and_images_pass_through(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        examples = [{"conversation": _conversation(_image(81, 41))}, {"conversation": _conversation(None)}]
        batch = inkling2_collate_fn(examples, processor, sequence_length=64, pad_to_max_length=True)

        input_ids = batch["input_ids"]
        assert input_ids.shape == batch["labels"].shape == batch["loss_mask"].shape == (2, 64)
        rendered = processor([example["conversation"] for example in examples], max_length=64)
        width = rendered["input_ids"].shape[1]
        supervised = rendered["assistant_masks"][:, 1:].bool()
        assert torch.equal(batch["loss_mask"][:, : width - 1].bool(), supervised)
        assert torch.equal(batch["labels"][:, : width - 1][supervised], rendered["input_ids"][:, 1:][supervised])
        assert bool((batch["labels"][batch["loss_mask"] == 0] == IGNORE_INDEX).all())

        visual = batch["visual_inputs"]
        assert isinstance(visual, Inkling2VisualInputs)
        model_inputs = visual.normalized_for_model()
        assert model_inputs["pixel_values"].shape == (6, 2, 40, 40, 3)
        assert model_inputs["image_grid_thw"].tolist() == [[1, 2, 3]]

    @pytest.mark.parametrize("row_effort", [None, "none"])
    def test_rows_render_with_their_tools_and_reasoning_effort(self, checkpoint: Path, row_effort) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint, reasoning_effort="low")
        call = {"id": "c", "type": "function", "function": {"name": "add", "arguments": {"a": 2, "b": 2}}}
        conversation = [
            {"role": "user", "content": [{"type": "image", "image": _image(81, 41)}, "What is 2 + 2 ?"]},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c", "content": "4"},
            {"role": "assistant", "content": "4"},
        ]
        example = {"conversation": conversation, "tools": TOOLS}
        if row_effort is not None:
            example["chat_template_kwargs"] = {"reasoning_effort": row_effort}

        batch = inkling2_collate_fn([example], processor, sequence_length=256)

        # The row's effort replaces the processor's.
        reference = _release_template_ids(
            processor, conversation, [6], tools=TOOLS, reasoning_effort=row_effort or "low"
        )
        input_ids = batch["input_ids"][0, : len(reference)].tolist()
        assert input_ids == reference
        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
        assert tokens[:3] == ["<|message_system|>", "tool_declare", "<|content_xml|>"]
        # Each position predicts the next token: the tool declaration, the effort line, the user turn and the tool
        # result are context, and both assistant turns are supervised from their header through end of sampling.
        assert batch["loss_mask"][0, : len(reference) - 1].tolist() == _assistant_turns(tokens)[1:]

    def test_template_arguments_the_template_does_not_read_are_rejected(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        example = {"conversation": _conversation(None), "chat_template_kwargs": {"enable_thinking": False}}
        with pytest.raises(ValueError, match="reads only tools and reasoning_effort"):
            inkling2_collate_fn([example], processor, sequence_length=64)
        with pytest.raises(ValueError, match="per conversation"):
            processor([_conversation(None)], reasoning_effort="none")
        with pytest.raises(ValueError, match="function name"):
            processor.render(_conversation(None), tools=[{"type": "function", "function": {"description": "x"}}])

    def test_text_only_batches_have_no_visual_inputs(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        batch = inkling2_collate_fn([{"conversation": _conversation(None)}], processor, sequence_length=32)
        assert batch["visual_inputs"] is None

    def test_in_batch_packing_is_rejected(self, checkpoint: Path) -> None:
        processor = Inkling2Processor.from_pretrained(checkpoint)
        with pytest.raises(ValueError, match="in-batch packing"):
            inkling2_collate_fn(
                [{"conversation": _conversation(None)}], processor, sequence_length=32, enable_in_batch_packing=True
            )
