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

"""Schema adapters for already-loaded Hugging Face datasets."""

import io
import json
import random
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from megatron.bridge.data.hf_datasets.token_utils import json2token
from megatron.bridge.utils.common_utils import resolve_path


HFDatasetAdapter = Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any] | None]


def _messages_example(prompt: str, answer: str, original_answers: list[str] | None = None) -> dict[str, Any]:
    example: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }
    if original_answers is not None:
        example["original_answers"] = original_answers
    return example


def _native_conversation_adapter(example: Mapping[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    messages_column = str(kwargs.get("messages_column", "messages"))
    conversation_column = str(kwargs.get("conversation_column", "conversation"))
    conversations_column = str(kwargs.get("conversations_column", "conversations"))
    schema_columns = {messages_column, conversation_column, conversations_column}
    extra = {key: value for key, value in example.items() if key not in schema_columns}
    if example.get(messages_column) is not None:
        return {"messages": example[messages_column], **extra}
    if example.get(conversation_column) is not None:
        return {"conversation": example[conversation_column], **extra}
    if example.get(conversations_column) is not None:
        return {"conversations": example[conversations_column], **extra}
    raise ValueError(
        "Hugging Face SFT rows must contain a messages, conversation, or conversations column, "
        "or configure a schema_adapter."
    )


def _squad_adapter(example: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    answers = example["answers"]["text"]
    prompt = f"Context: {example['context']} Question: {example['question']} Answer:"
    return _messages_example(prompt, answers[0], list(answers))


def _gsm8k_adapter(example: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    answer = str(example["answer"])
    final_answer = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
    return _messages_example(f"Question: {example['question']} Answer:", answer, [final_answer])


def _openmathinstruct2_adapter(example: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    return _messages_example(
        f"Problem: {example['problem']} Solution:",
        str(example["generated_solution"]),
        [str(example["expected_answer"])],
    )


def _strip_intermediate_boxed(text: str) -> str:
    marker = r"\boxed{"
    result: list[str] = []
    offset = 0
    while offset < len(text):
        index = text.find(marker, offset)
        if index == -1:
            result.append(text[offset:])
            break
        result.append(text[offset:index])
        depth = 0
        end = -1
        for cursor in range(index + len(marker) - 1, len(text)):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    end = cursor
                    break
        if end == -1:
            result.append(text[index:])
            break
        result.append(text[index + len(marker) : end])
        offset = end + 1
    return "".join(result)


def _openmathinstruct2_thinking_adapter(example: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    solution = str(example["generated_solution"])
    expected_answer = str(example["expected_answer"])
    marker = r"\boxed{"
    index = solution.rfind(marker)
    if index != -1:
        depth = 0
        end = -1
        for cursor in range(index + len(marker) - 1, len(solution)):
            if solution[cursor] == "{":
                depth += 1
            elif solution[cursor] == "}":
                depth -= 1
            if depth == 0:
                end = cursor
                break
        thinking = re.sub(r"\$?\s*$", "", solution[:index]).rstrip() if end != -1 else solution.rstrip()
    else:
        thinking = solution.rstrip()
    return {
        "messages": [
            {"role": "user", "content": example["problem"]},
            {
                "role": "assistant",
                "thinking": _strip_intermediate_boxed(thinking),
                "content": f"#### {expected_answer}",
            },
        ],
        "original_answers": [expected_answer],
    }


def _rdr_adapter(example: Mapping[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(kwargs.get("prompt", "Describe this image."))
    return {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": example["text"]}]},
        ]
    }


def _cord_v2_adapter(example: Mapping[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    ground_truth = json.loads(example["ground_truth"])
    if "gt_parses" in ground_truth:
        gt_jsons = ground_truth["gt_parses"]
    else:
        gt_jsons = [ground_truth["gt_parse"]]
    text = json2token(random.choice(gt_jsons), sort_json_key=True)
    prompt = str(kwargs.get("prompt", "Describe this image."))
    return {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]
    }


def _medpix_adapter(example: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image_id"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
        ]
    }


def _raven_adapter(example: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any] | None:
    images = example.get("images", [])
    texts = example.get("texts", [])
    if not images or not texts or not isinstance(texts[0], Mapping):
        return None
    prompt = texts[0].get("user")
    answer = texts[0].get("assistant")
    if prompt is None or answer is None:
        return None
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return {
        "conversation": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def _llava_video_adapter(example: Mapping[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any] | None:
    video = example.get("video")
    turns = example.get("conversations", [])
    video_root_path = kwargs.get("video_root_path")
    if video in (None, "") or not turns or video_root_path is None:
        return None

    conversation: list[dict[str, Any]] = []
    added_video = False
    for turn in turns:
        role = turn.get("from")
        value = str(turn.get("value", ""))
        if not value:
            continue
        if role == "human":
            content: list[dict[str, Any]] = []
            if not added_video:
                video_path = resolve_path(Path(str(video_root_path)) / str(video))
                content.append({"type": "video", "path": str(video_path)})
                added_video = True
            prompt = value.replace("<image>", "").replace("<video>", "").strip().lstrip("\n").rstrip()
            content.append({"type": "text", "text": prompt})
            conversation.append({"role": "user", "content": content})
        elif role == "gpt":
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": value.strip()}]})
    return {"conversation": conversation} if conversation else None


def _decode_audio(audio: Any) -> tuple[Any, int]:
    if isinstance(audio, Mapping) and audio.get("array") is not None:
        return audio["array"], int(audio["sampling_rate"])
    if not isinstance(audio, Mapping):
        raise ValueError("Audio examples must contain a Hugging Face audio mapping.")
    import soundfile as sf

    if audio.get("bytes") is not None:
        waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]))
    elif audio.get("path") is not None:
        waveform, sample_rate = sf.read(audio["path"])
    else:
        raise ValueError("Audio example has neither bytes, path, nor array data.")
    if getattr(waveform, "ndim", 1) > 1:
        waveform = waveform.mean(axis=1)
    return waveform, int(sample_rate)


def _audio_adapter(example: Mapping[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    audio_column = str(kwargs.get("audio_column", "audio"))
    text_column = str(kwargs.get("text_column", "text"))
    prompt = str(kwargs.get("prompt", "Transcribe the audio clip."))
    text = str(example[text_column])
    if bool(kwargs.get("remove_text_spaces", True)):
        text = text.replace(" ", "")
    waveform, sample_rate = _decode_audio(example[audio_column])
    return {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ],
        "audio": (waveform, sample_rate),
    }


def _cv17_adapter(example: Mapping[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "prompt": "Transcribe the Turkish audio clip.",
        "remove_text_spaces": False,
        "text_column": "transcription",
        **kwargs,
    }
    return _audio_adapter(example, values)


def prepare_hf_dataset_for_adapter(
    dataset: Any,
    *,
    adapter_name: str | None,
    adapter_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Apply dataset-level preparation required by a schema adapter."""
    if adapter_name not in {"cv17", "default_audio"} or not hasattr(dataset, "cast_column"):
        return dataset
    from datasets import Audio

    audio_column = str((adapter_kwargs or {}).get("audio_column", "audio"))
    return dataset.cast_column(audio_column, Audio(decode=False))


_ADAPTERS: dict[str, HFDatasetAdapter] = {
    "text_chat": _native_conversation_adapter,
    "chat": _native_conversation_adapter,
    "squad": _squad_adapter,
    "gsm8k": _gsm8k_adapter,
    "openmathinstruct2": _openmathinstruct2_adapter,
    "openmathinstruct2_thinking": _openmathinstruct2_thinking_adapter,
    "rdr": _rdr_adapter,
    "cord_v2": _cord_v2_adapter,
    "medpix": _medpix_adapter,
    "raven": _raven_adapter,
    "default_audio": _audio_adapter,
    "cv17": _cv17_adapter,
    "llava_video_178k": _llava_video_adapter,
}


def adapt_hf_dataset(
    dataset: Iterable[Mapping[str, Any]],
    *,
    adapter_name: str | None,
    adapter_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize an already-loaded dataset into canonical SFT examples.

    Args:
        dataset: Loaded Hugging Face rows.
        adapter_name: Optional registered schema adapter. Native conversation
            rows require no adapter.
        adapter_kwargs: Declarative adapter-specific options.

    Returns:
        Canonical text or multimodal conversation examples.
    """
    if adapter_name is None:
        adapter = _native_conversation_adapter
    else:
        try:
            adapter = _ADAPTERS[adapter_name]
        except KeyError as error:
            raise ValueError(f"Unknown Hugging Face schema adapter: {adapter_name}") from error
    kwargs = adapter_kwargs or {}
    examples = [adapted for row in dataset if (adapted := adapter(row, kwargs)) is not None]
    if not examples:
        raise ValueError("Hugging Face source produced no examples after schema adaptation.")
    return examples
