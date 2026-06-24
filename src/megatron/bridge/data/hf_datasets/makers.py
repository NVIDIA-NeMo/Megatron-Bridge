# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Built-in maker functions that transform HuggingFace datasets into
Bridge chat or multimodal conversation examples.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import concatenate_datasets, load_dataset

from megatron.bridge.data.hf_datasets.token_utils import json2token
from megatron.bridge.utils.common_utils import resolve_path


HF_MAKER_ALIASES = {
    "rdr": "make_rdr_dataset",
    "cord_v2": "make_cord_v2_dataset",
    "medpix": "make_medpix_dataset",
    "text_chat": "make_text_chat_dataset",
    "chat": "make_text_chat_dataset",
    "squad": "make_squad_dataset",
    "gsm8k": "make_gsm8k_dataset",
    "openmathinstruct2": "make_openmathinstruct2_dataset",
    "openmathinstruct2_thinking": "make_openmathinstruct2_thinking_dataset",
    "cv17": "make_cv17_dataset",
    "raven": "make_raven_dataset",
    "llava_video_178k": "make_llava_video_178k_dataset",
    "default_audio": "make_default_audio_dataset",
    "valor32k_avqa": "make_valor32k_avqa_dataset",
}


def _load_hf_dataset(
    path_or_dataset: str,
    subset: str | None = None,
    split: str = "train",
    **kwargs,
) -> Any:
    """Load a Hugging Face dataset with optional subset."""
    if subset is None:
        return load_dataset(path_or_dataset, split=split, **kwargs)
    return load_dataset(path_or_dataset, subset, split=split, **kwargs)


def _make_messages_example(
    prompt: str,
    answer: str,
    original_answers: list[str] | None = None,
) -> Dict[str, Any]:
    """Create a text-only chat example with optional evaluation answers."""
    example: Dict[str, Any] = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }
    if original_answers is not None:
        example["original_answers"] = original_answers
    return example


def _extract_final_answer(answer: str) -> str:
    """Extract the final numerical answer after the ``####`` delimiter."""
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def _strip_intermediate_boxed(text: str) -> str:
    """Replace all ``\\boxed{content}`` occurrences in text with ``content``."""
    marker = r"\boxed{"
    result = []
    i = 0
    while i < len(text):
        idx = text.find(marker, i)
        if idx == -1:
            result.append(text[i:])
            break
        result.append(text[i:idx])
        depth = 0
        end = -1
        for j in range(idx + len(marker) - 1, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            result.append(text[idx:])
            break
        result.append(text[idx + len(marker) : end])
        i = end + 1
    return "".join(result)


def make_squad_dataset(
    path_or_dataset: str = "rajpurkar/squad",
    subset: str | None = None,
    split: str = "train",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess SQuAD into text chat examples."""
    dataset = _load_hf_dataset(path_or_dataset, subset=subset, split=split, **kwargs)

    def format_example(example):
        prompt = f"Context: {example['context']} Question: {example['question']} Answer:"
        answers = example["answers"]["text"]
        return _make_messages_example(prompt=prompt, answer=answers[0], original_answers=answers)

    return [format_example(example) for example in dataset]


def make_gsm8k_dataset(
    path_or_dataset: str = "openai/gsm8k",
    subset: str | None = "main",
    split: str = "train",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess GSM8K into text chat examples."""
    dataset = _load_hf_dataset(path_or_dataset, subset=subset, split=split, **kwargs)

    def format_example(example):
        prompt = f"Question: {example['question']} Answer:"
        answer = example["answer"]
        return _make_messages_example(prompt=prompt, answer=answer, original_answers=[_extract_final_answer(answer)])

    return [format_example(example) for example in dataset]


def make_openmathinstruct2_dataset(
    path_or_dataset: str = "nvidia/OpenMathInstruct-2",
    subset: str | None = None,
    split: str = "train_1M",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess OpenMathInstruct-2 into text chat examples."""
    dataset = _load_hf_dataset(path_or_dataset, subset=subset, split=split, **kwargs)

    def format_example(example):
        prompt = f"Problem: {example['problem']} Solution:"
        return _make_messages_example(
            prompt=prompt,
            answer=example["generated_solution"],
            original_answers=[str(example["expected_answer"])],
        )

    return [format_example(example) for example in dataset]


def make_openmathinstruct2_thinking_dataset(
    path_or_dataset: str = "nvidia/OpenMathInstruct-2",
    subset: str | None = None,
    split: str = "train_1M",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load OpenMathInstruct-2 with reasoning in ``thinking`` and final answer in content."""
    dataset = _load_hf_dataset(path_or_dataset, subset=subset, split=split, **kwargs)

    def format_example(example):
        solution = example["generated_solution"]
        expected_answer = str(example["expected_answer"])

        marker = r"\boxed{"
        idx = solution.rfind(marker)
        if idx != -1:
            depth = 0
            end = -1
            for i in range(idx + len(marker) - 1, len(solution)):
                if solution[i] == "{":
                    depth += 1
                elif solution[i] == "}":
                    depth -= 1
                if depth == 0:
                    end = i
                    break
            thinking = re.sub(r"\$?\s*$", "", solution[:idx]).rstrip() if end != -1 else solution.rstrip()
        else:
            thinking = solution.rstrip()

        thinking = _strip_intermediate_boxed(thinking)

        return {
            "messages": [
                {"role": "user", "content": example["problem"]},
                {"role": "assistant", "thinking": thinking, "content": f"#### {expected_answer}"},
            ],
            "original_answers": [expected_answer],
        }

    return [format_example(example) for example in dataset]


def make_rdr_dataset(
    path_or_dataset: str = "quintend/rdr-items", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the RDR dataset for image-to-text fine-tuning.

    Returns a list of examples with a "conversation" field that includes an image and text.
    """
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["text"]}],
                },
            ],
        }

    return [format(example) for example in dataset]


def make_cord_v2_dataset(
    path_or_dataset: str = "naver-clova-ix/cord-v2", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the CORD-V2 dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        ground_truth = json.loads(example["ground_truth"])
        if "gt_parses" in ground_truth:
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
            gt_jsons = [ground_truth["gt_parse"]]

        text = random.choice([json2token(gt_json, sort_json_key=True) for gt_json in gt_jsons])

        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ],
        }

    return [format(example) for example in dataset]


def make_medpix_dataset(
    path_or_dataset: str = "mmoukouba/MedPix-VQA", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the MedPix dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
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
            ],
        }

    return [format(example) for example in dataset]


def make_text_chat_dataset(
    path_or_dataset: str,
    subset: str | None = None,
    split: str = "train",
    messages_column: str = "messages",
    conversation_column: str = "conversation",
    conversations_column: str = "conversations",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load a text-only HF chat dataset into the conversation-provider schema.

    The input dataset must already contain OpenAI-style ``messages``, a
    processor-ready ``conversation`` column, or a legacy ``conversations``
    column. Extra fields are preserved so collators can consume metadata such
    as tool schemas.
    """
    if subset is None:
        dataset = load_dataset(path_or_dataset, split=split, **kwargs)
    else:
        dataset = load_dataset(path_or_dataset, subset, split=split, **kwargs)

    schema_columns = {messages_column, conversation_column, conversations_column}

    def format_example(example):
        extra = {key: value for key, value in example.items() if key not in schema_columns}
        if messages_column in example and example[messages_column] is not None:
            return {"messages": example[messages_column], **extra}
        if conversation_column in example and example[conversation_column] is not None:
            return {"conversation": example[conversation_column], **extra}
        if conversations_column in example and example[conversations_column] is not None:
            return {"conversations": example[conversations_column], **extra}
        raise ValueError(
            f"Text chat dataset rows must contain '{messages_column}', '{conversation_column}', "
            f"or '{conversations_column}' columns."
        )

    return [format_example(example) for example in dataset]


def make_raven_dataset(
    path_or_dataset: str = "HuggingFaceM4/the_cauldron",
    subset: str = "raven",
    split: str = "train",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess the Raven subset from the Cauldron dataset.

    This subset follows the IDEFICS-style layout where each sample contains:
    - ``images``: a (possibly empty) list of PIL images
    - ``texts``: a list of conversation dictionaries. For Raven, ``texts[0]``
      is a *single* turn stored as a dictionary with two keys::

          {"user": "<question>", "assistant": "<answer>"}

      Only the first element is used.  The ``user`` string is taken as the
      user prompt, and ``assistant`` is the ground-truth answer.

    Conversation building policy:
    1. All images are placed at the beginning of the user turn followed by the
       textual prompt.
    2. The assistant turn contains the answer text.

    Examples missing either images or the required fields are filtered out.
    """
    if split != "train":
        raise ValueError("Raven dataset only supports train split. Please set `train.eval_iters=0`.")
    dataset = load_dataset(path_or_dataset, subset, split=split)

    def format(example):
        images = example.get("images", [])
        texts = example.get("texts", [])
        if not images or not texts or not isinstance(texts[0], dict):
            return None

        user_prompt = texts[0].get("user")
        assistant_answer = texts[0].get("assistant")
        if user_prompt is None or assistant_answer is None:
            return None

        user_content: List[Dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": user_prompt})

        assistant_content = [{"type": "text", "text": assistant_answer}]

        return {
            "conversation": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    formatted = (format(example) for example in dataset)
    # Filter out any None values from malformed rows.
    return [ex for ex in formatted if ex is not None]


def make_llava_video_178k_dataset(
    video_root_path: str,
    path_or_dataset: str = "lmms-lab/LLaVA-Video-178K",
    subsets: str | List[str] = "0_30_s_nextqa",
    split: str = "open_ended",
) -> List[Dict[str, Any]]:
    """Load and preprocess a subset of the *LLaVA-Video-178K* dataset.

    Each row contains:
    - ``video``: path or URL to the MP4 file.
    - ``conversations``: a **two-turn** list::

          [{"from": "human", "value": "<video>\n<question>"},
           {"from": "gpt",   "value": "<answer>"}]

      We map this schema to our internal multimodal conversation format:

      User turn  →  [video, user prompt]
      Assistant  →  answer text

    Note:
        Video files are assumed to be pre-downloaded and stored locally in the
        ``video_root_path`` directory. Rows with missing videos or empty
        conversations are filtered out from the final output.

    Args:
        video_root_path: Root directory where video files are stored locally.
        path_or_dataset: HF dataset path or local cache dir.
        subsets: Single subset name or list of the dataset's directory-style
            subsets to load.
        split: Split to load from the dataset. Note that "train" is automatically
            mapped to "open_ended".

    Returns:
        A list of dicts each containing a ``conversation`` field ready for
        downstream VLM processors.
    """
    if isinstance(subsets, str):
        subsets = [subsets]

    if split == "train":
        split = "open_ended"
    elif split in ("validation", "test"):
        raise ValueError("LLaVA-Video-178K dataset only supports train split. Please set `train.eval_iters=0`.")
    individual_datasets = [load_dataset(path_or_dataset, subset, split=split) for subset in subsets]
    dataset = concatenate_datasets(individual_datasets)

    # FIXME: right now we assume the video files are pre-downloaded and stored in the video_root_path
    # we need to modify this to download the video files from the hub if they are not present in the video_root_path

    def clean_prompt(val: str) -> str:
        # Remove placeholder tokens such as <image> or <video>
        val = val.replace("<image>", "").replace("<video>", "").strip()
        return val.lstrip("\n").rstrip()

    def format(example):
        video = example.get("video")
        convs = example.get("conversations", [])
        if video in (None, "") or not convs:
            return None

        conversation: List[Dict[str, Any]] = []

        first_human_handled = False
        for turn in convs:
            role = turn.get("from")
            value = turn.get("value", "")
            if not value:
                continue
            if role == "human":
                content: List[Dict[str, Any]] = []
                if not first_human_handled:
                    abs_path = resolve_path(Path(video_root_path) / video)
                    content.append({"type": "video", "path": str(abs_path)})
                    first_human_handled = True
                content.append({"type": "text", "text": clean_prompt(value)})
                conversation.append({"role": "user", "content": content})
            elif role == "gpt":
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": value.strip()}]})

        if not conversation:
            return None

        return {"conversation": conversation}

    formatted = (format(ex) for ex in dataset)
    return [ex for ex in formatted if ex is not None]


def make_default_audio_dataset(
    path_or_dataset: str,
    subset: str | None = None,
    split: str = "train",
    audio_column: str = "audio",
    text_column: str = "text",
    prompt: str = "Transcribe the audio clip.",
    remove_text_spaces: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess a HuggingFace audio dataset for audio-to-text fine-tuning.

    Formats each example into a conversation with an audio user turn and a text assistant turn.
    Works with any HF dataset that has audio and text columns.
    """
    dataset = load_dataset(path_or_dataset, subset, split=split)
    try:
        all_columns = dataset.column_names
    except Exception:
        first_example = dataset[0] if len(dataset) > 0 else {}
        all_columns = list(first_example.keys()) if isinstance(first_example, dict) else []
    if hasattr(dataset, "remove_columns"):
        columns_to_remove = [col for col in all_columns if col not in [audio_column, text_column]]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

    def format_example(example):
        text = example[text_column]
        if remove_text_spaces:
            text = text.replace(" ", "")
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            ],
            "audio": (example[audio_column]["array"], example[audio_column]["sampling_rate"]),
        }

    return [format_example(example) for example in dataset]


def make_valor32k_avqa_dataset(
    data_root: str,
    split: str = "train",
    max_audio_duration: float = 10.0,
    modality_filter: str = "all",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load Valor32k-AVQA v2.0 dataset for audio-visual QA finetuning.

    Expects a directory produced by ``tutorials/data/valor32k-avqa/prepare_valor32k_avqa.py``::

        data_root/
        ├── videos/                                  # 10s MP4 clips
        ├── audio/                                   # 16 kHz mono WAV
        └── combined_dataset_{split}_flattened.json

    Args:
        data_root: Root directory of the preprocessed dataset.
        split: ``"train"``, ``"val"``, or ``"test"``.
        max_audio_duration: Maximum audio duration in seconds.
        modality_filter: ``"all"``, ``"audio-visual"``, ``"audio"``, or ``"visual"``.
    """
    root = Path(data_root)
    # Map split names: "train"→"train", "validation"→"val", "test"→"test"
    split_name = "val" if split == "validation" else split
    qa_file = root / f"combined_dataset_{split_name}_flattened.json"
    if not qa_file.exists():
        raise FileNotFoundError(
            f"QA file not found: {qa_file}. Run tutorials/data/valor32k-avqa/prepare_valor32k_avqa.py first."
        )

    with open(qa_file) as f:
        qa_pairs = json.load(f)

    examples: List[Dict[str, Any]] = []
    for qa in qa_pairs:
        # Apply modality filter
        modality = qa.get("modality", "audio-visual")
        if modality_filter != "all" and modality != modality_filter:
            continue

        video_id = str(qa["video_id"])
        video_path = root / "videos" / f"{video_id}.mp4"
        audio_path = root / "audio" / f"{video_id}.wav"

        # For visual-only, skip audio requirement; for audio-only, skip video
        has_video = video_path.exists()
        has_audio = audio_path.exists()
        if modality in ("visual", "audio-visual") and not has_video:
            continue
        if modality in ("audio", "audio-visual") and not has_audio:
            continue

        # Build question with MCQ options
        question = qa["question"]
        options = qa.get("options", [])
        if options:
            option_labels = "ABCD"
            option_text = "\n".join(f"{option_labels[i]}. {opt}" for i, opt in enumerate(options))
            question = f"{question}\n{option_text}"

        # Build answer from correct option
        correct_idx = qa.get("correct_answer_idx", 0)
        if options and correct_idx < len(options):
            answer = options[correct_idx]
        else:
            answer = qa.get("rephrased_answers", [""])[0] if qa.get("rephrased_answers") else ""

        # Build conversation with video + question
        user_content = []
        if has_video:
            user_content.append({"type": "video", "path": str(video_path)})
        user_content.append({"type": "text", "text": question})

        example = {
            "conversation": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ],
        }
        if has_audio:
            example["audio_path"] = str(audio_path)
            example["max_audio_duration"] = max_audio_duration

        examples.append(example)

    if not examples:
        raise ValueError(f"No valid examples found in {qa_file}.")
    return examples


def make_cv17_dataset(
    path_or_dataset: str = "ysdede/commonvoice_17_tr_fixed",
    split: str = "train",
    prompt: str = "Transcribe the Turkish audio clip.",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess the CommonVoice 17 dataset for audio-to-text fine-tuning."""
    import io

    import soundfile as sf
    from datasets import Audio

    dataset = load_dataset(path_or_dataset, split=split)

    # Disable automatic audio decoding (avoids torchcodec dependency)
    # and decode manually using soundfile.
    if hasattr(dataset, "cast_column"):
        dataset = dataset.cast_column("audio", Audio(decode=False))

    # Be robust to simple list-like datasets used in tests without `column_names` attr
    try:
        all_columns = dataset.column_names  # type: ignore[attr-defined]
    except Exception:
        first_example = dataset[0] if len(dataset) > 0 else {}
        all_columns = list(first_example.keys()) if isinstance(first_example, dict) else []
    if hasattr(dataset, "remove_columns"):
        columns_to_remove = [col for col in all_columns if col not in ["audio", "transcription"]]
        dataset = dataset.remove_columns(columns_to_remove)

    def _decode_audio(audio_dict):
        """Decode audio bytes/path to numpy array using soundfile."""
        if isinstance(audio_dict, dict) and "array" in audio_dict:
            # Already decoded
            return audio_dict["array"], audio_dict["sampling_rate"]
        audio_bytes = audio_dict.get("bytes") if isinstance(audio_dict, dict) else None
        audio_path = audio_dict.get("path") if isinstance(audio_dict, dict) else None
        if audio_bytes is not None:
            waveform, sr = sf.read(io.BytesIO(audio_bytes))
        elif audio_path is not None:
            waveform, sr = sf.read(audio_path)
        else:
            raise ValueError("Audio example has neither 'bytes', 'path', nor 'array'")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform, sr

    def format(example):
        array, sr = _decode_audio(example["audio"])
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": example["transcription"]}]},
            ],
            "audio": (array, sr),
        }

    return [format(example) for example in dataset]


def get_hf_dataset_maker(maker_name: str):
    """Return a built-in Hugging Face dataset maker by name or alias."""
    registry = {
        "make_rdr_dataset": make_rdr_dataset,
        "make_cord_v2_dataset": make_cord_v2_dataset,
        "make_medpix_dataset": make_medpix_dataset,
        "make_text_chat_dataset": make_text_chat_dataset,
        "make_squad_dataset": make_squad_dataset,
        "make_gsm8k_dataset": make_gsm8k_dataset,
        "make_openmathinstruct2_dataset": make_openmathinstruct2_dataset,
        "make_openmathinstruct2_thinking_dataset": make_openmathinstruct2_thinking_dataset,
        "make_cv17_dataset": make_cv17_dataset,
        "make_raven_dataset": make_raven_dataset,
        "make_llava_video_178k_dataset": make_llava_video_178k_dataset,
        "make_default_audio_dataset": make_default_audio_dataset,
        "make_valor32k_avqa_dataset": make_valor32k_avqa_dataset,
    }
    resolved_name = HF_MAKER_ALIASES.get(maker_name, maker_name)
    try:
        return registry[resolved_name]
    except KeyError as err:
        raise ValueError(f"Unknown maker_name: {maker_name}") from err
