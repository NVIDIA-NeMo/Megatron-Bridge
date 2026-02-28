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
conversation-style examples consumable by VLM processors.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from datasets import concatenate_datasets, load_dataset

from megatron.bridge.data.vlm_datasets.token_utils import json2token
from megatron.bridge.utils.common_utils import resolve_path


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

 
def make_llava_pretrain_dataset(
    path_or_dataset: str = "liuhaotian/LLaVA-Pretrain",
    json_file: str = "blip_laion_cc_sbu_558k.json",
    split: str = "train",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess the *LLaVA-Pretrain* (558K) dataset.

    The dataset can be loaded in two ways:

    1. **Local JSON** (preferred): if ``path_or_dataset`` points to a local
       directory that contains ``json_file`` (default
       ``blip_laion_cc_sbu_558k.json``), the entries are read directly from
       that file.  Images are expected to be in the same directory::

           /path/to/LLaVA-Pretrain/
           ├── blip_laion_cc_sbu_558k.json
           └── 00453/004539375.jpg
           └── ...

    2. **HuggingFace Hub**: otherwise ``load_dataset(path_or_dataset, ...)``
       is used.

    Each entry contains:
    - ``image``: relative path to the image file (e.g. ``"00453/004539375.jpg"``).
    - ``conversations``: a two-turn list::

          [{"from": "human", "value": "<question>\\n<image>"},
           {"from": "gpt",   "value": "<answer>"}]

      The ``<image>`` placeholder is stripped from the human prompt and replaced
      with an actual image content entry.

    Args:
        path_or_dataset: HF dataset path **or** local directory containing
            ``json_file`` and the image tree.
        json_file: Name of the JSON annotation file inside
            ``path_or_dataset`` when loading from a local directory.
        split: Split to load when using ``load_dataset``.

    Returns:
        A list of dicts each containing a ``conversation`` field ready for
        downstream VLM processors.
    """
    dataset_root = Path(path_or_dataset)
    local_json = dataset_root / json_file
    if local_json.is_file():
        with open(local_json, "r") as f:
            dataset = json.load(f)
    else:
        dataset = load_dataset(path_or_dataset, split=split)
        dataset_root = None

    dataset = dataset[:100]  # FIXME: for testing only, remove this line for full dataset

    def clean_prompt(val: str) -> str:
        val = val.replace("<image>", "").replace("<video>", "").strip()
        return val.lstrip("\n").rstrip()

    def format(example):
        image = example.get("image")
        convs = example.get("conversations", [])
        if image in (None, "") or not convs:
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
                    if dataset_root is not None:
                        abs_path = resolve_path(dataset_root / image)
                        content.append({"type": "image", "image": str(abs_path)})
                    else:
                        content.append({"type": "image", "image": image})
                    first_human_handled = True
                content.append({"type": "text", "text": clean_prompt(value)})
                conversation.append({"role": "user", "content": content})
            elif role == "gpt":
                conversation.append(
                    {"role": "assistant", "content": [{"type": "text", "text": value.strip()}]}
                )

        if not conversation:
            return None

        return {"conversation": conversation}

    formatted = (format(ex) for ex in dataset)
    return [ex for ex in formatted if ex is not None]


def make_llava_video_178k_dataset(
    video_root_path: str,
    path_or_dataset: str = "lmms-lab/LLaVA-Video-178K",
    subsets: str | List[str] = "0_30_s_nextqa", # 0_30_s_academic_v0_1
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


def make_cv17_dataset(
    path_or_dataset: str = "ysdede/commonvoice_17_tr_fixed", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the CommonVoice 17 dataset for audio-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)
    # Be robust to simple list-like datasets used in tests without `column_names` attr
    try:
        all_columns = dataset.column_names  # type: ignore[attr-defined]
    except Exception:
        first_example = dataset[0] if len(dataset) > 0 else {}
        all_columns = list(first_example.keys()) if isinstance(first_example, dict) else []
    if hasattr(dataset, "remove_columns"):
        columns_to_remove = [col for col in all_columns if col not in ["audio", "transcription"]]
        dataset = dataset.remove_columns(columns_to_remove)

    def format(example):
        return {
            "conversation": [
                {"role": "user", "content": "<|audio_1|>Transcribe the Turkish audio clip."},
                {"role": "assistant", "content": example["transcription"]},
            ],
            "audio": (example["audio"]["array"], example["audio"]["sampling_rate"]),
        }

    return [format(example) for example in dataset]
