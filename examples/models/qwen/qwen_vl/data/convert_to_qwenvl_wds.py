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
Example Usage:

# Example dataset from https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain

python examples/models/qwen/qwen_vl/data/convert_to_qwenvl_wds.py \\
    --dataset-root=/path/to/LLaVA-Pretrain-LCS-558K \
    --json=blip_laion_cc_sbu_558k.json \
    --mediate-path=images \
    --max-samples-per-tar=1000

Dataset Structure:
/path/to/LLaVA-Pretrain-LCS-558K/
├── blip_laion_cc_sbu_558k.json
└── images/
    ├── 00453/
    │   └── 004539375.jpg
    └── ...

Each sample stores its image files as a pickled list of bytes in ``.jpgs``. It stores each video as the
frames that qwen-vl-utils samples, at the source resolution, as a pickled list of PNG bytes in ``.videos``.
The shards keep the source pixels, so the Qwen-VL collate resizes the media one time, as in the model card.
"""

import io
import json
import os
import pickle
from argparse import ArgumentParser

import webdataset as wds
from PIL import Image
from tqdm import tqdm


os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
from qwen_vl_utils.vision_process import VIDEO_READER_BACKENDS, get_video_reader_backend


def _video_frames(file_path):
    """Return the frames that qwen-vl-utils samples from a video file, as PNG bytes at the source resolution."""
    video, _, _ = VIDEO_READER_BACKENDS[get_video_reader_backend()]({"video": file_path})
    frames = []
    for frame in video:
        buffer = io.BytesIO()
        Image.fromarray(frame.permute(1, 2, 0).numpy()).save(buffer, format="PNG")
        frames.append(buffer.getvalue())
    return frames


def convert(dataset_dir, json_name, max_count=10000, mediate_path=""):
    """
    Here we provide an example to convert llava-pretrain dataset to webdataset
    """

    # Paths to the dataset files
    json_file = os.path.join(dataset_dir, json_name)
    output = os.path.join(dataset_dir, "wds")

    if not os.path.exists(output):
        os.mkdir(output)

    # Load data
    with open(json_file, "r") as f:
        data = json.load(f)

    has_idx = None
    with wds.ShardWriter(os.path.join(output, "pretrain-%d.tar"), maxcount=max_count) as shard_writer:
        for idx, entry in enumerate(tqdm(data)):
            # NOTE: read a dataset in sharegpt format
            images_data = []
            if "image" in entry:
                pop_item = entry.pop("image")
            elif "images" in entry:
                pop_item = entry.pop("images")
            else:
                pop_item = []

            if not isinstance(pop_item, list):
                pop_item = [pop_item]
            for image in pop_item:
                file_path = os.path.normpath(os.path.join(dataset_dir, mediate_path, image))
                with open(file_path, "rb") as image_file:
                    images_data.append(image_file.read())

            videos_data = []
            if "video" in entry:
                pop_item = entry.pop("video")
            elif "videos" in entry:
                pop_item = entry.pop("videos")
            else:
                pop_item = []

            if not isinstance(pop_item, list):
                pop_item = [pop_item]
            for video in pop_item:
                file_path = os.path.normpath(os.path.join(dataset_dir, mediate_path, video))
                videos_data.append(_video_frames(file_path))

            if has_idx is None:
                has_idx = "id" in entry
            assert has_idx == ("id" in entry), "All entries should either all contain idx or not."
            if "conversations" in entry:
                conv = json.dumps(entry["conversations"]).encode("utf-8")
            elif "messages" in entry:
                conv = json.dumps(entry["messages"]).encode("utf-8")
            else:
                conv = None
            assert conv is not None, "No conversation texts"

            sample = {
                "__key__": entry.pop("id", str(idx)),
                "jpgs": pickle.dumps(images_data),
                "videos": pickle.dumps(videos_data),
                "json": conv,
            }
            shard_writer.write(sample)

    return output


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dataset-root", required=True, type=str)
    argparser.add_argument("--json", default="dataset.json", type=str)
    argparser.add_argument("--max-samples-per-tar", default=10000, type=float)
    argparser.add_argument("--mediate-path", default="", type=str)
    args = argparser.parse_args()

    output_dir = convert(
        args.dataset_root, args.json, max_count=args.max_samples_per_tar, mediate_path=args.mediate_path
    )
    print(f"Dataset is successfully converted to wds, output dir: {output_dir}")
