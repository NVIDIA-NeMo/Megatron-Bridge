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

import json

from megatron.bridge.data.loaders import (
    get_blend_and_blend_per_split,
)


DATA_PATH = "/opt/data/datasets/train/test_text_document"


class TestDataLoaders:
    def test_get_blend_and_blend_per_split_data_paths(self):
        blend, blend_per_split = get_blend_and_blend_per_split(data_paths=[1.0, DATA_PATH])

        assert blend == ([DATA_PATH], [1.0])
        assert blend_per_split == None

    def test_get_blend_and_blend_per_split_data_args_path(self):
        # Generate data args file
        data_args_path = "/opt/data/datasets/input/data_args.txt"
        with open(data_args_path, "w") as data_args_file:
            data_args_file.write(f"0.5 {DATA_PATH} 0.5 {DATA_PATH}")
        blend, blend_per_split = get_blend_and_blend_per_split(data_args_path=data_args_path)

        assert blend == ([DATA_PATH, DATA_PATH], [0.5, 0.5])
        assert blend_per_split == None

    def test_get_blend_and_blend_per_split_per_split_data_args_path(self):
        blend, blend_per_split = get_blend_and_blend_per_split(
            train_data_paths=[0.5, DATA_PATH, 0.5, DATA_PATH],
            valid_data_paths=[1.0, DATA_PATH],
            test_data_paths=[1.0, DATA_PATH],
        )

        assert blend == None
        assert blend_per_split == [
            ([DATA_PATH, DATA_PATH], [0.5, 0.5]),
            ([DATA_PATH], [1.0]),
            ([DATA_PATH], [1.0]),
        ]

        split_data = {
            "train": [DATA_PATH],
            "valid": [DATA_PATH],
            "test": [DATA_PATH],
        }
        split_data_path = "/opt/data/datasets/input/split_data.json"
        with open(split_data_path, "w") as f:
            json.dump(split_data, f)

        blend, blend_per_split = get_blend_and_blend_per_split(per_split_data_args_path=split_data_path)

        assert blend == None
        assert blend_per_split == [
            ([DATA_PATH], None),
            ([DATA_PATH], None),
            ([DATA_PATH], None),
        ]
