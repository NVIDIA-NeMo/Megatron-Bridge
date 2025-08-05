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

from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data


class TestDataPackedSequence:
    def test_prepare_packed_sequence_data(self):
        input_path = "/home/data/toy/toy_markdown_document/cache/GPTDataset_indices/00f330e10f00bd69bdd77f3dc540c16d-GPTDataset-train-document_index.npy"
        output_path- "/home/data/toy/just_a_test"
        