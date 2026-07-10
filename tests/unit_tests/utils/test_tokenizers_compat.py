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

from tokenizers import processors


class TestTokenizersCompat:
    def test_roberta_processing_accepts_transformers_kwargs(self):
        """The installed tokenizers must accept the exact kwargs transformers' CLIPTokenizer passes."""
        # The assertion IS that this construction does not raise TypeError on the
        # cls= kwarg (renamed to cls_token in tokenizers 0.23). A returned instance
        # confirms the compatible API is installed.
        processors.RobertaProcessing(
            sep=("<|endoftext|>", 49407),
            cls=("<|startoftext|>", 49406),
            add_prefix_space=False,
            trim_offsets=False,
        )
