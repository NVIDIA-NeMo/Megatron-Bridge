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

import types

import torch

from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids, json2token


class _DummyTokenizer:
    def __init__(self, added_tokens):
        # Simulate HF tokenizer.added_tokens_decoder
        self.added_tokens_decoder = added_tokens


def test_extract_skipped_token_ids_finds_known_pads():
    # Map token id -> Token object string representation
    class _TokObj:
        def __init__(self, s):
            self.s = s

        def __str__(self):
            return self.s

    added = {10: _TokObj("<|vision_pad|>"), 11: _TokObj("<image>"), 12: _TokObj("<nonpad>")}

    proc = types.SimpleNamespace(tokenizer=_DummyTokenizer(added))
    ids = extract_skipped_token_ids(proc)
    assert isinstance(ids, torch.Tensor)
    vals = set(ids.tolist())
    assert 10 in vals and 11 in vals
    assert 12 not in vals


def test_json2token_roundtrip_basic():
    obj = {"a": "x", "b": [1, 2]}
    s = json2token(obj, sort_json_key=True)
    # Contains bracketed tags and separators
    assert "<s_a>" in s and "</s_a>" in s
    assert "<s_b>" in s and "</s_b>" in s
    assert "<sep/>" in s
