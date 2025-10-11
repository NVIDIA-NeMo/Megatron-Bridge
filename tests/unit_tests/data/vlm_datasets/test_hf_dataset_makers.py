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
from types import SimpleNamespace

import megatron.bridge.data.vlm_datasets.hf_dataset_makers as makers


class _DummyDataset(list):
    def remove_columns(self, cols):  # match datasets API used
        return self


def _monkeypatch_load_dataset(monkeypatch, rows):
    def _fake_load_dataset(path_or_dataset, split="train", **kwargs):  # noqa: ARG001 - interface
        return _DummyDataset(rows)

    monkeypatch.setattr(makers, "load_dataset", _fake_load_dataset)


def test_make_rdr_dataset(monkeypatch):
    rows = [
        {"image": SimpleNamespace(), "text": "a cat"},
        {"image": SimpleNamespace(), "text": "a dog"},
    ]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_rdr_dataset()
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["conversation"][0]["content"][0]["type"] == "image"


def test_make_cord_v2_dataset_variants(monkeypatch):
    gt = {"gt_parses": [{"x": 1}, {"y": 2}]}
    rows = [{"image": SimpleNamespace(), "ground_truth": json.dumps(gt)}]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_cord_v2_dataset()
    assert out and out[0]["conversation"][1]["role"] == "assistant"

    # alt structure with single gt_parse
    gt2 = {"gt_parse": {"a": 1}}
    rows2 = [{"image": SimpleNamespace(), "ground_truth": json.dumps(gt2)}]
    _monkeypatch_load_dataset(monkeypatch, rows2)
    out2 = makers.make_cord_v2_dataset()
    assert out2 and "<s_a>" in makers.json2token({"a": 1}, sort_json_key=True)


def test_make_medpix_dataset(monkeypatch):
    rows = [{"image_id": SimpleNamespace(), "question": "q?", "answer": "a"}]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_medpix_dataset()
    assert out and out[0]["conversation"][1]["content"][0]["type"] == "text"


def test_make_cv17_dataset(monkeypatch):
    rows = [{"audio": {"array": [0.1, 0.2], "sampling_rate": 16000}, "transcription": "hello"}]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_cv17_dataset()
    assert out and isinstance(out[0]["audio"], tuple)
