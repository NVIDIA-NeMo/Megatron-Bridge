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

import argparse
import os
import tempfile

import torch

from megatron.bridge.diffusion.models.wan.inference import utils as inf_utils


def test_str2bool_variants_and_errors():
    true_vals = ["yes", "true", "t", "y", "1", "TRUE", "Yes"]
    false_vals = ["no", "false", "f", "n", "0", "FALSE", "No"]
    for v in true_vals:
        assert inf_utils.str2bool(v) is True
    for v in false_vals:
        assert inf_utils.str2bool(v) is False
    assert inf_utils.str2bool(True) is True
    assert inf_utils.str2bool(False) is False
    try:
        inf_utils.str2bool("maybe")
    except argparse.ArgumentTypeError:
        pass
    else:
        assert False, "Expected argparse.ArgumentTypeError for invalid boolean string"


def test_cache_image_writes_file(tmp_path):
    # Small 3x8x8 image
    img = torch.rand(3, 8, 8)
    out_path = tmp_path / "test.png"
    saved = inf_utils.cache_image(img, str(out_path), nrow=1, normalize=False, value_range=(0.0, 1.0), retry=1)
    assert saved == str(out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_cache_video_uses_writer_and_returns_path(monkeypatch):
    # Stub imageio.get_writer to avoid codec dependency
    calls = {"frames": 0, "path": None}

    class _DummyWriter:
        def __init__(self, path, fps=None, codec=None, quality=None):
            calls["path"] = path

        def append_data(self, frame):
            calls["frames"] += 1

        def close(self):
            pass

    monkeypatch.setattr(
        inf_utils.imageio, "get_writer", lambda path, fps, codec, quality: _DummyWriter(path, fps, codec, quality)
    )

    # Stub make_grid to return a fixed CHW tensor regardless of input
    def _fake_make_grid(x, nrow, normalize, value_range):
        return torch.rand(3, 4, 5)

    monkeypatch.setattr(inf_utils.torchvision.utils, "make_grid", _fake_make_grid)

    # Build a tensor whose unbind(2) yields 2 slices so we expect 2 frames written
    vid = torch.rand(3, 3, 2, 2)  # shape chosen to exercise unbind(2)
    with tempfile.TemporaryDirectory() as td:
        out_file = os.path.join(td, "out.mp4")
        result = inf_utils.cache_video(
            vid, save_file=out_file, fps=5, suffix=".mp4", nrow=1, normalize=False, value_range=(0.0, 1.0), retry=1
        )
        assert result == out_file
        assert calls["path"] == out_file
        assert calls["frames"] == vid.shape[2]  # frames equal to number of unbinds on dim=2
