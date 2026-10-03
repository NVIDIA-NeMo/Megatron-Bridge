# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import sample_video_frames_to_data_urls


pytestmark = pytest.mark.unit


def test_sample_video_frames_preserves_source_timing_metadata(monkeypatch):
    class _Frame:
        def asnumpy(self):
            return np.zeros((2, 3, 3), dtype=np.uint8)

    class _VideoReader:
        def __len__(self):
            return 30

        def get_avg_fps(self):
            return 10.0

        def __getitem__(self, index):  # noqa: ARG002
            return _Frame()

    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(VideoReader=lambda path: _VideoReader()))

    frame_urls, metadata = sample_video_frames_to_data_urls("video.mp4", fps=2)

    assert len(frame_urls) == 6
    assert metadata.total_num_frames == 30
    assert metadata.fps == 10.0
    assert metadata.duration == 3.0
    assert metadata.video_backend == "decord"
    assert metadata.frames_indices == [0, 6, 12, 17, 23, 29]
