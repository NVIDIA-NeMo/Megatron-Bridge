# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import imageio
import numpy as np


def save_video(
    grid: np.ndarray,
    fps: int,
    H: int,
    W: int,
    video_save_quality: int,
    video_save_path: str,
    caption: str = None,
):
    ffmpeg_params = ["-s", f"{W}x{H}"]

    # Add caption as metadata if provided
    if caption is not None:
        ffmpeg_params.extend(
            [
                "-metadata",
                f"description={caption}",
            ]
        )

    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ffmpeg_params,
        "output_params": ["-f", "mp4"],
    }

    imageio.mimsave(video_save_path, grid, "mp4", **kwargs)
