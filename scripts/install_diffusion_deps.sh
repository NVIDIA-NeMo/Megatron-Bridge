# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#!/usr/bin/env bash
# Install the WAN diffusion codecs (imageio, imageio-ffmpeg, av) into an opt-in
# environment. All three carry CVEs and are excluded from the default project sync via
# [tool.uv] override-dependencies (each marked `sys_platform == 'never'`). The CI image
# installs them explicitly so offline diffusion tests do not need this helper.
#
# Install them directly with `uv pip install --no-config`: --no-config ignores the
# project's [tool.uv] config, so the `sys_platform == 'never'` override does not
# neutralize the install.
set -euo pipefail

uv pip install --no-config imageio imageio-ffmpeg av easydict
