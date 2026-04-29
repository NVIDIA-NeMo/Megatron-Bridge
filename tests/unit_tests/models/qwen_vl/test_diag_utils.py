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

import os
from unittest.mock import patch

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    thd_diag_align_enabled,
    thd_diag_enabled,
    thd_diag_mrope_enabled,
)


class TestDiagSwitches:
    @patch.dict(os.environ, {"THD_DIAG": "1"}, clear=False)
    def test_thd_diag_enabled_true(self):
        assert thd_diag_enabled() is True

    @patch.dict(os.environ, {"THD_DIAG": "false"}, clear=False)
    def test_thd_diag_enabled_false(self):
        assert thd_diag_enabled() is False

    @patch.dict(os.environ, {"THD_DIAG_ALIGN": "1"}, clear=False)
    def test_thd_diag_align_enabled_true(self):
        assert thd_diag_align_enabled() is True

    @patch.dict(os.environ, {"THD_DIAG_ALIGN": "False"}, clear=False)
    def test_thd_diag_align_enabled_false(self):
        assert thd_diag_align_enabled() is False

    @patch.dict(os.environ, {"THD_DIAG_MROPE": "1"}, clear=False)
    def test_thd_diag_mrope_enabled_true(self):
        assert thd_diag_mrope_enabled() is True

    @patch.dict(os.environ, {"THD_DIAG_MROPE": ""}, clear=False)
    def test_thd_diag_mrope_enabled_false(self):
        assert thd_diag_mrope_enabled() is False
