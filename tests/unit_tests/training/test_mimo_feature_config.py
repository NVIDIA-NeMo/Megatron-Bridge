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

import pytest

from megatron.bridge.training.config import MegatronMIMOFeatureConfig


@pytest.mark.unit
class TestMegatronMIMOFeatureConfig:
    def test_defaults_pack_off(self):
        cfg = MegatronMIMOFeatureConfig()
        # In-batch sequence packing is opt-in (off by default), and the pad id defaults to 0.
        assert cfg.pack_sequences_in_batch is False
        assert cfg.pad_token_id == 0
        cfg.finalize()  # default config is valid

    def test_pack_sequences_in_batch_toggle(self):
        cfg = MegatronMIMOFeatureConfig(pack_sequences_in_batch=True, pad_token_id=248044)
        assert cfg.pack_sequences_in_batch is True
        cfg.finalize()

    def test_pad_token_id_accepts_nonzero(self):
        MegatronMIMOFeatureConfig(pad_token_id=248044).finalize()

    def test_pad_token_id_rejects_negative(self):
        with pytest.raises(ValueError, match="pad_token_id"):
            MegatronMIMOFeatureConfig(pad_token_id=-1).finalize()
