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

import pytest

from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


@pytest.mark.parametrize(
    "hf_path",
    [
        "Qwen/attacker_processor",
        "nvidia/local-model",
        "google/custom-tokenizer",
        "meta-llama/custom-dataset",
        "./Qwen/attacker_processor",
        "/tmp/attacker_processor",
    ],
)
def test_is_safe_repo_defaults_to_no_remote_code(hf_path):
    """Test that omitted trust_remote_code never enables remote code."""
    assert is_safe_repo(hf_path=hf_path, trust_remote_code=None) is False


def test_is_safe_repo_explicit_trust_remote_code_wins():
    """Test that explicit trust_remote_code values are honored."""
    assert is_safe_repo(hf_path="attacker/repo", trust_remote_code=True) is True
    assert is_safe_repo(hf_path="Qwen/model", trust_remote_code=False) is False
