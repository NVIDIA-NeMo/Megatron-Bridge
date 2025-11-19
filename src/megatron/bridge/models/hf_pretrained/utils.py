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

import logging


logger = logging.getLogger(__name__)


SAFE_REPOS: list[str] = [
    "deepseek-ai",
    "gpt2",
    "google",
    "llava-hf",
    "meta-llama",
    "mistralai",
    "moonshotai",
    "nvidia",
    "openai",
    "Qwen",
]


def is_safe_repo(hf_path: str, trust_remote_code: bool | None) -> bool:
    """
    Determine whether remote code execution should be trusted for a given
    Hugging Face repository path.

    Args:
        trust_remote_code (bool): whther to define repo as safe w/o checking SAFE_REPOS.
        hf_path (str): path to HF's model or dataset.

    Returns:
        True if remote code execution is allowed; False otherwise.
    """
    if trust_remote_code is not None:
        if trust_remote_code is False:
            logger.warning(
                "`trust_remote_code=False`. Remote code may not be executed. "
                "Set `trust_remote_code=True` only if you fully trust the Hugging Face repository."
            )
        return trust_remote_code

    hf_repo = hf_path.split("/")[0]
    if hf_repo in SAFE_REPOS:
        return True

    return False
