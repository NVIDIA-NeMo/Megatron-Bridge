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

"""Dataset acquisition contracts of the agentic SFT example (``02_prepare_coderforge.py``)."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

SOURCE = Path(__file__).resolve().parents[3] / "examples/agentic_sft/02_prepare_coderforge.py"
SPEC = importlib.util.spec_from_file_location("agentic_sft_preparation", SOURCE)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def example():
    return {
        "trajectory_id": "trajectory-1",
        "image": "execution-environment",
        "license": "source-license",
        "messages": [
            {"role": "user", "content": "Run the test"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Check the result first.",
                "tool_calls": [{"id": "call-1", "function": {"name": "test", "arguments": '{"x":1}'}}],
            },
            {"role": "tool", "content": "passed", "tool_call_id": "call-1"},
        ],
        "tools": [{"type": "function", "function": {"name": "test"}}],
    }


@pytest.mark.parametrize("serialized", [False, True])
def test_preparation_retains_complete_trajectory_and_source_metadata(serialized):
    original = example()
    record = deepcopy(original)
    if serialized:
        record["messages"] = json.dumps(record["messages"])
        record["tools"] = json.dumps(record["tools"])
    before = deepcopy(record)
    result = prepare.canonical_record(record)
    assert record == before
    assert result["messages"] == original["messages"]
    assert result["tools"] == original["tools"]
    assert result["environment_image"] == original["image"]
    assert result["license"] == original["license"]
    assert "image" not in result


def test_task_group_is_split_consistently_for_different_trajectories():
    first = {**example(), "task_id": "shared-task"}
    second = {**first, "trajectory_id": "trajectory-2"}
    group = prepare.validation_group(first, None)
    assert group == prepare.validation_group(second, None)
    assert prepare.split_name(group, 1234, 0.05) == prepare.split_name(group, 1234, 0.05)
    assert prepare.validation_group(example(), None)[0] == "trajectory_id"
    with pytest.raises(ValueError, match="missing_task_id"):
        prepare.validation_group(example(), "task_id")


def test_invalid_serialization_does_not_expose_trajectory_text():
    record = example()
    record["messages"] = "sensitive malformed payload"
    with pytest.raises(ValueError, match="^invalid_messages_json$"):
        prepare.canonical_record(record)


def test_rejects_multimodal_content_and_missing_identifiers():
    record = example()
    record["messages"][0]["content"] = [{"type": "image"}]
    with pytest.raises(ValueError, match="nontext_content"):
        prepare.canonical_record(record)
    record = example()
    del record["trajectory_id"]
    with pytest.raises(ValueError, match="missing_trajectory_id"):
        prepare.canonical_record(record)
