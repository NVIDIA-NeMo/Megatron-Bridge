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

import argparse
import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "mfsdp" / "convert_checkpoints_fsdp.py"


@pytest.fixture(scope="module")
def conversion_cli():
    spec = importlib.util.spec_from_file_location("mfsdp_conversion_cli_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_parser_accepts_fsdp_dtensor(conversion_cli) -> None:
    parser = argparse.ArgumentParser()
    conversion_cli._add_common_args(parser)

    args = parser.parse_args(["--hf-model", "test/model", "--ckpt-format", "fsdp_dtensor"])

    assert args.ckpt_format == "fsdp_dtensor"


def test_common_parser_rejects_torch_dist(conversion_cli) -> None:
    parser = argparse.ArgumentParser()
    conversion_cli._add_common_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--hf-model", "test/model", "--ckpt-format", "torch_dist"])
