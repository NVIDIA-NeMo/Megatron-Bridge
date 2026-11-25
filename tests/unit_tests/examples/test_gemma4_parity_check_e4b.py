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

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from megatron.training import global_vars


_SCRIPT = Path(__file__).parents[3] / "examples" / "models" / "gemma" / "gemma4" / "parity_check_e4b.py"


class _StopAfterInit(Exception):
    pass


def _load_script():
    spec = importlib.util.spec_from_file_location("gemma4_parity_check_e4b", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parity_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))
    (tmp_path / "pretrain_gpt.py").write_text("")
    module = _load_script()
    monkeypatch.setattr(module, "MEGATRON_LM_ROOT", str(tmp_path))
    monkeypatch.setattr(
        module,
        "_parse",
        lambda: SimpleNamespace(
            hf_dir=str(tmp_path / "hf"),
            megatron_ckpt=str(tmp_path / "ckpt"),
            atol=3.0,
            tp=1,
            bf16=True,
            mode="text",
            vl_image_tokens=module.IMAGE_NUM_TOKENS,
        ),
    )
    global_vars.unset_global_variables()
    yield module
    global_vars.unset_global_variables()


class TestGemma4ParityCheckInit:
    """Tests for the Megatron initialization in parity_check_e4b.main."""

    def test_run_config_registered_before_initialize_megatron(self, parity_module, monkeypatch):
        """initialize_megatron sees a registered run config, with a DDP section, built from the parsed args."""
        seen = {}

        def fake_initialize_megatron(*args, **kwargs):
            seen["run_config"] = global_vars.is_run_config_initialized()
            seen["args"] = global_vars.get_args()
            raise _StopAfterInit

        monkeypatch.setattr("megatron.training.initialize.initialize_megatron", fake_initialize_megatron)

        with pytest.raises(_StopAfterInit):
            parity_module.main()

        assert seen["run_config"] is True
        assert seen["args"].tensor_model_parallel_size == 1
        assert seen["args"].seq_length == parity_module.SEQ
        run_config = global_vars.get_run_config()
        assert run_config.checkpoint.load == seen["args"].load
        assert run_config.ddp is not None
