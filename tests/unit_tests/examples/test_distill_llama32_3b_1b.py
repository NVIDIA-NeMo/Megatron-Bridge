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
import pathlib
import sys
from types import SimpleNamespace

import pytest

from megatron.bridge.training.tokenizers.config import TokenizerConfig


pytestmark = pytest.mark.unit

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "distillation" / "llama" / "distill_llama32_3b-1b.py"


def _load_example_module(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _recipe_config():
    tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=32000, use_tokenizer_vocab_size=True)
    return SimpleNamespace(model=None, tokenizer=tokenizer)


def test_main_uses_student_hf_tokenizer(monkeypatch):
    """main() hands distill a Hugging Face tokenizer matching the student checkpoint."""
    name = "distill_llama32_3b_1b_example"
    try:
        module = _load_example_module(name)
        hf_ids = []
        captured = {}

        class _FakeAutoBridge:
            @staticmethod
            def from_hf_pretrained(hf_id):
                hf_ids.append(hf_id)
                return SimpleNamespace(to_megatron_provider=lambda load_weights: SimpleNamespace(hf_id=hf_id))

        monkeypatch.setattr(module, "AutoBridge", _FakeAutoBridge)
        monkeypatch.setattr(module, "llama32_1b_pretrain_config", _recipe_config)
        monkeypatch.setattr(module, "llama32_3b_pretrain_config", _recipe_config)
        monkeypatch.setattr(module, "convert_to_distillation_provider", lambda student, teacher, kd: student)
        monkeypatch.setattr(module, "process_config_with_overrides", lambda config, **kwargs: config)
        monkeypatch.setattr(module, "get_rank_safe", lambda: 1)
        monkeypatch.setattr(module, "distill", lambda config: captured.setdefault("config", config))
        monkeypatch.setattr(
            module, "parse_cli_args", lambda: (argparse.Namespace(config_file="unused.yaml", debug=False), [])
        )

        module.main()

        cfg = captured["config"]
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model == hf_ids[0] == cfg.model.hf_id
    finally:
        sys.modules.pop(name, None)
