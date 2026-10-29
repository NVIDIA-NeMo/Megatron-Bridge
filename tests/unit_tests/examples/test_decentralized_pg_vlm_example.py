# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Contract test for the decentralized Qwen3-VL example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "examples" / "training_features" / "decentralized_pg" / "pretrain_qwen3_vl_simple.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("decentralized_pg_vlm_example_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decentralized_qwen3_vl_example_applies_overrides_after_parameterless_recipe(monkeypatch):
    module = _load_example_module()
    config = SimpleNamespace(
        model=SimpleNamespace(
            expert_model_parallel_size=4,
            seq_length=4096,
            share_embeddings_and_output_weights=True,
        ),
        dataset=SimpleNamespace(seq_length=4096),
        train=SimpleNamespace(train_iters=300_000, global_batch_size=32, micro_batch_size=2),
        scheduler=SimpleNamespace(lr_warmup_iters=500, lr_decay_iters=300_000),
        dist=SimpleNamespace(use_decentralized_pg=False, use_gloo_process_groups=True),
    )
    recipe_calls = []
    pretrain_calls = []

    monkeypatch.setattr(module, "qwen3_vl_30b_a3b_pretrain_mock_config", lambda: recipe_calls.append(()) or config)
    monkeypatch.setattr(module, "pretrain", lambda **kwargs: pretrain_calls.append(kwargs))
    monkeypatch.setattr(module.torch.distributed, "is_initialized", lambda: False)

    try:
        module.main()
    finally:
        sys.modules.pop(module.__name__, None)

    assert recipe_calls == [()]
    assert config.model.expert_model_parallel_size == 8
    assert config.model.seq_length == config.dataset.seq_length == 1024
    assert config.train.train_iters == 100
    assert config.train.global_batch_size == 32
    assert config.train.micro_batch_size == 1
    assert config.scheduler.lr_warmup_iters == 10
    assert config.scheduler.lr_decay_iters == 100
    assert config.model.share_embeddings_and_output_weights is False
    assert config.dist.use_decentralized_pg is True
    assert config.dist.use_gloo_process_groups is False
    assert pretrain_calls == [{"config": config, "forward_step_func": module.forward_step}]
