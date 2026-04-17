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

import importlib
import importlib.util
import pathlib
import sys
import types
import typing
from types import SimpleNamespace


if "megatron.energon" not in sys.modules:
    fake_energon = types.ModuleType("megatron.energon")
    fake_energon.WorkerConfig = object
    fake_energon.get_savable_loader = lambda *args, **kwargs: None
    fake_energon.get_train_dataset = lambda *args, **kwargs: None
    sys.modules["megatron.energon"] = fake_energon

if not hasattr(typing, "override"):
    typing.override = lambda func: func  # type: ignore[attr-defined]

if "megatron.bridge.recipes.common" not in sys.modules:
    fake_common = types.ModuleType("megatron.bridge.recipes.common")

    def _fake_sft_common_vlm():
        return SimpleNamespace(
            model=None,
            train=SimpleNamespace(train_iters=0, global_batch_size=0, micro_batch_size=0),
            optimizer=SimpleNamespace(lr=0.0),
            scheduler=SimpleNamespace(),
            dataset=SimpleNamespace(seq_length=0, hf_processor_path=None, pack_sequences_in_batch=True),
            tokenizer=SimpleNamespace(),
            checkpoint=SimpleNamespace(),
            ddp=SimpleNamespace(
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                check_for_nan_in_grad=False,
                use_distributed_optimizer=False,
                grad_reduce_in_fp32=False,
                average_in_collective=False,
                data_parallel_sharding_strategy=None,
            ),
            peft=None,
            mixed_precision=None,
        )

    fake_common._sft_common_vlm = _fake_sft_common_vlm
    sys.modules["megatron.bridge.recipes.common"] = fake_common

if "megatron.bridge.recipes.utils.optimizer_utils" not in sys.modules:
    fake_optimizer_utils = types.ModuleType("megatron.bridge.recipes.utils.optimizer_utils")

    def _fake_distributed_fused_adam_with_cosine_annealing(**kwargs):
        return SimpleNamespace(lr=kwargs["max_lr"]), SimpleNamespace()

    fake_optimizer_utils.distributed_fused_adam_with_cosine_annealing = _fake_distributed_fused_adam_with_cosine_annealing
    sys.modules["megatron.bridge.recipes.utils.optimizer_utils"] = fake_optimizer_utils

if "megatron.bridge.data.vlm_datasets.preloaded_provider" not in sys.modules:
    fake_preloaded = types.ModuleType("megatron.bridge.data.vlm_datasets.preloaded_provider")

    class _FakePreloadedVLMConversationProvider:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.pack_sequences_in_batch = False

    fake_preloaded.PreloadedVLMConversationProvider = _FakePreloadedVLMConversationProvider
    sys.modules["megatron.bridge.data.vlm_datasets.preloaded_provider"] = fake_preloaded

_RECIPE_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "src"
    / "megatron"
    / "bridge"
    / "recipes"
    / "qwen_omni"
    / "qwen3_omni.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_qwen3_omni_recipe_module", _RECIPE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_qwen3_omni_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_qwen3_omni_module)


class _FakeProvider:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_audio_model = False
        self.vit_gradient_checkpointing = False
        self.multimodal_attn_impl = "auto"
        self.transformer_impl = "transformer_engine"
        self.cuda_graph_impl = "none"
        self.attention_backend = "auto"

    def finalize(self):
        return None


class _FakeAutoBridge:
    @staticmethod
    def from_hf_pretrained(_hf_path: str):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        assert load_weights is False
        return _FakeProvider()


def test_qwen3_omni_sft_recipe_builds_config(monkeypatch):
    monkeypatch.setattr(_qwen3_omni_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_omni_module.qwen3_omni_30b_a3b_sft_config()

    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.peft is None

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_audio_model is False
    assert cfg.model.vit_gradient_checkpointing is False
    assert cfg.model.multimodal_attn_impl == "auto"

    assert cfg.dataset.seq_length == 4096
    assert cfg.dataset.pack_sequences_in_batch is False
    assert cfg.dataset.hf_processor_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    assert cfg.optimizer.lr == 5e-6


def test_qwen3_omni_preloaded_recipe_uses_preloaded_provider(monkeypatch):
    monkeypatch.setattr(_qwen3_omni_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_omni_module.qwen3_omni_30b_a3b_sft_preloaded_config()

    assert cfg.dataset is not None
    assert cfg.dataset.seq_length == 4096
    assert cfg.dataset.hf_processor_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert cfg.dataset.train_data_path is None
    assert cfg.dataset.valid_data_path is None
    assert cfg.dataset.test_data_path is None
    assert cfg.dataset.pack_sequences_in_batch is False
