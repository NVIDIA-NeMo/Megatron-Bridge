import importlib
from typing import Callable, List

import pytest


def _safe_overrides_for(name: str) -> dict:
    overrides = {
        "name": f"unit_{name}",
        "dir": ".",
        "mock": True,
        "train_iters": 10,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "seq_length": 64,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "context_parallelism": 1,
        "use_null_tokenizer": True,
    }

    # Large models/variants may set additional flags in recipes; keep harmless defaults
    lname = name.lower()
    if "70b" in lname or "405b" in lname:
        overrides.update(
            {
                "virtual_pipeline_parallelism": None,
                "sequence_parallelism": True,
            }
        )

    return overrides


class _FakeModelCfg:
    def finalize(self):
        return None


class _FakeBridge:
    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        return _FakeBridge()


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None

    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1
    assert cfg.dataset.sequence_length >= 1


def test_each_llama_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    if overrides.get("use_null_tokenizer") and hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1


# Dynamic parametrization from megatron.bridge.recipes.llama


def pytest_generate_tests(metafunc):
    if "recipe_func" in metafunc.fixturenames:
        llama_module = importlib.import_module("megatron.bridge.recipes.llama")
        exported: List[str] = getattr(llama_module, "__all__", [])
        funcs: List[Callable] = []
        ids: List[str] = []
        for name in exported:
            obj = getattr(llama_module, name, None)
            if callable(obj):
                funcs.append(obj)
                ids.append(name)
        if funcs:
            metafunc.parametrize("recipe_func", funcs, ids=ids)
