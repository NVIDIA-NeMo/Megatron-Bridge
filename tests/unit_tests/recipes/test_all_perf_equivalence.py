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

"""Comprehensive equivalence test: old perf pipeline vs new flat perf recipes.

For every model family, verifies that the old path (scripts/performance/configs/)
and the new path (src/megatron/bridge/recipes/<family>/<model>_perf.py) produce
identical ConfigContainer objects.

Run standalone (no conftest needed):
    uv run python -m pytest tests/unit_tests/recipes/test_all_perf_equivalence.py -v --noconftest
"""

import re
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest


_PERF_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "performance")
if _PERF_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _PERF_SCRIPTS_DIR)

from utils.utils import get_perf_optimized_recipe

from megatron.bridge.recipes import deepseek, gpt_oss, kimi, llama, nemotronh, qwen, qwen_vl


# ── Precision mapping (new short name -> old path name) ─────────────────

_PREC_NEW_TO_OLD = {
    "bf16": "bf16",
    "fp8cs": "fp8_cs",
    "fp8mx": "fp8_mx",
    "nvfp4": "nvfp4",
}

# ── Model -> family mapping ─────────────────────────────────────────────

_MODEL_TO_FAMILY = {
    "llama3_8b": "llama",
    "llama3_70b": "llama",
    "llama31_405b": "llama",
    "qwen3_235b_a22b": "qwen",
    "qwen3_30b_a3b": "qwen",
    "deepseek_v3": "deepseek",
    "nemotronh_56b": "nemotronh",
    "nemotron_3_nano": "nemotronh",
    "kimi_k2": "kimi",
    "gpt_oss_120b": "gpt_oss",
    "qwen3_vl_235b_a22b": "qwen_vl",
    "qwen3_vl_30b_a3b": "qwen_vl",
}

# H100 uses fp8_sc not fp8_cs for deepseek old path
_DEEPSEEK_H100_PREC_MAP = {
    "fp8cs": "fp8_sc",
    "fp8mx": "fp8_mx",
}

# Kimi H100 uses fp8_sc not fp8_cs
_KIMI_H100_PREC_MAP = {
    "fp8cs": "fp8_sc",
}


def _parse_recipe_name(name: str):
    """Parse a flat recipe function name into (model, task, variant, num_gpus, gpu, prec).

    Examples:
        llama3_70b_pretrain_64gpu_gb300_bf16_config -> (llama3_70b, pretrain, v1, 64, gb300, bf16)
        llama3_70b_pretrain_v2_64gpu_gb300_bf16_config -> (llama3_70b, pretrain, v2, 64, gb300, bf16)
        llama3_8b_sft_8gpu_gb200_bf16_config -> (llama3_8b, sft, v1, 8, gb200, bf16)
        llama3_70b_lora_8gpu_gb300_bf16_config -> (llama3_70b, lora, v1, 8, gb300, bf16)
    """
    name = name.removesuffix("_config")

    m = re.match(
        r"^(.+?)_(pretrain|sft|lora)(?:_(v2))?_(\d+)gpu_(\w+?)_(bf16|fp8cs|fp8mx|nvfp4)$",
        name,
    )
    if not m:
        raise ValueError(f"Cannot parse recipe name: {name}")

    model, task, variant, num_gpus, gpu, prec = m.groups()
    variant = variant or "v1"
    return model, task, variant, int(num_gpus), gpu, prec


def _get_old_recipe(model, task, gpu, prec_short, variant):
    """Build ConfigContainer via the old perf pipeline."""
    family = _MODEL_TO_FAMILY[model]
    prec = _PREC_NEW_TO_OLD[prec_short]

    # DeepSeek H100 uses fp8_sc naming in old path
    if family == "deepseek" and gpu == "h100" and prec_short in _DEEPSEEK_H100_PREC_MAP:
        prec = _DEEPSEEK_H100_PREC_MAP[prec_short]
    if family == "kimi" and gpu == "h100" and prec_short in _KIMI_H100_PREC_MAP:
        prec = _KIMI_H100_PREC_MAP[prec_short]

    return get_perf_optimized_recipe(
        model_family_name=family,
        model_recipe_name=model,
        train_task=task,
        gpu=gpu,
        compute_dtype=prec,
        mock=True,
        config_variant=variant,
    )


# ── Comparison logic ────────────────────────────────────────────────────

_MODEL_FIELDS = [
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "sequence_parallel",
    "seq_length",
    "cuda_graph_impl",
    "cuda_graph_scope",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "cpu_offloading",
    "cpu_offloading_weights",
    "cpu_offloading_num_layers",
    "init_model_with_meta_device",
    "gradient_accumulation_fusion",
    "should_pad_vocab",
    "apply_rope_fusion",
    "cross_entropy_fusion_impl",
    "use_te_rng_tracker",
]

_DDP_FIELDS = [
    "use_megatron_fsdp",
    "data_parallel_sharding_strategy",
    "keep_fp8_transpose_cache",
    "average_in_collective",
    "nccl_ub",
    "fsdp_manual_registration",
    "fsdp_double_buffer",
    "suggested_communication_unit_size",
    "check_for_nan_in_grad",
    "check_for_large_grads",
    "grad_reduce_in_fp32",
    "overlap_grad_reduce",
    "overlap_param_gather",
]


def _compare_dataclass(old, new, path=""):
    diffs = []
    if not is_dataclass(old) or not is_dataclass(new):
        if old != new:
            diffs.append(f"{path}: {old!r} != {new!r}")
        return diffs
    for f in fields(old):
        old_val = getattr(old, f.name, None)
        new_val = getattr(new, f.name, None)
        fp = f"{path}.{f.name}" if path else f.name
        if is_dataclass(old_val) and is_dataclass(new_val):
            diffs.extend(_compare_dataclass(old_val, new_val, fp))
        elif old_val != new_val:
            diffs.append(f"{fp}: {old_val!r} != {new_val!r}")
    return diffs


def _assert_configs_equal(old, new):
    diffs = []
    for f in _MODEL_FIELDS:
        ov = getattr(old.model, f, "MISSING")
        nv = getattr(new.model, f, "MISSING")
        if ov != nv:
            diffs.append(f"model.{f}: old={ov!r}  new={nv!r}")

    for f in ["train_iters", "global_batch_size", "micro_batch_size", "eval_iters"]:
        if getattr(old.train, f) != getattr(new.train, f):
            diffs.append(f"train.{f}: old={getattr(old.train, f)!r}  new={getattr(new.train, f)!r}")

    for f in _DDP_FIELDS:
        ov = getattr(old.ddp, f, "MISSING")
        nv = getattr(new.ddp, f, "MISSING")
        if ov != nv:
            diffs.append(f"ddp.{f}: old={ov!r}  new={nv!r}")

    for f in ["lr_decay_iters", "lr_warmup_iters"]:
        if getattr(old.scheduler, f) != getattr(new.scheduler, f):
            diffs.append(f"scheduler.{f}: old={getattr(old.scheduler, f)!r}  new={getattr(new.scheduler, f)!r}")

    if old.checkpoint.save != new.checkpoint.save:
        diffs.append(f"checkpoint.save: old={old.checkpoint.save!r}  new={new.checkpoint.save!r}")

    if old.logger.log_interval != new.logger.log_interval:
        diffs.append(f"logger.log_interval: old={old.logger.log_interval!r}  new={new.logger.log_interval!r}")

    if old.tokenizer.vocab_size != new.tokenizer.vocab_size:
        diffs.append(f"tokenizer.vocab_size: old={old.tokenizer.vocab_size!r}  new={new.tokenizer.vocab_size!r}")

    mp_diffs = _compare_dataclass(old.mixed_precision, new.mixed_precision, "mixed_precision")
    diffs.extend(mp_diffs)

    if old.comm_overlap is not None and new.comm_overlap is not None:
        diffs.extend(_compare_dataclass(old.comm_overlap, new.comm_overlap, "comm_overlap"))
    elif (old.comm_overlap is None) != (new.comm_overlap is None):
        diffs.append(f"comm_overlap: old={old.comm_overlap!r}  new={new.comm_overlap!r}")

    diffs.extend(_compare_dataclass(old.dataset, new.dataset, "dataset"))
    diffs.extend(_compare_dataclass(old.optimizer, new.optimizer, "optimizer"))
    diffs.extend(_compare_dataclass(old.dist, new.dist, "dist"))
    diffs.extend(_compare_dataclass(old.profiling, new.profiling, "profiling"))

    if old.peft is not None and new.peft is not None:
        diffs.extend(_compare_dataclass(old.peft, new.peft, "peft"))
    elif (old.peft is None) != (new.peft is None):
        diffs.append(f"peft: old={old.peft!r}  new={new.peft!r}")

    if old.rerun_state_machine.check_for_nan_in_loss != new.rerun_state_machine.check_for_nan_in_loss:
        diffs.append(
            f"rerun_state_machine.check_for_nan_in_loss: "
            f"old={old.rerun_state_machine.check_for_nan_in_loss!r}  "
            f"new={new.rerun_state_machine.check_for_nan_in_loss!r}"
        )

    if diffs:
        raise AssertionError("ConfigContainer mismatch:\n" + "\n".join(f"  {d}" for d in diffs))


# ── Collect all new recipe functions from each module ───────────────────


def _collect_perf_fns(module, prefix_list):
    """Collect all functions whose name matches a perf recipe pattern."""
    fns = []
    for name in sorted(dir(module)):
        if not name.endswith("_config") or not callable(getattr(module, name)):
            continue
        if not any(name.startswith(p) for p in prefix_list):
            continue
        # Must match the gpu pattern
        if not re.search(r"_\d+gpu_", name):
            continue
        fns.append((name, getattr(module, name)))
    return fns


_LLAMA_PRETRAIN_FNS = _collect_perf_fns(
    llama,
    [
        "llama3_70b_pretrain_",
        "llama3_8b_pretrain_",
        "llama31_405b_pretrain_",
    ],
)
_LLAMA_SFT_FNS = _collect_perf_fns(
    llama,
    [
        "llama3_8b_sft_",
        "llama3_70b_sft_",
    ],
)
_LLAMA_LORA_FNS = _collect_perf_fns(
    llama,
    [
        "llama3_70b_lora_",
    ],
)
_DEEPSEEK_FNS = _collect_perf_fns(deepseek, ["deepseek_v3_pretrain_"])
_QWEN_FNS = _collect_perf_fns(qwen, ["qwen3_235b_a22b_pretrain_", "qwen3_30b_a3b_pretrain_"])
_NEMOTRONH_FNS = _collect_perf_fns(nemotronh, ["nemotronh_56b_pretrain_", "nemotron_3_nano_pretrain_"])
_KIMI_FNS = _collect_perf_fns(kimi, ["kimi_k2_pretrain_"])
_GPT_OSS_FNS = _collect_perf_fns(gpt_oss, ["gpt_oss_120b_pretrain_"])
_QWEN_VL_FNS = _collect_perf_fns(qwen_vl, ["qwen3_vl_235b_a22b_pretrain_", "qwen3_vl_30b_a3b_pretrain_"])


def _make_cases(fn_list):
    cases = []
    for name, fn in fn_list:
        model, task, variant, _num_gpus, gpu, prec = _parse_recipe_name(name)
        cases.append(pytest.param(fn, model, task, variant, gpu, prec, id=name.removesuffix("_config")))
    return cases


# ── Parametrized tests ──────────────────────────────────────────────────


@pytest.mark.unit
class TestLlamaPretrain:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_LLAMA_PRETRAIN_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


@pytest.mark.unit
class TestLlamaSft:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_LLAMA_SFT_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


@pytest.mark.unit
class TestLlamaLora:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_LLAMA_LORA_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


@pytest.mark.unit
class TestDeepseekV3:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_DEEPSEEK_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


@pytest.mark.unit
class TestQwen3Moe:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_QWEN_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


@pytest.mark.unit
class TestNemotronH:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_NEMOTRONH_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


@pytest.mark.unit
class TestKimiK2:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_KIMI_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


@pytest.mark.unit
class TestGptOss:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_GPT_OSS_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


@pytest.mark.unit
class TestQwen3Vl:
    @pytest.mark.parametrize("new_fn,model,task,variant,gpu,prec", _make_cases(_QWEN_VL_FNS))
    def test_equivalence(self, new_fn, model, task, variant, gpu, prec):
        old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
        new_cfg = new_fn()
        _assert_configs_equal(old_cfg, new_cfg)


# ── Standalone runner ───────────────────────────────────────────────────

if __name__ == "__main__":
    all_fns = (
        _LLAMA_PRETRAIN_FNS
        + _LLAMA_SFT_FNS
        + _LLAMA_LORA_FNS
        + _DEEPSEEK_FNS
        + _QWEN_FNS
        + _NEMOTRONH_FNS
        + _KIMI_FNS
        + _GPT_OSS_FNS
        + _QWEN_VL_FNS
    )
    passed = failed = skipped = 0
    failures = []

    for name, fn in all_fns:
        try:
            model, task, variant, num_gpus, gpu, prec = _parse_recipe_name(name)
        except ValueError as e:
            print(f"SKIP {name}: {e}")
            skipped += 1
            continue

        try:
            old_cfg = _get_old_recipe(model, task, gpu, prec, variant)
            new_cfg = fn()
            _assert_configs_equal(old_cfg, new_cfg)
            print(f"PASS {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failed += 1
            failures.append((name, str(e)))

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed + failed + skipped}  PASSED: {passed}  FAILED: {failed}  SKIPPED: {skipped}")
    if failures:
        print("\nFailures:")
        for n, err in failures:
            print(f"  {n}: {err[:200]}")
    print(f"{'=' * 60}")
    sys.exit(1 if failed else 0)
