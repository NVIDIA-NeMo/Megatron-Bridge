"""Behavioral tests for library + explicit benchmark configuration composition."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "performance"))

from argument_parser import parse_cli_args
from perf_benchmark_overrides import prepare_perf_benchmark_config, validate_perf_benchmark_dependencies

from megatron.bridge.recipes.gpt.h100.vanilla_gpt import vanilla_gpt_pretrain_1gpu_h100_bf16_config
from megatron.bridge.recipes.llama.h100.execution import llama3_8b_pretrain_bf16, llama3_8b_pretrain_fp8
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.training.perf_benchmark_overrides import build_perf_benchmark_config


NAME = "llama3_8b_pretrain_8gpu_h100_bf16_config"


@pytest.fixture
def library_cfg(monkeypatch):
    """Keep unit tests offline while retaining real, strict config objects."""
    from megatron.bridge.recipes.llama import llama3

    cfg = vanilla_gpt_pretrain_1gpu_h100_bf16_config()
    cfg.mixed_precision = bf16_mixed()
    cfg.model.cuda_graph_impl = "none"
    monkeypatch.setattr(llama3, "llama3_8b_pretrain_config", lambda: copy.deepcopy(cfg))
    return cfg


def args_for(tmp_path, *extra):
    args, unknown = parse_cli_args().parse_known_args(
        [
            "-m",
            "llama",
            "-mr",
            "llama3_8b",
            "-g",
            "h100",
            "-ng",
            "8",
            "-c",
            "bf16",
            "--save_config_filepath",
            str(tmp_path / "cfg.yaml"),
            *extra,
        ]
    )
    assert not unknown
    return args


@pytest.mark.unit
def test_benchmark_changes_are_explicit_and_do_not_mutate_library(library_cfg):
    before = copy.deepcopy(library_cfg)
    cfg = build_perf_benchmark_config(NAME)
    assert cfg.train.train_iters == 50
    assert cfg.scheduler.lr_decay_iters == 50
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.checkpoint.save is None
    assert cfg.ddp.check_for_nan_in_grad is False
    assert library_cfg.train == before.train
    assert library_cfg.optimizer == before.optimizer
    assert library_cfg.checkpoint == before.checkpoint
    assert library_cfg.rng == before.rng


@pytest.mark.unit
def test_execution_selection_preserves_training_policy(library_cfg):
    protected = (
        "train",
        "optimizer",
        "mixed_precision",
        "dataset",
        "scheduler",
        "checkpoint",
        "rng",
        "rerun_state_machine",
    )
    before = {key: copy.deepcopy(getattr(library_cfg, key)) for key in protected}
    llama3_8b_pretrain_bf16(library_cfg)
    for key in protected:
        assert getattr(library_cfg, key) == before[key], key
    assert library_cfg.model.context_parallel_size == 2
    assert library_cfg.ddp.check_for_nan_in_grad is True


@pytest.mark.unit
def test_fp8_layout_does_not_silently_lower_precision(library_cfg):
    with pytest.raises(ValueError, match="requires an explicit FP8"):
        llama3_8b_pretrain_fp8(library_cfg)
    assert library_cfg.mixed_precision.fp8 is None


@pytest.mark.unit
def test_dotted_overrides_win_over_flags_including_false_and_none(library_cfg, tmp_path):
    cfg = prepare_perf_benchmark_config(
        args_for(tmp_path, "--max_steps", "100", "--context_parallel_size", "2"),
        [
            "train.train_iters=7",
            "model.context_parallel_size=1",
            "optimizer.use_precision_aware_optimizer=False",
            "checkpoint.load=null",
            "ddp.check_for_nan_in_grad=True",
            "env_vars.CUDA_DEVICE_MAX_CONNECTIONS=7",
        ],
        name=NAME,
    )
    assert cfg.train.train_iters == 7
    assert cfg.scheduler.lr_decay_iters == 50  # explicitly selected benchmark horizon
    assert cfg.model.context_parallel_size == 1
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.checkpoint.load is None
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 7


@pytest.mark.unit
def test_topology_does_not_weak_scale_batch(library_cfg, tmp_path):
    cfg = prepare_perf_benchmark_config(
        args_for(tmp_path, "--context_parallel_size", "1"),
        [],
        name=NAME,
    )
    assert cfg.train.global_batch_size == 128
    with pytest.raises(ValueError, match="GBS=3"):
        prepare_perf_benchmark_config(args_for(tmp_path), ["train.global_batch_size=3"], name=NAME)


@pytest.mark.unit
def test_incompatible_expert_grid_is_rejected(library_cfg):
    library_cfg.model.expert_model_parallel_size = 3
    with pytest.raises(ValueError, match="expert-parallel grid"):
        validate_perf_benchmark_dependencies(library_cfg, name=NAME, num_gpus=8)


@pytest.mark.unit
def test_fsdp_does_not_erase_selected_checkpoint(library_cfg, tmp_path):
    cfg = prepare_perf_benchmark_config(
        args_for(tmp_path, "--use_megatron_fsdp", "true", "--load_dir", "/checkpoints/resume"),
        [],
        name=NAME,
    )
    assert cfg.checkpoint.load == "/checkpoints/resume"


@pytest.mark.unit
def test_graph_toggle_requires_explicit_check_policy(library_cfg, tmp_path):
    with pytest.raises(ValueError, match="explicit compatible loss-check policy"):
        prepare_perf_benchmark_config(
            args_for(tmp_path, "--cuda_graph_impl", "local", "--cuda_graph_scope", "full_iteration"),
            ["rerun_state_machine.check_for_nan_in_loss=True"],
            name=NAME,
        )


@pytest.mark.unit
@pytest.mark.parametrize("override", ["optimizer.optimizer=sgd", "optimizer={optimizer:sgd}"])
def test_sgd_does_not_inherit_adam_launch_default(library_cfg, tmp_path, override):
    cfg = prepare_perf_benchmark_config(args_for(tmp_path), [override], name=NAME)
    assert cfg.optimizer.optimizer == "sgd"
    assert cfg.optimizer.use_precision_aware_optimizer is False
    cfg.optimizer.finalize()


@pytest.mark.unit
def test_incompatible_explicit_optimizer_selection_is_rejected(library_cfg, tmp_path):
    with pytest.raises(ValueError, match="Precision-aware optimizer requires Adam"):
        prepare_perf_benchmark_config(
            args_for(tmp_path),
            ["optimizer.optimizer=sgd", "optimizer.use_precision_aware_optimizer=True"],
            name=NAME,
        )


@pytest.mark.unit
def test_dictionary_optimizer_selection_is_final(library_cfg, tmp_path):
    cfg = prepare_perf_benchmark_config(
        args_for(tmp_path),
        ["optimizer={use_precision_aware_optimizer:false}"],
        name=NAME,
    )
    assert cfg.optimizer.use_precision_aware_optimizer is False


@pytest.mark.unit
@pytest.mark.parametrize("protect_env", [False, True])
def test_nccl_environment_uses_final_selection(library_cfg, tmp_path, protect_env):
    overrides = ["ddp.nccl_ub=False"]
    if protect_env:
        overrides.append("env_vars.NCCL_NVLS_ENABLE=7")
    cfg = prepare_perf_benchmark_config(
        args_for(tmp_path, "--nccl_ub", "true"),
        overrides,
        name=NAME,
    )
    assert cfg.ddp.nccl_ub is False
    assert cfg.env_vars["NCCL_NVLS_ENABLE"] == (7 if protect_env else 0)
    assert cfg.env_vars.get("NCCL_CTA_POLICY") != 1


@pytest.mark.unit
def test_dotted_topology_derives_same_environment_as_flags(library_cfg, tmp_path):
    name = "llama3_8b_pretrain_8gpu_h100_fp8cs_config"
    flag_cfg = prepare_perf_benchmark_config(
        args_for(tmp_path, "-c", "fp8_cs", "--tensor_model_parallel_size", "2"),
        ["model.sequence_parallel=True"],
        name=name,
    )
    dotted_cfg = prepare_perf_benchmark_config(
        args_for(tmp_path, "-c", "fp8_cs"),
        ["model.tensor_model_parallel_size=2", "model.sequence_parallel=True"],
        name=name,
    )
    assert flag_cfg.env_vars == dotted_cfg.env_vars
    assert dotted_cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 1


@pytest.mark.unit
def test_unselected_tuple_and_nested_fields_survive_composition(library_cfg):
    library_cfg.model.cp_comm_type = ("p2p", "a2a")
    before = copy.deepcopy(library_cfg.model.cp_comm_type)
    cfg = build_perf_benchmark_config(NAME)
    assert cfg.model.cp_comm_type == before
    assert type(cfg.model.cp_comm_type) is tuple


@pytest.mark.unit
def test_moe_dotted_parallelism_and_overlap_are_final(library_cfg, tmp_path, monkeypatch):
    from megatron.bridge.recipes.qwen.h100 import qwen3_moe

    library_cfg.model.num_moe_experts = 128
    monkeypatch.setattr(qwen3_moe, "qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config", lambda: copy.deepcopy(library_cfg))
    cfg = prepare_perf_benchmark_config(
        args_for(
            tmp_path,
            "-m",
            "qwen",
            "-mr",
            "qwen3_30b_a3b",
            "-ng",
            "16",
            "-c",
            "fp8_cs",
            "--expert_model_parallel_size",
            "16",
            "--moe_a2a_overlap",
            "true",
        ),
        ["model.expert_model_parallel_size=8", "comm_overlap.overlap_moe_expert_parallel_comm=False"],
        name="qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config",
    )
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.train.global_batch_size == 1024
    assert cfg.mixed_precision.fp8 is not None


@pytest.mark.unit
def test_explicit_dependent_setting_is_validated_not_overwritten(library_cfg, tmp_path):
    with pytest.raises(ValueError, match="Explicit ddp.average_in_collective=True conflicts"):
        prepare_perf_benchmark_config(
            args_for(tmp_path, "--nccl_ub", "true"),
            ["ddp.average_in_collective=True"],
            name=NAME,
        )


@pytest.mark.unit
def test_benchmark_module_imports_in_a_fresh_process():
    scripts = str(Path(__file__).resolve().parents[4] / "scripts" / "performance")
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {scripts!r}); import perf_benchmark_overrides"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
