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

"""Launch contracts of the agentic SFT training script (``04_train_qwen35_35b_a3b.py``).

The configuration is built on CPU from a synthetic Qwen3.5-shaped text config; nothing is trained and no weights or
tokenizer files are read.
"""

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5MoeTextConfig

from megatron.bridge.data.sft_processing import ChatSFTPreprocessingConfig
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit

if "dynamic_context_parallel" not in {field.name for field in fields(TransformerConfig)}:
    pytest.skip("The pinned Megatron-Core does not declare dynamic context parallelism.", allow_module_level=True)

_SOURCE = Path(__file__).resolve().parents[3] / "examples/agentic_sft/04_train_qwen35_35b_a3b.py"
_SPEC = importlib.util.spec_from_file_location("agentic_sft_train_under_test", _SOURCE)
train = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train
_SPEC.loader.exec_module(train)


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    """A directory holding the prepared 40-layer text config, as script 01 writes it."""
    directory = tmp_path_factory.mktemp("prepared") / "tokenizer"
    pattern = ["linear_attention"] * 3 + ["full_attention"]
    config = Qwen3_5MoeTextConfig(
        num_hidden_layers=40,
        hidden_size=2048,
        num_experts=256,
        moe_intermediate_size=512,
        layer_types=pattern * 10,
        mtp_num_hidden_layers=1,
    )
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    config.save_pretrained(directory)
    return directory


def _args(tmp_path, tokenizer_dir, *flags):
    args, overrides = train.parse_cli_args(
        [
            *flags,
            "--workspace",
            str(tmp_path),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )
    assert overrides == []
    return args


def _config(tmp_path, tokenizer_dir, mode, *flags):
    return train.build_config(_args(tmp_path, tokenizer_dir, mode, *flags))


def test_defaults_describe_the_full_model_on_four_ranks(tmp_path, tokenizer_dir):
    cfg = _config(tmp_path, tokenizer_dir, "online_dynamic_cp")
    model = cfg.model

    assert model.num_layers == 40
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (1, 1)
    assert (model.context_parallel_size, model.expert_model_parallel_size, model.expert_tensor_parallel_size) == (
        4,
        4,
        1,
    )
    assert model.seq_length == 65536
    assert model.max_seqlen_per_dp_cp_rank == 16384
    assert model.mtp_num_layers == 1
    assert model.mtp_loss_scaling_factor == 0.1
    assert model.calculate_per_token_loss is True
    assert model.dynamic_context_parallel is True
    assert model.min_dynamic_context_parallel_size == 1
    assert (cfg.train.micro_batch_size, cfg.train.global_batch_size, cfg.train.train_iters) == (1, 32, 100)
    assert cfg.ddp.average_in_collective is False


def test_static_mode_leaves_dynamic_context_parallel_off(tmp_path, tokenizer_dir):
    cfg = _config(tmp_path, tokenizer_dir, "online_static_cp")

    assert cfg.model.dynamic_context_parallel is False
    assert cfg.model.context_parallel_size == 4


def test_data_contract_is_assistant_only_generation_masks_with_online_packing(tmp_path, tokenizer_dir):
    dataset = _config(tmp_path, tokenizer_dir, "online_dynamic_cp").dataset

    assert dataset.preprocessing == ChatSFTPreprocessingConfig(loss_mode="assistant", mask_source="generation")
    assert dataset.enable_global_batch_packing is True
    assert dataset.dataloader_type == "single"
    assert dataset.seq_length == 65536
    assert dataset.dataset_root == str(tmp_path / "data")
    assert dataset.do_validation is True and dataset.do_test is False


def test_optimizer_schedule_matches_the_training_recipe(tmp_path, tokenizer_dir):
    cfg = _config(tmp_path, tokenizer_dir, "online_dynamic_cp", "--train-iters", "200")

    assert (cfg.optimizer.lr, cfg.optimizer.min_lr, cfg.optimizer.weight_decay) == (2e-5, 2e-6, 0.033)
    assert (cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2, cfg.optimizer.clip_grad) == (0.9, 0.95, 1.0)
    assert (cfg.scheduler.lr_warmup_iters, cfg.scheduler.lr_decay_iters, cfg.scheduler.lr_decay_style) == (
        20,
        200,
        "cosine",
    )
    assert cfg.scheduler.start_weight_decay == cfg.scheduler.end_weight_decay == 0.033
    for dtype in ("main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        assert getattr(cfg.optimizer, dtype) == torch.float32


@pytest.mark.parametrize("precision", ["mxfp8", "bf16"])
@pytest.mark.parametrize("offload", [True, False])
def test_precision_and_offload_combinations(tmp_path, tokenizer_dir, precision, offload):
    flags = ["--precision", precision, "--optimizer-cpu-offload" if offload else "--no-optimizer-cpu-offload"]
    cfg = _config(tmp_path, tokenizer_dir, "online_dynamic_cp", *flags)

    assert cfg.optimizer.optimizer_cpu_offload is offload
    assert cfg.optimizer.use_precision_aware_optimizer is offload
    if offload:
        assert cfg.optimizer.optimizer_offload_fraction == 1.0
    assert str(cfg.env_vars["OMP_NUM_THREADS"]) == ("12" if offload else "1")
    mxfp8 = precision == "mxfp8"
    assert cfg.model.moe_use_grouped_tensor is mxfp8
    assert cfg.model.use_transformer_engine_op_fuser is mxfp8
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == int(mxfp8)
    if mxfp8:
        assert cfg.mixed_precision.fp8_recipe == "mxfp8"
        # BF16 parameter storage and gather when the optimizer lives on the host; FP8 gather otherwise.
        assert cfg.mixed_precision.fp8_param_gather is (not offload)
        assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is (not offload)
    else:
        assert getattr(cfg.mixed_precision, "fp8", None) is None


@pytest.mark.parametrize("hybrid_ep", [True, False])
def test_hybrid_ep_selects_the_dispatcher_and_its_environment(tmp_path, tokenizer_dir, hybrid_ep):
    flag = "--hybrid-ep" if hybrid_ep else "--no-hybrid-ep"
    cfg = _config(tmp_path, tokenizer_dir, "online_dynamic_cp", flag)

    assert cfg.model.moe_token_dispatcher_type == ("flex" if hybrid_ep else "alltoall")
    assert ("USE_MNNVL" in cfg.env_vars) is hybrid_ep
    if hybrid_ep:
        assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
        assert cfg.model.moe_flex_dispatcher_num_sms == 32
        assert cfg.model.moe_hybridep_pad_uneven_dispatch_inputs is True
        assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 4


def test_recompute_and_overlap_toggles(tmp_path, tokenizer_dir):
    default = _config(tmp_path, tokenizer_dir, "online_dynamic_cp")
    assert (default.model.recompute_granularity, default.model.recompute_method) == ("full", "uniform")
    assert default.model.recompute_num_layers == 1
    assert default.ddp.overlap_grad_reduce is True and default.ddp.overlap_param_gather is True

    tuned = _config(tmp_path, tokenizer_dir, "online_dynamic_cp", "--no-full-recompute", "--no-ddp-overlap")
    assert tuned.model.recompute_granularity == "selective"
    assert tuned.model.recompute_modules == ["gdn_norm_out", "moe"]
    assert tuned.ddp.overlap_grad_reduce is False and tuned.ddp.overlap_param_gather is False


def test_shallower_prefix_keeps_whole_gdn_attention_cycles(tmp_path, tokenizer_dir):
    cfg = _config(tmp_path, tokenizer_dir, "online_static_cp", "--num-layers", "8")

    assert cfg.model.num_layers == 8
    assert cfg.model.linear_attention_freq == [1, 1, 1, 0] * 2


@pytest.mark.parametrize(
    "flags",
    [
        ["--num-layers", "6"],
        ["--num-layers", "0"],
        ["--seq-length", "65535"],
        ["--init-mode", "checkpoint"],
        ["--pretrained-checkpoint", "/some/checkpoint"],
    ],
)
def test_invalid_arguments_are_rejected_before_any_work(tmp_path, tokenizer_dir, flags):
    with pytest.raises(SystemExit):
        _args(tmp_path, tokenizer_dir, "online_dynamic_cp", *flags)


def test_more_layers_than_prepared_are_rejected(tmp_path, tokenizer_dir):
    with pytest.raises(ValueError, match="exceeds the prepared 40 layers"):
        _config(tmp_path, tokenizer_dir, "online_dynamic_cp", "--num-layers", "44")


def test_checkpoint_initialization_loads_weights_only(tmp_path, tokenizer_dir):
    cfg = _config(
        tmp_path,
        tokenizer_dir,
        "online_dynamic_cp",
        "--init-mode",
        "checkpoint",
        "--pretrained-checkpoint",
        str(tmp_path / "converted"),
    )

    assert cfg.checkpoint.pretrained_checkpoint == str(tmp_path / "converted")
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.load_optim is False and cfg.checkpoint.load_rng is False


def test_existing_checkpoints_are_never_resumed_by_accident(tmp_path, tokenizer_dir):
    args = _args(tmp_path, tokenizer_dir, "online_dynamic_cp")
    args.save_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="--resume found no checkpoint"):
        train.build_config(_args(tmp_path, tokenizer_dir, "online_dynamic_cp", "--resume"))

    (args.save_dir / "latest_checkpointed_iteration.txt").write_text("100")
    with pytest.raises(ValueError, match="use --resume or select a fresh --save-dir"):
        train.build_config(args)
    resumed = train.build_config(_args(tmp_path, tokenizer_dir, "online_dynamic_cp", "--resume"))
    assert resumed.checkpoint.load == str(args.save_dir)
    assert resumed.checkpoint.load_optim is True


def test_run_names_separate_experiments_that_cannot_resume_each_other(tmp_path, tokenizer_dir):
    names = {
        train.run_name(_args(tmp_path, tokenizer_dir, mode, *flags))
        for mode in train.MODES
        for flags in ([], ["--precision", "bf16"], ["--no-optimizer-cpu-offload"], ["--num-layers", "8"])
    }

    assert len(names) == len(train.MODES) * 4


def test_undeclared_config_fields_fail_instead_of_becoming_phantom_attributes():
    class Target:
        known = 1

    train._set(Target(), known=2)
    with pytest.raises(ValueError, match="no field 'unknown'"):
        train._set(Target(), unknown=3)
