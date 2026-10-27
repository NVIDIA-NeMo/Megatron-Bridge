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

"""Launch contracts of the agentic SFT training script (``train_qwen35_35b_a3b.py``).

The configuration is built on CPU from the architecture embedded in the script; nothing is trained and no weights or
tokenizer files are read.
"""

import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from megatron.bridge.data.sft_processing import PromptCompletionSFTPreprocessingConfig
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit

if "dynamic_context_parallel" not in {field.name for field in fields(TransformerConfig)}:
    pytest.skip("The pinned Megatron-Core does not declare dynamic context parallelism.", allow_module_level=True)

_SOURCE = Path(__file__).resolve().parents[3] / "examples/agentic_sft/train_qwen35_35b_a3b.py"
_SPEC = importlib.util.spec_from_file_location("agentic_sft_train_under_test", _SOURCE)
train = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train
_SPEC.loader.exec_module(train)


def _args(tmp_path, *flags):
    args, overrides = train.parse_cli_args([*flags, "--workspace", str(tmp_path)])
    assert overrides == []
    return args


def _config(tmp_path, mode, *flags):
    return train.build_config(_args(tmp_path, mode, *flags))


def test_embedded_architecture_is_the_full_35b_a3b_text_model():
    config = train.load_text_config(None)

    assert config.model_type == "qwen3_5_moe_text"
    assert (config.num_hidden_layers, config.hidden_size, config.num_experts) == (40, 2048, 256)
    assert config.layer_types == (["linear_attention"] * 3 + ["full_attention"]) * 10
    assert config.mtp_num_hidden_layers == 1


def test_model_matches_the_original_launcher_flags(tmp_path):
    model = _config(tmp_path, "online_dynamic_cp").model

    expected = {
        "num_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_query_groups": 2,
        "kv_channels": 256,
        "experimental_attention_variant": "gdn",
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "normalization": "RMSNorm",
        "layernorm_epsilon": 1e-6,
        "layernorm_zero_centered_gamma": True,
        "gated_linear_unit": True,
        "add_bias_linear": False,
        "qk_layernorm": True,
        "attention_output_gate": True,
        "num_moe_experts": 256,
        "moe_ffn_hidden_size": 512,
        "moe_router_topk": 8,
        "moe_shared_expert_intermediate_size": 512,
        "moe_shared_expert_gate": True,
        "moe_router_load_balancing_type": "global_aux_loss",
        "moe_aux_loss_coeff": 0.001,
        "moe_router_dtype": "fp32",
        "vocab_size": 248320,
        "position_embedding_type": "rope",
        "rotary_base": 10000000,
        "rotary_percent": 0.25,
        "share_embeddings_and_output_weights": False,
        "init_method_std": 0.02,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "mtp_num_layers": 1,
        "mtp_loss_scaling_factor": 0.1,
    }
    assert {name: getattr(model, name) for name in expected} == expected
    # Three Gated DeltaNet layers (1) then one attention layer (0), ten times.
    assert model.linear_attention_freq == [1, 1, 1, 0] * 10


def test_defaults_describe_four_ranks_with_dynamic_cp(tmp_path):
    cfg = _config(tmp_path, "online_dynamic_cp")
    model = cfg.model

    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (1, 1)
    assert (model.context_parallel_size, model.expert_model_parallel_size, model.expert_tensor_parallel_size) == (
        4,
        4,
        1,
    )
    assert model.seq_length == 65536
    assert model.max_seqlen_per_dp_cp_rank == 16384
    assert model.calculate_per_token_loss is True
    assert model.dynamic_context_parallel is True
    assert model.min_dynamic_context_parallel_size == 1
    assert (cfg.train.micro_batch_size, cfg.train.global_batch_size, cfg.train.train_iters) == (1, 32, 100)
    assert cfg.ddp.average_in_collective is False


def test_static_mode_leaves_dynamic_context_parallel_off(tmp_path):
    cfg = _config(tmp_path, "online_static_cp")

    assert cfg.model.dynamic_context_parallel is False
    assert cfg.model.context_parallel_size == 4


def test_data_is_mock_token_rows_read_through_the_local_tokenizer(tmp_path):
    cfg = _config(tmp_path, "online_dynamic_cp", "--train-iters", "10", "--global-batch-size", "8")

    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model == str(tmp_path / "data/mock/tokenizer")
    assert cfg.model.vocab_size == 248320
    dataset = cfg.dataset
    assert dataset.dataset_root == str(tmp_path / "data/mock")
    assert isinstance(dataset.preprocessing, PromptCompletionSFTPreprocessingConfig)
    assert dataset.preprocessing.loss_mode == "completion" and dataset.preprocessing.add_eos is False
    assert dataset.enable_global_batch_packing is True and dataset.dataloader_type == "single"
    assert dataset.seq_length == 65536
    assert dataset.do_validation is False and dataset.do_test is False


def test_one_pass_of_training_rows_by_default_and_validation_only_when_evaluating(tmp_path):
    default = _args(tmp_path, "online_static_cp", "--train-iters", "10", "--global-batch-size", "8")
    assert default.num_train_samples == 80
    assert train.num_validation_samples(default) == 0

    evaluating = _args(
        tmp_path, "online_static_cp", "--train-iters", "50", "--global-batch-size", "8", "--eval-iters", "2"
    )
    assert train.num_validation_samples(evaluating) == 2 * 8 * (50 // 25 + 1)
    assert _config(tmp_path, "online_static_cp", "--eval-iters", "2").dataset.do_validation is True


def test_optimizer_schedule_matches_the_training_recipe(tmp_path):
    cfg = _config(tmp_path, "online_dynamic_cp", "--train-iters", "200")

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
def test_precision_and_offload_combinations(tmp_path, precision, offload):
    flags = ["--precision", precision, "--optimizer-cpu-offload" if offload else "--no-optimizer-cpu-offload"]
    cfg = _config(tmp_path, "online_dynamic_cp", *flags)

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
def test_hybrid_ep_selects_the_dispatcher_and_its_environment(tmp_path, hybrid_ep):
    flag = "--hybrid-ep" if hybrid_ep else "--no-hybrid-ep"
    cfg = _config(tmp_path, "online_dynamic_cp", flag)

    assert cfg.model.moe_token_dispatcher_type == ("flex" if hybrid_ep else "alltoall")
    assert ("USE_MNNVL" in cfg.env_vars) is hybrid_ep
    if hybrid_ep:
        assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
        assert cfg.model.moe_flex_dispatcher_num_sms == 32
        assert cfg.model.moe_hybridep_pad_uneven_dispatch_inputs is True
        assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 4


def test_recompute_and_overlap_toggles(tmp_path):
    default = _config(tmp_path, "online_dynamic_cp")
    assert (default.model.recompute_granularity, default.model.recompute_method) == ("full", "uniform")
    assert default.model.recompute_num_layers == 1
    assert default.ddp.overlap_grad_reduce is True and default.ddp.overlap_param_gather is True

    tuned = _config(tmp_path, "online_dynamic_cp", "--no-full-recompute", "--no-ddp-overlap")
    assert tuned.model.recompute_granularity == "selective"
    assert tuned.model.recompute_modules == ["gdn_norm_out", "moe"]
    assert tuned.ddp.overlap_grad_reduce is False and tuned.ddp.overlap_param_gather is False


def test_shallower_prefix_keeps_whole_gdn_attention_cycles(tmp_path):
    cfg = _config(tmp_path, "online_static_cp", "--num-layers", "8")

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
        ["--resume"],
        ["--train-iters", "0"],
    ],
)
def test_invalid_arguments_are_rejected_before_any_work(tmp_path, flags):
    with pytest.raises(SystemExit):
        _args(tmp_path, "online_dynamic_cp", *flags)


def test_more_layers_than_the_model_has_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="exceeds the model's 40 layers"):
        _config(tmp_path, "online_dynamic_cp", "--num-layers", "44")


def test_nothing_is_saved_unless_a_save_directory_is_given(tmp_path):
    cfg = _config(tmp_path, "online_dynamic_cp")

    assert cfg.checkpoint.save is None and cfg.checkpoint.load is None
    assert cfg.checkpoint.pretrained_checkpoint is None


def test_checkpoint_initialization_loads_weights_only(tmp_path):
    cfg = _config(
        tmp_path,
        "online_dynamic_cp",
        "--init-mode",
        "checkpoint",
        "--pretrained-checkpoint",
        str(tmp_path / "converted"),
        "--save-dir",
        str(tmp_path / "run"),
    )

    assert cfg.checkpoint.pretrained_checkpoint == str(tmp_path / "converted")
    assert cfg.checkpoint.save == str(tmp_path / "run") and cfg.checkpoint.load is None
    assert cfg.checkpoint.load_optim is False and cfg.checkpoint.load_rng is False


def test_existing_checkpoints_are_never_resumed_by_accident(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="--resume found no checkpoint"):
        _config(tmp_path, "online_dynamic_cp", "--save-dir", str(run), "--resume")

    (run / "latest_checkpointed_iteration.txt").write_text("100")
    with pytest.raises(ValueError, match="use --resume or select a fresh --save-dir"):
        _config(tmp_path, "online_dynamic_cp", "--save-dir", str(run))
    resumed = _config(tmp_path, "online_dynamic_cp", "--save-dir", str(run), "--resume")
    assert resumed.checkpoint.load == str(run) and resumed.checkpoint.load_optim is True


def test_prepare_data_only_writes_the_mock_dataset_and_exits(tmp_path):
    train.main(
        [
            "online_static_cp",
            "--workspace",
            str(tmp_path),
            "--prepare-data-only",
            "--train-iters",
            "2",
            "--global-batch-size",
            "4",
            "--seq-length",
            "4096",
            "--max-seqlen-per-rank",
            "1024",
            "--num-layers",
            "4",
        ]
    )

    manifest = json.loads((tmp_path / "data/mock/manifest.json").read_text())
    assert len(manifest["splits"]["training"]["runtime_lengths"]) == 8
    assert (tmp_path / "data/mock/tokenizer/tokenizer.json").is_file()
    assert len((tmp_path / "data/mock/training.jsonl").read_text().splitlines()) == 8


def test_dry_run_builds_the_config_without_touching_the_filesystem(tmp_path):
    train.main(["online_dynamic_cp", "--workspace", str(tmp_path), "--dry-run"])

    assert list(tmp_path.iterdir()) == []


def test_undeclared_config_fields_fail_instead_of_becoming_phantom_attributes():
    class Target:
        known = 1

    train._set(Target(), known=2)
    with pytest.raises(ValueError, match="no field 'unknown'"):
        train._set(Target(), unknown=3)


def test_memory_headroom_callback_caps_the_allocator_on_the_current_device(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: calls.append((fraction, device))
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    (callback,) = train.build_callbacks(_args(tmp_path, "online_dynamic_cp"))
    callback.on_train_start(context=None)

    assert calls == [(0.9, 2)]
    assert train.build_callbacks(_args(tmp_path, "online_dynamic_cp", "--cuda-memory-fraction", "0")) == []


@pytest.mark.parametrize("fraction", ["-0.1", "1.5"])
def test_memory_fraction_must_be_a_fraction(tmp_path, fraction):
    with pytest.raises(SystemExit):
        _args(tmp_path, "online_dynamic_cp", "--cuda-memory-fraction", fraction)
