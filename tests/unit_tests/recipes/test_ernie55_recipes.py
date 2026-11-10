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

"""CPU tests of the ERNIE 5.5 recipe.

Every setting of the reference's run is pinned on the returned config, the config passes Bridge's runtime
validation (with the GPU-only checks stubbed), and the learning rate and weight decay that the scheduler gives
each parameter group match the reference's schedule.
"""

import pytest


# ERNIE 5.5 is pinned to Megatron-Core dev and cannot run on Megatron-Core main, which CI uses, so this test
# module is skipped unconditionally for now.
pytest.skip("ERNIE 5.5 is pinned to Megatron-Core dev and cannot run on Megatron-Core main.", allow_module_level=True)

import contextlib
import inspect
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# Bridge's conversion stack imports its LoRA layers, which need Transformer Engine; without TE, import it with a
# placeholder for them. The training stack imports Hugging Face `datasets`; without it the module is skipped.
allow_conversion_imports_without_te()
pytest.importorskip("megatron.bridge.models.conversion.model_bridge", reason="needs Bridge's conversion stack")
pytest.importorskip("megatron.bridge.training.config", reason="needs Bridge's training stack")

from megatron.bridge.training.tokenizers.config import TokenizerConfig


try:
    TokenizerConfig()
except AttributeError as error:  # Bridge's training configs follow Megatron-Core dev's (main renamed fields)
    pytest.skip(f"Bridge's training configs do not support this Megatron-Core build: {error}", allow_module_level=True)

from megatron.core.optimizer import _get_param_groups
from megatron.core.transformer import transformer_config as mcore_transformer_config
from megatron.core.transformer.enums import AttnBackend

import megatron.bridge.models.ernie.ernie55_bridge  # noqa: F401 - registers the bridge, as the package import does
from megatron.bridge.models.ernie.ernie55_config import Ernie55ModelConfig, ernie55_csa_compress_ratios
from megatron.bridge.models.ernie.ernie55_context_parallel import ernie55_per_row_lm_loss
from megatron.bridge.models.ernie.ernie55_data import Ernie55SFTDataset, Ernie55SFTDatasetConfig
from megatron.bridge.models.ernie.ernie55_step import ernie55_loss_function
from megatron.bridge.models.ernie.modeling_ernie55.context_parallel import load_cp_layout_api
from megatron.bridge.recipes.ernie import ernie55 as recipes
from megatron.bridge.recipes.ernie.gb200 import ernie55 as gb200_recipes
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import (
    OptimizerConfigOverrideProviderContext,
    runtime_config_update,
)
from megatron.bridge.training.optim import _get_scheduler
from tests.unit_tests.models.ernie.ernie55_hf_configs import FULL_SIZE_PATTERN, full_size_config
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import needs_hook


pytestmark = pytest.mark.unit

needs_cp = pytest.mark.skipif(
    load_cp_layout_api() is None, reason="ERNIE 5.5 context parallelism needs Megatron-Core dev (CP partition modes)"
)

RECIPE = "ernie55_sft_64gpu_gb300_bf16_config"
WORLD_SIZE = 64

# Settings of the reference's production command.
REFERENCE_SETTINGS = {
    "model.tensor_model_parallel_size": 1,
    "model.pipeline_model_parallel_size": 1,
    "model.virtual_pipeline_model_parallel_size": None,
    "model.expert_tensor_parallel_size": 1,
    "model.sequence_parallel": False,
    "model.vocab_size": 200027,
    "model.share_embeddings_and_output_weights": False,
    "model.hidden_dropout": 0.0,
    "model.attention_dropout": 0.0,
    "model.attention_backend": AttnBackend.local,
    "model.dsa_kernel_backend": "cudnn",
    "model.ernie_fused_ops": True,
    "model.csa_dense_mode": True,
    "model.apply_rope_fusion": False,
    "model.masked_softmax_fusion": False,
    "model.bias_activation_fusion": False,
    "model.bias_dropout_fusion": False,
    "model.gradient_accumulation_fusion": False,
    "model.persist_layer_norm": False,
    "model.cross_entropy_loss_fusion": False,
    "model.use_fused_mhc": False,
    "model.cuda_graph_impl": "none",
    "model.moe_token_dispatcher_type": "alltoall",
    "model.moe_grouped_gemm": True,
    "model.moe_permute_fusion": False,
    "model.moe_router_fusion": False,
    "model.moe_router_aux_loss_fusion": False,
    "model.moe_router_dtype": "fp32",
    "model.moe_expert_capacity_factor": None,
    "model.moe_pad_expert_input_to_capacity": False,
    "model.moe_aux_loss_coeff": 0.0,
    "model.moe_z_loss_coeff": 0.0,
    "model.moe_router_enable_expert_bias": True,
    "model.moe_router_bias_update_rate": 1e-4,
    "model.moe_router_skip_muon": True,
    "model.ernie_per_row_loss": True,
    "model.calculate_per_token_loss": False,
    "model.mtp_num_layers": 1,
    "model.mtp_loss_scaling_factor": 0.1,
    "model.dsa_indexer_loss_coeff": 0.01,
    "model.dsa_indexer_use_sparse_loss": True,
    "model.recompute_granularity": "full",
    "model.recompute_method": "uniform",
    "model.recompute_num_layers": 1,
    "optimizer.optimizer": "dist_muon",
    "optimizer.lr": 2e-5,
    "optimizer.min_lr": 2e-6,
    "optimizer.weight_decay": 0.1,
    "optimizer.clip_grad": 1.0,
    "optimizer.bf16": True,
    "optimizer.muon_momentum": 0.95,
    "optimizer.muon_nesterov": True,
    "optimizer.muon_scale_mode": "spectral",
    "optimizer.muon_extra_scale_factor": 0.2,
    "optimizer.muon_num_ns_steps": 5,
    "optimizer.muon_coefficient_type": "quintic",
    # The reference's row blocks: per-parameter qkv_layout blocks, orthogonalized one by one; MLA layouts stay whole.
    "optimizer.muon_split_qkv": True,
    "optimizer.muon_split_qkv_per_head": False,
    "optimizer.muon_split_mla_qkv": False,
    "optimizer.muon_fp32_matmul_prec": "medium",
    "optimizer.muon_tp_mode": "blockwise",
    "optimizer.muon_scalar_optimizer": "adam",
    "optimizer.adam_beta1": 0.9,
    "optimizer.adam_beta2": 0.95,
    "optimizer.adam_eps": 1e-8,
    "optimizer.decoupled_weight_decay": True,
    "optimizer.apply_wd_to_qk_layernorm": False,
    "optimizer.use_distributed_optimizer": False,
    "scheduler.lr_decay_style": "cosine",
    "scheduler.lr_warmup_init": 0.0,
    "scheduler.start_weight_decay": 0.1,
    "scheduler.end_weight_decay": 0.1,
    "scheduler.weight_decay_incr_style": "constant",
    "ddp.grad_reduce_in_fp32": True,
    "ddp.overlap_grad_reduce": False,
    "ddp.overlap_param_gather": False,
    "ddp.average_in_collective": False,
    "ddp.use_distributed_optimizer": False,
    "ddp.use_megatron_fsdp": False,
    "mixed_precision.bf16": True,
    "mixed_precision.params_dtype": torch.bfloat16,
    "mixed_precision.grad_reduce_in_fp32": True,
    "tokenizer.tokenizer_type": "NullTokenizer",
    "tokenizer.tokenizer_model": None,
    "tokenizer.vocab_size": 200027,
    "tokenizer.use_tokenizer_vocab_size": False,
    "train.micro_batch_size": 1,
    "validation.eval_iters": 0,
    "rng.seed": 42,
    "comm_overlap": None,
    "dist.enable_megatron_core_experimental": False,
    "dataset.pad_token_id": 151643,
    "dataset.path": None,
    "dataset.seed": 42,
}

RECIPE_SETTINGS = {
    "model.hybrid_layer_pattern": FULL_SIZE_PATTERN,
    "model.num_moe_experts": 256,
    "model.context_parallel_size": 64,
    "model.cp_partition_mode": "contiguous",
    "model.expert_model_parallel_size": 64,
    "model.seq_length": 262144,
    "model.deterministic_mode": False,
    "model.ernie_reference_rmsnorm": False,
    "train.train_iters": 2065,
    "train.global_batch_size": 2,
    "scheduler.lr_warmup_iters": 103,
    "scheduler.lr_decay_iters": 2065,
    "dataset.seq_length": 262144,
    "dataset.truncation": "split",
    "dataset.pack": False,
    "dataset.pad_to_multiple_of": 128,
    "dataset.mock_document_lengths": (262144,),
    "dataset.mock_prompt_fraction": 0.25,
    "checkpoint.pretrained_checkpoint": None,
    "checkpoint.save_interval": 2065,
    "checkpoint.save_optim": False,
    "checkpoint.save_rng": False,
}

RECIPE_ENVIRONMENT = {
    **COMMON_RECIPE_ENV_VARS,
    "CUDA_DEVICE_MAX_CONNECTIONS": 1,
    "NCCL_NVLS_ENABLE": 0,
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
}


def _get(cfg, path: str):
    value = cfg
    for name in path.split("."):
        value = getattr(value, name)
    return value


@pytest.fixture(autouse=True)
def _no_hf_path_from_the_environment(monkeypatch):
    monkeypatch.delenv(recipes.ERNIE55_HF_PATH_ENV, raising=False)


@pytest.fixture
def gpu_free_runtime(monkeypatch):
    """Run Bridge's runtime config update on CPU: GPU-only checks stubbed, process state restored afterwards."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for variable in RECIPE_ENVIRONMENT:
        os.environ.pop(variable, None)
    deterministic_calls = []
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda *args, **kwargs: deterministic_calls.append(args)
    )
    monkeypatch.setattr(
        torch.utils.deterministic, "fill_uninitialized_memory", torch.utils.deterministic.fill_uninitialized_memory
    )
    # The cuDNN DSA backend needs a Blackwell or Hopper GPU and cudnn-frontend at finalize time.
    monkeypatch.setattr(mcore_transformer_config, "_validate_dsa_kernel_backend_dependencies", lambda backend: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 3))
    monkeypatch.setitem(sys.modules, "cudnn", types.SimpleNamespace(DSA=types.SimpleNamespace()))
    return deterministic_calls


@contextlib.contextmanager
def single_rank_process_group(store: str):
    """A one-rank gloo world, which Megatron-Core's parameter grouping all-gathers over."""
    if torch.distributed.is_initialized():
        yield
        return
    torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


class TestBuiltInConfigs:
    def test_full_size_is_the_assumed_layout(self):
        reference = {
            key: value for key, value in full_size_config().items() if key not in ("model_type", "torch_dtype")
        }
        assert recipes.ERNIE55_FULL_SIZE_HF_CONFIG == {**reference, "dtype": "bfloat16"}


@needs_cp
class TestSettings:
    def test_reference_settings(self):
        cfg = getattr(recipes, RECIPE)()
        assert isinstance(cfg.model, Ernie55ModelConfig)
        expected = {**REFERENCE_SETTINGS, **RECIPE_SETTINGS}
        actual = {path: _get(cfg, path) for path in expected}
        assert actual == expected

    def test_launch_environment(self):
        assert getattr(recipes, RECIPE)().env_vars == RECIPE_ENVIRONMENT

    def test_fused_ops_switch_from_the_command_line(self):
        from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

        cfg = process_config_with_overrides(getattr(recipes, RECIPE)(), cli_overrides=["model.ernie_fused_ops=false"])
        assert cfg.model.transformer.ernie_fused_ops is False
        assert cfg.model.ernie_fused_ops is False

    def test_reference_rmsnorm_switch_from_the_command_line(self):
        from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

        flipped = not getattr(recipes, RECIPE)().model.ernie_reference_rmsnorm
        cfg = process_config_with_overrides(
            getattr(recipes, RECIPE)(), cli_overrides=[f"model.ernie_reference_rmsnorm={str(flipped).lower()}"]
        )
        assert cfg.model.transformer.ernie_reference_rmsnorm is flipped
        assert cfg.model.ernie_reference_rmsnorm is flipped

    def test_per_row_loss_is_selected_by_the_step(self):
        cfg = getattr(recipes, RECIPE)()
        loss_function = ernie55_loss_function(
            torch.ones(1, 8),
            config=cfg.model.transformer,
            cp_group=None,
            check_for_nan_in_loss=False,
            check_for_spiky_loss=False,
        )
        assert loss_function.func is ernie55_per_row_lm_loss


def _checkpoint_dir(tmp_path, **overrides):
    config = {**full_size_config(), **overrides}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@needs_cp
class TestSFTRecipe:
    """The fine-tuning recipe: the production settings plus the checkpoint's architecture and weights."""

    def test_without_a_checkpoint_the_recipe_builds_and_finetune_refuses_to_start(self, tmp_path):
        # Bridge's recipe matrix builds every recipe without arguments; finetuning then stops in setup because
        # there is nothing to load (finetune() sets _checkpoint_load_required).
        from megatron.bridge.training.setup import _should_load_checkpoint

        cfg = getattr(recipes, RECIPE)()
        assert cfg.checkpoint.pretrained_checkpoint is None
        cfg.checkpoint.load = str(tmp_path / "no_checkpoint_here")
        cfg._checkpoint_load_required = True
        with pytest.raises(FileNotFoundError, match="Finetuning requires"):
            _should_load_checkpoint(cfg, SimpleNamespace(checkpointing_context={}))

    def test_the_checkpoint_comes_from_hf_path(self, tmp_path):
        path = str(_checkpoint_dir(tmp_path))
        assert getattr(recipes, RECIPE)(hf_path=path).checkpoint.pretrained_checkpoint == path

    def test_the_checkpoint_comes_from_the_environment_too(self, tmp_path, monkeypatch):
        path = str(_checkpoint_dir(tmp_path))
        monkeypatch.setenv(recipes.ERNIE55_HF_PATH_ENV, path)
        assert getattr(recipes, RECIPE)().checkpoint.pretrained_checkpoint == path

    def test_the_production_shape_is_kept(self, tmp_path):
        cfg = recipes.ernie55_sft_64gpu_gb300_bf16_config(hf_path=str(_checkpoint_dir(tmp_path)))
        model = cfg.model
        assert (model.context_parallel_size, model.expert_model_parallel_size, model.seq_length) == (64, 64, 262144)
        assert model.cp_partition_mode == "contiguous"
        assert (cfg.train.global_batch_size, cfg.train.micro_batch_size) == (2, 1)

    def test_run_recipe_infers_the_training_loop_from_the_name(self, monkeypatch):
        # The sft recipe runs Bridge's finetune loop without --mode.
        import importlib.util

        path = Path(__file__).resolve().parents[3] / "scripts" / "training" / "recipe_metadata.py"
        spec = importlib.util.spec_from_file_location("ernie55_recipe_metadata", path)
        metadata = importlib.util.module_from_spec(spec)
        # The module's dataclasses resolve their string annotations through sys.modules[cls.__module__].
        monkeypatch.setitem(sys.modules, spec.name, metadata)
        spec.loader.exec_module(metadata)
        assert metadata.infer_recipe_mode(RECIPE) == "sft"


class TestArchitectureSource:
    @needs_cp
    def test_hf_path_is_read(self, tmp_path):
        cfg = getattr(recipes, RECIPE)(hf_path=str(_checkpoint_dir(tmp_path, n_routed_experts=128)))
        assert cfg.model.num_moe_experts == 128
        assert cfg.model.hybrid_layer_pattern == FULL_SIZE_PATTERN

    @needs_cp
    def test_environment_variable_is_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv(recipes.ERNIE55_HF_PATH_ENV, str(_checkpoint_dir(tmp_path, n_routed_experts=128)))
        assert getattr(recipes, RECIPE)().model.num_moe_experts == 128

    def test_recipe_builds_with_the_recipe_matrix_stand_in(self, monkeypatch):
        # Bridge's recipe matrix (test_all_recipe_factories.py) builds every recipe with defaults against this
        # offline AutoBridge stand-in.
        from tests.unit_tests.recipes.recipe_test_utils import _OfflineAutoBridge

        monkeypatch.setattr(recipes, "AutoBridge", _OfflineAutoBridge)
        factory = getattr(recipes, RECIPE)
        inspect.signature(factory).bind()
        cfg = factory()
        assert cfg.tokenizer.vocab_size == cfg.model.vocab_size


class TestData:
    @needs_cp
    def test_production_rows(self):
        config = getattr(recipes, RECIPE)().dataset
        assert isinstance(config, Ernie55SFTDatasetConfig)
        config.finalize()
        row = Ernie55SFTDataset(config, num_items=1, seed_offset=0)[0]
        assert [len(tokens) for tokens, _ in row] == [262144]
        tokens, _ = row[0]
        # Synthetic tokens stay below the reference tokenizer's special tokens, which start at 151643.
        assert all(1 <= token < 151643 for token in tokens)


@needs_hook
@needs_cp
class TestRuntimeValidation:
    def test_runtime_config_update(self, gpu_free_runtime):
        cfg = getattr(recipes, RECIPE)()
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        runtime_config_update(cfg)

        global_batch_size = cfg.train.global_batch_size
        assert cfg.data_parallel_size == WORLD_SIZE // cfg.model.context_parallel_size
        assert cfg.scheduler.lr_warmup_steps == cfg.scheduler.lr_warmup_iters * global_batch_size
        assert cfg.scheduler.lr_decay_steps == cfg.train.train_iters * global_batch_size
        assert cfg.scheduler.wd_incr_steps == cfg.train.train_iters * global_batch_size
        # dist_muon: Muon with the layer-wise distributed optimizer, not the ordinary distributed optimizer.
        assert cfg.optimizer.optimizer == "muon"
        assert cfg.optimizer.use_layer_wise_distributed_optimizer
        assert not cfg.optimizer.use_distributed_optimizer and not cfg.ddp.use_distributed_optimizer
        # muon_split_qkv survives validation (muon_split_qkv_per_head stays off): the model's qkv_layout row blocks
        # (per-head and gate/up) reach TensorParallelMuon.
        assert cfg.optimizer.muon_split_qkv
        assert not cfg.optimizer.muon_split_qkv_per_head and not cfg.optimizer.muon_split_mla_qkv
        assert cfg.optimizer.muon_tp_mode == "blockwise"
        transformer = cfg.model.transformer
        assert transformer.params_dtype == torch.bfloat16
        assert transformer.csa_compress_ratios == ernie55_csa_compress_ratios(cfg.model.hybrid_layer_pattern)
        for variable, value in RECIPE_ENVIRONMENT.items():
            assert os.environ[variable] == str(value)
        # Not deterministic mode: torch.use_deterministic_algorithms is never called.
        assert gpu_free_runtime == []

    def test_launch_script_overrides(self, gpu_free_runtime):
        # The overrides examples/models/ernie55/slurm_train.sh passes to run_recipe.py.
        from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

        overrides = [
            "checkpoint.save=/workspace/checkpoints/run",
            "checkpoint.load=/workspace/checkpoints/run",
            "logger.tensorboard_dir=/workspace/logs/run/tensorboard",
            "dataset.path=/data/train.jsonl",
            "train.train_iters=20",
        ]
        cfg = process_config_with_overrides(getattr(recipes, RECIPE)(), cli_overrides=overrides)
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        runtime_config_update(cfg)
        assert isinstance(cfg.dataset, Ernie55SFTDatasetConfig)
        assert cfg.dataset.path == "/data/train.jsonl"
        assert cfg.checkpoint.save == "/workspace/checkpoints/run"
        assert cfg.checkpoint.load == "/workspace/checkpoints/run"
        assert cfg.logger.tensorboard_dir == "/workspace/logs/run/tensorboard"
        assert cfg.train.train_iters == 20
        # The schedule keeps the recipe's length.
        assert cfg.scheduler.lr_decay_iters == RECIPE_SETTINGS["scheduler.lr_decay_iters"]
        assert cfg.model.attention_backend == AttnBackend.local
        assert cfg.mixed_precision.params_dtype == torch.bfloat16

    def test_realized_learning_rate_and_weight_decay(self, gpu_free_runtime, tmp_path):
        cfg = getattr(recipes, RECIPE)()
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        runtime_config_update(cfg)

        # Parameters named and shaped like an ERNIE 5.5 stage's, at toy sizes.
        shapes = {
            "embedding.word_embeddings.weight": (16, 8),
            "decoder.layers.0.self_attention.linear_q_down_proj.weight": (4, 8),
            "decoder.layers.0.self_attention.q_layernorm.weight": (4,),
            "decoder.layers.0.self_attention.vha_postmix_u": (2, 2),
            "decoder.layers.2.self_attention.core_attention.attn_sink": (2,),
            "decoder.layers.1.mlp.router.weight": (4, 8),
            "decoder.layers.1.mlp.router.routed_scaling_factor_param": (4,),
            "decoder.layers.1.mlp.experts.linear_fc1.weight0": (6, 8),
            "decoder.layers.0.hyper_connection.alpha_pre": (1,),
            "decoder.hc_head_scale": (1,),
            "output_layer.weight": (16, 8),
            "output_layer.multimax_ranges": (4,),
        }
        parameters = {name: torch.nn.Parameter(torch.zeros(shape)) for name, shape in shapes.items()}
        for embedding_or_output in ("embedding.word_embeddings.weight", "output_layer.weight"):
            parameters[embedding_or_output].is_embedding_or_output_parameter = True
        stage = SimpleNamespace(named_parameters=lambda: iter(parameters.items()))

        overrides = cfg.optimizer_config_override_provider.build_config_overrides(
            OptimizerConfigOverrideProviderContext(cfg.scheduler, cfg.optimizer, stage)
        )
        with single_rank_process_group(str(tmp_path / "store")):
            groups = _get_param_groups([stage], cfg.optimizer, overrides)
        scheduler = _get_scheduler(cfg.optimizer, cfg.scheduler, SimpleNamespace(param_groups=groups))
        group_of = {id(param): group for group in groups for param in group["params"]}

        # Weight decay 0.1 on every matrix (embedding and output layer included), none on 1-D tensors.
        for parameter_name, parameter in parameters.items():
            expected_weight_decay = 0.1 if parameter.dim() == 2 else 0.0
            assert group_of[id(parameter)]["weight_decay"] == pytest.approx(expected_weight_decay), parameter_name

        # Warmup from 0 over 103 steps to 2e-5, then cosine decay to 2e-6 at step 2065.
        steps_taken = 0
        for steps, expected_lr in [(0, 0.0), (1, 2e-5 / 103), (103, 2e-5), (2065, 2e-6)]:
            scheduler.step(increment=(steps - steps_taken) * cfg.train.global_batch_size)
            steps_taken = steps
            for group in groups:
                assert group["lr"] == pytest.approx(expected_lr, rel=1e-9, abs=1e-15), steps


GB200_RECIPE = "ernie55_sft_64gpu_gb200_bf16_config"

# What the GB200 recipe sets differently from the GB300 recipe.
GB200_SETTINGS = {
    **RECIPE_SETTINGS,
    "model.context_parallel_size": 8,
    "model.seq_length": 16384,
    "train.global_batch_size": 32,
    "dataset.seq_length": 16384,
    "dataset.pack": True,
    "dataset.pad_to_multiple_of": 16,
    "dataset.mock_document_lengths": (16384,),
    "dist.distributed_backend": "cpu:gloo,cuda:nccl",
}


class TestGB200Recipe:
    """The GB200 recipe: the GB300 recipe's settings at a shape that fits a GB200."""

    @needs_cp
    def test_settings(self):
        cfg = getattr(gb200_recipes, GB200_RECIPE)()
        assert isinstance(cfg.model, Ernie55ModelConfig)
        expected = {**REFERENCE_SETTINGS, **GB200_SETTINGS}
        actual = {path: _get(cfg, path) for path in expected}
        assert actual == expected
        assert cfg.env_vars == RECIPE_ENVIRONMENT

    def test_the_recipe_is_exported(self):
        import megatron.bridge.recipes as all_recipes
        import megatron.bridge.recipes.ernie as ernie_recipes

        factory = getattr(gb200_recipes, GB200_RECIPE)
        assert getattr(ernie_recipes, GB200_RECIPE) is factory
        assert getattr(all_recipes, GB200_RECIPE) is factory

    @needs_cp
    def test_the_checkpoint_comes_from_hf_path_or_the_environment(self, tmp_path, monkeypatch):
        path = str(_checkpoint_dir(tmp_path, n_routed_experts=128))
        cfg = getattr(gb200_recipes, GB200_RECIPE)(hf_path=path)
        assert cfg.checkpoint.pretrained_checkpoint == path
        assert cfg.model.num_moe_experts == 128
        monkeypatch.setenv(recipes.ERNIE55_HF_PATH_ENV, path)
        assert getattr(gb200_recipes, GB200_RECIPE)().checkpoint.pretrained_checkpoint == path

    def test_recipe_builds_with_the_recipe_matrix_stand_in(self, monkeypatch):
        # The GB200 module builds its config with the GB300 module's helpers, so that module's AutoBridge is the
        # one Bridge's recipe matrix (test_all_recipe_factories.py) replaces with its offline stand-in.
        from tests.unit_tests.recipes.recipe_test_utils import _OfflineAutoBridge

        assert not hasattr(gb200_recipes, "AutoBridge")
        monkeypatch.setattr(recipes, "AutoBridge", _OfflineAutoBridge)
        factory = getattr(gb200_recipes, GB200_RECIPE)
        inspect.signature(factory).bind()
        cfg = factory()
        assert cfg.tokenizer.vocab_size == cfg.model.vocab_size

    @needs_cp
    def test_synthetic_rows(self):
        config = getattr(gb200_recipes, GB200_RECIPE)().dataset
        assert isinstance(config, Ernie55SFTDatasetConfig)
        config.finalize()
        row = Ernie55SFTDataset(config, num_items=1, seed_offset=0)[0]
        assert [len(tokens) for tokens, _ in row] == [16384]
        tokens, _ = row[0]
        assert all(1 <= token < 151643 for token in tokens)

    @needs_cp
    def test_file_samples_are_split_and_packed_into_rows(self, tmp_path):
        # The 20000-token sample is split into 16384 + 3616 tokens. Padded to a multiple of 16 the pieces take
        # 16384, 5008, 5008, 5008, 3616 and 3008 tokens, which first fit (longest first) puts in three rows:
        # 16384 / 5008 + 5008 + 5008 / 3616 + 3008.
        config = getattr(gb200_recipes, GB200_RECIPE)().dataset
        path = tmp_path / "train.jsonl"
        path.write_text(
            "".join(
                json.dumps({"input_ids": [1] * length, "labels": [1] * length}) + "\n"
                for length in (5000, 20000, 5000, 3000, 5000)
            )
        )
        config.path = str(path)
        config.finalize()
        dataset = Ernie55SFTDataset(config, num_items=1, seed_offset=0)
        lengths = sorted(sorted(len(tokens) for tokens, _ in dataset[index]) for index in range(len(dataset)))
        assert lengths == [[3000, 3616], [5000, 5000, 5000], [16384]]

    @needs_hook
    @needs_cp
    def test_runtime_config_update(self, gpu_free_runtime):
        cfg = getattr(gb200_recipes, GB200_RECIPE)()
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        runtime_config_update(cfg)

        # 64 GPUs at CP8: four micro-batches on each of the 8 data-parallel ranks, over which the layer-wise
        # distributed optimizer spreads its state.
        assert cfg.data_parallel_size == 8
        assert cfg.train.global_batch_size == 4 * cfg.train.micro_batch_size * cfg.data_parallel_size
        assert cfg.optimizer.optimizer == "muon"
        assert cfg.optimizer.use_layer_wise_distributed_optimizer
        assert cfg.scheduler.lr_warmup_steps == 103 * 32
        assert cfg.scheduler.lr_decay_steps == 2065 * 32
        for variable, value in RECIPE_ENVIRONMENT.items():
            assert os.environ[variable] == str(value)

    @needs_hook
    @needs_cp
    def test_launch_script_overrides_keep_the_shape(self, gpu_free_runtime):
        # The overrides examples/models/ernie55/slurm_train.sh passes to run_recipe.py.
        from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

        overrides = [
            "checkpoint.save=/workspace/checkpoints/run",
            "checkpoint.load=/workspace/checkpoints/run",
            "logger.tensorboard_dir=/workspace/logs/run/tensorboard",
            "dataset.path=/data/train.jsonl",
            "train.train_iters=20",
        ]
        cfg = process_config_with_overrides(getattr(gb200_recipes, GB200_RECIPE)(), cli_overrides=overrides)
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        runtime_config_update(cfg)
        assert isinstance(cfg.dataset, Ernie55SFTDatasetConfig)
        assert (cfg.dataset.path, cfg.dataset.pack, cfg.dataset.pad_to_multiple_of) == ("/data/train.jsonl", True, 16)
        assert (cfg.model.context_parallel_size, cfg.model.seq_length, cfg.dataset.seq_length) == (8, 16384, 16384)
        assert cfg.train.train_iters == 20
        # The process group survives the OmegaConf round trip of the overrides.
        assert cfg.dist.distributed_backend == "cpu:gloo,cuda:nccl"
