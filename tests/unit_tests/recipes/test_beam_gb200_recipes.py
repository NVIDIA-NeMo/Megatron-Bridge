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

"""GB200 Beam recipes must preserve model and finetuning semantics after finalization."""

import math
from types import SimpleNamespace

import pytest
import torch

import megatron.bridge.recipes as recipes
import megatron.bridge.recipes.beam.gb200.beam as gb200
import megatron.bridge.recipes.beam.h200.beam as h200
from megatron.bridge.models.beam.beam_config import BeamModelConfig
from megatron.bridge.models.beam.modeling_beam import validate_beam_config
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import runtime_config_update
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global
from tests.unit_tests.recipes.test_beam_recipes import _BeamAutoBridge


pytestmark = pytest.mark.unit

RECIPES = [
    (gb200.beam_pretrain_64gpu_gb200_bf16_config, 64),
    (gb200.beam_sft_64gpu_gb200_bf16_config, 64),
    (gb200.beam_peft_32gpu_gb200_bf16_config, 32),
]
FINETUNING = [
    (gb200.beam_sft_64gpu_gb200_bf16_config, h200.beam_sft_128gpu_h200_bf16_config),
    (gb200.beam_peft_32gpu_gb200_bf16_config, h200.beam_peft_16gpu_h200_bf16_config),
]


@pytest.fixture(autouse=True)
def offline_bridge(monkeypatch):
    _BeamAutoBridge.calls = []
    # Configuration tests describe their target without requiring a local CUDA device.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name="NVIDIA GB200", major=10, minor=0),
    )
    for module in (gb200, h200):
        patch_recipe_module_global(monkeypatch, module, "AutoBridge", _BeamAutoBridge)


def _apply_runtime_config(cfg, world_size, monkeypatch, tmp_path):
    monkeypatch.setattr("megatron.bridge.training.config.get_world_size_safe", lambda: world_size)
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", "0")
    if cfg.peft is not None:
        # PEFT callers must supply this launch-time input; validation requires it.
        assert cfg.checkpoint.pretrained_checkpoint is None
        checkpoint_path = tmp_path / "iter_0000000"
        checkpoint_path.mkdir()
        cfg.checkpoint.pretrained_checkpoint = str(checkpoint_path)
    for name, value in cfg.env_vars.items():
        monkeypatch.setenv(name, str(value))
    runtime_config_update(cfg)


@pytest.mark.parametrize("recipe,world_size", RECIPES)
def test_parallel_meshes_and_batch_are_valid_after_finalization(recipe, world_size, monkeypatch, tmp_path):
    cfg = recipe()
    _apply_runtime_config(cfg, world_size, monkeypatch, tmp_path)
    model = cfg.model.transformer
    tp, cp, pp = model.tensor_model_parallel_size, model.context_parallel_size, model.pipeline_model_parallel_size
    ep, etp = model.expert_model_parallel_size, model.expert_tensor_parallel_size
    assert world_size % (pp * math.lcm(tp * cp, ep * etp)) == 0
    assert model.num_moe_experts % ep == 0
    assert model.num_attention_heads % tp == model.num_query_groups % tp == 0
    assert cfg.train.global_batch_size % (cfg.train.micro_batch_size * world_size // (tp * cp * pp)) == 0
    assert cfg.model.hybrid_layer_pattern.count("*") == 52
    assert cfg.model.hybrid_layer_pattern.count("E") == 49
    assert cfg.model.hybrid_layer_pattern.count("-") == 3
    assert model.sequence_parallel is (tp > 1)
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == ep
    assert model.recompute_granularity != "full"
    # Packed batches vary across ranks; eager HybridEP must pad them safely.
    assert model.moe_hybridep_pad_uneven_dispatch_inputs is (recipe is not RECIPES[0][0])
    assert getattr(recipes, recipe.__name__) is recipe


@pytest.mark.parametrize("recipe,world_size", RECIPES)
def test_runtime_precision_and_routing_preserve_training(recipe, world_size, monkeypatch, tmp_path):
    cfg = recipe()
    _apply_runtime_config(cfg, world_size, monkeypatch, tmp_path)
    model = cfg.model.transformer
    assert model.bf16 and cfg.optimizer.bf16
    assert model.params_dtype == torch.bfloat16
    assert cfg.ddp.grad_reduce_in_fp32
    assert not cfg.optimizer.use_precision_aware_optimizer
    assert {
        cfg.optimizer.main_params_dtype,
        cfg.optimizer.main_grads_dtype,
        cfg.optimizer.exp_avg_dtype,
        cfg.optimizer.exp_avg_sq_dtype,
    } == {torch.float32}
    assert cfg.ddp.check_for_nan_in_grad and cfg.ddp.check_for_large_grads
    assert cfg.rerun_state_machine.check_for_nan_in_loss
    assert not model.moe_router_force_load_balancing
    assert model.moe_router_dtype == "fp32" and model.moe_router_fp32_params
    assert model.moe_router_load_balancing_type == "none"
    assert model.moe_aux_loss_coeff == model.moe_router_bias_update_rate == 0.0
    assert model.post_self_attn_layernorm and model.post_mlp_layernorm and model.depth_scaled_residual
    assert model.ssmax and model.final_logit_softcapping == 30.0
    # A round trip must retain fields used by saved run configurations.
    restored = BeamModelConfig.from_dict(cfg.model.as_dict())
    assert restored.as_dict() == cfg.model.as_dict()
    model.overlap_moe_expert_parallel_comm = True
    with pytest.raises(ValueError, match="fine-grained"):
        validate_beam_config(model)


@pytest.mark.parametrize("recipe,reference", FINETUNING)
def test_finetuning_preserves_h200_hyperparameters_and_dataset_selection(recipe, reference):
    cfg, old = recipe(), reference()
    assert cfg.model.context_parallel_size == 1
    # Host collection scheduling is a performance setting, separate from training semantics.
    collection_fields = {"manual_gc", "manual_gc_interval"}
    assert {key: value for key, value in vars(cfg.train).items() if key not in collection_fields} == {
        key: value for key, value in vars(old.train).items() if key not in collection_fields
    }
    assert cfg.optimizer == old.optimizer
    assert cfg.scheduler == old.scheduler
    assert cfg.rng == old.rng
    assert cfg.tokenizer == old.tokenizer
    assert cfg.peft == old.peft
    assert cfg.dataset.seq_length == old.dataset.seq_length == 8192
    assert cfg.dataset.hf_dataset == old.dataset.hf_dataset
    assert cfg.dataset.seed == old.dataset.seed
    assert cfg.dataset.enable_offline_packing and not cfg.dataset.do_validation
    assert cfg.dataset.hf_validation_proportion is None
    # Sample padding follows the new TP mesh; PEFT pack membership may therefore change.
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == cfg.model.tensor_model_parallel_size


@pytest.mark.parametrize("recipe,world_size", RECIPES)
def test_local_checkpoint_override_keeps_model_and_tokenizer_together(recipe, world_size):
    cfg = recipe(hf_path="/checkpoints/beam")
    assert _BeamAutoBridge.calls == [("/checkpoints/beam", {})]
    assert cfg.tokenizer.tokenizer_model == "/checkpoints/beam"
    assert not cfg.tokenizer.hf_tokenizer_kwargs


def test_custom_peft_instance_is_preserved():
    custom = LoRA(dim=4, alpha=8, target_modules=["linear_proj"])
    assert gb200.beam_peft_32gpu_gb200_bf16_config(peft_scheme=custom).peft is custom


def test_unsupported_gpu_is_rejected_without_changing_dispatcher(monkeypatch, tmp_path):
    cfg = gb200.beam_pretrain_64gpu_gb200_bf16_config()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name="NVIDIA V100", major=7, minor=0),
    )
    with pytest.raises(ValueError, match="HybridEP is supported"):
        _apply_runtime_config(cfg, 64, monkeypatch, tmp_path)
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"


@pytest.mark.parametrize(
    "recipe,world_size",
    [(gb200.beam_sft_64gpu_gb200_bf16_config, 64), (gb200.beam_peft_32gpu_gb200_bf16_config, 32)],
)
def test_finetuning_selected_execution_preserves_unfused_math(recipe, world_size, monkeypatch, tmp_path):
    cfg = recipe()
    _apply_runtime_config(cfg, world_size, monkeypatch, tmp_path)
    assert cfg.train.manual_gc and cfg.train.manual_gc_interval == 10
    assert cfg.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == 0
    assert not cfg.model.bias_activation_fusion
    assert not cfg.model.moe_router_force_load_balancing
    assert cfg.model.recompute_granularity is None
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
