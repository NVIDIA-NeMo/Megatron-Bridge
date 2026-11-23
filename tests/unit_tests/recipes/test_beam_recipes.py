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

"""Offline tests for the Beam H200 recipes, built on a real ``BeamModelConfig`` with Beam's dimensions."""

import math
from unittest.mock import Mock

import pytest
import torch

import megatron.bridge.recipes as recipes_package
import megatron.bridge.recipes.beam.h200.beam as beam_recipes
from megatron.bridge.models.beam.beam_bridge import BeamBridge
from megatron.bridge.models.beam.beam_config import BeamModelConfig
from megatron.bridge.models.beam.modeling_beam import beam_hybrid_stack_spec
from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.mixed_precision import update_config_with_precision_overrides
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


pytestmark = pytest.mark.unit

# Published Beam architecture: 52 layers (3 dense + 49 MoE), full attention every sixth layer and on the last one.
BEAM_LAYERS = 52
BEAM_HF_CONFIG = {
    "architectures": ["BeamForCausalLM"],
    "model_type": "beam",
    "hidden_size": 5120,
    "intermediate_size": 20480,
    "moe_intermediate_size": 2560,
    "num_hidden_layers": BEAM_LAYERS,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "initializer_range": 0.02,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "n_group": 8,
    "topk_group": 4,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": True,
    "sliding_window": 512,
    "layer_types": [
        "full_attention" if i % 6 == 5 or i == BEAM_LAYERS - 1 else "sliding_attention" for i in range(BEAM_LAYERS)
    ],
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (BEAM_LAYERS - 3),
    "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}, "full_attention": None},
    "default_theta": 10000.0,
    "final_logit_softcapping": 30.0,
}

PRETRAIN = beam_recipes.beam_pretrain_128gpu_h200_bf16_config
SFT = beam_recipes.beam_sft_128gpu_h200_bf16_config
PEFT = beam_recipes.beam_peft_16gpu_h200_bf16_config
WORLD_SIZE = {PRETRAIN: 128, SFT: 128, PEFT: 16}


class _BeamAutoBridge:
    """Return a real ``BeamModelConfig`` converted from Beam's HF config, without reading the Hub."""

    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_hf_pretrained(cls, path: str, **kwargs: object) -> "_BeamAutoBridge":
        cls.calls.append((path, dict(kwargs)))
        return cls()

    def get_model_config(self) -> BeamModelConfig:
        hf_config = Mock(spec=list(BEAM_HF_CONFIG))
        for key, value in BEAM_HF_CONFIG.items():
            setattr(hf_config, key, value)
        return BeamBridge().hf_config_to_model_config(hf_config)


@pytest.fixture(autouse=True)
def _offline_beam_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    _BeamAutoBridge.calls = []
    patch_recipe_module_global(monkeypatch, beam_recipes, "AutoBridge", _BeamAutoBridge)


def _field(model: BeamModelConfig, name: str) -> object:
    """Read a flat recipe field from the Beam config or its nested transformer config."""
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, name):
        return getattr(transformer, name)
    return getattr(model, name)


_LAYOUT_FIELDS = {
    "tensor": "tensor_model_parallel_size",
    "pipeline": "pipeline_model_parallel_size",
    "context": "context_parallel_size",
    "expert": "expert_model_parallel_size",
    "expert_tensor": "expert_tensor_parallel_size",
}


def _layout(cfg) -> dict[str, int]:
    return {key: _field(cfg.model, name) for key, name in _LAYOUT_FIELDS.items()}


@pytest.mark.parametrize("recipe", [PRETRAIN, SFT, PEFT], ids=lambda f: f.__name__)
def test_recipe_builds_real_beam_config_and_fits_its_gpu_count(recipe) -> None:
    cfg = recipe()
    world = WORLD_SIZE[recipe]
    layout = _layout(cfg)
    tp, pp, cp, ep, etp = (layout[k] for k in ("tensor", "pipeline", "context", "expert", "expert_tensor"))

    assert isinstance(cfg.model, BeamModelConfig)
    # The dense TP*CP mesh and the expert EP*ETP mesh share each pipeline stage's GPUs.
    assert world % (pp * max(tp * cp, ep * etp)) == 0
    data_parallel = world // (tp * cp * pp)
    assert cfg.train.global_batch_size % (data_parallel * cfg.train.micro_batch_size) == 0
    assert _field(cfg.model, "num_moe_experts") % ep == 0
    assert _field(cfg.model, "num_attention_heads") % tp == 0
    assert _field(cfg.model, "num_query_groups") % tp == 0
    assert _field(cfg.model, "sequence_parallel") is (tp > 1)
    assert _field(cfg.model, "pipeline_dtype") == (torch.bfloat16 if pp > 1 else None)
    assert BEAM_LAYERS >= pp


@pytest.mark.parametrize("recipe", [PRETRAIN, SFT, PEFT], ids=lambda f: f.__name__)
def test_beam_execution_settings(recipe) -> None:
    cfg = recipe()
    model = cfg.model
    assert _field(model, "cp_comm_type") == "a2a"
    assert _field(model, "moe_token_dispatcher_type") == "alltoall"
    assert _field(model, "moe_router_force_load_balancing") is False
    assert _field(model, "recompute_granularity") == "full"
    assert _field(model, "recompute_method") == "uniform"
    assert _field(model, "recompute_num_layers") == 1
    assert _field(model, "transformer_impl") == "transformer_engine"
    assert _field(model, "virtual_pipeline_model_parallel_size") is None
    assert cfg.checkpoint.pretrained_checkpoint is None


@pytest.mark.parametrize(("recipe", "lr"), [(SFT, 5e-6), (PEFT, 1e-4)], ids=["sft", "peft"])
def test_finetuning_recipes_own_convergence_v2(recipe, lr) -> None:
    cfg = recipe()
    layout = _layout(cfg)
    data_parallel = WORLD_SIZE[recipe] // (layout["tensor"] * layout["context"] * layout["pipeline"])

    assert data_parallel in {1, 2, 4, 8}
    assert (cfg.train.global_batch_size, cfg.train.micro_batch_size, cfg.train.train_iters) == (8, 1, 100)
    assert _field(cfg.model, "seq_length") == cfg.dataset.seq_length == 8192
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 8192
    assert cfg.dataset.enable_offline_packing is True
    assert cfg.dataset.hf_dataset.dataset_name == "tulu3"
    assert cfg.dataset.hf_dataset.split == "train[:10000]"
    assert cfg.dataset.hf_dataset.load_kwargs == {"revision": beam_recipes.TULU3_REVISION}
    assert cfg.dataset.do_validation is False
    assert (cfg.validation.eval_interval, cfg.validation.eval_iters) == (0, 0)
    assert (cfg.dataset.seed, cfg.rng.seed) == (1234, 5678)

    opt, sched = cfg.optimizer, cfg.scheduler
    assert (opt.lr, opt.min_lr, opt.adam_beta1, opt.adam_beta2, opt.adam_eps, opt.clip_grad) == (
        lr,
        0.0,
        0.9,
        0.95,
        1e-8,
        1.0,
    )
    assert (sched.lr_warmup_iters, sched.lr_decay_iters, sched.lr_warmup_init, sched.lr_decay_style) == (
        10,
        100,
        0.0,
        "cosine",
    )
    assert (sched.start_weight_decay, sched.end_weight_decay, sched.weight_decay_incr_style) == (
        0.033,
        0.033,
        "constant",
    )
    assert opt.use_distributed_optimizer is True
    assert opt.use_precision_aware_optimizer is False
    assert {opt.main_grads_dtype, opt.main_params_dtype, opt.exp_avg_dtype, opt.exp_avg_sq_dtype} == {torch.float32}
    assert cfg.ddp.grad_reduce_in_fp32 is True
    # CP > 1 would force per-token loss for finetuning and change the contract's loss normalization.
    assert layout["context"] == 1


@pytest.mark.parametrize("recipe", [SFT, PEFT], ids=["sft", "peft"])
def test_pad_seq_to_mult_follows_resolved_topology(recipe) -> None:
    cfg = recipe()
    layout = _layout(cfg)
    tp, cp = layout["tensor"], layout["context"]
    cp_multiple = 2 * cp if cp > 1 else 1
    sp_multiple = cp * tp if _field(cfg.model, "sequence_parallel") and tp > 1 else 1
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == math.lcm(cp_multiple, sp_multiple)


def test_sft_reaches_dp8_without_context_parallelism() -> None:
    layout = _layout(SFT())
    assert (layout["tensor"], layout["pipeline"], layout["context"], layout["expert"]) == (8, 2, 1, 64)


def test_peft_adapts_attention_projections_only() -> None:
    cfg = PEFT()
    assert isinstance(cfg.peft, LoRA)
    assert (cfg.peft.dim, cfg.peft.alpha, cfg.peft.dropout) == (8, 16, 0.0)
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]

    attention = beam_hybrid_stack_spec(cfg.model).submodules.attention_layer.submodules.self_attention.submodules
    for target in cfg.peft.target_modules:
        assert hasattr(attention, target)


def test_peft_scheme_variants() -> None:
    dora = PEFT(peft_scheme="dora")
    assert isinstance(dora.peft, DoRA)
    assert (dora.peft.dim, dora.peft.alpha, dora.peft.target_modules) == (8, 16, ["linear_qkv", "linear_proj"])

    custom = LoRA(dim=4, alpha=8, target_modules=["linear_proj"])
    assert PEFT(peft_scheme=custom).peft is custom


def test_pretrain_keeps_beam_bounded_settings() -> None:
    cfg = PRETRAIN()
    layout = _layout(cfg)
    data_parallel = WORLD_SIZE[PRETRAIN] // (layout["tensor"] * layout["context"] * layout["pipeline"])

    assert (layout["tensor"], layout["pipeline"], layout["expert"]) == (2, 1, 128)
    assert (cfg.train.global_batch_size, cfg.train.micro_batch_size) == (256, 1)
    assert cfg.train.global_batch_size // data_parallel == 4
    assert _field(cfg.model, "seq_length") == cfg.dataset.seq_length == 4096
    assert (cfg.optimizer.lr, cfg.optimizer.min_lr) == (1e-4, 1e-5)
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_grads_dtype == cfg.optimizer.exp_avg_dtype == cfg.optimizer.exp_avg_sq_dtype
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.main_params_dtype == torch.float32
    # runtime_config_update copies the mixed-precision fields onto the DDP config, so both must agree.
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert update_config_with_precision_overrides(cfg.mixed_precision, cfg.ddp).grad_reduce_in_fp32 is False


def test_recipes_pin_the_hub_revision_unless_given_a_local_path() -> None:
    cfg = SFT()
    assert _BeamAutoBridge.calls[-1] == (
        beam_recipes.BEAM_HF_MODEL_ID,
        {"revision": beam_recipes.BEAM_HF_MODEL_REVISION},
    )
    assert cfg.tokenizer.tokenizer_model == beam_recipes.BEAM_HF_MODEL_ID
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"revision": beam_recipes.BEAM_HF_MODEL_REVISION}

    local = PEFT(hf_path="/checkpoints/beam")
    assert _BeamAutoBridge.calls[-1] == ("/checkpoints/beam", {})
    assert local.tokenizer.tokenizer_model == "/checkpoints/beam"


def test_recipes_are_exported_from_the_package() -> None:
    for recipe in (PRETRAIN, SFT, PEFT):
        assert getattr(recipes_package, recipe.__name__) is recipe
