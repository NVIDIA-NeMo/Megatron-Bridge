import copy
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from megatron.training.models.base import ModelConfig
from transformers import AutoConfig, Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

from megatron.bridge import AutoBridge
from megatron.bridge.engine.recipes import qwen35, registry
from megatron.bridge.engine.recipes.base import data_parallel_size, is_multimodal, min_gpus
from megatron.bridge.engine.recipes.registry import (
    RecipeEntry,
    apply_recipe_overrides,
    detect_hardware,
    resolve_recipe,
)
from megatron.bridge.engine.schedule import uses_mrope


def _recipe(tp=1, pp=1, cp=8, ep=16, etp=1):
    return SimpleNamespace(
        model=SimpleNamespace(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
        )
    )


@pytest.mark.parametrize(
    "parallelism, expected",
    [
        (dict(tp=1, pp=1, cp=8, ep=16, etp=1), 16),  # the Qwen3.5 recipe: EP grid is the larger one
        (dict(tp=2, pp=2, cp=1, ep=1, etp=None), 4),  # dense: ETP defaults to TP
        (dict(tp=1, pp=1, cp=8, ep=4, etp=1), 8),  # EP grid fits in the dense grid
        (dict(tp=2, pp=1, cp=4, ep=16, etp=1), 16),  # lcm(8, 16)
        (dict(tp=1, pp=2, cp=3, ep=4, etp=1), 24),  # lcm(6, 8)
    ],
)
def test_min_gpus_is_the_lcm_of_both_rank_grids(parallelism, expected):
    assert min_gpus(_recipe(**parallelism)) == expected


@pytest.mark.parametrize("world_size, dp", [(16, 2), (32, 4), (64, 8)])
def test_data_parallel_size_scales_with_multiples_of_the_unit(monkeypatch, world_size, dp):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    assert data_parallel_size(_recipe()) == dp


@pytest.mark.parametrize("world_size", [8, 24])
def test_data_parallel_size_rejects_world_sizes_below_or_off_the_unit(monkeypatch, world_size):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    with pytest.raises(ValueError, match="minimum GPU count 16"):
        data_parallel_size(_recipe())


def _entry(name, hardware=("GB300",), max_seq_length=131072, signature=None):
    calls = []

    def build(*, hf_path, hf_revision, seq_length):
        calls.append((hf_path, hf_revision, seq_length))
        return SimpleNamespace(name=name, hf_path=hf_path, seq_length=seq_length, calls=calls)

    return RecipeEntry(
        name=name,
        model_type="toy_moe",
        signature=signature if signature is not None else {"num_hidden_layers": 4},
        hardware=hardware,
        max_seq_length=max_seq_length,
        build=build,
    )


def _patch_registry(monkeypatch, entries, config=None):
    monkeypatch.setattr(registry, "REGISTRY", tuple(entries))
    config = config or SimpleNamespace(model_type="toy_moe", num_hidden_layers=4)
    monkeypatch.setattr(registry.AutoConfig, "from_pretrained", lambda *a, **k: config)


def test_resolve_recipe_takes_the_smallest_bucket_covering_seq_length(monkeypatch):
    short = _entry("short", max_seq_length=32768)
    long = _entry("long", max_seq_length=131072)
    _patch_registry(monkeypatch, [long, short])

    recipe = resolve_recipe("/models/toy", seq_length=20000, hardware="GB300")
    assert recipe.name == "short"
    assert recipe.calls == [("/models/toy", None, 20000)]  # the user's length, not the bucket

    assert resolve_recipe("/models/toy", seq_length=32768, hardware="GB300").name == "short"
    assert resolve_recipe("/models/toy", seq_length=32769, hardware="GB300").name == "long"


def test_resolve_recipe_reads_the_text_config_of_a_multimodal_wrapper(monkeypatch):
    text_config = SimpleNamespace(model_type="toy_moe", num_hidden_layers=4)
    wrapper = SimpleNamespace(model_type="toy_vl", text_config=text_config)
    _patch_registry(monkeypatch, [_entry("only")], config=wrapper)
    assert resolve_recipe("/models/toy-vl", seq_length=1024, hardware="GB300").name == "only"


@pytest.mark.parametrize(
    "kwargs, config, message",
    [
        (
            dict(seq_length=1024, hardware="GB300"),
            SimpleNamespace(model_type="other"),
            "No recipe registered for model",
        ),
        (
            dict(seq_length=1024, hardware="GB300"),
            SimpleNamespace(model_type="toy_moe", num_hidden_layers=8),
            "No recipe registered for model",
        ),
        (dict(seq_length=1024, hardware="H100"), None, "verified on H100"),
        (dict(seq_length=200000, hardware="GB300"), None, "covers seq_length 200000"),
        (dict(seq_length=0, hardware="GB300"), None, "seq_length must be positive"),
    ],
)
def test_resolve_recipe_errors_name_the_missing_match(monkeypatch, kwargs, config, message):
    _patch_registry(monkeypatch, [_entry("only")], config=config)
    with pytest.raises(ValueError, match=message):
        resolve_recipe("/models/toy", **kwargs)


def test_resolve_recipe_detects_hardware_when_not_given(monkeypatch):
    _patch_registry(monkeypatch, [_entry("only")])
    monkeypatch.setattr(registry, "detect_hardware", lambda: "GB300")
    assert resolve_recipe("/models/toy", seq_length=1024).name == "only"


@pytest.mark.parametrize(
    "device_name, family",
    [("NVIDIA GB300", "GB300"), ("NVIDIA GB200", "GB200"), ("NVIDIA H100 80GB HBM3", "H100")],
)
def test_detect_hardware_matches_the_device_family(monkeypatch, device_name, family):
    monkeypatch.setattr(registry.torch.cuda, "get_device_properties", lambda index: SimpleNamespace(name=device_name))
    assert detect_hardware() == family


def test_detect_hardware_rejects_unknown_devices(monkeypatch):
    monkeypatch.setattr(registry.torch.cuda, "get_device_properties", lambda index: SimpleNamespace(name="NVIDIA T4"))
    with pytest.raises(ValueError, match="pass hardware="):
        detect_hardware()


def test_registry_entry_matches_the_qwen35_recipe_function():
    (entry,) = registry.REGISTRY
    assert entry.build.__name__ == entry.name
    assert entry.hardware == ("GB300",)
    assert entry.max_seq_length == 131072


@dataclass
class _Opt:
    lr: float = 2e-5


@dataclass
class _Recipe:
    optimizer: _Opt = field(default_factory=_Opt)
    router_replay: bool = False


def test_apply_recipe_overrides_sets_nested_fields():
    recipe = _Recipe()
    apply_recipe_overrides(recipe, {"optimizer.lr": 1e-6, "router_replay": True})
    assert recipe.optimizer.lr == 1e-6
    assert recipe.router_replay is True


@pytest.mark.parametrize("path", ["optimizer.lrr", "optimizerr.lr", "nope"])
def test_apply_recipe_overrides_rejects_unknown_paths(path):
    recipe = _Recipe()
    with pytest.raises(ValueError, match="Unknown recipe field"):
        apply_recipe_overrides(recipe, {path: 1})
    assert recipe == _Recipe()


# A tiny Qwen3.5 MoE text model with the 35B-A3B layout: three GatedDeltaNet layers per attention layer, MTP 1.
_TINY_QWEN35_TEXT = dict(
    vocab_size=512,
    hidden_size=64,
    num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    full_attention_interval=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    attn_output_gate=True,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=16,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    max_position_embeddings=8192,
    mtp_num_hidden_layers=1,
    rope_parameters={
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
)


class _BuilderOnlyBridge:
    """Stands in for ``AutoBridge``: real builder-backed configs; the removed provider path fails the test."""

    def __init__(self, hf_config):
        self._hf_config = hf_config

    @classmethod
    def from_hf_config(cls, hf_config):
        return cls(hf_config)

    @classmethod
    def from_hf_pretrained(cls, hf_path, revision=None):
        return cls(AutoConfig.from_pretrained(hf_path, revision=revision))

    def get_model_config(self):
        return AutoBridge.from_hf_config(self._hf_config).get_model_config()

    def to_megatron_provider(self, load_weights=False):
        raise AssertionError("engine recipes must build builder-backed configs with get_model_config()")


def _assert_35b_a3b_execution_settings(model, seq_length):
    assert isinstance(model, ModelConfig)
    assert (model.seq_length, model.context_parallel_size, model.expert_model_parallel_size) == (seq_length, 8, 16)
    assert (model.gdn_kernel_backend, model.moe_flex_dispatcher_backend) == ("transformer_engine", "hybridep")
    assert model.recompute_modules == ["gdn_norm_out", "moe"]
    assert model.mtp_num_layers == 1


def test_qwen35_text_recipe_builds_a_builder_backed_config(monkeypatch, tmp_path):
    Qwen3_5MoeTextConfig(architectures=["Qwen3_5MoeForCausalLM"], **_TINY_QWEN35_TEXT).save_pretrained(tmp_path)
    monkeypatch.setattr(qwen35, "AutoBridge", _BuilderOnlyBridge)
    recipe = qwen35.qwen35_text_35b_a3b_rl_16gpu_gb200_bf16(hf_path=str(tmp_path), seq_length=4096)
    _assert_35b_a3b_execution_settings(recipe.model, 4096)
    assert not is_multimodal(recipe)
    assert not uses_mrope(recipe.model)
    # TrainingEngine.from_recipe deep-copies the recipe; the copy must keep the outer config.
    _assert_35b_a3b_execution_settings(copy.deepcopy(recipe).model, 4096)


def test_qwen35_vl_recipe_builds_a_builder_backed_config(monkeypatch, tmp_path):
    vision = dict(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
        num_position_embeddings=64,
        deepstack_visual_indexes=[],
    )
    Qwen3_5MoeConfig(
        architectures=["Qwen3_5MoeForConditionalGeneration"], text_config=_TINY_QWEN35_TEXT, vision_config=vision
    ).save_pretrained(tmp_path)
    monkeypatch.setattr(qwen35, "AutoBridge", _BuilderOnlyBridge)
    recipe = qwen35.qwen35_vl_35b_a3b_rl_16gpu_gb200_bf16(hf_path=str(tmp_path), seq_length=4096)
    _assert_35b_a3b_execution_settings(recipe.model, 4096)
    assert is_multimodal(recipe)
    # The built model only carries the nested TransformerConfig, so M-RoPE must come from the recipe's config.
    assert uses_mrope(recipe.model)
    assert not hasattr(recipe.model.transformer, "position_embedding_type")
    # The vision-freeze overrides and LoRA's target scoping read these flags from the recipe's config.
    assert recipe.model.freeze_vision_model is False
    assert recipe.model.vision_recompute_granularity == "full"
