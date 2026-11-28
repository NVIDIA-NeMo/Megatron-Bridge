"""Recipe defaults, explicit overrides, serialization, and training entrypoints."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step import (
    get_forward_step_reference,
    load_forward_step_reference,
    resolve_forward_step,
)
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
from megatron.bridge.utils.instantiate_utils import InstantiationException
from tests.unit_tests.training.test_config import (
    create_test_config_container,
    create_test_gpt_config,
    restore_get_world_size_safe,
)


pytestmark = pytest.mark.unit
STEP = f"{__name__}:example_forward_step"
CLASS_STEP = f"{__name__}:ExampleStep"
QWEN_STEP = "megatron.bridge.models.qwen_vl.qwen3_vl_step:forward_step"
VLM_STEP = "megatron.bridge.training.vlm_step:forward_step"


def example_forward_step(data_iterator, model):
    return model, data_iterator


class ExampleStep:
    def __init__(self, mode="pretrain"):
        self.mode = mode

    def __call__(self, data_iterator, model):
        return model, data_iterator


def config(reference=STEP, **dataset_fields):
    return SimpleNamespace(
        forward_step=reference,
        dataset=SimpleNamespace(**dataset_fields),
        checkpoint=SimpleNamespace(pretrained_checkpoint="checkpoint", load=None),
        inprocess_restart=None,
        env_vars={},
    )


def test_function_reference_is_loaded_without_executing_step():
    assert load_forward_step_reference(STEP) is example_forward_step


@pytest.mark.parametrize("mode", ["pretrain", "finetune"])
def test_class_reference_constructs_for_the_training_mode(mode):
    step = load_forward_step_reference(CLASS_STEP, mode=mode)
    assert isinstance(step, ExampleStep)
    assert step.mode == mode


@pytest.mark.parametrize("reference", [None, "", "not-an-import-path", "untrusted.module:step"])
def test_explicit_callable_wins_without_loading_config_reference(reference):
    step = Mock()
    assert resolve_forward_step(config(reference), step) is step


def test_missing_default_has_actionable_error():
    with pytest.raises(ValueError, match="set config.forward_step or pass forward_step_func"):
        resolve_forward_step(config(None))


@pytest.mark.parametrize("reference", ["", 3, False])
def test_invalid_default_fails_before_import(reference):
    with pytest.raises(ValueError, match="config.forward_step"):
        resolve_forward_step(config(reference))


@pytest.mark.parametrize("reference", ["no_separator", ":step", "module:", "module:step:extra"])
def test_invalid_import_reference(reference):
    with pytest.raises(ValueError, match="expected module:attribute"):
        load_forward_step_reference(reference)


def test_non_callable_target_is_rejected():
    with pytest.raises(InstantiationException, match="callable"):
        load_forward_step_reference(f"{__name__}:STEP")


def test_config_target_policy_is_not_bypassed():
    with pytest.raises(InstantiationException, match="not (in|allowed)"):
        load_forward_step_reference("os:system")


def test_qwen_native_packing_uses_effective_dataset():
    cfg = config(QWEN_STEP)
    assert get_forward_step_reference(cfg) == QWEN_STEP
    cfg.dataset.packing_buffer_size = 8
    assert get_forward_step_reference(cfg) == VLM_STEP
    with patch("megatron.bridge.training.forward_step.load_forward_step_reference") as loader:
        resolve_forward_step(cfg, mode="finetune")
    loader.assert_called_once_with(VLM_STEP, mode="finetune")
    assert cfg.forward_step == VLM_STEP


def test_explicit_step_bypasses_qwen_default_adaptation():
    cfg = config(QWEN_STEP, packing_buffer_size=8)
    override = Mock()
    assert resolve_forward_step(cfg, override) is override
    assert cfg.forward_step == QWEN_STEP


def test_finetune_uses_recipe_step_and_preserves_callbacks():
    from megatron.bridge.training.finetune import finetune

    cfg = config(CLASS_STEP)
    callbacks = [Mock()]
    with patch("megatron.bridge.training.finetune.pretrain") as train:
        finetune(cfg, callbacks=callbacks)
    args, kwargs = train.call_args
    assert args[0] is cfg
    assert isinstance(args[1], ExampleStep)
    assert args[1].mode == "finetune"
    assert kwargs == {"callbacks": callbacks}
    assert cfg._checkpoint_load_required


def test_pretrain_uses_recipe_step_after_runtime_setup():
    from megatron.bridge.training.pretrain import pretrain

    cfg = config()
    with (
        patch("megatron.bridge.training.pretrain.runtime_config_update") as runtime,
        patch("megatron.bridge.training.pretrain.GlobalState"),
        patch("megatron.bridge.training.pretrain.resolve_forward_step", wraps=resolve_forward_step) as resolve,
        patch("megatron.bridge.training.pretrain._pretrain") as train,
    ):
        calls = Mock()
        calls.attach_mock(runtime, "runtime")
        calls.attach_mock(resolve, "resolve")
        pretrain(cfg)
    assert [call[0] for call in calls.mock_calls] == ["runtime", "resolve"]
    assert train.call_args.kwargs["forward_step_func"] is example_forward_step
    assert train.call_args.kwargs["state"].cfg is cfg


def test_forward_step_survives_config_overrides_and_yaml(tmp_path):
    cfg, original_world_size, config_module = create_test_config_container(
        model_config=create_test_gpt_config(), world_size_override=1
    )
    try:
        cfg.forward_step = STEP
        configured = process_config_with_overrides(cfg, cli_overrides=[f"forward_step={CLASS_STEP}"])
        assert configured.forward_step == CLASS_STEP
        saved = tmp_path / "config.yaml"
        configured.to_yaml(str(saved))
        restored = ConfigContainer.from_yaml(str(saved))
        assert restored.forward_step == CLASS_STEP
        assert resolve_forward_step(restored, mode="finetune").mode == "finetune"
    finally:
        restore_get_world_size_safe(original_world_size, config_module)


def test_old_configs_have_no_required_step_field():
    field = next(field for field in fields(ConfigContainer) if field.name == "forward_step")
    assert field.default is None
    cfg, original_world_size, config_module = create_test_config_container(
        model_config=create_test_gpt_config(), world_size_override=1
    )
    try:
        saved = cfg.to_dict()
        saved.pop("forward_step")
        restored = ConfigContainer.from_dict(saved)
        assert restored.forward_step is None
    finally:
        restore_get_world_size_safe(original_world_size, config_module)
