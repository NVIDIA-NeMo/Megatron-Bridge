import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec, patch

import pytest


pytestmark = pytest.mark.unit


def _load_checkpointing_compat():
    # The adapter only uses the standard library. Exercise it without importing
    # Bridge's model registry or requiring the GPU runtime in the test process.
    helper_path = Path(__file__).resolve().parents[3] / "src/megatron/bridge/training/utils/mcore_checkpointing.py"
    spec = importlib.util.spec_from_file_location("_bridge_mcore_checkpointing_test", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = _load_checkpointing_compat()


@pytest.mark.parametrize("gtp_enabled", [False, True])
def test_legacy_padding_helpers_require_gtp_only_when_requested(gtp_enabled):
    state_dict = {"weight": object()}
    with patch.object(compat, "import_module", return_value=SimpleNamespace()):
        if gtp_enabled:
            with pytest.raises(RuntimeError, match="GTP padding helpers"):
                compat.apply_gtp_checkpoint_padding(state_dict, "checkpoint", None, gtp_enabled=True)
        else:
            before = state_dict.copy()
            compat.apply_gtp_checkpoint_padding(state_dict, "checkpoint", None, gtp_enabled=False)
            assert state_dict == before


def test_supported_padding_helpers_receive_model_precision():
    resolve = Mock(return_value=32)
    grant = Mock()
    model_config = SimpleNamespace(fp4=None, fp8="hybrid", fp8_recipe="mxfp8")
    with patch.object(
        compat,
        "import_module",
        return_value=SimpleNamespace(
            resolve_gtp_pad_for_alignment=resolve, grant_shape_mismatch_for_gtp_padding=grant
        ),
    ):
        state_dict = {"weight": object()}
        compat.apply_gtp_checkpoint_padding(state_dict, "checkpoint", model_config, gtp_enabled=True)
    resolve.assert_called_once_with(fp4=False, fp8_recipe="mxfp8", fp8=True)
    grant.assert_called_once_with(state_dict, "checkpoint", 32)


@pytest.mark.parametrize("gtp_enabled", [False, True])
def test_missing_gtp_module_preserves_only_ordinary_loads(gtp_enabled):
    missing = ModuleNotFoundError(name="megatron.core.tensor_parallel.gtp_api")
    with patch.object(compat, "import_module", side_effect=missing):
        if gtp_enabled:
            with pytest.raises(RuntimeError, match="GTP support"):
                compat.gtp_checkpoint_load_context(object(), gtp_enabled=True)
        else:
            with compat.gtp_checkpoint_load_context(object(), gtp_enabled=False):
                pass


def test_gtp_module_dependency_errors_are_not_hidden():
    missing = ModuleNotFoundError(name="transformer_engine")
    with patch.object(compat, "import_module", side_effect=missing), pytest.raises(ModuleNotFoundError) as caught:
        compat.gtp_checkpoint_load_context(object(), gtp_enabled=False)
    assert caught.value is missing


def test_supported_gtp_load_context_is_forwarded():
    module = object()
    context = nullcontext()
    create_context = Mock(return_value=context)
    with patch.object(
        compat,
        "import_module",
        return_value=SimpleNamespace(HAVE_GTP=True, gtp_native_fp8_load_context=create_context),
    ):
        assert compat.gtp_checkpoint_load_context(module, gtp_enabled=True) is context
    create_context.assert_called_once_with(module)


def test_tokenizer_assets_delegate_to_available_mcore_helper():
    save = Mock()
    tokenizer, config = object(), object()
    with patch.object(compat, "import_module", return_value=SimpleNamespace(save_tokenizer_assets=save)):
        compat.save_tokenizer_assets(tokenizer, config, "checkpoint")
    save.assert_called_once_with(tokenizer, config, "checkpoint")


def test_missing_tokenizer_save_support_is_explicit():
    with (
        patch.object(compat, "import_module", return_value=SimpleNamespace()),
        pytest.raises(RuntimeError, match="checkpoint.save_tokenizer_assets=False"),
    ):
        compat.save_tokenizer_assets(object(), object(), "checkpoint")


def test_strict_tokenizer_save_forwards_error_policy():
    def save_assets(tokenizer, config, checkpoint_path, *, raise_on_error=False):
        pass

    save = create_autospec(save_assets)
    tokenizer, config = object(), object()
    with patch.object(compat, "import_module", return_value=SimpleNamespace(save_tokenizer_assets=save)):
        compat.save_tokenizer_assets(tokenizer, config, "checkpoint", raise_on_error=True)
    save.assert_called_once_with(tokenizer, config, "checkpoint", raise_on_error=True)


@pytest.mark.parametrize("raise_on_error", [False, True])
def test_legacy_tokenizer_save_does_not_silently_discard_strict_policy(raise_on_error):
    def save_assets(tokenizer, config, checkpoint_path):
        pass

    save = create_autospec(save_assets)
    tokenizer, config = object(), object()
    with patch.object(compat, "import_module", return_value=SimpleNamespace(save_tokenizer_assets=save)):
        if raise_on_error:
            with pytest.raises(RuntimeError, match="supports raise_on_error=True"):
                compat.save_tokenizer_assets(tokenizer, config, "checkpoint", raise_on_error=True)
            save.assert_not_called()
        else:
            compat.save_tokenizer_assets(tokenizer, config, "checkpoint")
            save.assert_called_once_with(tokenizer, config, "checkpoint")


def test_strict_tokenizer_save_errors_propagate():
    error = OSError("tokenizer write failed")
    save = Mock(side_effect=error)
    with (
        patch.object(compat, "import_module", return_value=SimpleNamespace(save_tokenizer_assets=save)),
        pytest.raises(OSError) as caught,
    ):
        compat.save_tokenizer_assets(object(), object(), "checkpoint", raise_on_error=True)
    assert caught.value is error
