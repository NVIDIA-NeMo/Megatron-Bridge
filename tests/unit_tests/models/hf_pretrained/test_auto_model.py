"""Verify remote AutoModel loader arguments without downloading a checkpoint."""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.auto_model import PreTrainedAutoModel


pytestmark = pytest.mark.unit


def test_auto_model_loader_preserves_source_revision_and_trust() -> None:
    wrapper = PreTrainedAutoModel(
        model_name_or_path="test/decision-model",
        device="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        revision="pinned-source-revision",
    )
    wrapper.config = Mock()
    model = Mock()
    model.to.return_value = model
    with patch(
        "megatron.bridge.models.hf_pretrained.auto_model.AutoModel.from_pretrained", return_value=model
    ) as loader:
        assert wrapper.model is model
    loader.assert_called_once_with(
        "test/decision-model",
        trust_remote_code=True,
        config=wrapper.config,
        revision="pinned-source-revision",
        torch_dtype=torch.float32,
    )
    model.to.assert_called_once_with("cpu")


def test_auto_model_wrapper_is_preserved_for_checkpoint_reload() -> None:
    wrapper = PreTrainedAutoModel(model_name_or_path="test/decision-model", device="cpu")
    bridge = AutoBridge(wrapper)
    assert bridge._pretrained_wrapper_cls is PreTrainedAutoModel
