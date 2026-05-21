import inspect

import pytest
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl._mcore_compat import (
    ensure_mtp_checkpointed_forward_accepts_padding_mask,
)


@pytest.mark.unit
def test_mtp_checkpointed_forward_accepts_padding_mask() -> None:
    ensure_mtp_checkpointed_forward_accepts_padding_mask()

    parameters = inspect.signature(MultiTokenPredictionLayer._checkpointed_forward).parameters
    assert "padding_mask" in parameters
