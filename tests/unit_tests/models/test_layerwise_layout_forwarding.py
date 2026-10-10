from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.unit
@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("layout", [None, False, True])
def test_get_model_forwards_layerwise_layout(monkeypatch, layerwise, layout):
    from megatron.bridge.models import model_provider as module

    model = [torch.nn.Linear(2, 2)]
    config = SimpleNamespace(use_cpu_initialization=True, init_model_with_meta_device=False, fp16=False, bf16=False)
    monkeypatch.setattr(module, "_create_model", Mock(return_value=model))
    monkeypatch.setattr(module, "_finalize_model_quantization", Mock())
    monkeypatch.setattr(module, "_print_num_params", Mock())
    monkeypatch.setattr(module, "get_model_config", Mock(return_value=config))
    monkeypatch.setattr(module, "correct_amax_history_if_needed", None)
    wrap = Mock(return_value=model)
    monkeypatch.setattr(module, "_ddp_wrap", wrap)
    options = {} if layout is None else {"use_layer_wise_param_layout": layout}
    result = module.get_model(
        SimpleNamespace(),
        SimpleNamespace(),
        pg_collection=SimpleNamespace(),
        use_layer_wise_distributed_optimizer=layerwise,
        **options,
    )
    assert result is model
    assert wrap.call_args.kwargs["use_layer_wise_distributed_optimizer"] is layerwise
    assert wrap.call_args.kwargs["use_layer_wise_param_layout"] is (True if layout is None else layout)
