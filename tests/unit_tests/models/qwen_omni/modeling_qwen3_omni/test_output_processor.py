# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace
from typing import Callable, cast

import pytest
import torch
from torch import Tensor

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model import Qwen3OmniThinkerModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel


pytestmark = pytest.mark.unit


def _thinker(language_model: Callable[..., object]) -> Qwen3OmniThinkerModel:
    # This proxy supplies the attributes used by the pre_process=False branch,
    # avoiding multimodal encoder construction and process-group initialization.
    return cast(
        Qwen3OmniThinkerModel,
        SimpleNamespace(
            pg_collection=None,
            pre_process=False,
            config=SimpleNamespace(sequence_parallel=False),
            language_model=language_model,
        ),
    )


def _forward(
    thinker: Qwen3OmniThinkerModel,
    *,
    output_processor: Callable[..., object] | None = None,
    output_processor_context: object | None = None,
) -> object:
    return Qwen3OmniThinkerModel.forward(
        thinker,
        torch.tensor([[1, 2]]),
        position_ids=torch.zeros(3, 1, 2, dtype=torch.long),
        output_processor=output_processor,
        output_processor_context=output_processor_context,
    )


def _hidden(kwargs: dict[str, object]) -> Tensor:
    hidden = kwargs["hidden_states"]
    assert isinstance(hidden, Tensor)
    return hidden


def _processor(kwargs: dict[str, object]) -> Callable[..., object]:
    processor = kwargs["output_processor"]
    assert callable(processor)
    # The callback was supplied through the typed test forward boundary above.
    return cast(Callable[..., object], processor)


def test_default_output_processor_preserves_result() -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    def language_model(**kwargs: object) -> object:
        calls.append(kwargs)
        assert kwargs.get("output_processor") is None
        assert kwargs.get("output_processor_context") is None
        return sentinel

    assert _forward(_thinker(language_model)) is sentinel
    assert len(calls) == 1


def test_output_processor_and_empty_context_are_forwarded() -> None:
    context: dict[str, object] = {}
    result = {"log_probs": torch.ones(1, 2), "entropy": torch.zeros(1, 2)}

    def processor(**kwargs: object) -> dict[str, Tensor]:
        assert kwargs["context"] is context
        return result

    def language_model(**kwargs: object) -> object:
        assert kwargs["output_processor"] is processor
        assert kwargs["output_processor_context"] is context
        return _processor(kwargs)(hidden_states=torch.zeros(2, 1, 3), context=context)

    assert _forward(_thinker(language_model), output_processor=processor, output_processor_context=context) is result


def test_callback_gradient_flows_through_language_model() -> None:
    hidden = torch.tensor([[[1.0, 2.0]]], requires_grad=True)

    def processor(**kwargs: object) -> Tensor:
        return _hidden(kwargs).square()

    def language_model(**kwargs: object) -> object:
        return _processor(kwargs)(hidden_states=hidden, context=kwargs["output_processor_context"])

    result = _forward(_thinker(language_model), output_processor=processor)
    assert isinstance(result, Tensor)
    torch.autograd.backward(result.sum())
    torch.testing.assert_close(hidden.grad, 2 * hidden)


def test_thinker_and_vl_forward_preserve_callback_gradient() -> None:
    hidden = torch.randn(2, 1, 4, requires_grad=True)
    context = object()
    calls: list[dict[str, object]] = []

    def processor(**kwargs: object) -> Tensor:
        assert kwargs["context"] is context
        calls.append(kwargs)
        return _hidden(kwargs).square()

    def preprocess(**kwargs: object) -> tuple[Tensor, None, None, None, None]:
        return hidden, None, None, None, None

    def decoder(**kwargs: object) -> Tensor:
        return _hidden(kwargs)

    def postprocess(**kwargs: object) -> object:
        return _processor(kwargs)(hidden_states=_hidden(kwargs), context=kwargs["output_processor_context"])

    # This proxy implements the VL forward attributes exercised by the test;
    # no constructor or model-parallel state is needed for this callback chain.
    language = cast(
        Qwen3VLGPTModel,
        SimpleNamespace(
            _preprocess=preprocess,
            decoder=decoder,
            mtp_process=False,
            config=SimpleNamespace(sequence_parallel=False),
            _postprocess=postprocess,
        ),
    )

    def language_model(**kwargs: object) -> object:
        # kwargs are the real Thinker forward's language-model call arguments.
        return _invoke_vl(language, kwargs)

    result = _forward(_thinker(language_model), output_processor=processor, output_processor_context=context)
    assert isinstance(result, Tensor)
    torch.autograd.backward(result.sum())
    torch.testing.assert_close(hidden.grad, 2 * hidden)
    assert len(calls) == 1


def test_output_processor_error_is_not_silenced() -> None:
    original = RuntimeError("processor failure")

    def processor(**kwargs: object) -> object:
        raise original

    def language_model(**kwargs: object) -> object:
        return _processor(kwargs)(hidden_states=torch.zeros(2, 1, 3))

    with pytest.raises(RuntimeError) as caught:
        _forward(_thinker(language_model), output_processor=processor)
    assert caught.value is original


def _invoke_vl(language: Qwen3VLGPTModel, kwargs: dict[str, object]) -> object:
    # Torch module calls expose dynamic keyword dictionaries; this unbound
    # forward is invoked with the real Thinker language-model call arguments.
    forward = cast(Callable[..., object], Qwen3VLGPTModel.forward)
    return forward(language, **kwargs)


def test_native_vl_forwards_structured_callback_output() -> None:
    hidden = torch.randn(2, 1, 4, requires_grad=True)
    context: dict[str, object] = {}

    def processor(**kwargs: object) -> dict[str, Tensor]:
        assert kwargs["context"] is context
        value = _hidden(kwargs)
        return {"log_probs": value.sum(dim=-1), "entropy": value.square().sum(dim=-1)}

    def preprocess(**kwargs: object) -> tuple[Tensor, None, None, None, None]:
        return hidden, None, None, None, None

    def decoder(**kwargs: object) -> Tensor:
        return _hidden(kwargs)

    def postprocess(**kwargs: object) -> object:
        return _processor(kwargs)(hidden_states=_hidden(kwargs), context=kwargs["output_processor_context"])

    # The proxy provides only the attributes used by this native VL forward.
    language = cast(
        Qwen3VLGPTModel,
        SimpleNamespace(
            _preprocess=preprocess,
            decoder=decoder,
            mtp_process=False,
            config=SimpleNamespace(sequence_parallel=False),
            _postprocess=postprocess,
        ),
    )

    def language_model(**kwargs: object) -> object:
        return _invoke_vl(language, kwargs)

    result = _forward(_thinker(language_model), output_processor=processor, output_processor_context=context)
    assert isinstance(result, dict) and set(result) == {"log_probs", "entropy"}
    assert isinstance(result["log_probs"], Tensor)
    torch.autograd.backward(result["log_probs"].sum())
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
