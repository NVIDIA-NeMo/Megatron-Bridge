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

import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


_SCRIPT = Path(__file__).parents[3] / "examples" / "conversion" / "hf_to_megatron_generate_text.py"
_SCRIPT_GLOBALS = runpy.run_path(_SCRIPT)
_build_parser = _SCRIPT_GLOBALS["build_parser"]
_decode_completion = _SCRIPT_GLOBALS["_decode_completion"]
_hf_revision_kwargs = _SCRIPT_GLOBALS["_hf_revision_kwargs"]
_maybe_gather_tensor_parallel_logits = _SCRIPT_GLOBALS["_maybe_gather_tensor_parallel_logits"]
_run_megatron_forward = _SCRIPT_GLOBALS["_run_megatron_forward"]
_text_forward_step = _SCRIPT_GLOBALS["text_forward_step"]
_tokenize_prompt = _SCRIPT_GLOBALS["_tokenize_prompt"]
_InferenceMode = _SCRIPT_GLOBALS["InferenceMode"]


@pytest.mark.unit
def test_tokenize_raw_prompt() -> None:
    tokenizer = MagicMock()
    tokenizer.encode.return_value = torch.tensor([[1, 2]])

    input_ids = _tokenize_prompt(tokenizer, "hello", apply_chat_template=False, thinking_mode="adaptive")

    torch.testing.assert_close(input_ids, torch.tensor([[1, 2]]))
    tokenizer.encode.assert_called_once_with("hello", return_tensors="pt")
    tokenizer.apply_chat_template.assert_not_called()


@pytest.mark.unit
def test_tokenize_chat_prompt() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {"input_ids": torch.tensor([[3, 4]])}

    input_ids = _tokenize_prompt(tokenizer, "hello", apply_chat_template=True, thinking_mode="disabled")

    torch.testing.assert_close(input_ids, torch.tensor([[3, 4]]))
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        thinking_mode="disabled",
    )
    tokenizer.encode.assert_not_called()


@pytest.mark.unit
def test_decode_completion_excludes_prompt_and_special_tokens() -> None:
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "The sky appears blue."

    text = _decode_completion(tokenizer, torch.tensor([[10, 11, 20, 21]]), prompt_length=2)

    assert text == "The sky appears blue."
    tokenizer.decode.assert_called_once_with([20, 21], skip_special_tokens=True)


@pytest.mark.unit
def test_hf_revision_is_parsed_and_forwarded() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret

    args = _build_parser().parse_args(["--hf_model_path", "org/model", "--hf-revision", revision])

    assert args.hf_revision == revision
    assert _hf_revision_kwargs(args.hf_revision) == {"revision": revision}
    assert _hf_revision_kwargs(None) == {}


@pytest.mark.unit
def test_text_forward_step_passes_static_context_and_gather_request() -> None:
    inference_context = object()
    batch = {
        "tokens": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.arange(3).unsqueeze(0),
        "inference_context": inference_context,
    }
    model = MagicMock(return_value=torch.randn(1, 3, 16))

    _text_forward_step(iter([batch]), model)

    assert model.call_args.kwargs["inference_context"] is inference_context
    assert model.call_args.kwargs["runtime_gather_output"] is True


@pytest.mark.unit
def test_generation_forward_activates_inference_mode() -> None:
    forward = MagicMock(return_value="output")
    context = MagicMock()

    with patch.object(_InferenceMode, "active", return_value=context) as active:
        result = _run_megatron_forward(forward, forward_only=True)

    assert result == "output"
    active.assert_called_once_with()
    context.__enter__.assert_called_once_with()
    context.__exit__.assert_called_once()
    forward.assert_called_once_with(forward_only=True)


@pytest.mark.unit
def test_generation_skips_tp_gather_for_complete_vocabulary() -> None:
    full_logits = torch.randn(1, 1, 128)

    with patch.object(torch.distributed, "all_gather") as all_gather:
        result = _maybe_gather_tensor_parallel_logits(full_logits, 128, 2, object())

    assert result is full_logits
    all_gather.assert_not_called()


@pytest.mark.unit
def test_generation_gathers_tp_vocabulary_shards() -> None:
    local_logits = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64)
    tp_group = object()

    def mock_all_gather(outputs, tensor, group):
        assert group is tp_group
        outputs[0].copy_(tensor)
        outputs[1].copy_(tensor + 64)

    with patch.object(torch.distributed, "all_gather", side_effect=mock_all_gather) as all_gather:
        result = _maybe_gather_tensor_parallel_logits(local_logits, 128, 2, tp_group)

    assert result.shape == (1, 1, 128)
    assert torch.equal(result[..., :64], local_logits)
    assert torch.equal(result[..., 64:], local_logits + 64)
    all_gather.assert_called_once()
