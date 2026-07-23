import importlib.util
from pathlib import Path

import torch
from PIL import Image


_HELPER_PATH = (
    Path(__file__).parents[3]
    / "skills"
    / "create-model-verification-card"
    / "scripts"
    / "verify_hf_inference.py"
)
_SPEC = importlib.util.spec_from_file_location("model_verification_hf_inference", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)


class _FakeProcessor:
    chat_template = "template"

    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {"input_ids": torch.tensor([[1, 2]])}


def test_prepare_vlm_inputs_preserves_image_and_prompt(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="red").save(image_path)
    processor = _FakeProcessor()

    inputs = _HELPER._prepare_vlm_inputs(processor, "What is shown?", str(image_path))

    assert inputs["input_ids"].tolist() == [[1, 2]]
    assert processor.messages[0]["content"][1] == {"type": "text", "text": "What is shown?"}
    assert isinstance(processor.messages[0]["content"][0]["image"], Image.Image)
    assert processor.kwargs == {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }


def test_move_inputs_casts_only_floating_tensors() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        "metadata": "kept",
    }

    moved = _HELPER._move_inputs(inputs, torch.device("cpu"), torch.bfloat16)

    assert moved["input_ids"].dtype == torch.int64
    assert moved["pixel_values"].dtype == torch.bfloat16
    assert moved["metadata"] == "kept"
