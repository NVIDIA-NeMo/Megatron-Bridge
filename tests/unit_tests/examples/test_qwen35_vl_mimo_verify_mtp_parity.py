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

"""Unit tests for the Qwen3.5-VL MegatronMIMO MTP parity verification script."""

import importlib.util
import pathlib
import sys
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "examples" / "megatron_mimo" / "qwen35_vl" / "verify_mtp_parity.py"
)
_IMAGE_TOKEN_ID = 2040


@pytest.fixture(scope="module")
def parity():
    name = "qwen35_vl_mimo_verify_mtp_parity_under_test"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def _hf_config():
    return SimpleNamespace(
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=2041,
        vision_start_token_id=2042,
        vision_end_token_id=2043,
        vision_config=SimpleNamespace(spatial_merge_size=2, in_channels=3, temporal_patch_size=2, patch_size=14),
    )


class TestBuildBatch:
    def test_layout(self, parity):
        batch = parity.build_batch(_hf_config(), seq_length=64, image_grid_thw=(1, 4, 4), seed=0)

        input_ids = batch["input_ids"]
        assert input_ids.shape == (1, 64)
        image_positions = (input_ids[0] == _IMAGE_TOKEN_ID).nonzero().flatten().tolist()
        assert len(image_positions) == 4
        assert input_ids[0, image_positions[0] - 1] == 2042 and input_ids[0, image_positions[-1] + 1] == 2043
        assert int((input_ids < 2040).sum()) == 64 - 6
        assert batch["position_ids"].shape == (3, 1, 64)
        assert batch["loss_mask"][0, :-1].eq(1).all() and batch["loss_mask"][0, -1] == 0
        assert torch.equal(batch["labels"][0, :-1], input_ids[0, 1:])
        encoder_inputs = batch["modality_inputs"]["images"]["qwen_visual"]
        assert encoder_inputs["hidden_states"].shape == (16, 3 * 2 * 14 * 14)
        assert encoder_inputs["grid_thw"].tolist() == [[1, 4, 4]]

    def test_is_deterministic_for_a_seed(self, parity):
        first = parity.build_batch(_hf_config(), seq_length=64, image_grid_thw=(1, 4, 4), seed=7)
        second = parity.build_batch(_hf_config(), seq_length=64, image_grid_thw=(1, 4, 4), seed=7)

        assert torch.equal(first["input_ids"], second["input_ids"])
        assert torch.equal(
            first["modality_inputs"]["images"]["qwen_visual"]["hidden_states"],
            second["modality_inputs"]["images"]["qwen_visual"]["hidden_states"],
        )

    def test_rejects_grid_not_divisible_by_merge(self, parity):
        with pytest.raises(ValueError, match="multiples of spatial_merge_size"):
            parity.build_batch(_hf_config(), seq_length=64, image_grid_thw=(1, 3, 4), seed=0)

    def test_rejects_sequence_without_room_for_text(self, parity):
        with pytest.raises(ValueError, match="leaves no room for text"):
            parity.build_batch(_hf_config(), seq_length=7, image_grid_thw=(1, 4, 4), seed=0)


def _save_results(
    output_dir: pathlib.Path,
    *,
    mimo_language: dict | None = None,
    mimo_images: dict | None = None,
    edit: Callable[[dict, dict, dict], None] | None = None,
):
    """Save matching standard and MegatronMIMO results; ``edit(standard, language, images)`` alters them first."""
    generator = torch.Generator().manual_seed(0)
    per_token = torch.rand(1, 8, generator=generator)
    language_grads = {"decoder.weight": torch.randn(4, 4, generator=generator), "mtp.layers.0.eh_proj.weight": None}
    image_grads = {"patch_embed.proj.weight": torch.tensor([0.0, 1.0, 2.0, 3.0])}
    language_weights = {"decoder.weight": "bf16:(4, 4):aa", "mtp.layers.0.eh_proj.weight": "bf16:(4, 8):bb"}
    image_weights = {"patch_embed.proj.weight": "bf16:(4,):cc"}
    standard = {
        "weights": {"language": language_weights, "images": image_weights},
        "masked": {
            "per_token": per_token,
            "lm_loss": 7.5,
            "mtp_losses": [7.25],
            "grads": {"language": language_grads, "images": image_grads},
        },
        "unmasked": {"per_token": per_token, "lm_loss": 7.5, "mtp_losses": [7.3]},
    }
    language = {
        "weights": dict(language_weights),
        "grads": {k: None if v is None else v.clone() for k, v in language_grads.items()},
        "per_token": per_token.clone(),
        "lm_loss": 7.5,
        "mtp_losses": [7.25],
    }
    images = {"weights": dict(image_weights), "grads": {k: v.clone() for k, v in image_grads.items()}}
    language.update(mimo_language or {})
    images.update(mimo_images or {})
    if edit is not None:
        edit(standard, language, images)
    torch.save(standard, output_dir / "standard.pt")
    torch.save(language, output_dir / "mimo_language.pt")
    torch.save(images, output_dir / "mimo_images.pt")


class TestCompareResults:
    def test_matching_paths_pass(self, parity, tmp_path):
        _save_results(tmp_path)

        assert parity.compare_results(tmp_path, min_cosine=0.999, loss_rtol=1e-3) == []

    def test_differing_weight_fails(self, parity, tmp_path):
        _save_results(
            tmp_path,
            mimo_language={"weights": {"decoder.weight": "bf16:(4, 4):aa", "mtp.layers.0.eh_proj.weight": "x"}},
        )

        failures = parity.compare_results(tmp_path, min_cosine=0.999, loss_rtol=1e-3)

        assert failures == ["language weights differ: ['mtp.layers.0.eh_proj.weight']"]

    def test_differing_mtp_loss_fails(self, parity, tmp_path):
        _save_results(tmp_path, mimo_language={"mtp_losses": [7.3]})

        failures = parity.compare_results(tmp_path, min_cosine=0.999, loss_rtol=1e-3)

        assert len(failures) == 1 and failures[0].startswith("mtp_1 loss differs")

    def test_differing_gradient_fails(self, parity, tmp_path):
        _save_results(tmp_path, mimo_images={"grads": {"patch_embed.proj.weight": torch.tensor([3.0, 2.0, 1.0, 0.0])}})

        failures = parity.compare_results(tmp_path, min_cosine=0.999, loss_rtol=1e-3)

        assert len(failures) == 1 and failures[0].startswith("images gradient patch_embed.proj.weight differs")

    def test_gradient_in_one_path_only_fails(self, parity, tmp_path):
        _save_results(
            tmp_path,
            mimo_language={
                "grads": {"decoder.weight": torch.zeros(4, 4), "mtp.layers.0.eh_proj.weight": torch.ones(4, 8)}
            },
        )

        failures = parity.compare_results(tmp_path, min_cosine=0.999, loss_rtol=1e-3)

        assert "language gradient mtp.layers.0.eh_proj.weight exists in only one path" in failures
        assert any(failure.startswith("language gradient decoder.weight differs") for failure in failures)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("side", ["standard", "mimo"])
    @pytest.mark.parametrize(
        ("quantity", "failure"),
        [
            ("mtp_losses", "mtp_1 loss is not finite"),
            ("lm_loss", "lm loss is not finite"),
            ("per_token", "lm loss is not finite"),
            ("grads", "images gradient patch_embed.proj.weight is not finite"),
        ],
    )
    def test_nonfinite_result_fails(self, parity, tmp_path, quantity, failure, side, value):
        def inject(standard, language, images):
            if quantity == "grads":
                grads = standard["masked"]["grads"]["images"] if side == "standard" else images["grads"]
                grads["patch_embed.proj.weight"][0] = value
                return
            target = standard["masked"] if side == "standard" else language
            if quantity == "mtp_losses":
                target["mtp_losses"] = [value]
            elif quantity == "lm_loss":
                target["lm_loss"] = value
            else:
                target["per_token"][0, 0] = value

        _save_results(tmp_path, edit=inject)

        failures = parity.compare_results(tmp_path, min_cosine=0.999, loss_rtol=1e-3)

        assert len(failures) == 1 and failures[0].startswith(failure)

    def test_identical_infinite_gradients_fail(self, parity, tmp_path):
        def inject(standard, language, images):
            standard["masked"]["grads"]["images"]["patch_embed.proj.weight"][0] = float("inf")
            images["grads"]["patch_embed.proj.weight"][0] = float("inf")

        _save_results(tmp_path, edit=inject)

        failures = parity.compare_results(tmp_path, min_cosine=0.999, loss_rtol=1e-3)

        assert failures == ["images gradient patch_embed.proj.weight is not finite"]

    def test_identical_zero_gradients_pass(self, parity, tmp_path):
        def zero(standard, language, images):
            standard["masked"]["grads"]["images"]["patch_embed.proj.weight"].zero_()
            images["grads"]["patch_embed.proj.weight"].zero_()

        _save_results(tmp_path, edit=zero)

        assert parity.compare_results(tmp_path, min_cosine=0.999, loss_rtol=1e-3) == []

    @pytest.mark.parametrize(("mtp_loss", "exit_code"), [(7.25, 0), (float("nan"), 1)])
    def test_compare_exit_status(self, parity, tmp_path, monkeypatch, mtp_loss, exit_code):
        _save_results(tmp_path, mimo_language={"mtp_losses": [mtp_loss]})
        monkeypatch.setattr(sys, "argv", ["verify_mtp_parity.py", "compare", "--output-dir", str(tmp_path)])

        with pytest.raises(SystemExit) as exited:
            parity.main()

        assert exited.value.code == exit_code
