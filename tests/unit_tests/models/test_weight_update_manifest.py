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

from collections.abc import Callable
from typing import cast

import pytest
import torch

from megatron.bridge.models.conversion import (
    WeightUpdateManifest,
    WeightUpdateMode,
    WeightUpdateRecorder,
    WeightUpdateTensor,
)


def _weights() -> list[tuple[str, torch.Tensor]]:
    return [
        ("model.layers.0.mlp.experts.0.weight", torch.empty(4, 8, dtype=torch.bfloat16)),
        ("model.layers.0.mlp.experts.1.weight", torch.empty(4, 8, dtype=torch.bfloat16)),
    ]


def _record(weights: list[tuple[str, torch.Tensor]]) -> WeightUpdateRecorder:
    recorder = WeightUpdateRecorder()
    for name, tensor in weights:
        recorder.record(name, tensor)
    return recorder


def _manifest(*, update_mode: WeightUpdateMode = "full") -> WeightUpdateManifest:
    recorder = _record(_weights())
    return recorder.build_manifest(
        model_id="org/model",
        model_config_id="config-sha256:abc123",
        update_mode=update_mode,
        base_version="41",
        target_version="42",
    )


def _validate(manifest: WeightUpdateManifest, recorder: WeightUpdateRecorder) -> None:
    manifest.validate(
        recorder,
        expected_model_id="org/model",
        expected_model_config_id="config-sha256:abc123",
        expected_update_mode=manifest.update_mode,
        current_version="41",
        expected_target_version="42",
    )


def test_recorder_tracks_one_shot_stream_without_retaining_payloads() -> None:
    weights = _weights()
    recorder = WeightUpdateRecorder()

    streamed = list(recorder.track(iter(weights)))

    assert [id(tensor) for _, tensor in streamed] == [id(tensor) for _, tensor in weights]
    assert recorder.tensors == (
        WeightUpdateTensor("model.layers.0.mlp.experts.0.weight", (4, 8), "bfloat16"),
        WeightUpdateTensor("model.layers.0.mlp.experts.1.weight", (4, 8), "bfloat16"),
    )
    assert not any(isinstance(value, torch.Tensor) for value in recorder.__dict__.values())


def test_manifest_round_trip_and_validate_complete_update() -> None:
    manifest = _manifest()

    assert manifest.schema_version == 1
    assert manifest.base_version == "41"
    assert manifest.target_version == "42"
    assert len(manifest.metadata_digest) == 64
    assert WeightUpdateManifest.from_json(manifest.to_json()) == manifest
    _validate(manifest, _record(_weights()))


def test_manifest_rejects_missing_or_reordered_tensor() -> None:
    manifest = _manifest(update_mode="delta")

    with pytest.raises(ValueError, match="expected 2 tensors, received 1"):
        _validate(manifest, _record(_weights()[:1]))
    with pytest.raises(ValueError, match="tensor 0 metadata mismatch"):
        _validate(manifest, _record(list(reversed(_weights()))))


@pytest.mark.parametrize(
    ("context_override", "value"),
    [
        ("expected_model_id", "other/model"),
        ("expected_model_config_id", "config-sha256:different"),
        ("expected_update_mode", "delta"),
        ("current_version", "40"),
        ("expected_target_version", "43"),
    ],
)
def test_manifest_rejects_consumer_context_mismatch(context_override: str, value: str) -> None:
    manifest = _manifest()
    context = {
        "expected_model_id": "org/model",
        "expected_model_config_id": "config-sha256:abc123",
        "expected_update_mode": "full",
        "current_version": "41",
        "expected_target_version": "42",
    }
    context[context_override] = value

    with pytest.raises(ValueError, match="weight-update context mismatch"):
        manifest.validate(
            _record(_weights()),
            expected_model_id=context["expected_model_id"],
            expected_model_config_id=context["expected_model_config_id"],
            expected_update_mode=cast(WeightUpdateMode, context["expected_update_mode"]),
            current_version=context["current_version"],
            expected_target_version=context["expected_target_version"],
        )


def test_manifest_rejects_tampered_metadata() -> None:
    tampered_json = _manifest().to_json().replace('"target_version":"42"', '"target_version":"43"')

    with pytest.raises(ValueError, match="metadata digest mismatch"):
        WeightUpdateManifest.from_json(tampered_json)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.replace('"schema_version":1', '"schema_version":true'), "schema_version"),
        (lambda payload: payload.replace('"shape":[4,8]', '"shape":[true,8]', 1), "list of integers"),
        (lambda payload: payload.replace('{"base_version"', '{"unknown":1,"base_version"'), "unknown"),
        (
            lambda payload: payload.replace('"base_version":"41"', '"base_version":"41","base_version":"41"', 1),
            "duplicate JSON field",
        ),
    ],
)
def test_manifest_parser_rejects_noncanonical_structure(mutate: Callable[[str], str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        WeightUpdateManifest.from_json(mutate(_manifest().to_json()))


def test_manifest_parser_normalizes_invalid_utf8_to_value_error() -> None:
    with pytest.raises(ValueError, match="invalid weight-update manifest"):
        WeightUpdateManifest.from_json(b"\xff")


def test_manifest_rejects_duplicate_tensor_names() -> None:
    duplicate = [("duplicate", torch.empty(2)), ("duplicate", torch.empty(2))]

    with pytest.raises(ValueError, match="duplicate tensor name"):
        WeightUpdateManifest.from_weights(
            duplicate,
            model_id="org/model",
            model_config_id="config-sha256:abc123",
            update_mode="full",
            base_version=None,
            target_version="42",
        )


def test_manifest_requires_base_version_for_incremental_updates() -> None:
    with pytest.raises(ValueError, match="base_version is required"):
        WeightUpdateManifest.from_weights(
            _weights(),
            model_id="org/model",
            model_config_id="config-sha256:abc123",
            update_mode="sparse",
            base_version=None,
            target_version="42",
        )
