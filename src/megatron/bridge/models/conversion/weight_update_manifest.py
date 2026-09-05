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

"""Structural manifests for streamed Hugging Face weight updates."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal, cast

import torch


WeightUpdateMode = Literal["full", "delta", "adapter", "sparse"]
_WEIGHT_UPDATE_MODES = ("full", "delta", "adapter", "sparse")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_config_id",
        "update_mode",
        "base_version",
        "target_version",
        "tensors",
        "metadata_digest",
    }
)
_TENSOR_FIELDS = frozenset({"name", "shape", "dtype"})


@dataclass(frozen=True)
class WeightUpdateTensor:
    """Structural metadata for one ordered tensor in a weight update."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor names must not be empty")
        if any(type(dimension) is not int or dimension < 0 for dimension in self.shape):
            raise ValueError(f"tensor {self.name} has an invalid shape: {self.shape}")
        if not self.dtype:
            raise ValueError(f"tensor {self.name} has an empty dtype")


class WeightUpdateRecorder:
    """Record ordered tensor metadata without retaining tensor payloads.

    Use :meth:`track` around a one-shot export stream on the producer and
    :meth:`record` as tensor frames arrive on the consumer. The recorder keeps
    names, shapes, and dtypes only.
    """

    def __init__(self) -> None:
        self._tensors: list[WeightUpdateTensor] = []
        self._names: set[str] = set()

    @property
    def tensors(self) -> tuple[WeightUpdateTensor, ...]:
        """Return the recorded ordered tensor metadata."""
        return tuple(self._tensors)

    def record(self, name: str, tensor: torch.Tensor) -> None:
        """Record metadata for one tensor without retaining its payload."""
        metadata = _tensor_metadata(name, tensor)
        if metadata.name in self._names:
            raise ValueError(f"duplicate tensor name: {metadata.name}")
        self._names.add(metadata.name)
        self._tensors.append(metadata)

    def track(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield a weight stream unchanged while recording its metadata."""
        for name, tensor in weights:
            self.record(name, tensor)
            yield name, tensor

    def build_manifest(
        self,
        *,
        model_id: str,
        model_config_id: str,
        update_mode: WeightUpdateMode,
        base_version: str | None,
        target_version: str,
    ) -> "WeightUpdateManifest":
        """Build a manifest trailer for the recorded stream."""
        return WeightUpdateManifest(
            model_id=model_id,
            model_config_id=model_config_id,
            update_mode=update_mode,
            base_version=base_version,
            target_version=target_version,
            tensors=self.tensors,
        )


@dataclass(frozen=True)
class WeightUpdateManifest:
    """Versioned structural contract for one complete HF-coordinate update.

    The manifest contains no tensor payloads. Send it as a trailer after the
    recorded tensor stream, then validate it against the consumer's recorder
    and expected context before activating staged weights.

    ``metadata_digest`` detects accidental corruption of manifest metadata. It
    is neither a tensor-content checksum nor an authentication mechanism.
    """

    model_id: str
    model_config_id: str
    update_mode: WeightUpdateMode
    base_version: str | None
    target_version: str
    tensors: tuple[WeightUpdateTensor, ...]
    schema_version: int = field(default=1, init=False)
    metadata_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.model_config_id:
            raise ValueError("model_config_id must not be empty")
        if self.update_mode not in _WEIGHT_UPDATE_MODES:
            raise ValueError(f"unsupported update_mode: {self.update_mode}")
        if not self.target_version:
            raise ValueError("target_version must not be empty")
        if self.update_mode != "full" and not self.base_version:
            raise ValueError(f"base_version is required for {self.update_mode} updates")
        if self.base_version == self.target_version:
            raise ValueError("base_version and target_version must differ")
        if not self.tensors:
            raise ValueError("a weight update must contain at least one tensor")

        names: set[str] = set()
        for tensor in self.tensors:
            if tensor.name in names:
                raise ValueError(f"duplicate tensor name: {tensor.name}")
            names.add(tensor.name)

        object.__setattr__(self, "metadata_digest", self._compute_metadata_digest())

    @classmethod
    def from_weights(
        cls,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        model_id: str,
        model_config_id: str,
        update_mode: WeightUpdateMode,
        base_version: str | None,
        target_version: str,
    ) -> "WeightUpdateManifest":
        """Build a manifest by consuming an ordered weight iterable.

        Use :class:`WeightUpdateRecorder` instead when tensor payloads must be
        sent in the same one-shot pass.
        """
        recorder = WeightUpdateRecorder()
        for name, tensor in weights:
            recorder.record(name, tensor)
        return recorder.build_manifest(
            model_id=model_id,
            model_config_id=model_config_id,
            update_mode=update_mode,
            base_version=base_version,
            target_version=target_version,
        )

    def validate(
        self,
        inventory: WeightUpdateRecorder,
        *,
        expected_model_id: str,
        expected_model_config_id: str,
        expected_update_mode: WeightUpdateMode,
        current_version: str | None,
        expected_target_version: str,
    ) -> None:
        """Validate consumer context and the complete recorded inventory.

        Args:
            inventory: Metadata recorded while the consumer staged tensors.
            expected_model_id: Model identity expected by the consumer.
            expected_model_config_id: Model-config identity expected by the consumer.
            expected_update_mode: Update mode allowed by the consumer operation.
            current_version: Consumer version before this update is activated.
            expected_target_version: Target version requested by the consumer operation.

        Raises:
            ValueError: If context or tensor metadata does not match exactly.
        """
        expected_context = {
            "model_id": expected_model_id,
            "model_config_id": expected_model_config_id,
            "update_mode": expected_update_mode,
            "base_version": current_version,
            "target_version": expected_target_version,
        }
        actual_context = {
            "model_id": self.model_id,
            "model_config_id": self.model_config_id,
            "update_mode": self.update_mode,
            "base_version": self.base_version,
            "target_version": self.target_version,
        }
        if actual_context != expected_context:
            raise ValueError(f"weight-update context mismatch: expected {expected_context}, received {actual_context}")

        actual_tensors = inventory.tensors
        if len(actual_tensors) != len(self.tensors):
            raise ValueError(f"expected {len(self.tensors)} tensors, received {len(actual_tensors)}")
        for index, (expected_tensor, actual_tensor) in enumerate(zip(self.tensors, actual_tensors)):
            if actual_tensor != expected_tensor:
                raise ValueError(
                    f"tensor {index} metadata mismatch: expected {expected_tensor}, received {actual_tensor}"
                )

    def to_json(self) -> str:
        """Serialize this manifest as canonical JSON."""
        payload = self._metadata_payload()
        payload["metadata_digest"] = self.metadata_digest
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> "WeightUpdateManifest":
        """Deserialize and strictly validate a manifest from JSON."""
        try:
            decoded = json.loads(payload, object_pairs_hook=_object_without_duplicate_keys)
            if not isinstance(decoded, dict):
                raise TypeError("manifest must be a JSON object")
            _require_exact_fields(decoded, _MANIFEST_FIELDS, "manifest")

            schema_version = decoded["schema_version"]
            if type(schema_version) is not int or schema_version != 1:
                raise ValueError(f"unsupported schema_version: {schema_version}")

            tensors_payload = decoded["tensors"]
            if not isinstance(tensors_payload, list):
                raise TypeError("tensors must be a list")
            tensors = tuple(_tensor_from_json(tensor) for tensor in tensors_payload)

            update_mode = decoded["update_mode"]
            if not isinstance(update_mode, str):
                raise TypeError("update_mode must be a string")
            base_version = decoded["base_version"]
            if base_version is not None and not isinstance(base_version, str):
                raise TypeError("base_version must be a string or null")

            manifest = cls(
                model_id=_required_json_string(decoded, "model_id"),
                model_config_id=_required_json_string(decoded, "model_config_id"),
                update_mode=cast(WeightUpdateMode, update_mode),
                base_version=base_version,
                target_version=_required_json_string(decoded, "target_version"),
                tensors=tensors,
            )
            digest = _required_json_string(decoded, "metadata_digest")
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid weight-update manifest: {error}") from error

        if digest != manifest.metadata_digest:
            raise ValueError("weight-update manifest metadata digest mismatch")
        return manifest

    def _metadata_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_config_id": self.model_config_id,
            "update_mode": self.update_mode,
            "base_version": self.base_version,
            "target_version": self.target_version,
            "tensors": [
                {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype} for tensor in self.tensors
            ],
        }

    def _compute_metadata_digest(self) -> str:
        canonical = json.dumps(self._metadata_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _tensor_metadata(name: str, tensor: torch.Tensor) -> WeightUpdateTensor:
    return WeightUpdateTensor(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype).removeprefix("torch."),
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _require_exact_fields(payload: dict[str, object], expected: frozenset[str], context: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"invalid {context} fields: missing={missing}, unknown={unknown}")


def _required_json_string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _tensor_from_json(payload: object) -> WeightUpdateTensor:
    if not isinstance(payload, dict):
        raise TypeError("tensor metadata must be an object")
    _require_exact_fields(payload, _TENSOR_FIELDS, "tensor metadata")
    name = _required_json_string(payload, "name")
    dtype = _required_json_string(payload, "dtype")
    shape_payload = payload["shape"]
    if not isinstance(shape_payload, list) or not all(type(dimension) is int for dimension in shape_payload):
        raise TypeError(f"tensor {name} shape must be a list of integers")
    return WeightUpdateTensor(name=name, shape=tuple(shape_payload), dtype=dtype)
