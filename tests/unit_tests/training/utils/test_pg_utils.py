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

#
# Test purpose:
# - Cover the previously untested helpers in
#   `megatron.bridge.training.utils.pg_utils`.
# - Exercise `get_pg_collection`'s success path, list-of-models path, MPU
#   fallback path, and re-raise behavior.
# - Exercise `DistTrainProcessGroupCollection`'s constructor (with and
#   without a language model module), `has_language_model`, and
#   `get_language_model_cp_size` (including the error path).
#

from unittest import mock

import pytest
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.training.utils.pg_utils import (
    DistTrainProcessGroupCollection,
    get_pg_collection,
)


# -----------------------------------------------------------------------------
# get_pg_collection
# -----------------------------------------------------------------------------


class TestGetPgCollection:
    """Tests for `get_pg_collection`.

    The helper mirrors the style of `get_model_config` but for
    `pg_collection`. It must:

    - return the model's pg_collection if attached,
    - use `model[0]` when given a list,
    - fall back to `ProcessGroupCollection.use_mpu_process_groups()` when
      the model has no `pg_collection` attribute,
    - re-raise any other RuntimeError unchanged.
    """

    def test_returns_pg_collection_attached_to_single_model(self):
        sentinel = object()
        with mock.patch(
            "megatron.bridge.training.utils.pg_utils.get_attr_wrapped_model",
            return_value=sentinel,
        ) as mock_get_attr:
            model = mock.MagicMock(name="model")
            result = get_pg_collection(model)

        assert result is sentinel
        mock_get_attr.assert_called_once_with(model, "pg_collection", allow_none=False)

    def test_uses_first_chunk_when_passed_a_list(self):
        sentinel = object()
        with mock.patch(
            "megatron.bridge.training.utils.pg_utils.get_attr_wrapped_model",
            return_value=sentinel,
        ) as mock_get_attr:
            chunk_a = mock.MagicMock(name="chunk_a")
            chunk_b = mock.MagicMock(name="chunk_b")
            result = get_pg_collection([chunk_a, chunk_b])

        assert result is sentinel
        # `model_ref` should be the first chunk; the second chunk is ignored.
        called_model = mock_get_attr.call_args.args[0]
        assert called_model is chunk_a

    def test_falls_back_to_mpu_when_pg_collection_attribute_missing(self):
        # `get_attr_wrapped_model` raises RuntimeError with this exact
        # substring when the requested attribute does not exist.
        missing_error = RuntimeError("couldn't find attribute pg_collection on the wrapped model")
        mpu_sentinel = object()

        with (
            mock.patch(
                "megatron.bridge.training.utils.pg_utils.get_attr_wrapped_model",
                side_effect=missing_error,
            ),
            mock.patch.object(
                ProcessGroupCollection,
                "use_mpu_process_groups",
                return_value=mpu_sentinel,
            ) as mock_use_mpu,
        ):
            result = get_pg_collection(mock.MagicMock(name="model"))

        assert result is mpu_sentinel
        mock_use_mpu.assert_called_once_with()

    def test_reraises_runtime_error_when_message_does_not_match(self):
        unrelated_error = RuntimeError("something else went wrong")

        with (
            mock.patch(
                "megatron.bridge.training.utils.pg_utils.get_attr_wrapped_model",
                side_effect=unrelated_error,
            ),
            mock.patch.object(
                ProcessGroupCollection,
                "use_mpu_process_groups",
            ) as mock_use_mpu,
        ):
            with pytest.raises(RuntimeError, match="something else went wrong"):
                get_pg_collection(mock.MagicMock(name="model"))

        # MPU fallback must NOT be invoked for unrelated RuntimeErrors.
        mock_use_mpu.assert_not_called()


# -----------------------------------------------------------------------------
# DistTrainProcessGroupCollection
# -----------------------------------------------------------------------------


def _make_pg_collection_with(**fields_set) -> ProcessGroupCollection:
    """Build a bare ProcessGroupCollection and set a few fields on it.

    All ProcessGroupCollection fields are `init=False`, so we instantiate
    without args and then set the fields the test cares about.
    """
    pg = ProcessGroupCollection()
    for name, value in fields_set.items():
        setattr(pg, name, value)
    return pg


class TestDistTrainProcessGroupCollection:
    """Tests for `DistTrainProcessGroupCollection`."""

    def test_inherits_process_group_collection(self):
        """The subclass relationship is preserved (so callers can `isinstance`)."""
        source = _make_pg_collection_with(tp=mock.sentinel.tp_group)
        result = DistTrainProcessGroupCollection(pg_collection=source)
        assert isinstance(result, ProcessGroupCollection)

    def test_copies_set_fields_from_source_collection(self):
        """All fields present on the source pg_collection are copied over."""
        source = _make_pg_collection_with(
            tp=mock.sentinel.tp_group,
            pp=mock.sentinel.pp_group,
            cp=mock.sentinel.cp_group,
        )

        result = DistTrainProcessGroupCollection(pg_collection=source)

        assert result.tp is mock.sentinel.tp_group
        assert result.pp is mock.sentinel.pp_group
        assert result.cp is mock.sentinel.cp_group

    def test_unset_fields_default_to_none(self):
        """Fields not present on the source resolve to None on the new instance.

        The constructor uses `getattr(pg_collection, field.name, None)` so a
        ProcessGroupCollection field that was never assigned does not raise —
        it becomes `None` on the wrapper.
        """
        source = _make_pg_collection_with(tp=mock.sentinel.tp_group)

        result = DistTrainProcessGroupCollection(pg_collection=source)

        # `dp` was never set on the source.
        assert result.dp is None

    def test_no_language_model_by_default(self):
        """When language_model_module_name is None, the wrapper has no LM."""
        source = _make_pg_collection_with(tp=mock.sentinel.tp_group)

        result = DistTrainProcessGroupCollection(pg_collection=source)

        assert result.has_language_model() is False
        assert result.language_model_module_name is None
        assert result.language_model_collection is None

    def test_language_model_attaches_source_collection(self):
        """When LM module is named, the source pg_collection is exposed as the LM collection."""
        source = _make_pg_collection_with(tp=mock.sentinel.tp_group)

        result = DistTrainProcessGroupCollection(
            pg_collection=source,
            language_model_module_name="llm",
        )

        assert result.has_language_model() is True
        assert result.language_model_module_name == "llm"
        # The wrapper attaches the SAME source object as the LM collection.
        assert result.language_model_collection is source

    def test_get_language_model_cp_size_returns_lm_cp_size(self):
        """When LM is configured, return the CP group size from the LM collection."""
        cp_group = mock.MagicMock(name="cp_group")
        cp_group.size.return_value = 8
        source = _make_pg_collection_with(cp=cp_group)

        result = DistTrainProcessGroupCollection(
            pg_collection=source,
            language_model_module_name="llm",
        )

        assert result.get_language_model_cp_size() == 8
        cp_group.size.assert_called_once_with()

    def test_get_language_model_cp_size_raises_without_lm(self):
        """No LM ⇒ get_language_model_cp_size raises a clear ValueError."""
        source = _make_pg_collection_with(tp=mock.sentinel.tp_group)
        result = DistTrainProcessGroupCollection(pg_collection=source)

        with pytest.raises(ValueError, match="No language model specified"):
            result.get_language_model_cp_size()
