# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import fields
from typing import Optional, Union

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_attr_wrapped_model


def get_pg_collection(model: Union[MegatronModule, list[MegatronModule]]) -> ProcessGroupCollection:
    """Return the ProcessGroupCollection from a model or list of model chunks.

    This mirrors the style of utility accessors like `get_model_config`, but for
    retrieving the communication process group collection from the model wrapper.

    Args:
        model: A MegatronModule or a list of MegatronModule chunks.

    Returns:
        ProcessGroupCollection: The model's process group collection.
    """
    if isinstance(model, list):
        model_ref = model[0]
    else:
        model_ref = model

    # Prefer pg_collection attached to the wrapped model, but fall back to the
    # default MPU-based process groups if it is not present.
    try:
        return get_attr_wrapped_model(model_ref, "pg_collection", allow_none=False)
    except RuntimeError as e:
        # get_attr_wrapped_model raises a RuntimeError with this exact message
        # when the requested attribute does not exist on the wrapped model.
        if "couldn't find attribute pg_collection" in str(e):
            return ProcessGroupCollection.use_mpu_process_groups()
        raise


class DistTrainProcessGroupCollection(ProcessGroupCollection):
    """Process group collection for dist train."""

    def __init__(self, pg_collection: ProcessGroupCollection, language_model_module_name: Optional[str] = None):
        """Initialize the dist train process group collection.

        Args:
            pg_collection: The process group collection.
            language_model_module_name: The name of the language model module.
        """
        for field_info in fields(ProcessGroupCollection):
            setattr(
                self,
                field_info.name,
                getattr(pg_collection, field_info.name, None),
            )
        self.language_model_module_name = language_model_module_name
        if language_model_module_name is not None:
            self.language_model_collection = pg_collection
        else:
            self.language_model_collection = None

    def get_language_model_cp_size(self) -> int:
        """Get context parallel size for the language model.

        Returns:
            Context parallel size for the language model.

        Raises:
            ValueError: If no language model is specified for this collection.
        """
        if self.language_model_collection is None:
            raise ValueError("No language model specified for this collection")
        return self.language_model_collection.cp.size()

    def has_language_model(self) -> bool:
        """Check if this rank has a language model.

        Returns:
            True if this rank has a language model, False otherwise.
        """
        return self.language_model_module_name is not None
