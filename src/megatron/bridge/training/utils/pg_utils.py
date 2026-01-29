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


from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_attr_wrapped_model


def get_pg_collection(model: MegatronModule | list[MegatronModule]) -> ProcessGroupCollection:
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
