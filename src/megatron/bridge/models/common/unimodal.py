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

from typing import Any


def _dist_utils() -> Any:
    from megatron.training.models import dist_utils

    return dist_utils


def _ddp_wrap(*args: Any, **kwargs: Any) -> Any:
    return _dist_utils()._ddp_wrap(*args, **kwargs)


def _print_num_params(*args: Any, **kwargs: Any) -> Any:
    return _dist_utils()._print_num_params(*args, **kwargs)


def _wrap_with_mp_wrapper(*args: Any, **kwargs: Any) -> Any:
    return _dist_utils()._wrap_with_mp_wrapper(*args, **kwargs)


def build_virtual_pipeline_stages(*args: Any, **kwargs: Any) -> Any:
    """Build virtual pipeline stages using Megatron training utilities."""
    return _dist_utils().build_virtual_pipeline_stages(*args, **kwargs)


def to_empty_if_meta_device(*args: Any, **kwargs: Any) -> Any:
    """Move meta-device modules to empty tensors using Megatron training utilities."""
    return _dist_utils().to_empty_if_meta_device(*args, **kwargs)


def unimodal_build_distributed_models(*args: Any, **kwargs: Any) -> Any:
    """Build distributed unimodal models using Megatron training utilities."""
    return _dist_utils().unimodal_build_distributed_models(*args, **kwargs)
