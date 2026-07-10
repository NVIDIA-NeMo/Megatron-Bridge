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

_MODEL_ALIASES = {
    ("nemodiag", "nemodiag_v0"): ("deepseek", "deepseek_v3"),
}


def resolve_model_alias(model_family_name: str, model_recipe_name: str) -> tuple[str, str]:
    """Resolve a public model alias to the recipe implementation that backs it.

    Args:
        model_family_name: Model family requested by the caller.
        model_recipe_name: Model recipe requested by the caller.

    Returns:
        The implementation family and recipe names.
    """
    return _MODEL_ALIASES.get(
        (model_family_name, model_recipe_name),
        (model_family_name, model_recipe_name),
    )
