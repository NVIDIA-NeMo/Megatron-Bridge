# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import importlib


def dynamic_import(full_path):
    """
    Dynamically import a class or function from a given full path.

    :param full_path: The full path to the class or function (e.g., "package.module.ClassName")
    :return: The imported class or function
    :raises ImportError: If the module or attribute cannot be imported
    :raises AttributeError: If the attribute does not exist in the module
    """
    try:
        # Split the full path into module path and attribute name
        module_path, attribute_name = full_path.rsplit(".", 1)
    except ValueError as e:
        raise ImportError(
            f"Invalid full path '{full_path}'. It should contain both module and attribute names."
        ) from e

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}'.") from e

    # Retrieve the attribute from the module
    try:
        attribute = getattr(module, attribute_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' does not have an attribute '{attribute_name}'.") from e

    return attribute
