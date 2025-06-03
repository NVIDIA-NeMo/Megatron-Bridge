#!/usr/bin/env python3
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

import functools
import logging
import nemo_run as run

from nemo_lm.training.config import ConfigContainer

logger: logging.Logger = logging.getLogger(__name__)


def prepare_config_for_nemo_run(config: ConfigContainer) -> ConfigContainer:
    """
    Prepares a pure ConfigContainer instance for use with NeMo Run by patching
    fields that are not directly serializable by nemo_run, such as functools.partial objects.

    Args:
        config: The ConfigContainer dataclass instance.

    Returns:
        The patched ConfigContainer instance compatible for execution with nemo_run.
    """
    model_cfg = config.model_config
    patched_fields = []

    if hasattr(model_cfg, 'init_method') and isinstance(model_cfg.init_method, functools.partial):
        original_partial = model_cfg.init_method
        model_cfg.init_method = run.Partial(original_partial.func, *original_partial.args, **original_partial.keywords)
        patched_fields.append("model_config.init_method")

    if hasattr(model_cfg, 'output_layer_init_method') and isinstance(
        model_cfg.output_layer_init_method, functools.partial
    ):
        original_partial = model_cfg.output_layer_init_method
        model_cfg.output_layer_init_method = run.Partial(
            original_partial.func, *original_partial.args, **original_partial.keywords
        )
        patched_fields.append("model_config.output_layer_init_method")

    # Check for other potential functools.partial objects in the model config
    for field_name in ['bias_init_method', 'weight_init_method']:
        if hasattr(model_cfg, field_name):
            field_value = getattr(model_cfg, field_name)
            if isinstance(field_value, functools.partial):
                original_partial = field_value
                setattr(
                    model_cfg,
                    field_name,
                    run.Partial(original_partial.func, *original_partial.args, **original_partial.keywords),
                )
                patched_fields.append(f"model_config.{field_name}")

    if patched_fields:
        logger.info(f"Wrapped the following fields with run.Partial: {', '.join(patched_fields)}")

    return config
