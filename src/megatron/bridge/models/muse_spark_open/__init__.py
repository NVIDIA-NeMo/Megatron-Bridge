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

"""Muse Spark Open model support with HF perception and an MCore Hybrid decoder."""

from megatron.bridge.models.muse_spark_open.modeling_muse_spark_open.model import MuseSparkModel
from megatron.bridge.models.muse_spark_open.muse_spark_open_bridge import MuseSparkBridge, MuseSparkTextBridge
from megatron.bridge.models.muse_spark_open.muse_spark_open_builder import MuseSparkModelBuilder
from megatron.bridge.models.muse_spark_open.muse_spark_open_config import MuseSparkModelConfig


__all__ = ["MuseSparkBridge", "MuseSparkTextBridge", "MuseSparkModel", "MuseSparkModelBuilder", "MuseSparkModelConfig"]

# The checkpoint's config predates the MuseSpark config type rename in Transformers.
# Register its alias only when a Transformers build with Muse Spark Open is installed.
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkConfig
except ImportError:
    pass
else:
    if "muse_spark" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("muse_spark", MuseSparkConfig)
