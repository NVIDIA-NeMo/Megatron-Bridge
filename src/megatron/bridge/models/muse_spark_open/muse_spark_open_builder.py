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

"""Builder for Muse Spark Open pipeline stages."""

from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.hybrid import HybridModelBuilder
from megatron.training.vocab_utils import calculate_padded_vocab_size

from megatron.bridge.models.muse_spark_open.modeling_muse_spark_open.model import MuseSparkModel
from megatron.bridge.models.muse_spark_open.muse_spark_open_config import MuseSparkModelConfig


class MuseSparkModelBuilder(HybridModelBuilder):
    """Build the decoder and HF perception modules in a single model stage."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MuseSparkModel:
        """Construct the requested pipeline stage from the imported config."""
        config = self._model_config
        if not isinstance(config, MuseSparkModelConfig):
            raise TypeError("MuseSparkModelBuilder requires MuseSparkModelConfig.")
        if vp_stage is not None:
            raise ValueError("The initial Muse Spark Open builder does not support virtual pipeline stages.")
        config.finalize()
        if config.vocab_size is None:
            raise ValueError("vocab_size must be set before building Muse Spark Open.")
        vocab_size = config.vocab_size
        if config.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size, config.make_vocab_size_divisible_by, config.transformer.tensor_model_parallel_size
            )
        return MuseSparkModel(
            config,
            vocab_size=vocab_size,
            pg_collection=pg_collection,
            pre_process=is_pp_first_stage(pg_collection.pp) if pre_process is None else pre_process,
            post_process=is_pp_last_stage(pg_collection.pp) if post_process is None else post_process,
        )
