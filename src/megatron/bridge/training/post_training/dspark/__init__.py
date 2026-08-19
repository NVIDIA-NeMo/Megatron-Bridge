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

"""DSpark speculative-decoding draft training (architecture-agnostic core).

This package holds the pure-``torch`` pieces of DSpark draft training: the
sequential (Markov / RNN) and confidence heads (:mod:`heads`) and the three-term
CE / L1 / confidence objective with its acceptance metrics (:mod:`loss`). The
parallel-aware draft backbone and the Megatron-Core forward-step / provider wiring
are separate, GPU-validated concerns. See ``docs/training/dspark-speculative-decoding.md``.
"""

from megatron.bridge.training.post_training.dspark.heads import (
    ConfidenceHead,
    GatedMarkovHead,
    RNNHead,
    VanillaMarkov,
    build_markov_head,
)
from megatron.bridge.training.post_training.dspark.loss import (
    DSparkForwardOutput,
    assert_per_token_loss_config,
    dspark_loss,
)


__all__ = [
    "ConfidenceHead",
    "DSparkForwardOutput",
    "GatedMarkovHead",
    "RNNHead",
    "VanillaMarkov",
    "assert_per_token_loss_config",
    "build_markov_head",
    "dspark_loss",
]
