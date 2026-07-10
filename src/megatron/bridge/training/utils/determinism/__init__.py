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

"""Determinism debugging utilities for Megatron-Bridge.

Prototype scope (cross-process / "reference" mode):
- :mod:`signature` — stable cross-process tensor signatures.
- :mod:`collective_trace` — fingerprint stream of ``torch.distributed`` collectives.
- :mod:`diff_streams` — offline first-divergence diff of two jobs' streams.

See ``scripts/performance/perf_leaderboard/design_determinism_debug_tool.md``.
"""

from megatron.bridge.training.utils.determinism import collective_trace, module_scope, op_trace
from megatron.bridge.training.utils.determinism.signature import TensorSignature, tensor_signature


__all__ = [
    "collective_trace",
    "op_trace",
    "module_scope",
    "TensorSignature",
    "tensor_signature",
]
