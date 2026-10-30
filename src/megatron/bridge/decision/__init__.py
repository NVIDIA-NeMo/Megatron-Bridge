# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Post-training for System One / Jev-style decision models (Cloudflare Clef, Contrastive-LM CLM).

Decision models read a ``state`` plus a schema of typed questions (``noul`` / ``choice`` / ``score``) and return a
probability for every allowed option in a single forward pass.

* :mod:`.records` - SystemOne records, option orders and targets.
* :mod:`.losses` - SFT (label-smoothed CE + Brier + RPS) and RL (exact expected reward / GRPO + KL) objectives.
* :mod:`.metrics` - accuracy, joint exact match, decision reward, Brier, ECE; before/after tables.
* :mod:`.clm_prompting`, :mod:`.clm_heads` - CLM prompt layout and projection heads (``contrastive-lm`` compatible).
* :mod:`.clef_head` - Clef joint schema head and record encoding (adapted from ``Cloudflare/clef``).
* :mod:`.clm_trainer` - CLM head post-training / evaluation on cached encoder embeddings.
* :mod:`.megatron_glue` - Megatron-Core glue: hidden-state output processors, vocab-parallel row gathers, LoRA I/O.
"""

from megatron.bridge.decision.losses import DecisionLoss, DecisionLossConfig, DecisionTargets
from megatron.bridge.decision.metrics import DecisionEvaluator
from megatron.bridge.decision.records import QuestionTarget, load_records, question_targets


__all__ = [
    "DecisionEvaluator",
    "DecisionLoss",
    "DecisionLossConfig",
    "DecisionTargets",
    "QuestionTarget",
    "load_records",
    "question_targets",
]
