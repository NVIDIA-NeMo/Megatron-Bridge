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

import os
import torch
from typing import Any
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0

EMA_DIRNAME = "ema_state"


def has_ema_state(user_state: dict[str, Any] | None) -> bool:
    return (
        isinstance(user_state, dict)
        and "ema_state" in user_state
        and isinstance(user_state["ema_state"], dict)
        and len(user_state["ema_state"]) > 0
    )

def _ema_rank_filename(checkpoint_name: str, rank: int | None = None) -> str:
    if rank is None:
        rank = get_rank_safe()
    return os.path.join(checkpoint_name, EMA_DIRNAME, f"rank_{rank:05d}.pt")

def save_ema_user_state(checkpoint_name: str, user_state: dict[str, Any]) -> bool:
    if not has_ema_state(user_state):
        return False

    ema_dir = os.path.join(checkpoint_name, EMA_DIRNAME)
    os.makedirs(ema_dir, exist_ok=True)

    rank = get_rank_safe()
    final_path = _ema_rank_filename(checkpoint_name, rank)
    tmp_path = final_path + ".tmp"

    payload = {
        "ema_state": {
            name: tensor.detach().float().cpu()
            for name, tensor in user_state["ema_state"].items()
        },
        "ema_updates": int(user_state.get("ema_updates", 0)),
        "ema_skipped_iters": int(user_state.get("ema_skipped_iters", 0)),
    }
    print_rank_0(
        f"[EMA SAVE] updates={payload['ema_updates']} "
        f"skipped={payload['ema_skipped_iters']} "
        f"num_tensors={len(payload['ema_state'])}"
    )

    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    return True

def load_ema_user_state(checkpoint_name: str, user_state: dict[str, Any]) -> bool:
    path = _ema_rank_filename(checkpoint_name)

    if not os.path.exists(path):
        return False

    payload = torch.load(path, map_location="cpu", weights_only=False)

    user_state["ema_state"] = payload["ema_state"]
    user_state["ema_updates"] = int(payload.get("ema_updates", 0))
    user_state["ema_skipped_iters"] = int(payload.get("ema_skipped_iters", 0))

    return True