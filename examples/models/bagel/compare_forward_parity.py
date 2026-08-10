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

"""Compare one official BAGEL forward with its Bridge/MCore counterpart."""

import argparse
import json
import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)
LOSS_ATOL = 1e-2
MIN_COSINE = 0.9999


def parse_args() -> argparse.Namespace:
    """Parse forward-parity arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-prefix", type=Path, required=True)
    parser.add_argument("--bridge-prefix", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_outputs(prefix: Path, world_size: int, bridge: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and concatenate per-rank CE and active MSE values."""
    ce_values = []
    mse_values = []
    for rank in range(world_size):
        payload = torch.load(f"{prefix}.rank{rank}.pt", map_location="cpu", weights_only=True)
        ce_values.append(payload["ce"].float())
        mse = payload["mse"].float()
        if bridge:
            mse = mse[payload["mse_mask"].bool()]
        mse_values.append(mse)
    return torch.cat(ce_values), torch.cat(mse_values)


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    """Calculate direct BF16 forward-parity metrics."""
    if reference.shape != actual.shape:
        raise ValueError(f"Forward shapes differ: official={reference.shape}, bridge={actual.shape}")
    difference = (reference - actual).abs()
    cosine = torch.nn.functional.cosine_similarity(reference.flatten(), actual.flatten(), dim=0)
    official_mean = reference.mean().item()
    bridge_mean = actual.mean().item()
    return {
        "shape": list(reference.shape),
        "official_mean": official_mean,
        "bridge_mean": bridge_mean,
        "mean_loss_abs_diff": abs(official_mean - bridge_mean),
        "max_abs_diff": difference.max().item(),
        "mean_abs_diff": difference.mean().item(),
        "cosine": cosine.clamp(-1, 1).item(),
    }


def main() -> None:
    """Write metrics and fail when the fixed BF16 parity contract is missed."""
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    official_ce, official_mse = load_outputs(args.official_prefix, args.world_size, False)
    bridge_ce, bridge_mse = load_outputs(args.bridge_prefix, args.world_size, True)
    report = {
        "ce": metrics(official_ce, bridge_ce),
        "mse": metrics(official_mse, bridge_mse),
        "loss_atol": LOSS_ATOL,
        "min_cosine": MIN_COSINE,
    }
    official_loss = report["ce"]["official_mean"] + report["mse"]["official_mean"]
    bridge_loss = report["ce"]["bridge_mean"] + report["mse"]["bridge_mean"]
    report["loss"] = {
        "official": official_loss,
        "bridge": bridge_loss,
        "abs_diff": abs(official_loss - bridge_loss),
    }
    report["passed"] = report["loss"]["abs_diff"] < LOSS_ATOL and all(
        report[name]["mean_loss_abs_diff"] < LOSS_ATOL and report[name]["cosine"] > MIN_COSINE
        for name in ("ce", "mse")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote BAGEL forward parity report to %s", args.output)
    if not report["passed"]:
        raise RuntimeError("BAGEL forward parity missed the fixed BF16 tolerance")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
