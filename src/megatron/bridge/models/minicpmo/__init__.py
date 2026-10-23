# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o 4.5 understanding-stage Bridge registration."""

from megatron.bridge.models.minicpmo.minicpmo_bridge import MiniCPMOBridge
from megatron.bridge.models.minicpmo.minicpmo_config import MiniCPMOModelConfig


__all__ = ["MiniCPMOBridge", "MiniCPMOModelConfig"]
