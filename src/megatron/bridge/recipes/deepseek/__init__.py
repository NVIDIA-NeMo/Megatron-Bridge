"""DeepSeek recipe exports.

This module re-exports AutoBridge-based pretrain config helpers for DeepSeek
models (V2, V2-Lite, V3, V4).
"""

# DeepSeek V2/V2-Lite
from .deepseek_v2 import (
    deepseek_v2_lite_pretrain_config,
    deepseek_v2_pretrain_config,
)

# DeepSeek V3
from .deepseek_v3 import (
    deepseek_v3_pretrain_config,
    deepseek_v3_pretrain_config_32nodes,
)

# DeepSeek V4
from .deepseek_v4 import (
    deepseek_v4_pretrain_config,
)


__all__ = [
    # DeepSeek V2/V2-Lite
    "deepseek_v2_pretrain_config",
    "deepseek_v2_lite_pretrain_config",
    # DeepSeek V3
    "deepseek_v3_pretrain_config",
    "deepseek_v3_pretrain_config_32nodes",
    # DeepSeek V4
    "deepseek_v4_pretrain_config",
]
