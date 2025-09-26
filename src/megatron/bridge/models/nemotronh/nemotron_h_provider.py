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

import logging
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.utils.common_utils import get_rank_safe


logger = logging.getLogger(__name__)


@dataclass
class NemotronHModelProvider(MambaModelProvider):
    """Configuration for Nemotron-H models."""

    seq_length: int = 8192
    mamba_num_groups: int = 8
    mamba_head_dim: int = 64
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    activation_func: callable = lambda x: torch.pow(F.relu(x), 2)
    masked_softmax_fusion: bool = True
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    attention_softmax_in_fp32: bool = False
    first_last_layers_bf16: bool = True
    is_hybrid_model: bool = True


@dataclass
class NemotronHModelProvider4B(NemotronHModelProvider):
    """Configuration for a 4B parameter Nemotron-H model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    num_layers: int = 52
    hidden_size: int = 3072
    mamba_num_heads: int = 112
    kv_channels: int = 128
    mamba_state_dim: int = 128
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 32
    use_mamba_mem_eff_path: bool = False


@dataclass
class NemotronHModelProvider8B(NemotronHModelProvider):
    """Configuration for a 8B parameter Nemotron-H model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    num_layers: int = 52
    hidden_size: int = 4096
    mamba_state_dim: int = 128
    mamba_num_heads: int = 128
    ffn_hidden_size: int = 21504
    num_attention_heads: int = 32


@dataclass
class NemotronHModelProvider47B(NemotronHModelProvider):
    """Configuration for a 47B parameter Nemotron-H model."""

    hybrid_override_pattern: str = (
        "M-M-M-M-M-M-M-M-M*-M-M-M-M-M-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-M-M---MM---M-M*-M-M-M-M-M-"
    )
    num_layers: int = 98
    hidden_size: int = 8192
    mamba_state_dim: int = 256
    mamba_num_heads: int = 256
    ffn_hidden_size: int = 30720
    num_attention_heads: int = 64


@dataclass
class NemotronHModelProvider56B(NemotronHModelProvider):
    """Configuration for a 56B parameter Nemotron-H model."""

    hybrid_override_pattern: str = (
        "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-"
        "M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    )
    num_layers: int = 118
    hidden_size: int = 8192
    mamba_state_dim: int = 256
    mamba_num_heads: int = 256
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 64


@dataclass
class NemotronNanoModelProvider9Bv2(NemotronHModelProvider):
    """Configuration for a 9B parameter Nemotron Nano v2 model."""

    hybrid_override_pattern: str = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
    num_layers: int = 56
    hidden_size: int = 4480
    mamba_num_heads: int = 128
    mamba_state_dim: int = 128
    ffn_hidden_size: int = 15680
    num_attention_heads: int = 40
    mamba_head_dim: int = 80
    seq_length: int = 131072


@dataclass
class NemotronNanoModelProvider12Bv2(NemotronHModelProvider):
    """Configuration for the Nemotron Nano v2 12B model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-"
    num_layers: int = 62
    hidden_size: int = 5120
    mamba_num_heads: int = 128
    mamba_state_dim: int = 128
    ffn_hidden_size: int = 20480
    num_attention_heads: int = 40
    mamba_head_dim: int = 80
    seq_length: int = 131072


# -----------------------------------------------------------------------------
# Deprecated aliases (to be removed in a future release)
# -----------------------------------------------------------------------------


def _warn_deprecated(old_cls: str, new_cls: str) -> None:
    if get_rank_safe() == 0:
        warnings.warn(
            f"{old_cls} is deprecated and will be removed in a future release. Use {new_cls} instead.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class NemotronHModel4BProvider(NemotronHModelProvider4B):
    """Deprecated alias for ``NemotronHModelProvider4B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``NemotronHModelProvider4B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("NemotronHModel4BProvider", "NemotronHModelProvider4B")
        super().__post_init__()


@dataclass
class NemotronHModel8BProvider(NemotronHModelProvider8B):
    """Deprecated alias for ``NemotronHModelProvider8B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``NemotronHModelProvider8B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("NemotronHModel8BProvider", "NemotronHModelProvider8B")
        super().__post_init__()


@dataclass
class NemotronHModel47BProvider(NemotronHModelProvider47B):
    """Deprecated alias for ``NemotronHModelProvider47B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``NemotronHModelProvider47B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("NemotronHModel47BProvider", "NemotronHModelProvider47B")
        super().__post_init__()


@dataclass
class NemotronHModel56BProvider(NemotronHModelProvider56B):
    """Deprecated alias for ``NemotronHModelProvider56B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``NemotronHModelProvider56B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("NemotronHModel56BProvider", "NemotronHModelProvider56B")
        super().__post_init__()


@dataclass
class NemotronNano9Bv2Provider(NemotronNanoModelProvider9Bv2):
    """Deprecated alias for ``NemotronNanoModelProvider9Bv2``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``NemotronNanoModelProvider9Bv2`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("NemotronNano9Bv2Provider", "NemotronNanoModelProvider9Bv2")
        super().__post_init__()


@dataclass
class NemotronNano12Bv2Provider(NemotronNanoModelProvider12Bv2):
    """Deprecated alias for ``NemotronNanoModelProvider12Bv2``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``NemotronNanoModelProvider12Bv2`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("NemotronNano12Bv2Provider", "NemotronNanoModelProvider12Bv2")
        super().__post_init__()
