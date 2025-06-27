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

import torch
from rich.table import Table


def weights_verification_table(bridge, megatron_model) -> Table:
    """
    Returns a table comparing weights between a Hugging Face model and a Megatron-LM model.

    Args:
        bridge (CausalLMBridge): The bridge object containing model information.
        megatron_model: The Megatron-LM model instance.

    Returns:
        Table: A rich Table object with the comparison.
    """
    table = Table(title="Hugging Face Weights Verification")
    table.add_column("Weight Name", style="cyan")
    table.add_column("Shape")
    table.add_column("DType")
    table.add_column("Device")
    table.add_column("Matches Original", justify="center")

    # Check each weight against the original HF-model
    for name, param in bridge(megatron_model, show_progress=True):
        original_param = bridge.hf_pretrained.state[name]
        table.add_row(
            name,
            str(tuple(param.shape)),
            str(param.dtype).replace("torch.", ""),
            str(param.device),
            "✅" if torch.allclose(param, original_param.to(param.device), atol=1e-6) else "❌",
        )

    return table
