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

"""Names of the ERNIE 5.5 reference implementation's loaded-weight manifests, translated to Bridge names.

The reference implementation (ERNIE 5.5 on a GPT-style Megatron-Core decoder, one transformer layer per
Hugging Face layer) can dump a manifest per rank after it loads a Hugging Face checkpoint: for every
parameter and buffer, its name, shape, dtype and the sha256 of its raw bytes. This module translates
the manifest names (observed in the manifests, not taken from the reference code) to Bridge's hybrid
names, so that an external parity check can verify that every tensor Bridge loads from the same
checkpoint hash-matches the manifest. No test in this repository uses it.

Translation, for reference layer ``i`` (Bridge hybrid layers ``2i`` and ``2i + 1``):

* ``input_layernorm``, ``self_attention.*`` and ``self_attention_hyper_connection.*`` belong to the
  attention layer ``decoder.layers.{2i}`` (under ``inner_layer`` / ``hyper_connection``);
  ``pre_mlp_layernorm``, ``mlp.*`` and ``mlp_hyper_connection.*`` to ``decoder.layers.{2i + 1}``.
* Attention renames: ``vha.U``/``vha.V`` are the core's ``vha_postmix_u``/``vha_postmix_v``; the
  output gate ``output_gate.gate_proj`` is ``linear_output_gate`` (on the attention in H layers, on the
  core in D layers); in D layers the sink ``core_attention.softmax_offset`` is ``core_attention.attn_sink``
  and the indexer projections ``wq_b``/``wk``/``weights_proj`` are ``linear_wq_b``/``linear_wk``/
  ``linear_weights_proj``.
* ``decoder.final_layernorm`` is ``decoder.final_norm``; the MTP body's MoE layer
  (``mtp.layers.{k}.mtp_model_layer.mlp``) is the body's hybrid layer 1.
* Grouped experts keep their local names (``linear_fc1.weight{e}``) when Bridge runs the same expert
  parallelism.
* ``local_tokens_per_expert`` buffers are not compared: they are non-persistent token counters (bf16 in
  the reference, int64 in Megatron-Core dev), empty after a load.

Dtypes that differ by design: the D-layer sink (reference ``softmax_offset``) is fp32 in Bridge and bf16
in the reference (:data:`DTYPE_DEVIATIONS`, reference-name suffixes); compare it after casting to the
reference dtype. The H-layer sink is fp32 in both.
"""

from __future__ import annotations

import re


__all__ = ["DTYPE_DEVIATIONS", "reference_to_bridge_name"]

# Reference-name suffixes whose Bridge dtype differs by design; compared in the reference dtype.
DTYPE_DEVIATIONS = ("self_attention.core_attention.softmax_offset",)

_LAYER = re.compile(r"decoder\.layers\.(\d+)\.(.+)")
_MTP_BODY_ROUTER = re.compile(r"mtp\.layers\.(\d+)\.mtp_model_layer\.mlp\.(router\..+)")
_SHARED_ATTENTION_RENAMES = {
    "vha.U": "core_attention.vha_postmix_u",
    "vha.V": "core_attention.vha_postmix_v",
}
_H_ATTENTION_RENAMES = {**_SHARED_ATTENTION_RENAMES, "output_gate.gate_proj.weight": "linear_output_gate.weight"}
_D_ATTENTION_RENAMES = {
    **_SHARED_ATTENTION_RENAMES,
    "output_gate.gate_proj.weight": "core_attention.linear_output_gate.weight",
    "core_attention.softmax_offset": "core_attention.attn_sink",
    "core_attention.indexer.wq_b.weight": "core_attention.indexer.linear_wq_b.weight",
    "core_attention.indexer.wk.weight": "core_attention.indexer.linear_wk.weight",
    "core_attention.indexer.weights_proj.weight": "core_attention.indexer.linear_weights_proj.weight",
}


def reference_to_bridge_name(name: str, attention: str | tuple[str, ...]) -> str | None:
    """The Bridge name of a reference manifest entry, or None for an entry that is not compared.

    Args:
        name: Manifest name.
        attention: ``"H"``/``"D"`` per Hugging Face layer (``Ernie55HFLayout.attention``).

    Raises:
        KeyError: For a name this translation does not know.
    """
    if name.endswith(".local_tokens_per_expert"):
        return None
    if name == "decoder.final_layernorm.weight":
        return "decoder.final_norm.weight"
    match = _MTP_BODY_ROUTER.fullmatch(name)
    if match:
        return f"mtp.layers.{match.group(1)}.mtp_model_layer.layers.1.inner_layer.mlp.{match.group(2)}"
    match = _LAYER.fullmatch(name)
    if match is None:
        if name.startswith(("embedding.", "output_layer.", "decoder.hc_head_", "mtp.layers.")):
            return name
        raise KeyError(f"unknown reference manifest name {name!r}")
    layer, rest = int(match.group(1)), match.group(2)
    attention_layer, mlp_layer = f"decoder.layers.{2 * layer}", f"decoder.layers.{2 * layer + 1}"
    head, _, tail = rest.partition(".")
    if rest == "input_layernorm.weight":
        return f"{attention_layer}.inner_layer.input_layernorm.weight"
    if head == "self_attention_hyper_connection":
        return f"{attention_layer}.hyper_connection.{tail}"
    if head == "mlp_hyper_connection":
        return f"{mlp_layer}.hyper_connection.{tail}"
    if rest == "pre_mlp_layernorm.weight":
        return f"{mlp_layer}.inner_layer.pre_mlp_layernorm.weight"
    if head == "mlp":
        return f"{mlp_layer}.inner_layer.mlp.{tail}"
    if head == "self_attention":
        renames = _H_ATTENTION_RENAMES if attention[layer] == "H" else _D_ATTENTION_RENAMES
        return f"{attention_layer}.inner_layer.self_attention.{renames.get(tail, tail)}"
    raise KeyError(f"unknown reference manifest name {name!r}")
