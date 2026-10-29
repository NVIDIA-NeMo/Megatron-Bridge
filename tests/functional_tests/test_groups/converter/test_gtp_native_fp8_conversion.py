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

"""Two-GPU native FP8 HF import and gathered dequantized export coverage."""

import gc
from types import SimpleNamespace

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.config import set_experimental_flag
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping
from tests.functional_tests.utils import initialize_distributed


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """This module generates its tensors and needs no downloaded data."""
    yield


class _NativeFP8Bridge(MegatronModelBridge):
    """Use the production conversion lifecycle for a single real TE parameter."""

    def provider_bridge(self, hf_pretrained):
        return None

    def mapping_registry(self):
        return MegatronMappingRegistry(ColumnParallelMapping("weight", "hf.weight"))


class _InMemoryHFState(dict):
    """Small HF source with the key-order interface used by task construction."""

    @property
    def source(self):
        return self

    def get_all_keys(self):
        return self.keys()


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("fp8_format", ["fp8", "mxfp8"])
def test_gtp_native_fp8_hf_import_and_dequantized_export(fp8_format):
    """Import twice into real GTP/TE storage and compare against independent TE casts."""
    if not HAVE_GTP:
        pytest.skip("GTP requires TransformerEngine >= 2.19")
    initialize_distributed()
    if torch.distributed.get_world_size() != 2:
        pytest.skip("This test requires exactly two distributed ranks")
    if fp8_format == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Native MXFP8 requires Blackwell or newer (CUDA device capability >= 10)")

    import transformer_engine_torch as tex
    from megatron.core.fp8_utils import is_float8tensor
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import reset_gtp_state
    from megatron.core.tensor_parallel.gtp_api import attach_gtp_to_presharded_module
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    def make_quantizer():
        if fp8_format == "mxfp8":
            quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
            quantizer.optimize_for_gemm = False
        else:
            quantizer = Float8Quantizer(
                scale=torch.ones(1, dtype=torch.float32, device="cuda"),
                amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
            quantizer.set_usage(rowwise=True, columnwise=False)
        return quantizer

    def quantized_payload(weight, _block_size=None):
        quantized = make_quantizer()(weight)
        metadata = quantized.get_metadata()
        values = metadata["rowwise_data" if fp8_format == "mxfp8" else "data"]
        scales = metadata["rowwise_scale_inv" if fp8_format == "mxfp8" else "fp8_scale_inv"]
        assert isinstance(values, torch.Tensor) and isinstance(scales, torch.Tensor)
        return values.contiguous().view(torch.float8_e4m3fn), scales

    set_experimental_flag(True)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        gtp_remat_size=2,
    )
    model = torch.nn.Module()
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model.pg_collection = pg
        model.config = SimpleNamespace(share_embeddings_and_output_weights=False, pipeline_model_parallel_size=1)
        model.weight = torch.nn.Parameter(
            make_quantizer()(torch.zeros(128, 128, dtype=torch.bfloat16, device="cuda")), requires_grad=False
        )
        attach_gtp_to_presharded_module(model, pg.gtp_remat, pad_length=0, replica_group=pg.dp_cp)
        param = model.weight
        assert is_float8tensor(param)
        assert param.is_gtp_weight_remat and param._gtp_native_fp8
        assert param.group.size() == 2 and param.shape == (128, 128)
        original_class = type(param)
        original_quantizer = param._quantizer
        metadata = param.get_metadata()
        storage_pointers = {
            name: tensor.data_ptr() for name, tensor in metadata.items() if isinstance(tensor, torch.Tensor)
        }
        assert storage_pointers, "The native FP8 parameter must expose real TE storage"
        bridge = _NativeFP8Bridge()
        previous_export = None

        for phase in range(2):
            # Identical HF source on both ranks, with distinct rows and non-FP8-exact values.
            source = (torch.arange(256 * 128, device="cuda").reshape(256, 128).float() / 37 + phase).sin()
            source = (source * (phase + 1)).bfloat16()
            expected = make_quantizer()(source).dequantize()
            hf = SimpleNamespace(
                state=_InMemoryHFState({"hf.weight": source}),
                config=SimpleNamespace(),
                model_name_or_path="in-memory-native-fp8",
            )

            bridge.load_weights_hf_to_megatron(hf, [model])
            assert model.weight is param and type(param) is original_class
            assert param._quantizer is original_quantizer
            assert param.quantized is param and param.is_gtp_weight_remat
            assert param.group is pg.gtp_remat and param.pad_length == 0
            current_metadata = param.get_metadata()
            assert {name: current_metadata[name].data_ptr() for name in storage_pointers} == storage_pointers

            exported = dict(
                bridge.stream_weights_megatron_to_hf(
                    [model], hf, cpu=False, show_progress=False, merge_adapter_weights=False
                )
            )
            assert set(exported) == {"hf.weight"}
            actual = exported["hf.weight"]
            assert actual.shape == source.shape and not is_float8tensor(actual)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert not torch.equal(actual, source), "The reference must exercise actual FP8 quantization"
            if previous_export is not None:
                assert not torch.equal(actual, previous_export), "The second import must update native FP8 storage"
            previous_export = actual.clone()

            # Exercise the separate quantized exporter with real TE casts after the GTP gather.
            expected_values, expected_scales = quantized_payload(expected)
            quantized = dict(
                bridge.stream_weights_megatron_to_hf_quant(
                    [model],
                    hf,
                    lambda _name: True,
                    quantized_payload,
                    cpu=False,
                    show_progress=False,
                )
            )
            assert set(quantized) == {"hf.weight", "hf.weight_scale_inv"}
            assert quantized["hf.weight"].shape == source.shape
            assert torch.equal(quantized["hf.weight"].view(torch.uint8), expected_values.view(torch.uint8))
            assert torch.equal(quantized["hf.weight_scale_inv"], expected_scales)
        torch.distributed.barrier()
    finally:
        reset_gtp_state()
        del model
        parallel_state.destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
