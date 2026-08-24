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

import io
import os
import pickle

import pytest
import torch
from webdataset.autodecode import DecodingError

from megatron.bridge.diffusion.data.common.safe_decoder import SafeDiffusionSampleDecoder


class _WriteMarkerPayload:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


@pytest.mark.parametrize("extension", ["pickle", "pkl", "pyd"])
@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_safe_decoder_loads_plain_tensor_pickles(extension, protocol):
    decoder = SafeDiffusionSampleDecoder()
    value = {"embedding": torch.arange(4, dtype=torch.float32), "shape": [1, 4]}

    restored = decoder.decode(f"sample.{extension}", pickle.dumps(value, protocol=protocol))

    assert restored["shape"] == value["shape"]
    assert torch.equal(restored["embedding"], value["embedding"])


def test_safe_decoder_loads_weights_only_pth():
    decoder = SafeDiffusionSampleDecoder()
    value = {"embedding": torch.arange(4, dtype=torch.float32), "shape": [1, 4]}
    buffer = io.BytesIO()
    torch.save(value, buffer)

    restored = decoder.decode("sample.pth", buffer.getvalue())

    assert restored["shape"] == value["shape"]
    assert torch.equal(restored["embedding"], value["embedding"])


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_safe_decoder_rejects_pickle_payloads_for_all_protocols(protocol, tmp_path):
    decoder = SafeDiffusionSampleDecoder()
    marker = tmp_path / f"pickle-{protocol}-executed"
    payload = _WriteMarkerPayload(str(marker))

    with pytest.raises(DecodingError) as error:
        decoder.decode("sample.pickle", pickle.dumps(payload, protocol=protocol))

    assert isinstance(error.value.__cause__, pickle.UnpicklingError)
    assert not marker.exists()


def test_safe_decoder_rejects_executable_pth_payload(tmp_path):
    decoder = SafeDiffusionSampleDecoder()
    marker = tmp_path / "pth-executed"
    buffer = io.BytesIO()
    torch.save(_WriteMarkerPayload(str(marker)), buffer)

    with pytest.raises(DecodingError) as error:
        decoder.decode("sample.pth", buffer.getvalue())

    assert isinstance(error.value.__cause__, pickle.UnpicklingError)
    assert not marker.exists()
