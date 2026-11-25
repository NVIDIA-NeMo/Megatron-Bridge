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

"""Check every HF conversion write, all model parameters, and the live SD-MTP router buffers."""

import contextlib
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


class WeightAudit:
    """Account for source keys, live model parameters, shared SD-MTP ownership and bias values."""

    def __init__(self, auto, hf, output):
        from megatron.bridge.models.ernie.ernie55_mappings import (
            check_ernie55_checkpoint_keys,
            ernie55_hf_param_dtypes,
        )

        self.auto, self.hf, self.output = auto, Path(hf), Path(output)
        self.output.mkdir(exist_ok=True, parents=True)
        self.index_path = self.hf / "model.safetensors.index.json"
        self.index = json.loads(self.index_path.read_text())["weight_map"]
        self.config = auto.hf_pretrained.config
        check_ernie55_checkpoint_keys(self.config, self.index)
        self.expected = set(ernie55_hf_param_dtypes(self.config))
        self.mapped, self.checked, self.bias = set(), {}, {}
        self.covered = set()
        self.witnesses = {}
        self.bias_names = {}
        self.single_source_names = {}
        self.pending = None
        self.rank = dist.get_rank()

    def verify_hf_body_copies(self):
        """Match redundant on-disk SD-MTP copies to the canonical last-layer bytes, distributed across ranks."""
        import struct

        from megatron.bridge.models.ernie.ernie55_mappings import (
            ernie55_mtp_body_copy_keys,
        )

        optional = sorted(set(self.index) & ernie55_mtp_body_copy_keys(self.config, self.expected))
        # The shared-state capture accepts BF16 or FP32 model-only weights.
        # Additional block scales would describe an independently quantized checkpoint.
        extra = set(self.index) - self.expected - set(optional)
        if extra:
            raise ValueError(f"Expected BF16 model-only weights; unexpected extra keys: {sorted(extra)[:10]}")
        with contextlib.ExitStack() as stack:
            shards = {}

            def locate(key):
                shard = self.index[key]
                if shard not in shards:
                    handle = stack.enter_context((self.hf / shard).open("rb"))
                    count = struct.unpack("<Q", handle.read(8))[0]
                    header = json.loads(handle.read(count))
                    shards[shard] = (handle, 8 + count, header)
                handle, start, header = shards[shard]
                data = header[key]
                lo, hi = data["data_offsets"]
                return handle, start + lo, hi - lo, data["shape"], data["dtype"]

            duplicates = []
            for key in optional[self.rank :: dist.get_world_size()]:
                _, _, layer, suffix = key.split(".", 3)
                last = self.config.num_hidden_layers - 1
                if int(layer) <= last:
                    raise ValueError(f"Expected an MTP copy: {key}")
                canonical = f"model.layers.{last}." + suffix.removeprefix("transformer_layer.")
                left, a, n, *left_meta = locate(key)
                right, b, m, *right_meta = locate(canonical)
                if m != n or left_meta != right_meta:
                    raise ValueError(f"Incompatible redundant MTP copy {key} and {canonical}")
                for offset in range(0, n, 4 << 20):
                    length = min(4 << 20, n - offset)
                    left.seek(a + offset)
                    block = left.read(length)
                    right.seek(b + offset)
                    if len(block) != length or block != right.read(length):
                        raise ValueError(f"Unequal redundant MTP copy {key} and {canonical}, offset {offset}")
                duplicates.append(key)
        return optional, duplicates

    @torch.no_grad()
    def flush(self):
        """Check the pending conversion after Bridge has copied it into the live parameter."""
        if self.pending is None:
            return
        task, expected = self.pending
        self.pending = None
        target = task.param_weight
        assert expected.shape == target.shape
        source = task.mapping.hf_param
        keys = [source] if isinstance(source, str) else list(source.values())
        assert not set(keys) - self.expected, (task.param_name, keys)
        # All coordinates, even for the embedding and output projection.
        # Slice along rows to bound temporary HBM and respect non-contiguous mapped layouts.
        a, b = target.detach(), expected.detach()
        if a.ndim == 0:
            a, b = a.reshape(1), b.reshape(1)
        step = max(1, 4_194_304 // max(1, a.numel() // max(1, a.shape[0])))
        for i in range(0, len(a), step):
            src = b[i : i + step].to(device=a.device, dtype=a.dtype)
            if not torch.equal(a[i : i + step], src):
                raise ValueError(f"Wrong runtime values after importing {keys} into {task.param_name} at row {i}")
        self.mapped.update(keys)
        self.covered.add((target.data_ptr(), target.numel()))
        # A small, independent source-derived witness lets us resolve every live
        # name again after the optimizer and the input pipeline have been initialized.
        stride = max(1, (target.numel() - 1) // 3)
        witness = expected.detach().reshape(-1)[::stride][:4].to(device=target.device, dtype=target.dtype).clone()
        self.witnesses[task.param_name] = (target, stride, witness)
        if isinstance(source, str):
            self.single_source_names[source] = task.param_name
        self.checked[task.global_param_name] = [
            keys,
            str(target.dtype),
            list(target.shape),
        ]
        if isinstance(source, str) and source.endswith(".e_score_correction_bias"):
            assert source not in self.bias and target.dtype == torch.float32 and target.shape == (512,)
            self.bias[source] = target
            self.bias_names[source] = task.param_name

    def load(self, model):
        """Load using AutoBridge and check every local live write and the union of global source keys."""
        from megatron.bridge.models.ernie.modeling_ernie55.mtp import (
            check_ernie55_mtp_aliases,
        )

        cls = type(self.auto._model_bridge)
        original = cls._convert_loaded_hf_weight
        if "_convert_loaded_hf_weight" in cls.__dict__:
            raise TypeError("Unexpected override on ERNIE conversion bridge")

        def convert(task, source):
            self.flush()
            value = original(task, source)
            if value is not None:
                self.pending = task, value
            return value

        cls._convert_loaded_hf_weight = staticmethod(convert)
        try:
            self.auto.load_hf_weights(model)
            self.flush()
        finally:
            del cls._convert_loaded_hf_weight
        self.models = model
        for part in model:
            missing = [n for n, p in part.named_parameters() if (p.data_ptr(), p.numel()) not in self.covered]
            if missing:
                raise ValueError(f"Model parameters without a checked HF import: {missing[:24]}, total {len(missing)}")
        self.alias_count = check_ernie55_mtp_aliases(model)
        assert self.alias_count > 0
        required = sorted(self.expected)
        marks = torch.tensor([int(k in self.mapped) for k in required], device="cuda", dtype=torch.uint8)
        dist.all_reduce(marks, op=dist.ReduceOp.MAX)
        absent = [k for k, ok in zip(required, marks.tolist()) if not ok]
        if absent:
            raise ValueError(f"{len(absent)} checkpoint keys never loaded; examples: {absent[:24]}")
        optional, duplicate_shard = self.verify_hf_body_copies()
        report = dict(
            duplicate_keys=len(optional),
            local_duplicate_keys_verified=duplicate_shard,
            index_sha256=hashlib.sha256(self.index_path.read_bytes()).hexdigest(),
            source_keys=len(self.index),
            registry_keys=len(required),
            union_checked=len(required),
            local_live_targets=len(self.checked),
            shared_mtp_parameters=self.alias_count,
            local_source_keys=len(self.mapped),
            targets=self.checked,
        )
        with gzip.open(self.output / f"mapping-{self.rank:02}.json.gz", "xt") as out:
            json.dump(report, out)
        self.check_bias("after-load")
        return model

    def verify_raw_fp32(self, live):
        """Independently compare native FP32 disk bytes to the actual pre-forward tensors in HF orientation."""
        from safetensors import safe_open

        receipts = {}
        errors = []
        for shard in sorted(set(self.index.values())):
            with safe_open(str(self.hf / shard), framework="pt", device="cpu") as store:
                for key in store.keys():
                    if key not in self.expected or store.get_slice(key).get_dtype() != "F32":
                        continue
                    # The Base F32 state is replicated. Only the hyper-connection matrices
                    # below transpose in the released formats. No conversion code is called here.
                    tensor = live[self.single_source_names[key]].detach()
                    source = store.get_tensor(key).contiguous()
                    if tensor.dtype != torch.float32 or source.dtype != torch.float32:
                        raise TypeError(f"FP32 HF tensor {key} reached the forward as {tensor.dtype}")
                    value = tensor.cpu()
                    if key.endswith((".hc_head_fn", ".mapping_proj.weight")):
                        value = value.T
                    if value.shape != source.shape:
                        raise ValueError(f"Direct FP32 shape mismatch for {key}: {value.shape} and {source.shape}")
                    bits = source.view(torch.int32)
                    actual = value.contiguous().view(torch.int32)
                    bad = int(torch.count_nonzero(actual != bits))
                    if bad:
                        errors.append((key, bad))
                    receipts[key] = [
                        source.numel(),
                        int(torch.count_nonzero(bits.bitwise_and(65535))),
                        hashlib.sha256(memoryview(source.numpy())).hexdigest(),
                        bad,
                    ]
        with gzip.open(self.output / f"raw-fp32-{self.rank:02}.json.gz", "xt") as stream:
            json.dump(receipts, stream)
        if not receipts or errors:
            raise ValueError(f"Direct on-disk FP32 check failed, {len(errors)} mismatching tensors: {errors[:8]}")

    def check_bias(self, stage):
        """Compare every live router buffer with its on-disk value and save its full values."""
        from safetensors import safe_open

        from megatron.bridge.models.ernie.modeling_ernie55.mtp import (
            check_ernie55_mtp_aliases,
        )

        assert check_ernie55_mtp_aliases(self.models) == self.alias_count
        from megatron.core.utils import unwrap_model

        (root,) = unwrap_model(self.models)
        live = dict(root.named_parameters(remove_duplicate=False))
        live.update(root.named_buffers(remove_duplicate=False))
        confirmed = 0
        if stage == "before-forward":
            self.verify_raw_fp32(live)
            for name, (original, stride, witness) in self.witnesses.items():
                tensor = live[name]
                if (
                    tensor.data_ptr() != original.data_ptr()
                    or tensor.dtype != witness.dtype
                    or not torch.equal(tensor.detach().reshape(-1)[::stride][:4], witness)
                ):
                    raise ValueError(f"Live target differs from the checked source before the actual forward: {name}")
                confirmed += 1
            (self.output / f"before-forward-{self.rank:02}.json").write_text(
                json.dumps({"live_rechecked_by_name": confirmed, "shared_mtp_parameters": self.alias_count, "ga": 8})
            )
        self.bias = {key: live[name] for key, name in self.bias_names.items()}
        keys = sorted(self.bias, key=lambda k: int(k.split(".")[2]))
        assert len(keys) == 43 and len({x.data_ptr() for x in self.bias.values()}) == 43
        assert set(keys) == {k for k in self.index if k.endswith(".e_score_correction_bias")}
        values = [self.bias[k].detach().cpu().clone() for k in keys]
        for shard in {self.index[k] for k in keys}:
            with safe_open(str(self.hf / shard), framework="pt") as handle:
                for k, value in zip(keys, values):
                    if self.index[k] == shard:
                        loaded = handle.get_tensor(k)
                        assert torch.equal(value, loaded), (
                            stage,
                            k,
                            value.dtype,
                            loaded.dtype,
                        )
        np.savez(
            self.output / f"bias-{stage}-{self.rank:02}.npz",
            keys=np.asarray(keys),
            values=torch.stack(values).numpy(),
        )
        if stage == "before-forward":
            from megatron.core.num_microbatches_calculator import get_num_microbatches

            assert get_num_microbatches() == 8
            if self.rank == 0:
                print(
                    f"GA=8 checked at live forward, {self.alias_count} shared parameters, 43 distinct FP32 biases.",
                    flush=True,
                )
