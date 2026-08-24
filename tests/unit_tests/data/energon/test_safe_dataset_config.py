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

import pytest
from megatron.energon.dataset_config import load_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.default_generic_webdataset import DefaultGenericWebdatasetFactory

from megatron.bridge.data.energon import base_energon_datamodule


@pytest.mark.parametrize("field", ["sample_loader", "part_filter"])
def test_dataset_yaml_rejects_python_hooks_before_import(field, tmp_path):
    marker = tmp_path / "dataset-python-executed"
    metadata_dir = tmp_path / ".nv-meta"
    metadata_dir.mkdir()
    module_path = metadata_dir / "evil.py"
    module_path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    config_path = metadata_dir / "dataset.yaml"
    config_path.write_text(
        "sample_loader: evil.py\npart_filter:\n  - json\n" if field == "sample_loader" else "part_filter: evil.py\n"
    )
    default_kwargs = {"path": EPath(tmp_path)}
    if field == "part_filter":
        default_kwargs["sample_loader"] = lambda sample: sample

    with pytest.raises(ValueError, match="cannot load Python files"):
        load_config(
            EPath(config_path),
            default_type=DefaultGenericWebdatasetFactory,
            default_kwargs=default_kwargs,
        )

    assert not marker.exists()


def test_factory_guard_preserves_callable_hooks(monkeypatch, tmp_path):
    captured = {}

    def original_init(self, path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(base_energon_datamodule, "_energon_factory_init", original_init)

    def sample_loader(sample):
        return sample

    def part_filter(_part):
        return True

    base_energon_datamodule._secure_energon_factory_init(
        object(), EPath(tmp_path), sample_loader=sample_loader, part_filter=part_filter
    )

    assert captured["sample_loader"] is sample_loader
    assert captured["part_filter"] is part_filter
