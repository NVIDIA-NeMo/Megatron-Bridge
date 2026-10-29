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

import importlib
from typing import Any, Callable

import pytest
from megatron.energon import CaptioningSample, StandardWebdatasetFactory
from megatron.energon import dataset_config as energon_dataset_config
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
        energon_dataset_config.load_config(
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


@pytest.mark.parametrize("nested", [False, True])
def test_dataset_yaml_rejects_serialized_functions_before_execution(nested, tmp_path):
    marker = tmp_path / "serialized-function-executed"
    module_path = tmp_path / "evil_config.py"
    module_path.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef payload(sample=None):\n    return sample\n"
    )
    config = {"__module__": "evil_config", "__function__": "payload"}
    default_kwargs = None
    if nested:
        config = {"sample_loader": config}
        default_kwargs = {"path": EPath(tmp_path), "part_filter": ["json"]}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ValueError, match="cannot resolve serialized Python functions"):
            energon_dataset_config.load_config(
                config,
                default_type=DefaultGenericWebdatasetFactory,
                default_kwargs=default_kwargs,
            )

    assert not marker.exists()


def test_dataset_yaml_rejects_serialized_class_before_execution(tmp_path):
    marker = tmp_path / "serialized-class-executed"
    config = {
        "unused": {
            "__module__": "subprocess",
            "__class__": "Popen",
            "args": ["/usr/bin/touch", str(marker)],
        }
    }

    with pytest.raises(ValueError, match="cannot instantiate serialized Python classes"):
        energon_dataset_config.load_config(
            config,
            default_type=DefaultGenericWebdatasetFactory,
            default_kwargs={"path": EPath(tmp_path), "field_map": {}},
        )

    assert not marker.exists()


def test_dataset_yaml_allows_packaged_factory_and_sample_classes(monkeypatch, tmp_path):
    captured = {}

    def original_init(self, path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(base_energon_datamodule, "_energon_factory_init", original_init)
    config = {
        "__module__": "megatron.energon",
        "__class__": "StandardWebdatasetFactory",
        "sample_type": {
            "__module__": "megatron.energon",
            "__class__": "CaptioningSample",
        },
        "field_map": {"image": "jpg", "caption": "txt"},
    }

    dataset = energon_dataset_config.load_config(
        config,
        default_type=DefaultGenericWebdatasetFactory,
        default_kwargs={"path": EPath(tmp_path)},
    )

    assert isinstance(dataset, StandardWebdatasetFactory)
    assert dataset.__sample_type__ is CaptioningSample
    assert captured["field_map"] == config["field_map"]


def test_dataset_yaml_allows_legacy_chatml_factory_alias(monkeypatch, tmp_path):
    def original_init(self, path, **kwargs):
        return None

    monkeypatch.setattr(base_energon_datamodule, "_energon_factory_init", original_init)
    dataset = energon_dataset_config.load_config(
        {
            "__module__": "megatron.bridge.models.qwen_vl.data.energon",
            "__class__": "ChatMLWebdataset",
        },
        default_type=DefaultGenericWebdatasetFactory,
        default_kwargs={"path": EPath(tmp_path)},
    )

    from megatron.bridge.data.energon.task_encoder_utils import ChatMLWebdataset

    assert isinstance(dataset, ChatMLWebdataset)


def _trusted_metadataset_joiner(*samples: Any) -> tuple[Any, ...]:
    return samples


def test_non_dataset_config_preserves_serialized_joiner_functions():
    joiner = energon_dataset_config.load_config(
        {
            "__module__": __name__,
            "__function__": "_trusted_metadataset_joiner",
        },
        default_type=Callable[..., tuple[Any, ...]],
    )

    assert joiner is _trusted_metadataset_joiner


def test_energon_guards_are_reload_safe():
    reloaded = importlib.reload(base_energon_datamodule)
    original_attribute = reloaded._ORIGINAL_METHOD_ATTRIBUTE

    assert getattr(reloaded._secure_energon_factory_init, original_attribute) is reloaded._energon_factory_init
    assert reloaded._energon_factory_init is not reloaded._secure_energon_factory_init
    assert getattr(reloaded._secure_energon_load_config, original_attribute) is reloaded._energon_load_config
    assert reloaded._energon_load_config is not reloaded._secure_energon_load_config
