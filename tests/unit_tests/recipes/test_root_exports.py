import builtins
import types
from pathlib import Path


_ROOT_INIT = Path(__file__).parents[3] / "src" / "megatron" / "bridge" / "recipes" / "__init__.py"

_NEMOTRON_VL_EXPORTS = {
    "megatron.bridge.recipes.nemotron_vl": [
        "nemotron_nano_v2_vl_12b_sft_config",
        "nemotron_nano_v2_vl_12b_peft_config",
    ],
    "megatron.bridge.recipes.nemotron_vl.h100": [
        "nemotron_nano_v2_vl_12b_sft_4gpu_h100_bf16_config",
        "nemotron_nano_v2_vl_12b_peft_2gpu_h100_bf16_config",
    ],
}


def test_root_exports_nemotron_vl_recipes():
    """Test that generic recipe lookup can resolve every public Nemotron VL recipe."""

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = types.ModuleType(name)
        module.__all__ = _NEMOTRON_VL_EXPORTS.get(name, [])
        for export in module.__all__:
            setattr(module, export, object())
        return module

    stub_builtins = vars(builtins).copy()
    stub_builtins["__import__"] = fake_import
    root_namespace = {"__builtins__": stub_builtins}

    source = _ROOT_INIT.read_text(encoding="utf-8")
    exec(compile(source, _ROOT_INIT, "exec"), root_namespace)

    expected_exports = [export for exports in _NEMOTRON_VL_EXPORTS.values() for export in exports]
    assert all(export in root_namespace for export in expected_exports)
