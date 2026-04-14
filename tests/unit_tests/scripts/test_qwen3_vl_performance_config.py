import importlib
import sys
from pathlib import Path


SCRIPTS_PERF_PATH = Path(__file__).parents[3] / "scripts" / "performance"
if str(SCRIPTS_PERF_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PERF_PATH))


class _FakeModelCfg:
    """Minimal fake model provider for Qwen3-VL perf config tests."""

    def __init__(self):
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_vision_projection = False
        self.moe_token_dispatcher_type = None
        self.moe_flex_dispatcher_backend = None
        self.moe_hybridep_num_sms = None
        self.moe_router_fusion = False
        self.moe_permute_fusion = False
        self.moe_grouped_gemm = False
        self.moe_router_padding_for_fp8 = False
        self.moe_shared_expert_overlap = False
        self.moe_router_force_load_balancing = False
        self.apply_rope_fusion = False

    def finalize(self):
        return None


class _FakeAutoBridge:
    """Fake AutoBridge used to avoid HF/network access in perf config tests."""

    @staticmethod
    def from_hf_pretrained(_hf_path: str):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()


def test_qwen3_vl_235b_perf_config_disables_rope_fusion(monkeypatch):
    """Qwen3-VL perf configs should not re-enable unsupported fused RoPE."""
    qwen3_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen3_vl")
    monkeypatch.setattr(qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    qwen3_vl_perf_module = importlib.import_module("configs.qwen_vl.qwen3_vl_pretrain")
    cfg = qwen3_vl_perf_module.qwen3_vl_235b_a22b_pretrain_config_h100(precision="bf16", mock=True)

    assert cfg.model.apply_rope_fusion is False
