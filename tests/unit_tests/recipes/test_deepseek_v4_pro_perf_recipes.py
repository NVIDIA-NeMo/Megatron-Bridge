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

"""Guard the DeepSeek V4 Pro benchmark topology, its reduced-depth proxy and the BF16 benchmark."""

import importlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pytest
import torch

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.models.deepseek.deepseek_v4_config import DeepSeekV4ModelConfig
from megatron.bridge.models.deepseek.deepseek_v4_data import FixedTHDMockDatasetConfig, _FixedTHDMockDataset
from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config,
    deepseek_v4_pro_pretrain_64gpu_vrnvl72_fp8mx_proxy_config,
    deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config,
    deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config,
    deepseek_v4_pro_pretrain_256gpu_vrnvl72_fp8mx_config,
)
from megatron.bridge.perf_recipes.deepseek.gb300.deepseek_v4 import (
    _deepseek_v4_pro_pretrain_256gpu_gb300_common,
)
from megatron.bridge.training.config import (
    ConfigContainer,
    _validate_and_sync_distributed_optimizer_settings,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names, is_full_iteration_cuda_graph
from tests.unit_tests.recipes.recipe_test_utils import (
    DeepSeekV4BuilderOnlyBridge,
    patch_recipe_construction_dependencies,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("recipe_factory", "num_gpus", "logical_layers", "pp", "vp"),
    [
        (deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config, 256, 61, 4, 4),
        (deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config, 64, 16, 1, None),
        (deepseek_v4_pro_pretrain_256gpu_vrnvl72_fp8mx_config, 256, 61, 4, 4),
        (deepseek_v4_pro_pretrain_64gpu_vrnvl72_fp8mx_proxy_config, 64, 16, 1, None),
    ],
    ids=["full-pro", "proxy", "rubin-full-pro", "rubin-proxy"],
)
def test_pro_model_and_batch_contract(
    recipe_factory: Callable[[], ConfigContainer],
    num_gpus: int,
    logical_layers: int,
    pp: int,
    vp: int | None,
) -> None:
    cfg = recipe_factory()
    model = cfg.model

    assert model.num_layers == 2 * logical_layers
    assert model.is_hybrid_model is True
    assert model.mtp_num_layers == 1
    assert model.mtp_hybrid_override_pattern == "WE"
    assert model.hidden_size == 7168
    assert model.num_attention_heads == 128
    assert model.num_moe_experts == 384
    assert model.moe_router_topk == 6
    assert model.moe_ffn_hidden_size == 3072
    assert model.seq_length == cfg.dataset.seq_length == 65536
    assert (model.tensor_model_parallel_size, model.context_parallel_size) == (1, 16)
    assert (model.expert_model_parallel_size, model.expert_tensor_parallel_size) == (64, 1)
    assert (model.pipeline_model_parallel_size, model.virtual_pipeline_model_parallel_size) == (pp, vp)
    assert model.sequence_parallel is False
    assert model.attention_cp_layout == model.linear_cp_layout == "contiguous"
    assert cfg.train.global_batch_size == 256
    assert cfg.train.micro_batch_size == 1

    dense_mesh = model.tensor_model_parallel_size * model.context_parallel_size * pp
    expert_mesh = model.expert_model_parallel_size * model.expert_tensor_parallel_size * pp
    assert num_gpus % dense_mesh == num_gpus % expert_mesh == 0
    assert num_gpus // dense_mesh == 4
    assert num_gpus // expert_mesh == 1
    assert cfg.train.global_batch_size // (cfg.train.micro_batch_size * (num_gpus // dense_mesh)) == 64

    # Preserve the reference HCA/CSA prefix, including its non-windowed first blocks.
    main_pattern = model.hybrid_layer_pattern.partition("/")[0].replace("|", "")
    assert main_pattern == ("HEHE" + "CEHE" * 30)[: 2 * logical_layers]
    assert main_pattern.count("E") == logical_layers
    # One ratio per hybrid layer, MTP included: H -> 128, C -> 4, W and E -> 0.
    ratios = {"H": 128, "C": 4, "W": 0, "E": 0}
    assert model.csa_compress_ratios == [ratios[symbol] for symbol in main_pattern + "WE"]


@pytest.mark.parametrize(
    "recipe_factory",
    [deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config, deepseek_v4_pro_pretrain_256gpu_vrnvl72_fp8mx_config],
    ids=["gb300", "rubin"],
)
def test_full_pro_pipeline_chunks_match_reference_logical_blocks(recipe_factory) -> None:
    cfg = recipe_factory()
    main_pattern = cfg.model.hybrid_layer_pattern.partition("/")[0]
    counts = [3] + [4] * 14 + [2]
    layout = cfg.model.pipeline_model_parallel_layout

    assert [len(stage) // 2 for stage in main_pattern.split("|")] == counts
    assert [stage.count("decoder") // 2 for stage in layout] == counts
    assert layout[0][0] == "embedding"
    assert layout[-1][-2:] == ["mtp", "loss"]
    assert sum(stage.count("embedding") for stage in layout) == 1
    assert sum(stage.count("mtp") for stage in layout) == 1
    assert sum(stage.count("loss") for stage in layout) == 1


def test_proxy_preserves_parent_settings_outside_depth_and_pipeline() -> None:
    full = deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config()
    proxy = deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config()
    depth_fields = {
        "num_layers",
        "hybrid_layer_pattern",
        "csa_compress_ratios",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "pipeline_model_parallel_layout",
    }
    assert {key: value for key, value in vars(proxy.model).items() if key not in depth_fields} == {
        key: value for key, value in vars(full.model).items() if key not in depth_fields
    }
    for field in fields(full):
        if field.name not in {"model", "comm_overlap", "ddp"}:
            assert getattr(proxy, field.name) == getattr(full, field.name), field.name

    assert proxy.model.pipeline_model_parallel_layout is None
    assert "|" not in proxy.model.hybrid_layer_pattern
    assert proxy.model.virtual_pipeline_model_parallel_size is None
    assert proxy.comm_overlap.overlap_p2p_comm is False


@pytest.mark.parametrize(
    ("gb300_factory", "rubin_factory"),
    [
        (deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config, deepseek_v4_pro_pretrain_256gpu_vrnvl72_fp8mx_config),
        (
            deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config,
            deepseek_v4_pro_pretrain_64gpu_vrnvl72_fp8mx_proxy_config,
        ),
    ],
    ids=["full-pro", "proxy"],
)
def test_rubin_differs_from_gb300_only_in_mok_comm_sms(gb300_factory, rubin_factory) -> None:
    gb300 = gb300_factory()
    rubin = rubin_factory()

    assert rubin.model.moe_megakernel_backend_config == {
        **gb300.model.moe_megakernel_backend_config,
        "fwd_num_comm_sms": 40,
        "bwd_num_comm_sms": 40,
    }
    assert {key: value for key, value in vars(rubin.model).items() if key != "moe_megakernel_backend_config"} == {
        key: value for key, value in vars(gb300.model).items() if key != "moe_megakernel_backend_config"
    }
    for field in fields(gb300):
        if field.name != "model":
            assert getattr(rubin, field.name) == getattr(gb300, field.name), field.name


@pytest.mark.parametrize(
    ("recipe_factory", "comm_sms"),
    [
        (deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config, 28),
        (deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config, 28),
        (deepseek_v4_pro_pretrain_256gpu_vrnvl72_fp8mx_config, 40),
        (deepseek_v4_pro_pretrain_64gpu_vrnvl72_fp8mx_proxy_config, 40),
    ],
    ids=["full-pro", "proxy", "rubin-full-pro", "rubin-proxy"],
)
def test_pro_optimizer_and_execution_contract(recipe_factory: Callable[[], ConfigContainer], comm_sms: int) -> None:
    cfg = recipe_factory()

    assert cfg.optimizer.optimizer == "muon"
    assert cfg.optimizer.muon_coefficient_type == "deepseekv4"
    assert cfg.optimizer.muon_num_ns_steps == 10
    # Configuration normalization must preserve compact layer-wise Muon ownership.
    for _ in range(2):
        _validate_and_sync_distributed_optimizer_settings(cfg)
        assert cfg.optimizer.use_distributed_optimizer is False
        assert cfg.optimizer.use_layer_wise_distributed_optimizer is True
        assert cfg.optimizer.use_layer_wise_param_layout is False
        assert cfg.ddp.use_distributed_optimizer is False
        # mHC cannot wait for an overlapped parameter gather on MCore main.
        assert cfg.ddp.overlap_param_gather is cfg.optimizer.overlap_param_gather is False
    assert cfg.comm_overlap.overlap_param_gather is False
    assert cfg.comm_overlap.overlap_param_gather_with_optimizer_step is False
    assert cfg.optimizer.overlap_param_gather_with_optimizer_step is False
    assert cfg.optimizer.lr == 2e-4
    assert cfg.optimizer.min_lr == 2e-5
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.train.train_iters == cfg.scheduler.lr_decay_iters == 50
    assert cfg.scheduler.lr_warmup_iters == 0

    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["mla_up_proj", "mhc"]
    assert cfg.model.mhc_recompute_layer_num == 2
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_warmup_steps == 0
    assert cuda_graph_module_names(cfg.model) == []
    assert not is_full_iteration_cuda_graph(cfg.model)
    assert cfg.model.use_te_rng_tracker is cfg.rng.te_rng_tracker is False
    assert cfg.comm_overlap.align_param_gather is False
    assert cfg.ddp.align_param_gather is False
    assert cfg.model.cuda_graph_thd_max_packed_sequences == cfg.dataset.metadata_capacity == 8
    assert cfg.model.output_projection_groups == 16
    assert cfg.model.output_projection_lora_rank == 1024
    assert cfg.model.moe_num_hash_layers == 3
    assert cfg.model.hash_moe_vocab_size == 129280
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules is None
    assert cfg.model.moe_paged_stash is False
    assert cfg.model.moe_megakernel_backend == "mok"
    assert cfg.model.moe_megakernel_backend_config == {
        "fwd_num_comm_sms": comm_sms,
        "bwd_num_comm_sms": comm_sms,
        "minibatch_size": 4096,
        "macrobatch_size": 32768,
        "schedule_capacity_multiplier": 0.0625,
        "all_gather_top_experts_chunk_bytes": 2048,
    }
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.dsa_kernel_backend == "cudnn"
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1
    # Legacy NCCL backend: PyTorch 26.11's default nccl2 breaks MoK's barrier and Muon's parameter gather.
    assert cfg.env_vars["TORCH_DIST_USE_NCCL2"] == 0


@pytest.mark.parametrize(
    ("num_gpus", "gpu", "variant", "num_layers", "pp"),
    [
        (256, "gb300", None, 122, 4),
        (64, "gb300", "proxy", 32, 1),
        (256, "vrnvl72", None, 122, 4),
        (64, "vrnvl72", "proxy", 32, 1),
    ],
    ids=["full-pro", "proxy", "rubin-full-pro", "rubin-proxy"],
)
def test_pro_recipes_are_available_through_performance_selector(
    monkeypatch: pytest.MonkeyPatch, num_gpus, gpu, variant, num_layers, pp
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts" / "performance"))
    from utils.utils import get_perf_recipe_by_name, list_available_config_variants

    cfg = get_perf_recipe_by_name("deepseek_v4_pro", "pretrain", num_gpus, gpu, "fp8_mx", config_variant=variant)
    assert cfg.model.num_layers == num_layers
    assert cfg.model.pipeline_model_parallel_size == pp
    assert variant in list_available_config_variants(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v4_pro",
        gpu=gpu,
        compute_dtype="fp8_mx",
        task="pretrain",
    )


def test_fixed_thd_samples_have_shifted_targets_and_static_packed_boundaries() -> None:
    dataset = _FixedTHDMockDataset(
        num_samples=2, seq_length=32, vocab_size=128, seed=1234, eod=127, metadata_capacity=8
    )
    sample = dataset[0]
    assert len(dataset) == 2
    assert sample["tokens"].shape == sample["labels"].shape == (32,)
    assert torch.equal(sample["tokens"][1:-1], sample["labels"][:-2])
    assert sample["tokens"][-1].item() == 0
    assert sample["labels"][-2:].tolist() == [127, 0]
    assert sample["loss_mask"].sum().item() == 31
    assert sample["loss_mask"][-1].item() == 0
    assert sample["position_ids"].tolist() == list(range(31)) + [0]
    assert sample["padding_mask"].tolist() == [False] * 31 + [True]
    assert sample["pad_between_seqs"].item() is True
    for suffix in ("q", "kv"):
        assert sample[f"cu_seqlens_{suffix}"].dtype == torch.int32
        assert sample[f"cu_seqlens_{suffix}"].tolist() == [0] + [31] * 8
        assert sample[f"cu_seqlens_{suffix}_padded"].tolist() == [0] + [32] * 8
        assert sample[f"max_seqlen_{suffix}"].item() == 32
    assert all(torch.equal(value, dataset[0][key]) for key, value in sample.items())
    assert not torch.equal(sample["tokens"], dataset[1]["tokens"])
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(IndexError):
        dataset[2]


def test_fixed_thd_dataset_config_retains_boundaries_and_separates_split_seeds() -> None:
    @dataclass
    class Tokenizer:
        eod: int = 0

    cfg = FixedTHDMockDatasetConfig(seq_length=32, vocab_size=128, seed=1234, metadata_capacity=8)
    context = DatasetBuildContext(train_samples=2, valid_samples=1, test_samples=0, tokenizer=Tokenizer())
    train, valid, test = cfg.build_datasets(context)
    assert cfg.enable_offline_packing is True
    assert cfg.pad_to_max_length is True
    assert cfg.offline_packing_specs.pad_cu_seqlens is True
    assert cfg.offline_packing_specs.packed_sequence_size == 32
    assert cfg.persistent_workers is False
    assert len(train) == 2
    assert len(valid) == 1
    assert test is None
    assert not torch.equal(train[0]["tokens"], valid[0]["tokens"])
    replay, _, _ = cfg.build_datasets(context)
    assert torch.equal(train[0]["tokens"], replay[0]["tokens"])


def test_fixed_thd_dataset_config_rejects_missing_tokenizer() -> None:
    cfg = FixedTHDMockDatasetConfig(seq_length=32, vocab_size=128, seed=1234, metadata_capacity=8)
    with pytest.raises(ValueError, match="requires the configured tokenizer"):
        cfg.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0))


@pytest.mark.parametrize(("seq_length", "vocab_size", "metadata_capacity"), [(1, 128, 8), (32, 1, 8), (32, 128, 0)])
def test_fixed_thd_dataset_config_rejects_invalid_dimensions(
    seq_length: int, vocab_size: int, metadata_capacity: int
) -> None:
    with pytest.raises(ValueError, match="DSv4 fixed THD data requires"):
        FixedTHDMockDatasetConfig(
            seq_length=seq_length, vocab_size=vocab_size, seed=1234, metadata_capacity=metadata_capacity
        )


@pytest.fixture
def _offline_recipe_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)
    module = importlib.import_module("megatron.bridge.recipes.deepseek.gb300.deepseek_v4")
    monkeypatch.setattr(module, "AutoBridge", DeepSeekV4BuilderOnlyBridge)


@pytest.mark.usefixtures("_offline_recipe_construction")
def test_bf16_preserves_full_pro_workload_and_optimizer() -> None:
    bf16 = deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config()
    mxfp8 = _deepseek_v4_pro_pretrain_256gpu_gb300_common()

    assert isinstance(bf16.model, DeepSeekV4ModelConfig)
    assert bf16.model.hidden_size == 7168
    assert bf16.model.num_moe_experts == 384
    assert bf16.model.num_layers == 122  # 61 logical attention/MoE pairs.
    assert bf16.model.hybrid_layer_pattern == mxfp8.model.hybrid_layer_pattern
    assert bf16.model.pipeline_model_parallel_layout == mxfp8.model.pipeline_model_parallel_layout
    assert bf16.model.seq_length == bf16.dataset.seq_length == 4096
    assert bf16.train.global_batch_size == 4096
    assert bf16.train.micro_batch_size == 1
    assert bf16.model.tensor_model_parallel_size == 1
    assert bf16.model.pipeline_model_parallel_size == 4
    assert bf16.model.virtual_pipeline_model_parallel_size == 4
    assert bf16.model.expert_model_parallel_size == 64
    assert bf16.model.expert_tensor_parallel_size == bf16.model.context_parallel_size == 1
    assert bf16.model.moe_router_force_load_balancing is True
    assert bf16.optimizer.optimizer == "adam"
    assert asdict(bf16.optimizer) == asdict(mxfp8.optimizer)
    assert asdict(bf16.scheduler) == asdict(mxfp8.scheduler)
    assert bf16.optimizer.main_params_dtype == torch.float32
    assert bf16.optimizer.main_grads_dtype == torch.bfloat16
    assert bf16.optimizer.exp_avg_dtype == bf16.optimizer.exp_avg_sq_dtype == torch.bfloat16


@pytest.mark.usefixtures("_offline_recipe_construction")
def test_bf16_effective_precision_disables_quantized_compute_and_gather() -> None:
    cfg = deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config()
    # Exercise precision propagation on the strict builder-backed recipe config,
    # including clearing pre-existing FP8 settings.
    model = cfg.model
    model.fp8 = "e4m3"
    model.fp8_param = True
    cfg.mixed_precision.finalize()
    cfg.mixed_precision.setup(model, cfg.optimizer, cfg.ddp)

    assert model.bf16 is True
    assert model.params_dtype == model.pipeline_dtype == torch.bfloat16
    assert model.fp8 is None
    assert model.fp4 is None
    assert cfg.model.quant_recipe is None
    assert model.fp8_param is False
    assert cfg.ddp.fp8_param_gather is False
    assert cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag is False
    assert model.fp8_dot_product_attention is False
    assert model.fp8_multi_head_attention is False
    assert cfg.model.moe_router_padding_for_fp8 is False
    assert cfg.model.moe_router_padding_for_quantization is False
    assert cfg.ddp.grad_reduce_in_fp32 is False


@pytest.mark.usefixtures("_offline_recipe_construction")
def test_bf16_uses_supported_eager_recompute_combination() -> None:
    cfg = deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config()
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["mla_up_proj", "mhc", "moe"]
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.use_te_rng_tracker is cfg.rng.te_rng_tracker is False
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules == []
    assert cfg.model.moe_paged_stash is False
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is False
