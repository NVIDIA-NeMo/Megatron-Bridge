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

"""Guard the logged Pro topology and the GB300/Rubin debugging proxies."""

from collections.abc import Callable
from dataclasses import dataclass, fields, make_dataclass
from pathlib import Path

import pytest
import torch

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import DeepSeekV4HybridModelProvider
from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config,
    deepseek_v4_pro_pretrain_64gpu_vr200_fp8mx_proxy_config,
    deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config,
)
from megatron.bridge.perf_recipes.deepseek.gb300._deepseek_v4_compat import (
    _MCORE_CAPABILITIES,
    FixedTHDMockDatasetConfig,
    _FixedTHDMockDataset,
    configure_dsv4_dev_model,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names, is_full_iteration_cuda_graph


pytestmark = pytest.mark.unit


def _build_recipe(recipe_factory: Callable[[], ConfigContainer]) -> ConfigContainer:
    available = {field.name for field in fields(DeepSeekV4HybridModelProvider)}
    missing = sorted(set(_MCORE_CAPABILITIES) - available)
    if missing:
        pytest.skip(f"DSv4 development MCore required; missing declared fields: {', '.join(missing)}")
    return recipe_factory()


@pytest.mark.parametrize(
    ("recipe_factory", "num_gpus", "logical_layers", "pp", "vp"),
    [
        (deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config, 256, 61, 4, 4),
        (deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config, 64, 16, 1, None),
        (deepseek_v4_pro_pretrain_64gpu_vr200_fp8mx_proxy_config, 64, 16, 1, None),
    ],
    ids=["full-pro", "gb300-proxy", "vr200-proxy"],
)
def test_pro_log_model_and_batch_contract(
    recipe_factory: Callable[[], ConfigContainer],
    num_gpus: int,
    logical_layers: int,
    pp: int,
    vp: int | None,
) -> None:
    cfg = _build_recipe(recipe_factory)
    model = cfg.model

    assert model.num_layers == 2 * logical_layers
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
    assert model.cp_partition_mode == "contiguous"
    assert cfg.train.global_batch_size == 256
    assert cfg.train.micro_batch_size == 1

    dense_mesh = model.tensor_model_parallel_size * model.context_parallel_size * pp
    expert_mesh = model.expert_model_parallel_size * model.expert_tensor_parallel_size * pp
    assert num_gpus % dense_mesh == num_gpus % expert_mesh == 0
    assert num_gpus // dense_mesh == 4
    assert num_gpus // expert_mesh == 1
    assert cfg.train.global_batch_size // (cfg.train.micro_batch_size * (num_gpus // dense_mesh)) == 64

    # Preserve the HCA/CSA prefix from the logs, including their non-windowed first blocks.
    main_pattern = model.hybrid_layer_pattern.partition("/")[0].replace("|", "")
    expected_pattern = ("HEHE" + "CEHE" * 30)[: 2 * logical_layers]
    assert main_pattern == expected_pattern
    assert main_pattern.count("E") == logical_layers


def test_full_pro_pipeline_chunks_match_logged_logical_blocks() -> None:
    cfg = _build_recipe(deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config)
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
    full = _build_recipe(deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config)
    proxy = _build_recipe(deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config)
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


def test_rubin_proxy_preserves_gb300_contract_outside_mok_comm_sms() -> None:
    gb300 = _build_recipe(deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config)
    rubin = _build_recipe(deepseek_v4_pro_pretrain_64gpu_vr200_fp8mx_proxy_config)

    assert {key: value for key, value in vars(rubin.model).items() if key != "moe_megakernel_backend_config"} == {
        key: value for key, value in vars(gb300.model).items() if key != "moe_megakernel_backend_config"
    }
    for section in fields(gb300):
        if section.name != "model":
            assert getattr(rubin, section.name) == getattr(gb300, section.name), section.name

    # The Flash Rubin recipe changes these for its grouped-GLU path; Pro uses MoK.
    assert rubin.model.activation_func_clamp_value == 10.0
    assert rubin.model.use_transformer_engine_op_fuser is False


@pytest.mark.parametrize(
    ("recipe_factory", "comm_sms"),
    [
        (deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config, 28),
        (deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config, 28),
        (deepseek_v4_pro_pretrain_64gpu_vr200_fp8mx_proxy_config, 40),
    ],
    ids=["full-pro", "gb300-proxy", "vr200-proxy"],
)
def test_pro_log_optimizer_and_execution_contract(
    recipe_factory: Callable[[], ConfigContainer], comm_sms: int
) -> None:
    cfg = _build_recipe(recipe_factory)

    assert set(_MCORE_CAPABILITIES) <= {field.name for field in fields(cfg.model)}
    assert {"muon_coefficient_type", "muon_num_ns_steps"} <= {field.name for field in fields(cfg.optimizer)}

    assert cfg.optimizer.optimizer == "muon"
    assert cfg.optimizer.muon_coefficient_type == "deepseekv4"
    assert cfg.optimizer.muon_num_ns_steps == 10
    assert cfg.optimizer.use_distributed_optimizer is False
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
    assert cfg.model.mhc_recompute_attn_cuda_graph_split is True
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(cfg.model) == ["attn"]
    assert not is_full_iteration_cuda_graph(cfg.model)
    assert cfg.model.cuda_graph_warmup_steps == 1
    assert cfg.model.cuda_graph_dynamic_microbatches is True
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


@pytest.mark.parametrize("hardware", ["gb300", "vr200"])
def test_proxy_is_available_through_performance_selector(monkeypatch: pytest.MonkeyPatch, hardware: str) -> None:
    _build_recipe(deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts" / "performance"))
    from utils.utils import get_perf_recipe_by_name, list_available_config_variants

    cfg = get_perf_recipe_by_name("deepseek_v4_pro", "pretrain", 64, hardware, "fp8_mx", config_variant="proxy")
    assert cfg.model.num_layers == 32
    assert cfg.model.pipeline_model_parallel_size == 1
    assert "proxy" in list_available_config_variants(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v4_pro",
        gpu=hardware,
        compute_dtype="fp8_mx",
        task="pretrain",
    )


@pytest.mark.parametrize(
    "recipe_factory",
    [
        deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config,
        deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config,
        deepseek_v4_pro_pretrain_64gpu_vr200_fp8mx_proxy_config,
    ],
)
def test_pro_recipes_reject_unsupported_mcore(recipe_factory: Callable[[], ConfigContainer]) -> None:
    available = {field.name for field in fields(DeepSeekV4HybridModelProvider)}
    if set(_MCORE_CAPABILITIES) <= available:
        pytest.skip("Installed MCore declares the DSv4 development capabilities.")
    with pytest.raises(RuntimeError, match="d211e50e888fcb8640ddc9923e98940800afcfd3"):
        recipe_factory()


def test_development_capability_guard_requires_declared_dataclass_fields() -> None:
    @dataclass
    class UnsupportedModel:
        num_layers: int = 32

        @property
        def mhc_recompute_layer_num(self) -> int:
            return 2

    # A readable property is insufficient: this setting must survive config serialization.
    with pytest.raises(RuntimeError, match="mhc_recompute_layer_num"):
        configure_dsv4_dev_model(UnsupportedModel())


def test_development_compression_ratios_cover_attention_moe_and_mtp() -> None:
    config_type = make_dataclass(
        "DevelopmentModel",
        [(name, object, None) for name in _MCORE_CAPABILITIES]
        + [
            ("enable_hyper_connections", bool, False),
            ("num_residual_streams", int, 1),
            ("hybrid_stack_spec", object, None),
            ("hybrid_layer_pattern", str, "HEHE|CEHE"),
            ("mtp_hybrid_override_pattern", str, "WE"),
            ("mtp_num_layers", int, 1),
            ("csa_compress_ratios", object, None),
        ],
    )
    model = config_type()
    configure_dsv4_dev_model(model)
    assert model.csa_compress_ratios == [128, 0, 128, 0, 4, 0, 128, 0, 0, 0]
    assert model.o_groups == 16
    assert model.o_lora_rank == 1024
    assert model.moe_n_hash_layers == 3
    assert model.actual_vocab_size == 129280
    assert model.enable_hyper_connections is True
    assert model.num_residual_streams == 4
    assert callable(model.hybrid_stack_spec)


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
