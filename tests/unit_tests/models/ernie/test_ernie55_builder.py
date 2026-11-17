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

"""CPU tests of :class:`Ernie55ModelBuilder`: a tiny ERNIE 5.5 built from a Hugging Face config, end to end.

The config goes through the real bridge mapping (:class:`Ernie55Bridge`) and :class:`Ernie55ModelConfig`; only
the Transformer Engine leaves of the stack spec are replaced by the CPU stand-ins of ``test_ernie55_mhc_mtp.py``
(the builder accepts any ERNIE 5.5 stack spec).
"""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec as mcore_hybrid_stack_spec
from megatron.core.optimizer import optimizer as mcore_optimizer

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# Bridge's conversion stack imports its LoRA layers, which need Transformer Engine; without TE, import it with a
# placeholder for them.
allow_conversion_imports_without_te()
pytest.importorskip("megatron.bridge.models.conversion.model_bridge", reason="needs Bridge's conversion stack")

from megatron.bridge.models.ernie.ernie55_bridge import Ernie55Bridge
from megatron.bridge.models.ernie.ernie55_builder import Ernie55ModelBuilder
from megatron.bridge.models.ernie.ernie55_config import Ernie55ModelConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    MULTIMAX_GRAD_NORM_GROUP,
    Ernie55HCASelfAttention,
    Ernie55HybridModel,
    Ernie55HybridStack,
    Ernie55LatentMQASelfAttention,
    check_ernie55_mtp_aliases,
)
from megatron.bridge.models.hybrid.hybrid_builder import BridgeHybridModelBuilder
from tests.unit_tests.models.ernie.ernie55_hf_configs import proxy_config
from tests.unit_tests.models.ernie.test_ernie55_bridge_config import _hf
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import (  # noqa: F401 - pytest fixtures
    _cpu_nvtx,
    _cpu_spec,
    _fresh_trackers,
    _randomize_,
    needs_hook,
    pg_collection,
)


# Every ERNIE 5.5 model with H layers needs Megatron-Core's _post_o_group_proj hook.
pytestmark = [pytest.mark.unit, needs_hook]

VOCAB = 97

# The proxy's layout (H, D, H, D with a dense first layer, one shared-body D MTP depth) at CPU sizes.
TINY = proxy_config(
    vocab_size=VOCAB,
    hidden_size=64,
    intermediate_size=24,
    num_attention_heads=4,
    head_dim=16,
    v_head_dim=16,
    qk_rope_head_dim=8,
    qk_nope_head_dim=8,
    q_lora_rank=32,
    o_groups=2,
    o_lora_rank=8,
    rope_theta=10000.0,
    compress_rope_theta=1000.0,
    sliding_window=5,
    hybrid_mla_num_attention_heads=4,
    hybrid_mla_q_lora_rank=32,
    hybrid_mla_kv_lora_rank=16,
    hybrid_mla_qk_nope_head_dim=8,
    hybrid_mla_qk_rope_head_dim=8,
    hybrid_mla_v_head_dim=8,
    index_n_heads=4,
    index_head_dim=16,
    index_topk=6,
    n_routed_experts=8,
    num_experts_per_tok=4,
    moe_intermediate_size=8,
    vha_rank=2,
)


def _tiny_config(**overrides) -> Ernie55ModelConfig:
    config = Ernie55Bridge().hf_config_to_model_config(_hf(TINY))
    # CPU: the PyTorch reference attention, CPU parameters, the all-gather dispatcher and the stand-in leaves.
    config.dsa_kernel_backend = "none"
    config.use_cpu_initialization = True
    config.moe_token_dispatcher_type = "allgather"
    config.moe_grouped_gemm = False
    config.gradient_accumulation_fusion = False
    config.hybrid_stack_spec = _cpu_spec()
    config.seq_length = 1024
    for name, value in overrides.items():
        setattr(config, name, value)
    config.finalize()
    return config


def _build(pg_collection, config: Ernie55ModelConfig | None = None, **kwargs) -> Ernie55HybridModel:
    torch.manual_seed(0)
    return Ernie55ModelBuilder(config or _tiny_config()).build_model(pg_collection, **kwargs)


def test_builds_the_ernie_model(pg_collection):
    model = _build(pg_collection)
    assert type(model) is Ernie55HybridModel
    assert isinstance(model.decoder, Ernie55HybridStack)
    assert "".join(model.decoder.layer_type_list) == "H-DEHEDE"
    attentions = [
        type(module)
        for module in model.modules()
        if isinstance(module, (Ernie55HCASelfAttention, Ernie55LatentMQASelfAttention))
    ]
    # Two H and two D layers in the decoder, the shared-body D layer of the MTP depth.
    assert attentions.count(Ernie55HCASelfAttention) == 2
    assert attentions.count(Ernie55LatentMQASelfAttention) == 3
    assert model.embedding.word_embeddings.weight.shape[0] == VOCAB
    assert model.output_layer.weight.shape[0] == VOCAB
    assert model.output_layer.multimax_ranges.shape == (4,)
    assert model.ernie55_mtp_aliases and check_ernie55_mtp_aliases(model) == len(model.ernie55_mtp_aliases)


def test_forward_and_backward(pg_collection):
    model = _build(pg_collection, _tiny_config(dsa_indexer_loss_coeff=0.01, dsa_indexer_use_sparse_loss=True))
    _randomize_(model, seed=1)
    model.train()
    # SBHD input: Megatron-Core main has no reference (non-cuDNN) path for packed THD H layers.
    seq = 300
    tokens = torch.randint(1, VOCAB, (1, seq), generator=torch.Generator().manual_seed(2))
    loss_mask = torch.ones(1, seq)
    loss_mask[0, -1] = 0
    losses = model(
        input_ids=tokens,
        position_ids=torch.arange(seq).unsqueeze(0),
        attention_mask=None,
        labels=tokens.roll(-1, dims=1),
        loss_mask=loss_mask,
    )
    loss = (losses * loss_mask).sum() / loss_mask.sum()
    assert torch.isfinite(loss)
    loss.backward()
    missing = [name for name, param in model.named_parameters() if param.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(param.grad).all() for param in model.parameters())


def test_every_pipeline_stage_is_the_ernie_model(pg_collection, monkeypatch):
    monkeypatch.setattr(
        mcore_optimizer,
        "SEPARATE_GRAD_NORM_GROUPS",
        tuple(group for group in mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS if group != MULTIMAX_GRAD_NORM_GROUP),
    )
    # Megatron-Core runs mHC MTP only on the last stage, so a middle stage of this single-rank test has no MTP.
    config = _tiny_config(
        hybrid_layer_pattern="H-DEHEDE",
        mtp_num_layers=None,
        ernie_mtp_shared_last_layer=False,
        csa_compress_ratios=None,
    )
    model = _build(pg_collection, config, pre_process=False, post_process=False)
    assert type(model) is Ernie55HybridModel
    assert getattr(model, "output_layer", None) is None
    # Every stage registers the multimax group: the optimizer's per-group collectives need it on every rank.
    assert MULTIMAX_GRAD_NORM_GROUP in mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS


def test_build_distributed_models_keeps_the_sd_mtp_sharing(pg_collection):
    models = Ernie55ModelBuilder(_tiny_config()).build_distributed_models(
        pg_collection, wrap_with_ddp=False, mixed_precision_wrapper=None
    )
    assert len(models) == 1 and type(models[0]) is Ernie55HybridModel
    assert check_ernie55_mtp_aliases(models) == len(models[0].ernie55_mtp_aliases) > 0


@pytest.mark.parametrize("flag", ["use_megatron_fsdp", "use_torch_fsdp2"])
def test_sd_mtp_rejects_wrappers_that_split_parameters(pg_collection, flag):
    builder = Ernie55ModelBuilder(_tiny_config())
    with pytest.raises(ValueError, match=flag):
        builder.build_distributed_models(pg_collection, wrap_with_ddp=False, **{flag: True})


def test_sd_mtp_reads_megatron_fsdp_from_the_ddp_config(pg_collection):
    """Bridge's setup passes Megatron-FSDP through the DDP config, not as a keyword."""
    builder = Ernie55ModelBuilder(_tiny_config())
    with pytest.raises(ValueError, match="use_megatron_fsdp"):
        builder.build_distributed_models(pg_collection, SimpleNamespace(use_megatron_fsdp=True), wrap_with_ddp=False)


def test_peft_excluded_module_paths_cover_both_sides_of_every_alias(pg_collection):
    """The model names the SD-MTP body and the decoder layers it aliases, and nothing else."""
    model = _build(pg_collection)
    excluded = model.peft_excluded_module_paths()
    assert excluded
    for alias, source in model.ernie55_mtp_aliases.items():
        assert alias.rpartition(".")[0] in excluded
        assert source.rpartition(".")[0] in excluded
    for path in excluded:
        model.get_submodule(path)
    shared_layers = {int(path.split(".")[2]) for path in excluded if path.startswith("decoder.layers.")}
    num_layers = len(model.decoder.layers)
    assert shared_layers == set(range(num_layers - len(shared_layers), num_layers))
    assert len(shared_layers) < num_layers
    assert all(path.startswith(("mtp.", "decoder.layers.")) for path in excluded)


def test_lora_pre_wrap_hook_keeps_the_sd_mtp_sharing(pg_collection):
    """Bridge's LoRA skips the aliased modules, adapts the other targets, and the alias check still passes.

    The CPU stand-ins for the TE linears are plain modules LoRA does not know, so the targets are the routed
    experts' linears (Megatron-Core parallel linears); the MTP body's experts are aliased and must stay untouched.
    Bridge's LoRA layers import Transformer Engine, so this test (and only this test) needs it.
    """
    pytest.importorskip("transformer_engine.pytorch", reason="Bridge's LoRA layers import Transformer Engine")
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.peft.lora_layers import LoRALinear

    lora = LoRA(
        target_modules=["*.experts.local_experts.*.linear_fc1", "*.experts.local_experts.*.linear_fc2"],
        dim=4,
        alpha=8,
        dropout=0.0,
    )
    config = _tiny_config()
    config.pre_wrap_hooks.append(lambda chunks: lora(chunks, training=True))
    models = Ernie55ModelBuilder(config).build_distributed_models(
        pg_collection, wrap_with_ddp=False, mixed_precision_wrapper=None
    )
    model = models[0]
    assert check_ernie55_mtp_aliases(models) == len(model.ernie55_mtp_aliases) > 0
    adapted = {name for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    excluded = model.peft_excluded_module_paths()
    assert adapted and not adapted & excluded
    assert not any(name.startswith("mtp.") for name in adapted)
    assert any(name.startswith("decoder.layers.") for name in adapted)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(".adapter." in name for name in trainable)
    assert not any(name.rpartition(".")[0].rpartition(".adapter")[0] in excluded for name in trainable)


class _ModuleWrapper(torch.nn.Module):
    """An adapter-like wrapper that moves the base module to ``to_wrap``."""

    def __init__(self, module):
        super().__init__()
        self.to_wrap = module


def test_an_adapter_on_an_aliased_module_is_reported(pg_collection):
    """A PEFT method that ignores the exclusions is caught with a message that names them."""
    model = _build(pg_collection)
    owner_name = sorted(model.ernie55_mtp_aliases)[0].rpartition(".")[0]
    parent_name, _, child = owner_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, _ModuleWrapper(getattr(parent, child)))
    with pytest.raises(RuntimeError, match="peft_excluded_module_paths"):
        check_ernie55_mtp_aliases(model)


def test_wrapping_that_splits_an_alias_is_caught(pg_collection, monkeypatch):
    """A wrapper that re-creates an MTP parameter (FSDP-like) fails the build instead of training two copies."""
    build_wrapped = BridgeHybridModelBuilder.build_distributed_models

    def split_one_alias(self, *args, **kwargs):
        models = build_wrapped(self, *args, **kwargs)
        model = models[0]
        owner_name, _, leaf = sorted(model.ernie55_mtp_aliases)[0].rpartition(".")
        owner = model.get_submodule(owner_name)
        owner._parameters[leaf] = torch.nn.Parameter(owner._parameters[leaf].detach().clone())
        return models

    monkeypatch.setattr(BridgeHybridModelBuilder, "build_distributed_models", split_one_alias)
    with pytest.raises(RuntimeError, match="no longer share"):
        Ernie55ModelBuilder(_tiny_config()).build_distributed_models(
            pg_collection, wrap_with_ddp=False, mixed_precision_wrapper=None
        )


def test_rejects_a_non_ernie_stack_spec(pg_collection):
    config = _tiny_config()
    config.hybrid_stack_spec = mcore_hybrid_stack_spec
    with pytest.raises(TypeError, match="Ernie55HybridStack"):
        Ernie55ModelBuilder(config).build_model(pg_collection)


def test_rejects_a_padded_vocabulary(pg_collection):
    config = _tiny_config(should_pad_vocab=True, make_vocab_size_divisible_by=128)
    with pytest.raises(ValueError, match="must not pad"):
        Ernie55ModelBuilder(config).build_model(pg_collection)


def test_default_spec_is_the_ernie_spec():
    config = _tiny_config(hybrid_stack_spec=None)
    spec = Ernie55ModelBuilder(config)._resolve_hybrid_stack_spec()
    assert spec.module is Ernie55HybridStack


def test_builder_needs_the_ernie_config():
    with pytest.raises(TypeError, match="Ernie55ModelConfig"):
        Ernie55ModelBuilder(object())
