from __future__ import annotations

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.utils import is_torch_min_version
from transformers import ApertusForCausalLM
from transformers.activations import XIELUActivation

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.utils import unwrap_model

_ROPE_DEFAULTS = {
    "rope_type": "llama3",
    "original_max_position_embeddings": 8192,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
}


class MCoreXIELU(XIELUActivation):
    def __init__(self, *, config):
        super().__init__(dtype=config.params_dtype, with_vector_loads=False)
        if self._xielu_cuda_obj is None:
            raise RuntimeError("CUDA xIELU is required. Install rubber-duck-debug/xielu.")
        self.to(
            device=torch.device("cpu")
            if getattr(config, "use_cpu_initialization", False)
            else torch.device("cuda", torch.cuda.current_device())
        )
        self.alpha_p.sum_gradients_across_tp_domain = True
        self.alpha_n.sum_gradients_across_tp_domain = True
        self._sync_runtime_scalars()
        self.register_load_state_dict_post_hook(lambda m, _: m._sync_runtime_scalars())

    @torch.no_grad()
    def _sync_runtime_scalars(self):
        """Refresh CUDA-kernel host caches after beta/eps buffers change."""
        if getattr(self.beta, "is_meta", False) or getattr(self.eps, "is_meta", False):
            return
        self._beta_scalar = float(self.beta.detach().cpu().float().item())
        self._eps_scalar = float(self.eps.detach().cpu().float().item())


def get_apertus_decoder_block_spec(config, vp_stage=None, pp_rank=None):
    assert is_torch_min_version("2.4.0a0"), "Torch RMSNorm requires PyTorch >= 2.4"
    block_spec = get_gpt_decoder_block_spec(
        config, use_transformer_engine=True, normalization="RMSNorm", vp_stage=vp_stage, pp_rank=pp_rank
    )
    for layer in block_spec.layer_specs:
        attn = layer.submodules.self_attention.submodules
        attn.q_layernorm = attn.k_layernorm = (lambda config, hidden_size, eps=1e-5, **_: torch.nn.RMSNorm(hidden_size, eps=eps))
        get_submodules(layer.submodules.mlp).activation_func = MCoreXIELU
    return block_spec


@MegatronModelBridge.register_bridge(source=ApertusForCausalLM, target=GPTModel, model_type="apertus")
class ApertusBridge(MegatronModelBridge):
    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        if hidden_act != "xielu":
            return super().hf_to_megatron_activation(hidden_act)
        return lambda _: (_ for _ in ()).throw(RuntimeError("expected MCoreXIELU"))

    def provider_bridge(self, hf_pretrained):
        provider = super().provider_bridge(hf_pretrained)
        config = hf_pretrained.config
        rope = {**(getattr(config, "rope_scaling", None) or {}), **(getattr(config, "rope_parameters", None) or {})}
        rope_type = rope.get("rope_type", rope.get("type", "llama3"))
        factor = float(rope.get("factor", 1.0))
        theta = float(rope.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
        if config.hidden_act != "xielu":
            raise ValueError(f"Expected hidden_act='xielu', got {config.hidden_act!r}")
        if config.attention_bias:
            raise ValueError("Apertus attention_bias=True is unsupported")
        if rope_type != "llama3":
            raise ValueError(f"Unsupported Apertus RoPE type: {rope_type!r}")

        provider.apertus_rope_scaling = {
            "rope_type": rope_type,
            "type": rope_type,
            "factor": factor,
            "original_max_position_embeddings": int(rope.get("original_max_position_embeddings", 8192)),
            "low_freq_factor": float(rope.get("low_freq_factor", 1.0)),
            "high_freq_factor": float(rope.get("high_freq_factor", 4.0)),
        }
        provider.normalization = "RMSNorm"
        provider.qk_layernorm = True
        provider.gated_linear_unit = False
        provider.use_te_activation_func = False
        provider.bias_activation_fusion = False
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.rotary_interleaved = False
        provider.position_embedding_type = "rope"
        provider.rotary_base = theta
        provider.rope_scaling = True
        provider.rope_scaling_factor = factor
        provider.transformer_layer_spec = get_apertus_decoder_block_spec
        return provider

    def load_weights_hf_to_megatron(self, hf_pretrained, megatron_model, allowed_mismatched_params=None):
        models = super().load_weights_hf_to_megatron(
            hf_pretrained, megatron_model, allowed_mismatched_params=allowed_mismatched_params
        )
        [
            m._sync_runtime_scalars()
            for model in unwrap_model(models)
            for m in model.modules()
            if isinstance(m, MCoreXIELU)
        ]
        return models

    @classmethod
    def megatron_to_hf_config(cls, provider):
        config = super().megatron_to_hf_config(provider)
        theta = float(provider.rotary_base)
        rope = {
            **(getattr(provider, "apertus_rope_scaling", None) or _ROPE_DEFAULTS),
            "factor": float(provider.rope_scaling_factor),
        }
        rope["rope_type"] = rope["type"] = rope.get("rope_type", rope.get("type", "llama3"))
        config.update(
            hidden_act="xielu",
            attention_bias=False,
            rope_theta=theta,
            rope_scaling=rope,
            rope_parameters={**rope, "rope_theta": theta},
        )
        return config

    def mapping_registry(self) -> MegatronMappingRegistry:
        L, H = "decoder.layers.*", "model.layers.*"
        auto = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            f"{L}.self_attention.linear_proj.weight": f"{H}.self_attn.o_proj.weight",
            f"{L}.self_attention.q_layernorm.weight": f"{H}.self_attn.q_norm.weight",
            f"{L}.self_attention.k_layernorm.weight": f"{H}.self_attn.k_norm.weight",
            f"{L}.mlp.linear_fc2.weight": f"{H}.mlp.down_proj.weight",
        }
        repl = {
            f"{L}.self_attention.linear_qkv.layer_norm_weight": f"{H}.attention_layernorm.weight",
            f"{L}.mlp.linear_fc1.layer_norm_weight": f"{H}.feedforward_layernorm.weight",
            **{f"{L}.mlp.activation_func.{n}": f"{H}.mlp.act_fn.{n}" for n in ("alpha_p", "alpha_n", "beta", "eps")},
        }
        qkv = QKVMapping(
            f"{L}.self_attention.linear_qkv.weight",
            q=f"{H}.self_attn.q_proj.weight",
            k=f"{H}.self_attn.k_proj.weight",
            v=f"{H}.self_attn.v_proj.weight",
        )
        qkv._tp_mapping = ColumnParallelMapping(qkv.megatron_param, qkv.megatron_param)
        return MegatronMappingRegistry(
            *(AutoMapping(m, h) for m, h in auto.items()),
            *(ReplicatedMapping(m, h) for m, h in repl.items()),
            qkv,
            ColumnParallelMapping(f"{L}.mlp.linear_fc1.weight", f"{H}.mlp.up_proj.weight"),
        )
