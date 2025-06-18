import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import LlamaForCausalLM

from megatron.hub.bridge import MegatronModelBridge
from megatron.hub.bridge.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.hub.bridge.state_bridge import MegatronStateBridge
from megatron.hub.bridge.weight_bridge import (
    GatedMLPWeightBridge,
    QKVWeightBridge,
    TPAwareWeightBridge,
)
from megatron.hub.models.llama.llama_provider import LlamaModelProvider


@MegatronModelBridge.impl(source=LlamaForCausalLM, target=GPTModel)
class MegatronCausalLlamaBridge(MegatronModelBridge):
    """
    Megatron-Hub Bridge for Llama Causal LM.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> LlamaModelProvider:
        hf_config = hf_pretrained.config

        # if getattr(hf_config, 'rope_scaling', None) is not None and hf_config.rope_scaling.get('rope_type') == 'llama3':
        #     # Apply Llama3.1 customize rope scaling
        #     cls = partial(Llama31Config, scale_factor=source.rope_scaling.get("factor", 8.0))
        # else:
        #     cls = LlamaConfig

        provider = LlamaModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            num_query_groups=hf_config.num_key_value_heads,
            seq_length=hf_config.max_position_embeddings,
            rotary_base=hf_config.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            fp16=(self.dtype_from_hf(hf_config) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config),
            generation_config=hf_pretrained.generation_config,
            vocab_size=hf_config.vocab_size,
        )

        provider.gradient_accumulation_fusion = False
        provider.variable_seq_lengths = True

        return provider

    def state_bridge(self) -> MegatronStateBridge:
        return MegatronStateBridge(
            # ------------------------------------------------------------------
            # Embedding & output projection – column-parallel
            # ------------------------------------------------------------------
            TPAwareWeightBridge(
                megatron="embedding.word_embeddings.weight",
                to="model.embed_tokens.weight",
            ),
            TPAwareWeightBridge(
                megatron="output_layer.weight",
                to="lm_head.weight",
            ),
            # ------------------------------------------------------------------
            # LayerNorm (replicated across TP ranks)
            # ------------------------------------------------------------------
            TPAwareWeightBridge(
                megatron="decoder.final_layernorm.weight",
                to="model.norm.weight",
            ),
            TPAwareWeightBridge(
                megatron="decoder.layers.*.input_layernorm.weight",
                to="model.layers.*.input_layernorm.weight",
            ),
            TPAwareWeightBridge(
                megatron="decoder.layers.*.pre_mlp_layernorm.weight",
                to="model.layers.*.post_attention_layernorm.weight",
            ),
            # ------------------------------------------------------------------
            # Attention – fused QKV & output projection
            # ------------------------------------------------------------------
            QKVWeightBridge(
                megatron="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            TPAwareWeightBridge(
                megatron="decoder.layers.*.self_attention.linear_proj.weight",
                to="model.layers.*.self_attn.o_proj.weight",
            ),
            # ------------------------------------------------------------------
            # MLP – gated projection & output projection
            # ------------------------------------------------------------------
            GatedMLPWeightBridge(
                megatron="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            TPAwareWeightBridge(
                megatron="decoder.layers.*.mlp.linear_fc2.weight",
                to="model.layers.*.mlp.down_proj.weight",
            ),
        )
