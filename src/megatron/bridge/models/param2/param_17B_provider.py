from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


def _param2_transformer_layer_spec(
    config: "Param2ModelProvider",
    vp_stage: Optional[int] = None,
):
    """
    Build a hybrid GPT decoder block spec for Param2:
      - layer 0  : dense
      - layers 1+: MoE

    Force TransformerEngine spec so RMSNorm is instantiated via TE/TENorm
    rather than legacy FusedLayerNorm.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    return get_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=True,
        normalization=config.normalization,
        qk_l2_norm=getattr(config, "qk_l2_norm", False),
        vp_stage=vp_stage,
    )


@dataclass
class Param2ModelProvider(GPTModelProvider):
    """Provider for BharatGen Param2-17B-A2.4B-Thinking."""

    # Core architecture
    num_layers: int = 21
    hidden_size: int = 2048
    ffn_hidden_size: int = 9216
    num_attention_heads: int = 32
    num_query_groups: int = 8
    kv_channels: Optional[int] = 64

    # Token / sequence
    vocab_size: int = 128008
    seq_length: int = 4096
    max_position_embeddings: int = 4096
    make_vocab_size_divisible_by: int = 8

    # Attention / norm / activation
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    qk_l2_norm: bool = False
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 1_000_000.0
    position_embedding_type: str = "rope"

    # Dropout / init
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    init_method_std: float = 0.02

    # Dtype
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True
    fp16: bool = False

    # Embedding tying
    share_embeddings_and_output_weights: bool = True

    # MoE core
    num_moe_experts: int = 64
    moe_ffn_hidden_size: int = 2048
    moe_router_topk: int = 6
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True

    # Hybrid layout: layer 0 dense, others MoE
    moe_layer_freq: list[int] = field(default_factory=lambda: [0] + [1] * 20)

    # Router behavior
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_pre_softmax: bool = False
    moe_router_score_function: str = "sigmoid"
    moe_router_dtype: str = "fp32"
    moe_router_topk_scaling_factor: float = 2.5
    moe_router_enable_expert_bias: bool = True
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1

    # Shared experts
    moe_shared_expert_intermediate_size: int = 4096
    moe_shared_expert_gate: bool = False
    moe_shared_expert_overlap: bool = False

    # Force TE full layer spec for RMSNorm support
    transformer_layer_spec: Callable = _param2_transformer_layer_spec
    use_transformer_engine_full_layer_spec: bool = True

    # Disable alternate spec paths
    use_kitchen: bool = False
    use_kitchen_attention: bool = False
    kitchen_attention_backend: Optional[str] = None
    use_te_activation_func: bool = False
    multi_latent_attention: bool = False

    # HF metadata / round-trip
    model_type: str = "param2moe"
    architectures: tuple[str, ...] = ("Param2MoEForCausalLM",)
    hidden_act: str = "silu"
    use_rmsnorm: bool = True
    use_qk_norm: bool = True
    use_qkv_bias: bool = False
    use_bias: bool = False
    embedding_dropout: float = 0.0
    output_dropout: float = 0.0
    output_router_logits: bool = False
    pad_token_id: int = 0
    eos_token_id: int = 3
    partial_rotary_factor: float = 1.0
    rope_scaling: dict | None = None
    max_window_layers: int = 20
    num_nextn_predict_layers: int = 0
    mtp_loss_scaling_factor: float = 0.0
    router_dtype: str = "fp32"
    score_function: str = "sigmoid"
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1

    # Bridge convenience fields
    num_shared_experts: int = 2
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True

    def __post_init__(self):
        super().__post_init__()

        expected = [0] + [1] * (self.num_layers - 1)
        if self.moe_layer_freq is None or len(self.moe_layer_freq) != self.num_layers:
            self.moe_layer_freq = expected

        if self.max_position_embeddings is None:
            self.max_position_embeddings = self.seq_length
        if self.seq_length is None:
            self.seq_length = self.max_position_embeddings
