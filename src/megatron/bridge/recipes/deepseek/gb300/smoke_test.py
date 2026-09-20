import torch
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec as _get_exp_attn_spec,
)


def _minimal_pro_provider() -> MLAModelProvider:
    provider = MLAModelProvider(
        # ---- 基础架构 ----
        num_layers=7,
        hidden_size=2048, 
        ffn_hidden_size=2048,
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=512,
        vocab_size=129280,
        seq_length=512,  
        layernorm_epsilon=1e-6,
        init_method_std=0.02,
        # ---- MLA ----
        q_lora_rank=1024,
        kv_lora_rank=512,  
        qk_head_dim=448, 
        qk_pos_emb_head_dim=64,
        v_head_dim=512,
        # ---- MoE ----
        num_moe_experts=8,
        moe_router_topk=6,
        moe_ffn_hidden_size=2048,
        moe_router_score_function="sqrtsoftplus",
        moe_router_topk_scaling_factor=1.5,
        moe_aux_loss_coeff=0.0,
        mtp_num_layers=None, 
    )

    # ---- Attention ----
    provider.experimental_attention_variant = "dsv4_hybrid"
    provider.multi_latent_attention = True
    provider.transformer_layer_spec = _get_exp_attn_spec  # 换成 DSv4 实验 attention spec
    provider.qk_layernorm = True
    provider.normalization = "RMSNorm"
    provider.add_bias_linear = False
    provider.rotary_percent = 1.0

    if hasattr(provider, "output_projection_groups"):
        provider.output_projection_groups = 8
        provider.output_projection_lora_rank = 1024
    else:
        provider.o_groups = 8
        provider.o_lora_rank = 1024

    # ---- RoPE ----
    provider.apply_rope_fusion = True
    provider.rope_type = "yarn"
    provider.rotary_base = 10000.0
    provider.csa_compress_rotary_base = 160000.0
    provider.rotary_scaling_factor = 16.0
    provider.original_max_position_embeddings = 65536
    provider.beta_fast = 32.0
    provider.beta_slow = 1.0
    provider.mscale = 1.0
    provider.mscale_all_dim = 1.0

    # ---- CSA ----
    provider.csa_compress_ratios = [0,0,0,4,128,4,128]
    provider.csa_window_size = 128
    provider.dsa_indexer_n_heads = 64
    provider.dsa_indexer_head_dim = 128
    provider.dsa_indexer_topk = 512
    provider.dsa_kernel_backend = "none" 
    if hasattr(provider, "apply_dsa_kernel_fusion"):
        provider.apply_dsa_kernel_fusion = False

    # ---- Hyper-Connections ----
    if hasattr(provider, "enable_hyper_connections"):
        provider.enable_hyper_connections = True
        provider.num_residual_streams = 4
    else:
        provider.enable_mhc_connections = True
        provider.mhc_num_residual_streams = 4
    provider.use_fused_mhc = False
    provider.mhc_sinkhorn_iterations = 20

    # ---- MoE ----
    provider.gated_linear_unit = True
    provider.moe_grouped_gemm = True
    provider.moe_router_pre_softmax = False
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "noaux_tc"
    provider.moe_shared_expert_overlap = True
    provider.moe_router_enable_expert_bias = True
    provider.moe_router_dtype = "fp32"
    provider.moe_permute_fusion = True
    provider.norm_topk_prob = True
    provider.moe_n_hash_layers = 0
    provider.actual_vocab_size = 129280
    provider.activation_func_clamp_value = 10.0
    provider.moe_layer_freq = [1] * 7
    provider.moe_shared_expert_intermediate_size = 2048

    # ---- Others ----
    provider.share_embeddings_and_output_weights = False
    provider.gradient_accumulation_fusion = False
    provider.bias_dropout_fusion = True
    provider.cross_entropy_fusion_impl = "te"
    provider.cross_entropy_loss_fusion = True
    provider.masked_softmax_fusion = True
    provider.persist_layer_norm = True
    provider.hidden_dropout = 0.0
    provider.attention_softmax_in_fp32 = False
    provider.make_vocab_size_divisible_by = 1280

    return provider

def deepseek_v4_pro_smoketest_1gpu_config() -> ConfigContainer:
    cfg = _pretrain_common()
    cfg.model = _minimal_pro_provider()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 512
    cfg.model.params_dtype = torch.bfloat16

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = False
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.cuda_graph_impl = "none"

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.tokenizer.make_vocab_size_divisible_by = cfg.model.make_vocab_size_divisible_by
    cfg.tokenizer.tensor_model_parallel_size = 1
    cfg.tokenizer.rank = 0

    cfg.dataset.blend = None
    cfg.dataset.seq_length = 512
    cfg.dataset.num_workers = 0
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.dataloader_type = "single"

    cfg.train.train_iters = 1
    cfg.train.global_batch_size = 1
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False

    cfg.validation.eval_interval = 0
    cfg.checkpoint.save_interval = 0
    cfg.logger.log_interval = 1


    cfg.dist.enable_megatron_core_experimental = True
    cfg.ddp.use_megatron_fsdp = False
    return cfg

if __name__ == "__main__":
    cfg = deepseek_v4_pro_smoketest_1gpu_config()
    pretrain(cfg, forward_step)