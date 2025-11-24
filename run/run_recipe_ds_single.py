from megatron.bridge.recipes.deepseek import deepseek_v3_pretrain_config
from megatron.bridge.training.gpt_step import forward_step  # For text-only models
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.comm_overlap import CommOverlapConfig
import torch
from megatron.core.transformer.enums import AttnBackend

def main() -> None:

    cfg = deepseek_v3_pretrain_config(
        # Data
        data_paths=["/lustre/fsw/coreai_dlalgo_llm/datasets/RedPajama2/kenlm_perp_head_gopher_linefilter_decompressed/bin_idx/nemo/head_01_text_document"],
        global_batch_size=16,
        micro_batch_size=1,
        seq_length=4096,
        # Const LR
        lr=3e-4,
        min_lr=3e-4,
        lr_warmup_iters=0,
        # Parallelism
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        context_parallel_size=2,
        sequence_parallel=True,
        # Training parameters
        train_iters=50,
        eval_interval=50,
        mtp_num_layers=0,
        recompute_granularity=None,
        recompute_modules=None,
        # Log
        wandb_project="nemo2_mbridge_comparison",
        wandb_exp_name="dsv3_2layers_mbridge_tp2pp2ep2cp2",
        wandb_entity="joc",
    )

    cfg.checkpoint.pretrained_checkpoint = '/aot/checkpoints/dsv3/mbridge-2l'
    cfg.checkpoint.load_optim = False
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.num_layers = 2
    cfg.model.moe_layer_freq = [0, 1]
    cfg.model.num_moe_experts = 16
    cfg.logger.log_interval = 1

    # DDP settings - must be set via mixed_precision and comm_overlap configs
    # These will be applied to cfg.ddp during runtime_config_update()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig()
    
    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_param_gather = True
    cfg.comm_overlap.delay_wgrad_compute = False
    
    # Deterministic mode
    cfg.model.deterministic_mode = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.attention_backend = AttnBackend.auto
    import os
    os.environ["WANDB_API_KEY"] = "f37880d4fc7a812145caab826f6fd1bf2dbd169c"
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    pretrain(config=cfg, forward_step_func=forward_step)

if __name__ == "__main__":
    main()
