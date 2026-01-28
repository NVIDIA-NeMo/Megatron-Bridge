from megatron.bridge.recipes.deepseek import deepseek_v3_pretrain_config
from megatron.bridge.training.gpt_step import forward_step  # For text-only models
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.comm_overlap import CommOverlapConfig
import torch
from megatron.core.transformer.enums import AttnBackend
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run deterministic recipe for toy dataset")
    parser.add_argument("--data-paths", type=str, nargs="+", required=True, help="Paths to data")
    parser.add_argument("--pretrained-checkpoint", type=str, required=True, help="Path to pretrained checkpoint")

    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=2)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=2)
    parser.add_argument("--expert-model-parallel-size", type=int, default=2)
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)

    parser.add_argument("--wandb-api-key", type=str, required=True, help="Wandb API key")
    parser.add_argument("--wandb-project", type=str, default="nemo2_mbridge_comparison_new")
    parser.add_argument("--wandb-exp-name", type=str, default="dsv3_2layers_mbridge_final_bs16_tp2pp2ep2cp2sp_100steps")
    parser.add_argument("--wandb-entity", type=str, default="ao-tang96-none")

    return parser.parse_args()

def main() -> None:

    args = parse_args()
    # Initialize the default config for the recipe
    cfg = deepseek_v3_pretrain_config(
        # Data
        data_paths=args.data_paths,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        # Const LR
        lr=3e-4,
        min_lr=3e-4,
        lr_warmup_iters=0,
        # Parallelism
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        pipeline_dtype=torch.bfloat16,
        context_parallel_size=1, # CP fixed to 1 since unfused attention does not apply CP
        sequence_parallel=True,
        # Training parameters
        train_iters=args.train_iters,
        eval_interval=args.eval_interval,
        mtp_num_layers=0,
        recompute_granularity=None,
        recompute_modules=None,
        # Log
        wandb_project=args.wandb_project,
        wandb_exp_name=args.wandb_exp_name,
        wandb_entity=args.wandb_entity,
    )

    # Load from checkpoint
    cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    cfg.checkpoint.load_optim = False

    # Model configuration relating to the toy deepseek v3 model
    # here we use the 2-layer toy model with reduced hidden size and experts
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.num_layers = 2
    cfg.model.moe_layer_freq = [0, 1]
    cfg.model.num_moe_experts = 16
    cfg.model.hidden_size = 7168 // 2 # 3584
    cfg.model.ffn_hidden_size = 18432 // 2 # 921
    cfg.model.gradient_accumulation_fusion = False
    cfg.model.moe_aux_loss_coeff = 1e-4

    # DDP settings - must be set via mixed_precision and comm_overlap configs
    # These will be applied to cfg.ddp during runtime_config_update()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig()
    
    cfg.comm_overlap.overlap_grad_reduce = False
    cfg.comm_overlap.overlap_param_gather = False
    cfg.comm_overlap.delay_wgrad_compute = False
    
    # Scheduler and optimizer settings
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.optimizer.weight_decay = 0.1
    cfg.mixed_precision.fp8_recipe = None

    # Deterministic mode
    cfg.logger.log_interval = 1
    cfg.model.deterministic_mode = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.attention_backend = AttnBackend.unfused
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    pretrain(config=cfg, forward_step_func=forward_step)

if __name__ == "__main__":
    main()

