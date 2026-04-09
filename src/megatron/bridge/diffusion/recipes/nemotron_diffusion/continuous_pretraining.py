from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer, TokenizerConfig


def _nemotron_diffusion_cpt_config(
    hf_path,
    tensor_model_parallel_size,
    micro_batch_size,
    tokenizer_model,
    data_paths=None,
    data_args_path=None,
    peft=None,
) -> ConfigContainer:
    cfg = _pretrain_common()

    # Model configuration
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=True, hf_path=hf_path)
    cfg.model.freeze_vision_model = True
    cfg.model.seq_length = 4096

    # Parallel settings
    cfg.model.tensor_model_parallel_size = tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Training config
    cfg.train.train_iters = 500000
    cfg.train.eval_interval = 5000
    cfg.train.eval_iters = 10
    cfg.train.save_interval = 5000
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Optimizer with WSD scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=0,
        lr_decay_iters=None,
        max_lr=1e-5,
        min_lr=1e-6,
    )
    scheduler_cfg.lr_decay_style = "WSD"
    scheduler_cfg.lr_warmup_fraction = 0.01
    scheduler_cfg.lr_warmup_iters = 0
    scheduler_cfg.lr_wsd_decay_iters = 100000
    cfg.optimizer = opt_cfg
    cfg.optimizer.adam_beta2 = 0.95
    cfg.scheduler = scheduler_cfg

    # Dataset configuration
    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths=data_paths,
        data_args_path=data_args_path,
    )
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = blend
    cfg.dataset.blend_per_split = blend_per_split
    cfg.dataset.split = "950,50,0"
    cfg.dataset.num_workers = 10
    cfg.dataset.dataloader_type = "cyclic"
    cfg.dataset.mmap_bin_files = False

    # DDP settings
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Distributed timeout
    cfg.dist.distributed_timeout_minutes = 240

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Tokenizer
    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=tokenizer_model,
    )

    # PEFT (optional, None for full CPT)
    cfg.peft = peft

    return cfg


def nemotron_diffusion_3b_finetune_config(
    data_paths=None,
    data_args_path=None,
    hf_path=None,
    peft=None,
) -> ConfigContainer:
    """Return a CPT config for Nemotron-Diffusion 3B. Default: TP=1, MBS=1."""
    return _nemotron_diffusion_cpt_config(
        hf_path=hf_path or "mistralai/Ministral-3-3B-Base-2512",
        tensor_model_parallel_size=1,
        micro_batch_size=1,
        tokenizer_model="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/yongganf/miscs/models/Nemotron-H-8B-Base-8K",
        data_paths=data_paths,
        data_args_path=data_args_path,
        peft=peft,
    )


def nemotron_diffusion_8b_finetune_config(
    data_paths=None,
    data_args_path=None,
    hf_path=None,
    peft=None,
) -> ConfigContainer:
    """Return a CPT config for Nemotron-Diffusion 8B. Default: TP=4, MBS=1."""
    return _nemotron_diffusion_cpt_config(
        hf_path=hf_path or "mistralai/Ministral-3-8B-Base-2512",
        tensor_model_parallel_size=4,
        micro_batch_size=1,
        tokenizer_model="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/yongganf/miscs/models/Nemotron-H-8B-Base-8K",
        data_paths=data_paths,
        data_args_path=data_args_path,
        peft=peft,
    )


def nemotron_diffusion_14b_finetune_config(
    data_paths=None,
    data_args_path=None,
    hf_path=None,
    peft=None,
) -> ConfigContainer:
    """Return a CPT config for Nemotron-Diffusion 14B. Default: TP=8, MBS=1."""
    return _nemotron_diffusion_cpt_config(
        hf_path=hf_path or "mistralai/Ministral-3-14B-Base-2512",
        tensor_model_parallel_size=8,
        micro_batch_size=1,
        tokenizer_model="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/yongganf/miscs/models/Nemotron-H-8B-Base-8K",
        data_paths=data_paths,
        data_args_path=data_args_path,
        peft=peft,
    )
