from torch.distributed.elastic.multiprocessing.errors import record

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.recipes.llama.llama3 import llama32_1b_pretrain_config as pretrain_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


@record
def main():
    cfg = pretrain_config()

    # Load pre-converted checkpoint
    cfg.checkpoint.pretrained_checkpoint = "/tmp/checkpoints/llama_32_1b"

    # Configure tokenizer
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = "meta-llama/Llama-3.2-1B"

    # FinetuningDatasetConfig
    cfg.dataset = HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=8192,
        dataloader_type="batch",
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=8192,
            tokenizer_model_name="meta-llama/Llama-3.2-1B",
        ),
        dataset_kwargs={"pad_to_max_length": True},
    )
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False
    cfg.model.context_parallel_size = 2

    # Short test run (10 iterations for demo
    cfg.train.train_iters = 10
    cfg.scheduler.lr_warmup_iters = 1
    cfg.train.global_batch_size = 8
    cfg.logger.log_interval = 1
    cfg.train.eval_iters = 0
    cfg.checkpoint.save = None

    # BF16 is used by default (no mixed_precision setting needed)
    print(cfg)

    finetune(cfg, forward_step)


if __name__ == "__main__":
    main()
