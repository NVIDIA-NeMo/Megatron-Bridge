from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.recipes.llama.llama3 import llama31_8b_finetune_config as finetune_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


if __name__ == "__main__":
    cfg = finetune_config(peft=None)

    # Configure tokenizer
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = "meta-llama/Llama-3.1-8B"

    # FinetuningDatasetConfig
    cfg.dataset = HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=2048,
        dataloader_type="cyclic",
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=8192,
            tokenizer_model_name="meta-llama/Llama-3.1-8B",
        ),
        num_workers=0,
    )
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False
    cfg.model.num_layers = 2
    cfg.model.context_parallel_size = 2
    cfg.model.tensor_model_parallel_size = 1

    # Short test run (10 iterations for demo
    cfg.train.global_batch_size = 1
    cfg.train.micro_batch_size = 1
    cfg.logger.log_interval = 1
    cfg.train.eval_iters = 0
    cfg.checkpoint.save = None

    # BF16 is used by default (no mixed_precision setting needed)
    print(cfg)

    finetune(cfg, forward_step)
