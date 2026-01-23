# Finetune
torchrun --nproc_per_node=8 finetune_gemma3_vl.py \
    --recipe gemma3_vl_4b_finetune_config \
    --pretrained-checkpoint /models/gemma-3-4b-it \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=2 \
    model.context_parallel_size=1

# Lora
torchrun --nproc_per_node=8 finetune_gemma3_vl.py \
    --recipe gemma3_vl_4b_finetune_config \
    --pretrained-checkpoint /models/gemma-3-4b-it \
    --peft_scheme lora \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=2 \
    model.context_parallel_size=1
