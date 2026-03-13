#!/bin/bash

torchrun --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path moonshotai/Kimi-K2.5 \
    --trust_remote_code \
    --image_path https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
    --prompt "Describe this image." \
    --tp 1 \
    --ep 4 \
    --pp 1 \
    --num_layers 4
