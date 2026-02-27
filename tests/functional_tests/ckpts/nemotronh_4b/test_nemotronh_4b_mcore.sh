LOAD_DIR=/workspace/test_ckpts/nemotronh_4b_mbridge
SAVE_DIR=/workspace/test_ckpts/nemotronh_4b_mcore

CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 /opt/Megatron-Bridge/3rdparty/Megatron-LM/pretrain_mamba.py \
    --init-method-std 0.014 \
    --disable-bias-linear \
    --use-rope-scaling \
    --squared-relu \
    --qk-layernorm \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --use-rotary-position-embeddings \
    --hybrid-override-pattern "M-M-M-M*-M-M-M-M*-M-M-M-M*" \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --num-layers 26 \
    --hidden-size 3072 \
    --num-attention-heads 32 \
    --mamba-num-heads 112 \
    --ffn-hidden-size 12288 \
    --kv-channels 128 \
    --group-query-attention \
    --position-embedding-type none \
    --attention-backend fused \
    --num-query-groups 8 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --train-iters 10 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 151936 \
    --save-interval 5 \
    --eval-interval 5 \
    --eval-iters 4 \
    --load ${LOAD_DIR} \
    --save ${SAVE_DIR} \
    --ckpt-format torch_dist \
    --log-progress \
    --bf16 \
    --lr 4.5e-4 \
    --min-lr 4.5e-5 \
    --num-workers 2 \
    --tensorboard-dir /workspace/tb \
    --log-interval 1 \
    --log-throughput \
    --no-load-optim \
    --no-load-rng

echo rm -rf ${LOAD_DIR}
echo rm -rf ${SAVE_DIR}
