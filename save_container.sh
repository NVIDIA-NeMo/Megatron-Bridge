#!/bin/bash

#SBATCH -A coreai_dlalgo_llm 
#SBATCH -p batch
#SBATCH -J coreai_dlalgo_llm-deterministic.save_container
#SBATCH -t 1:00:00 
#SBATCH -N 1 
#SBATCH --mem=0 
#SBATCH --dependency=singleton 
ACCOUNT="coreai_dlalgo_llm"

WORKDIR=$(pwd)

MOUNTS="\
/lustre:/lustre,$WORKDIR:/opt/Megatron-Bridge,$WORKDIR/3rdparty/Megatron-LM:/opt/megatron-lm"

ORIGINAL_CONTAINER_NAME="nvcr.io#nvidia/nemo:25.11"
# With this weekend's build, [9.18.0.45], cudnn has deterministic f16 MLA support.
# NEW_CONTAINER_NAME="/lustre/fsw/coreai_dlalgo_llm/zhiyul/containers/nemo-25.11-cudnn9.18.0.45.sqsh"
NEW_CONTAINER_NAME="/lustre/fsw/coreai_dlalgo_llm/zhiyul/containers/nemo-25.11-cudnn9.18.0.76-new.sqsh"


srun -N1 \
 -n1 \
 -A ${ACCOUNT} \
 -J coreai_dlalgo_llm-deterministic.save_container \
 -t 1:00:00 \
 -p batch \
 --export=ALL,HOME=/tmp \
 --no-container-mount-home \
 --container-mounts ${MOUNTS} \
 --container-image=${ORIGINAL_CONTAINER_NAME} \
 --container-writable \
 --container-save=${NEW_CONTAINER_NAME} \
 --pty bash
#  --pty bash -c "echo 'success'"
#  --pty bash -c "pip install -q catalogue ninja build && \
#     pip install -e /opt/megatron-lm --no-build-isolation && \
#     pip install -e /opt/Megatron-Bridge[recipes] --no-build-isolation && \
#  echo 'success'"
#  --pty bash -c "echo 'success'"
# --pty bash -c "echo 'success'"
