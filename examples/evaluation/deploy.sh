# Unset SLURM/PMI/PMIX env vars to prevent MPI initialization issues
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

MEGATRON_CHECKPOINT=$1
NUM_REPLICAS=$2
NUM_GPUS=$3
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --megatron_checkpoint "$MEGATRON_CHECKPOINT" \
  --model_id megatron_model \
  --host 0.0.0.0 \
  --port 8000 \
  --num_gpus "$NUM_GPUS" \
  --num_replicas "$NUM_REPLICAS"
