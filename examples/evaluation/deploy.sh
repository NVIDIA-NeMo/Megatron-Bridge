# Unset SLURM/PMI/PMIX env vars to prevent MPI initialization issues
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done
export RAY_enable_infeasible_task_early_exit=true

python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --megatron_checkpoint /nemo-workspace/pagaray/megatron_bridge_ci/weekly_test/2026-01-10/llama/llama3_8b/checkpoints/iter_0009600/  \
  --model_id megatron_model \
  --host 0.0.0.0 \
  --port 8000 \
  --num_gpus 8 \
  --num_replicas 1 \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --context_parallel_size 1