# Unset SLURM/PMI/PMIX env vars to prevent MPI initialization issues
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

OUTPUT_DIR=$1
PARALLELISM=$2

# Install missing dependency for lm-evaluation-harness
uv pip install math_verify --quiet

cat << EVAL_EOF > _temp_eval_script.py
import subprocess
import time

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.api import check_endpoint, evaluate

# Configuration
endpoint_url = "http://0.0.0.0:8000/v1/completions/"
endpoint_type = "completions"
model_id = "megatron_model"
eval_task = "mmlu"
limit_samples = 100
parallelism = $PARALLELISM
request_timeout = 1000
temperature = None
top_p = None
top_k = None
# Use local filesystem to avoid NFS SQLite lock contention on GCP (NFSv3).
# lm-eval's --use_cache writes a SQLite WAL that requires POSIX locks which
# are extremely slow over NFS, causing minutes-long stalls between batches.
pvc_output_dir = "/$OUTPUT_DIR/results/"
output_dir = "/tmp/eval_results/"

# Check server readiness
server_ready = check_endpoint(
    endpoint_url=endpoint_url,
    endpoint_type=endpoint_type,
    model_name=model_id,
)
if not server_ready:
    raise RuntimeError(
        "Server is not ready to accept requests. Check the deployment logs for errors."
    )

# Build configs
api_endpoint = ApiEndpoint(
    url=endpoint_url,
    type=endpoint_type,
    model_id=model_id,
)
target_cfg = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(
    limit_samples=limit_samples,
    parallelism=parallelism,
    request_timeout=request_timeout,
    temperature=temperature,
    top_p=top_p,
)
eval_cfg = EvaluationConfig(
    type=eval_task,
    params=eval_params,
    output_dir=output_dir,
)

if __name__ == "__main__":
    # Run evaluation
    result = evaluate(target_cfg=target_cfg, eval_cfg=eval_cfg)

    # Copy results from local tmp to PVC output dir (lm_cache stays in /tmp)
    import os, shutil
    os.makedirs(pvc_output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item == "lm_cache_rank0.db":
            continue  # skip large SQLite cache
        src = os.path.join(output_dir, item)
        dst = os.path.join(pvc_output_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"Results copied to {pvc_output_dir}")

    # Shutdown Ray server
    print("Evaluation completed. Shutting down Ray server...")
    subprocess.run(["ray", "stop", "--force"], check=False, timeout=30)
    print("Ray server shutdown command sent.")
    time.sleep(5)
EVAL_EOF

uv run --active --no-sync python _temp_eval_script.py
