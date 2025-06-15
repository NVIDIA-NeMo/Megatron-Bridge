#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --megatron-lm-ref)
        MCORE_REF="$2"
        shift 2
        ;;
    --export-deploy-ref)
        EXPORT_DEPLOY_REF="$2"
        shift 2
        ;;
    --nemo-ref)
        NEMO_REF="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --megatron-lm-ref MCORE_REF --export-deploy-ref EXPORT_DEPLOY_REF --nemo-ref NEMO_REF"
        exit 1
        ;;
    esac
done

# Check if required arguments are provided
if [ -z "$MCORE_REF" ] || [ -z "$EXPORT_DEPLOY_REF" ] || [ -z "$NEMO_REF" ]; then
    echo "Error: --megatron-lm-ref and --export-deploy-ref and --nemo-ref are required"
    echo "Usage: $0 --megatron-lm-ref MCORE_REF --export-deploy-ref EXPORT_DEPLOY_REF --nemo-ref NEMO_REF"
    exit 1
fi

source ${UV_PROJECT_ENVIRONMENT}/bin/activate

# Nemo-run has conflicting dependencies to export-deploy:
# They collide on nvidia-pytriton (export-deploy) and torchx (nemo-run)
# via urllib3.
uv pip install --no-cache-dir --upgrade nemo_run

# We need nemo-toolkit until migration is complete.
uv pip install --no-cache-dir --upgrade nemo-toolkit[automodel,common-only,nlp-only,eval,multimodal-only]@git+https://github.com/NVIDIA/NeMo.git@${NEMO_REF} pandas==2.3.0 "opentelemetry-sdk<=1.27.0"

# megatron-core and export-deploy are dependencies, but for development
# we override with latest VCS commits
uv pip uninstall megatron-core nemo-export-deploy
uv pip install --no-cache-dir --upgrade \
    "numpy<2.0.0" "megatron_core@git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_REF}"
uv pip install --no-cache-dir --upgrade \
    "numpy<2.0.0" "NeMo-Export-Deploy${INSTALL_EVAL:+[te,trtllm]}@git+https://github.com/NVIDIA-NeMo/NeMo-Export-Deploy.git@${EXPORT_DEPLOY_REF}"
