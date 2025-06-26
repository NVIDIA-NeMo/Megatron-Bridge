#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --export-deploy-ref)
        EXPORT_DEPLOY_REF="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --export-deploy-ref EXPORT_DEPLOY_REF"
        exit 1
        ;;
    esac
done

# Check if required arguments are provided
if [ -z "$EXPORT_DEPLOY_REF" ]; then
    echo "Error: --export-deploy-ref"
    echo "Usage: $0 --export-deploy-ref EXPORT_DEPLOY_REF"
    exit 1
fi

source ${UV_PROJECT_ENVIRONMENT}/bin/activate

# Nemo-run has conflicting dependencies to export-deploy:
# They collide on nvidia-pytriton (export-deploy) and torchx (nemo-run)
# via urllib3.
uv pip install --no-cache-dir --upgrade nemo_run
