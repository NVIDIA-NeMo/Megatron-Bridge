#!/bin/bash

set -eoxu pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $SCRIPT_DIR/..

STASHED=False
if ! uv lock --check > /dev/null 2>&1 ; then
    echo "Lock file is up to date. Will temporarily stash changes."
    git stash push
    STASHED=True
fi

docker build -t megatron-bridge -f docker/Dockerfile.ci .

if [ $STASHED = True ]; then
    echo "Restoring stashed changes."
    git stash pop
fi

docker run \
    --rm \
    -v $(pwd):/workdir/ \
    -w /workdir/ \
    megatron-bridge \
    uv lock --upgrade