#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
installer="$repo_root/docker/common/install_mok.sh"
dockerfile="${1:-$repo_root/docker/Dockerfile.ci}"
temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

# Both CPU platforms inherit the selected base image; the final install must
# follow every uv sync, which would otherwise remove this Docker-only package.
grep -Fqx 'FROM ${BASE_IMAGE} AS base' "$dockerfile"
grep -Fqx 'FROM base AS mok-wheel' "$dockerfile"
grep -Fqx 'FROM base AS megatron_bridge' "$dockerfile"
grep -Fqx 'ARG MOK_ARCH=ALL' "$dockerfile"
grep -Fq -- '--mount=type=bind,from=mok-wheel,source=/opt/mok-wheels' "$dockerfile"
last_sync=$(grep -nE 'uv sync([[:space:]]|$)' "$dockerfile" | tail -1 | cut -d: -f1)
wheel_install=$(grep -n 'uv pip install .* /opt/mok-wheels/\*.whl' "$dockerfile" | cut -d: -f1)
((wheel_install > last_sync))
revision=$(sed -n 's/^ARG MOK_COMMIT=//p' "$dockerfile")
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]

# Mock external source/build tools, retaining the real shell validation and
# cuobjdump output checks. This is not a native compilation or runtime test.
export MOK_TEST_DIR="$temporary_dir"
export CUDA_HOME="$temporary_dir/cuda"
mkdir -p "$CUDA_HOME/bin"
cat > "$CUDA_HOME/bin/cuobjdump" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == --list-elf && -f "$2" ]]
for target in $MOK_TEST_CUBINS; do
    echo "ELF file 1: kernel.${target}.cubin"
done
SH
chmod +x "$CUDA_HOME/bin/cuobjdump"

git() {
    echo git >> "$MOK_TEST_DIR/calls"
    if [[ "$1" == init ]]; then
        mkdir -p "$2/scripts"
        echo 'touch "$MOK_TEST_DIR/prepared"' > "$2/scripts/prepare_thunderkittens.sh"
    fi
}
uv() {
    echo "uv $*" >> "$MOK_TEST_DIR/calls"
    case "$1" in
        build)
            [[ -f "$MOK_TEST_DIR/prepared" ]]
            [[ "$MOK_NVCC" == "$CUDA_HOME/bin/nvcc" ]]
            [[ "$LIBRARY_PATH" == "$CUDA_HOME/lib64/stubs"* ]]
            [[ " $* " == *' --no-build-isolation '* ]]
            while [[ "$1" != --out-dir ]]; do shift; done
            mkdir -p "$2"
            touch "$2/mixture_of_kittens.whl"
            ;;
        pip)
            [[ "$2" == install && " $* " == *' --no-deps '* ]]
            touch "$MOK_TEST_DIR/_C.so"
            ;;
        run)
            cat > /dev/null
            echo "$MOK_TEST_DIR/_C.so"
            ;;
        *) return 1 ;;
    esac
}
export -f git uv

run_installer() {
    bash "$installer" "$1" "$repo_root/docker/patches/mok.patch" "$temporary_dir/wheels" \
        > "$temporary_dir/output" 2>&1
}
expect_failure() {
    if run_installer "$1"; then
        echo "Installer unexpectedly accepted revision=$1 MOK_ARCH=${MOK_ARCH:-ALL}" >&2
        cat "$temporary_dir/output" >&2
        exit 1
    fi
}

unset MOK_ARCH
expect_failure main
grep -q 'MOK_COMMIT must be a full Git commit' "$temporary_dir/output"
[[ ! -e "$temporary_dir/calls" ]]
export MOK_ARCH=SM90
expect_failure "$revision"
grep -q 'Unsupported MOK_ARCH' "$temporary_dir/output"
[[ ! -e "$temporary_dir/calls" ]]

unset MOK_ARCH
export MOK_TEST_CUBINS='sm_100a sm_103a'
run_installer "$revision"
grep -q '^uv build ' "$temporary_dir/calls"
grep -q '^uv pip install ' "$temporary_dir/calls"
for missing_target in sm_100a sm_103a; do
    export MOK_TEST_CUBINS="${MOK_TEST_CUBINS/$missing_target/}"
    expect_failure "$revision"
    export MOK_TEST_CUBINS='sm_100a sm_103a'
done

export MOK_ARCH=SM100 MOK_TEST_CUBINS=sm_100a
run_installer "$revision"
export MOK_ARCH=SM103 MOK_TEST_CUBINS=sm_103a
run_installer "$revision"
echo 'MoK installer regression checks passed'
