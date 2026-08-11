# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The published ARC image predates non-root ownership of NeMo Run's canonical
# experiment directory, has no passwd entry for the target UID, and contains
# source directories without traversal bits. Keep these target-only adaptations
# on top of that exact image; repository source and test dependencies remain unchanged.
ARG BASE_IMAGE=766267172432.dkr.ecr.us-east-1.amazonaws.com/megatron-bridge@sha256:2c211aac77076f00566e89c4fb355530999c864395e4e066df81063a780d934a
FROM ${BASE_IMAGE}

ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532

USER root
RUN if ! getent group "${RUNTIME_GID}" >/dev/null; then \
      groupadd --gid "${RUNTIME_GID}" repository-e2e; \
    fi && \
    if ! getent passwd "${RUNTIME_UID}" >/dev/null; then \
      useradd --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" \
        --home-dir /tmp/regent-cache/home --no-create-home \
        --shell /usr/sbin/nologin repository-e2e; \
    fi && \
    find /opt/Megatron-Bridge -type d -exec chmod a+rx {} + && \
    chown "${RUNTIME_UID}:${RUNTIME_GID}" /opt/Megatron-Bridge && \
    install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" \
      /nemo_run /tmp/regent-cache /tmp/regent-cache/home

ENV HOME=/tmp/regent-cache/home \
    UV_CACHE_DIR=/tmp/regent-cache/uv

USER ${RUNTIME_UID}:${RUNTIME_GID}
RUN touch /nemo_run/.repository-e2e-write-probe && \
    rm /nemo_run/.repository-e2e-write-probe && \
    touch /opt/Megatron-Bridge/.repository-e2e-write-probe && \
    rm /opt/Megatron-Bridge/.repository-e2e-write-probe && \
    test -r /opt/Megatron-Bridge/tests/unit_tests/Launch_Unit_Tests_Core.sh && \
    test "$(getent passwd "$(id -u)" | cut -d: -f6)" = "${HOME}" && \
    uv cache dir | grep -Fx /tmp/regent-cache/uv
