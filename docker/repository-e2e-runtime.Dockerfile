# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The published ARC image predates non-root ownership of NeMo Run's canonical
# experiment directory. Keep this target-only adaptation on top of that exact
# image; repository source and test dependencies remain unchanged.
ARG BASE_IMAGE=766267172432.dkr.ecr.us-east-1.amazonaws.com/megatron-bridge@sha256:2c211aac77076f00566e89c4fb355530999c864395e4e066df81063a780d934a
FROM ${BASE_IMAGE}

ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532

USER root
RUN install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" /nemo_run

USER ${RUNTIME_UID}:${RUNTIME_GID}
RUN touch /nemo_run/.repository-e2e-write-probe && \
    rm /nemo_run/.repository-e2e-write-probe
