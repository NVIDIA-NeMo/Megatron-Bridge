# Known Issues

This page lists known issues and limitations in the current release.

## 26.04

- The [`av`](https://pypi.org/project/av/) (PyAV) Python package is no longer installed by default in the NeMo Framework 26.04 container (`nvcr.io/nvidia/nemo:26.04`) to mitigate a CVE in the bundled FFmpeg binaries. Workflows that rely on `av` for video decoding (for example, certain multimodal data pipelines and `qwen-vl-utils` video paths) must reinstall it at runtime — see [`docker/common/README.md`](../../docker/common/README.md#reinstalling-pyav-av-at-runtime) for instructions.

## 26.02

- AWS EKS only: Due to AWS-OFI-NCCL v1.17.0 long-running jobs suffer a memory leak that causes performance regression over time. This can be mitigated by upgrading to [v1.17.3](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.3).
- Context parallelism with sequence packing are not yet supported for Qwen 3 VL in the r0.3.0 release. **Fixed in 26.02.01 (r0.3.1).**
- DeepEP is not supported in the current NeMo framework 26.02 container (nvcr.io/nvidia/nemo:26.02), which results in reduced DSv3 performance compared to the NeMo framework 25.09 container (nvcr.io/nvidia/nemo:25.09) on H100 machines. For optimal H100 performance, we recommend using the NeMo framework 25.09 container.

## 25.11

- Deepseek V3 on H100 has an issue when using DeepEP and fails with `RuntimeError: DeepEP error: timeout (dispatch CPU)`.
- MODEL_TFLOP/s/GPU is printed as 0 to stdout for all Hybrid models, such as Nemotron-H 56B.

## 25.09

- **Pretraining DeepSeek in subchannel FP8 precision is not working.** Pretraining DeepSeek with current scaling FP8 is a workaround, but MTP loss does not converge.
