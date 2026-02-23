# Known Issues

This page lists known issues and limitations in the current release.

## 26.02

- AWS EKS only: Due to AWS-OFI-NCCL v1.17.0 long-running jobs suffer a memory leak that causes performance regression over time. This can be mitigated by upgrading to [v1.17.3](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.3).

## 25.11

- Deepseek V3 on H100 has an issue when using DeepEP and fails with `RuntimeError: DeepEP error: timeout (dispatch CPU)`.
- MODEL_TFLOP/s/GPU is printed as 0 to stdout for all Hybrid models, such as Nemotron-H 56B.

## 25.09

- **Pretraining DeepSeek in subchannel FP8 precision is not working.** Pretraining DeepSeek with current scaling FP8 is a workaround, but MTP loss does not converge.
