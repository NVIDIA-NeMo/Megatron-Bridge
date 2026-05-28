## Description: <br>
Practical guidance for training MoE VLMs in Megatron Bridge, comparing FSDP and 3D-parallel approaches using rounded lessons from Qwen3-VL, Qwen3-Next, and other multimodal experiments. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers training Mixture-of-Experts vision-language models with Megatron Bridge, selecting between FSDP and 3D-parallel strategies, and tuning parallelism, recompute, and CUDA-graph settings for optimal throughput on NVIDIA GPU clusters. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Megatron Bridge Performance Tuning Guide](docs/performance-guide.md) <br>
- [Megatron Bridge Performance Summary](docs/performance-summary.md) <br>
- [MoE Optimization Documentation](docs/training/moe-optimization.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
NVSkills-Eval 3-Tier evaluation using the external profile. Tier 1 static validation (9 checks), Tier 2 deduplication (2 checks), and Tier 3 live agent evaluation (not available in this report). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
v0.2.0rc6-1468-gbbcfbcea (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
