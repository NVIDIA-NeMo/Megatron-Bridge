## Description: <br>
Systematic workflow for MoE training optimization in Megatron Bridge, based on the Megatron-Core MoE paper. Covers the Three Walls framework, parallel folding, recompute strategy, dispatcher choice, and CUDA-graph bring-up. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers performing full MoE throughput tuning sweeps or diagnosing MoE throughput regressions after a commit or config change in Megatron Bridge. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/abs/2603.07685) <br>
- [NVIDIA Megatron Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
NVSkills-Eval 3-Tier Evaluation with external profile. Tier 1 static validation (9 checks, 11 findings). Tier 2 deduplication (2 checks, 0 findings). Tier 3 live agent evaluation not available. <br>

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
