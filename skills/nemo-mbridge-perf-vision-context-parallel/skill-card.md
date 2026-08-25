## Description: <br>
Operational guide for sharding a VLM vision encoder across the language model's context-parallel ranks in Megatron-Bridge, including config knobs, code anchors, load-balance pitfalls, and measured impact. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers optimizing Vision-Language Model (VLM) training throughput and memory usage when running with context parallelism greater than 1 in Megatron-Bridge. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Megatron-Bridge Performance Tuning Guide](docs/performance-guide.md) <br>
- [Nemotron Omni Modeling Implementation](src/megatron/bridge/models/nemotron_omni/modeling_nemotron_omni.py) <br>
- [MCore Context Parallel Helpers](3rdparty/Megatron-LM/megatron/core/models/multimodal/context_parallel.py) <br>
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 evaluation tasks (2 positive, 1 negative) from a curated skill-evaluator dataset. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct answers against the reference answer. <br>
- Discoverability: Whether the right skill is loaded and activated when needed. <br>
- Effectiveness: Whether the skill helps the agent complete the user's goal and expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage, measuring routing quality and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 71% → 98% (+27 points) | 56% → 94% (+39 points) |
| Security | 100% → 100% (±0 points) | 67% → 100% (+33 points) |
| Correctness | 27% → 93% (+67 points) | 60% → 93% (+33 points) |
| Discoverability | 100% → 100% (±0 points) | 58% → 96% (+38 points) |
| Effectiveness | 30% → 98% (+68 points) | 40% → 81% (+41 points) |
| Efficiency | 98% → 100% (+2 points) | 53% → 100% (+47 points) |

## Testing Completed: <br>
**[x] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
1.0.0+b7643bd (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
