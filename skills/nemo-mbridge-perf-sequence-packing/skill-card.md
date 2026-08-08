## Description: <br>
Validate and use packed sequences and long-context training in Megatron-Bridge, including offline LLM packing, collate-time VLM packing, Energon online packing, and CP constraints. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers configuring sequence packing and long-context training in Megatron-Bridge for LLM and VLM finetuning workloads on NVIDIA GPUs. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Packed Sequences Documentation](docs/training/packed-sequences.md) <br>
- [Performance Tuning Guide](docs/performance-guide.md) <br>
- [Hierarchical Context Parallel](docs/training/hierarchical-context-parallel.md) <br>
- [Multi-Token Prediction](docs/training/multi-token-prediction.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Code, Shell commands] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 task (1 positive) in isolated k8s-sandbox pods with dataset digest sha256:57d3c088. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the skill produces correct answers against reference answers. <br>
- Discoverability: Whether the right skill is discovered and activated when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal and expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill) | Codex (Baseline → Skill) |
|---|---:|---:|
| Overall | 40% → 96% (+56 pts) | 68% → 96% (+29 pts) |
| Security | 100% → 100% (±0 pts) | 100% → 100% (±0 pts) |
| Correctness | 0% → 100% (+100 pts) | 100% → 100% (±0 pts) |
| Discoverability | 50% → 98% (+48 pts) | 50% → 88% (+38 pts) |
| Effectiveness | 0% → 81% (+81 pts) | 89% → 95% (+6 pts) |
| Efficiency | 50% → 100% (+50 pts) | 0% → 100% (+100 pts) |

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
