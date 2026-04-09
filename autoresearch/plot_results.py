#!/usr/bin/env python3
"""
Plot experiment results from autoresearch/experiments/*/results.md.

Reads the structured results.md files and generates a line plot showing
how Avg score scales with denoising steps per experiment.

Usage:
    python autoresearch/plot_results.py
    python autoresearch/plot_results.py --output autoresearch/results_plot.png
"""

import argparse
import re
from pathlib import Path
from typing import Dict

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

# Mapping from benchmark names in results.md to short names
BENCHMARK_MAP = {
    "MBPP (pass@1)": "MBPP",
    "MBPP+ (pass@1)": "MBPP+",
    "GSM8k (8-shot CoT, strict)": "GSM8k Strict",
    "GSM8k (8-shot CoT, flexible)": "GSM8k Flex",
    "**Avg**": "Avg",
}


def parse_results_md(path: Path) -> Dict[int, Dict[str, float]]:
    """Parse a results.md file. Returns {steps: {metric: value}}."""
    text = path.read_text()
    results = {}

    pattern = r"## Metrics — (\d+) denoising steps? per block\s*\n(.*?)(?=\n## |\n---|\Z)"
    for match in re.finditer(pattern, text, re.DOTALL):
        steps = int(match.group(1))
        section = match.group(2)
        metrics = {}

        for row in re.finditer(r"\|\s*\*{0,2}(.+?)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)%\*{0,2}\s*\|", section):
            name = row.group(1).strip().strip("*")
            value = float(row.group(2))
            short_name = BENCHMARK_MAP.get(name, name)
            metrics[short_name] = value

        if metrics:
            results[steps] = metrics

    return results


def collect_all_experiments() -> Dict[str, Dict[int, Dict[str, float]]]:
    """Collect results from all experiment directories."""
    all_results = {}
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        results_file = exp_dir / "results.md"
        if results_file.exists():
            parsed = parse_results_md(results_file)
            if parsed:
                all_results[exp_dir.name] = parsed
    return all_results


def plot_steps_scaling(all_results: Dict, output_path: str):
    """Create a line plot showing how Avg scales with denoising steps per experiment."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(all_results), 3)))

    for exp_idx, (exp_name, exp_data) in enumerate(all_results.items()):
        steps_sorted = sorted(exp_data.keys())
        avgs = [exp_data[s].get("Avg", 0) for s in steps_sorted]
        ax.plot(steps_sorted, avgs, "o-", label=exp_name, color=colors[exp_idx], linewidth=2, markersize=8)
        for s, a in zip(steps_sorted, avgs):
            ax.annotate(f"{a:.1f}%", (s, a), textcoords="offset points", xytext=(5, 8), fontsize=8)

    ax.set_xlabel("Denoising Steps per Block", fontsize=11)
    ax.set_ylabel("Avg Score (%)", fontsize=11)
    ax.set_title("Quality vs Denoising Steps", fontsize=13, fontweight="bold")
    ax.set_xticks(sorted({s for exp in all_results.values() for s in exp}))
    ax.set_ylim(0, 80)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main():
    """Generate results plot from experiment data."""
    parser = argparse.ArgumentParser(description="Plot autoresearch experiment results")
    parser.add_argument(
        "--output", default=str(EXPERIMENTS_DIR.parent / "results_plot.png"), help="Output path for the plot"
    )
    args = parser.parse_args()

    all_results = collect_all_experiments()
    if not all_results:
        print("No experiments found in", EXPERIMENTS_DIR)
        return

    print(f"Found {len(all_results)} experiments: {', '.join(all_results.keys())}")
    plot_steps_scaling(all_results, args.output)


if __name__ == "__main__":
    main()
