#!/usr/bin/env python3
"""
Collect evaluation results from megatron_eval_results.

Usage:
    python collect_results.py ministral_3b/seed_42 ministral_8b/seed_42
    python collect_results.py ministral_3b/seed_42 --csv
    python collect_results.py ministral_3b/seed_42 --detailed
    python collect_results.py ministral_3b/seed_42 --no-postprocess
"""

import os
import ast
import json
import glob
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

BASE_DIR = "/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_eval_results"

TASK_CONFIGS = {
    "humaneval-ns0":      {"task_key": "humaneval",      "metrics": ["pass@1,create_test"],                                         "use_postprocessing": True},
    "humaneval_plus-ns0": {"task_key": "humaneval_plus", "metrics": ["pass@1,create_test"],                                         "use_postprocessing": True},
    "mbpp-ns3":           {"task_key": "mbpp",           "metrics": ["pass_at_1,none"]},
    "mbpp_plus-ns3":      {"task_key": "mbpp_plus",      "metrics": ["pass_at_1,none"]},
    "gsm8k_cot-ns8":      {"task_key": "gsm8k_cot",      "metrics": ["exact_match,strict-match", "exact_match,flexible-extract"]},
    "minerva_math-ns4":   {"task_key": "minerva_math",   "metrics": ["exact_match,none", "math_verify,none"]},
}


# --- sanitize (inlined from sanitize.py) ---

def _refine_text(text: str) -> str:
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def _syntax_check(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False


def _extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > 100:
        lines = lines[:100]
    max_valid_lines = 0
    max_valid_snippet = ""
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i:j+1])
            if _syntax_check(current_snippet):
                valid_line_count = sum(1 for line in lines[i:j+1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    max_valid_snippet = current_snippet
    return max_valid_snippet


def _get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    deps.add(child.attr)
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps


def _get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]
    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)
    return visited


def sanitize(text: str, entrypoint: Optional[str] = None) -> str:
    text = _refine_text(text)
    code = _extract_longest_valid_code(text)
    tree = ast.parse(code)

    definitions = {}
    imports = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = ('class', node)
        elif isinstance(node, ast.FunctionDef):
            if any(isinstance(n, ast.Return) for n in ast.walk(node)):
                definitions[node.name] = ('function', node)
        elif isinstance(node, ast.Assign):
            targets = node.targets
            if targets and isinstance(targets[0], ast.Name):
                definitions[targets[0].id] = ('variable', node)

    if entrypoint:
        name2deps = _get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = _get_function_dependency(entrypoint, name2deps)

    sanitized_output = []
    for node in imports:
        sanitized_output.append(ast.unparse(node))
    for name, (_, node) in definitions.items():
        if not entrypoint or name in reachable:
            sanitized_output.append(ast.unparse(node))

    return "\n".join(sanitized_output)


# --- postprocessing (inlined from postprocess_code.py) ---

def _compute_pass_at_1(samples_file: str) -> Optional[float]:
    try:
        import evaluate as hf_evaluate
        pass_at_k = hf_evaluate.load("code_eval")

        data = []
        with open(samples_file) as f:
            for line in f:
                data.append(json.loads(line))

        references = [sample['target'] for sample in data]
        predictions = [
            [sanitize(
                sample['doc']['prompt'] + "\n" + sample['resps'][0][0].split('```python\n', 1)[-1].split('```')[0],
                sample['doc']["entry_point"]
            )]
            for sample in data
        ]

        pass_at_1s = [
            pass_at_k.compute(references=[ref], predictions=[pred], k=[1])[0]["pass@1"]
            for ref, pred in zip(references, predictions)
        ]

        # write cleaned file
        cleaned = [
            {"task_id": sample['doc']['task_id'], "completion": pred, "pass_at_1": res}
            for sample, pred, res in zip(data, predictions, pass_at_1s)
        ]
        with open(samples_file + '.cleaned', 'w') as f:
            for item in cleaned:
                f.write(json.dumps(item) + '\n')

        return sum(pass_at_1s) / len(pass_at_1s)
    except Exception as e:
        print(f"  Warning: postprocessing failed for {os.path.basename(samples_file)}: {e}")
        return None


# --- result collection ---

def find_latest_results_file(task_dir):
    subdirs = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
    if not subdirs:
        return None
    results_dir = os.path.join(task_dir, subdirs[0])
    json_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    return max(json_files, key=os.path.getmtime) if json_files else None


def find_samples_file(task_dir, task_key):
    subdirs = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
    if not subdirs:
        return None
    results_dir = os.path.join(task_dir, subdirs[0])
    files = glob.glob(os.path.join(results_dir, f"samples_{task_key}_*.jsonl"))
    files = [f for f in files if not f.endswith('.cleaned')]
    return max(files, key=os.path.getmtime) if files else None


def extract_metrics(json_file, task_key, metrics):
    try:
        with open(json_file) as f:
            data = json.load(f)
        task_results = data.get("results", {}).get(task_key, {})
        return {m: task_results.get(m) for m in metrics}
    except Exception:
        return {m: None for m in metrics}


def collect_experiment(exp_name, base_dir=BASE_DIR, postprocess=True):
    exp_dir = os.path.join(base_dir, exp_name)
    if not os.path.exists(exp_dir):
        print(f"Warning: not found: {exp_dir}")
        return None

    exp_results = {"exp_name": exp_name}
    for task_name, config in TASK_CONFIGS.items():
        task_dir = os.path.join(exp_dir, task_name)
        if not os.path.exists(task_dir):
            exp_results[task_name] = {m: None for m in config["metrics"]}
            continue

        if config.get("use_postprocessing") and postprocess:
            samples_file = find_samples_file(task_dir, config["task_key"])
            if samples_file:
                print(f"  {task_name}: postprocessing {os.path.basename(samples_file)} ...")
                score = _compute_pass_at_1(samples_file)
                if score is not None:
                    exp_results[task_name] = {"pass@1": score}
                    continue
            # fall through to raw JSON metrics if postprocessing fails or no samples

        latest = find_latest_results_file(task_dir)
        if latest:
            exp_results[task_name] = extract_metrics(latest, config["task_key"], config["metrics"])
            print(f"  {task_name}: {os.path.basename(latest)}")
        else:
            exp_results[task_name] = {m: None for m in config["metrics"]}

    return exp_results


def fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"


def print_table(all_results, csv=False):
    sep = "," if csv else "\t"
    headers = ["experiment"]
    for task_name, config in TASK_CONFIGS.items():
        short = task_name.replace("-ns8","").replace("-ns0","").replace("-ns3","").replace("-ns4","")
        if config.get("use_postprocessing"):
            headers.append(f"{short}_pass@1")
        else:
            for m in config["metrics"]:
                metric_name = m.split(',')[0]
                suffix = m.split(',')[1] if ',' in m else ''
                headers.append(f"{short}_{metric_name}_{suffix}" if suffix else f"{short}_{metric_name}")
    print(sep.join(headers))

    for r in all_results:
        if r is None:
            continue
        row = [r["exp_name"]]
        for task_name, config in TASK_CONFIGS.items():
            tr = r.get(task_name, {})
            if config.get("use_postprocessing"):
                v = tr.get("pass@1") or tr.get("pass@1,create_test")
                row.append(fmt(v))
            else:
                for m in config["metrics"]:
                    row.append(fmt(tr.get(m)))
        print(sep.join(row))


def print_detailed(all_results):
    for r in all_results:
        if r is None:
            continue
        print(f"\n{'='*60}\nExperiment: {r['exp_name']}\n{'='*60}")
        for task_name, config in TASK_CONFIGS.items():
            tr = r.get(task_name, {})
            print(f"  {task_name}:")
            for k, v in tr.items():
                print(f"    {k}: {fmt(v)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_names", nargs="+", help="e.g. ministral_3b/seed_42")
    parser.add_argument("--base-dir", default=BASE_DIR)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true", help="Skip code execution postprocessing for humaneval")
    args = parser.parse_args()

    all_results = []
    for exp in args.exp_names:
        print(f"\nCollecting: {exp}")
        all_results.append(collect_experiment(exp, args.base_dir, postprocess=not args.no_postprocess))

    print()
    print_table(all_results, csv=args.csv)
    if args.detailed:
        print_detailed(all_results)


if __name__ == "__main__":
    main()
