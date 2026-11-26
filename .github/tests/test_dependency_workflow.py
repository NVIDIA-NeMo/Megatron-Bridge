# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the dependency workflow's branch policy and pinning without network access."""

import json
import os
import subprocess
import unittest
from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "_update_dependencies.yml"
JOBS = yaml.safe_load(WORKFLOW.read_text())["jobs"]
PREFLIGHT = {step.get("id"): step for step in JOBS["pre-flight"]["steps"]}
TIP = "a" * 40
BASE = "b" * 40
RELEASE_RULE = {
    "pattern": "[rv][0-9].[0-9].[0-9]",
    "restrictsPushes": True,
    "blocksCreations": True,
    "requiresApprovingReviews": True,
    "requiresCodeOwnerReviews": True,
}
MOCK_GITHUB = """
const {script, branches, relation} = JSON.parse(require("fs").readFileSync(0, "utf8"));
const output = {};
const core = {setOutput: (name, value) => { output[name] = value; }};
const context = {repo: {owner: "NVIDIA-NeMo", repo: "Megatron-Bridge"}};
const repos = {
  listBranches() {},
  async compareCommitsWithBasehead({basehead}) {
    if (basehead !== process.env.EXPECTED_BASE + "..." + process.env.TIP) throw Error("incorrect compare");
    return {data: {status: relation}};
  },
  async getBranch({owner, repo, branch}) {
    if (owner !== "NVIDIA" || repo !== "Megatron-LM") throw Error("incorrect dependency repository");
    return {data: branches[branch]};
  },
};
const github = {
  rest: {repos},
  async paginate(method, params) {
    if (method !== repos.listBranches || !params.protected) throw Error("expected protected branches");
    return Object.values(branches).filter(branch => branch.protected);
  },
  async graphql(query, {ref}) {
    if (!ref.startsWith("refs/heads/")) throw Error("expected a qualified branch");
    return {repository: {ref: branches[ref.slice("refs/heads/".length)] || null}};
  },
};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
new AsyncFunction("github", "context", "core", script)(github, context, core)
  .then(() => process.stdout.write(JSON.stringify(output)))
  .catch(() => {process.exitCode = 1;});
"""


def branch(name: str, **rule: object) -> dict[str, object]:
    """Construct a public branch response with an effective classic protection rule."""
    return {
        "name": name,
        "protected": True,
        "branchProtectionRule": dict(RELEASE_RULE, **rule),
        "target": {"oid": TIP},
        "commit": {"sha": TIP},
    }


def resolve(step: str, branches: dict, *, relation: str = "ahead", **env: str) -> dict | None:
    """Execute the workflow's actual GitHub Script against controlled branch metadata."""
    result = subprocess.run(
        ["node", "-e", MOCK_GITHUB],
        input=json.dumps({"script": PREFLIGHT[step]["with"]["script"], "branches": branches, "relation": relation}),
        env=dict(os.environ, TIP=TIP, **env),
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout) if result.returncode == 0 else None


class DependencyRefs(unittest.TestCase):
    """Require protected, immutable Bridge and Megatron-LM inputs throughout dependency updates."""

    def test_latest_bridge_release_uses_effective_rule(self) -> None:
        """Skip higher names under weaker rules and aliases outside the classic release pattern."""
        branches = {
            "r0.5.0": branch("r0.5.0"),
            "r0.7.0": branch("r0.7.0"),
            "r0.8.0": branch("r0.8.0", blocksCreations=False),
            "r0.9.0": branch("r0.9.0", restrictsPushes=False),
            "r2.0.0": branch("r2.0.0", requiresCodeOwnerReviews=False),
            "r3.0.0": branch("r3.0.0", requiresApprovingReviews=False),
            "r4.0.0": branch("r4.0.0", pattern="r*"),
            "r9.10.0": branch("r9.10.0"),
            "r9.9.9-test": branch("r9.9.9-test"),
        }
        self.assertEqual(resolve("target", branches, TARGET_BRANCH="latest-release"), {"branch": "r0.7.0", "sha": TIP})
        self.assertIsNone(resolve("target", branches, TARGET_BRANCH="r0.8.0"))

    def test_merge_target_must_retain_policy_and_tested_history(self) -> None:
        """Allow normal main progress and reject replacement histories or changed protection."""
        branches = {"main": branch("main", pattern="main", restrictsPushes=False, blocksCreations=False)}
        inputs = {"TARGET_BRANCH": "main", "EXPECTED_BASE": BASE}
        self.assertEqual(resolve("target", branches, **inputs), {"branch": "main", "sha": TIP})
        for relation in ("behind", "diverged"):
            self.assertIsNone(resolve("target", branches, relation=relation, **inputs))
        self.assertIsNone(resolve("target", {"main": branch("main", pattern="*")}, **inputs))
        check = next(s for s in JOBS["merge"]["steps"] if s["name"].startswith("Recheck"))
        self.assertEqual(check["with"]["script"], PREFLIGHT["target"]["with"]["script"])

    def test_mcore_release_is_numeric_canonical_and_protected(self) -> None:
        """Resolve multi-digit upstream versions while excluding aliases and unprotected heads."""
        names = ["core_r0.9.0", "core_r0.20.0", "core_r9.00.0", "core_r1.0.0-test", "core_r0.21.0"]
        branches = {name: branch(name) for name in names}
        branches["core_r0.21.0"]["protected"] = False
        inputs = {"TARGET_BRANCH": "r0.7.0", "MCORE_BRANCH": "latest-release"}
        self.assertEqual(resolve("mcore", branches, **inputs), {"branch": "core_r0.20.0", "sha": TIP})
        self.assertIsNone(resolve("mcore", branches, TARGET_BRANCH="main", MCORE_BRANCH="core_r0.20.0"))

    def test_build_and_dev_restore_consume_preflight_shas(self) -> None:
        """Pin the container checkout, submodule fetch and dev restoration to preflight outputs."""
        update = JOBS["update-lockfile"]
        self.assertEqual(update["steps"][0]["with"]["ref"], "${{ env.TARGET_SHA }}")
        for variable, output in (("TARGET_SHA", "target-sha"), ("MCORE_SHA", "mcore-sha")):
            self.assertEqual(update["env"][variable], "${{ needs.pre-flight.outputs." + output + " }}")
        self.assertEqual(JOBS["finish-dev"]["env"]["TARGET_SHA"], update["env"]["TARGET_SHA"])
        restore = next(s["run"] for s in JOBS["finish-dev"]["steps"] if s.get("id") == "restore")
        self.assertIn('git restore --source="$TARGET_SHA"', restore)


if __name__ == "__main__":
    unittest.main()
