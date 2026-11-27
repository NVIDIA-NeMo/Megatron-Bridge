# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["pyyaml>=6.0,<7"]
# ///
"""Exercise release guards without credentials, uploads, or package builds."""

import json
import os
import re
import subprocess
import tempfile
import unittest
from itertools import product
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load(next((ROOT / ".github/workflows").glob("release.y*ml")).read_text())
JOBS = WORKFLOW["jobs"]
FREEZE = yaml.safe_load((ROOT / ".github/workflows/release-freeze.yml").read_text())["jobs"]
DOCS = yaml.safe_load((ROOT / ".github/workflows/release-docs.yml").read_text())["jobs"]


def evaluate(expression: str, context: dict[str, object]) -> bool:
    """Evaluate the boolean and equality subset used by release job guards."""
    source = expression.strip().removeprefix("${{").removesuffix("}}")
    source = source.replace("!cancelled()", "True").replace("!failure()", "True")
    source = re.sub(r"\b(?:github|inputs|needs)\.[\w.-]+", lambda match: repr(context[match[0]]), source)
    source = source.replace("&&", " and ").replace("||", " or ")
    source = re.sub(r"!(?!=)", "not ", source)
    return bool(eval(" ".join(source.split()), {"__builtins__": {}}, {}))


class ReleaseGuards(unittest.TestCase):
    """Cover publication eligibility and unsuccessful or skipped dependencies."""

    def setUp(self) -> None:
        """Create a successful manual release context for the canonical repo."""
        self.context = {
            "github.repository": "NVIDIA-NeMo/Megatron-Bridge",
            "github.event_name": "workflow_dispatch",
            "inputs.dry-run": False,
            "needs.validate-inputs.result": "success",
            "needs.release-workflow-tests.result": "success",
            "needs.bump.result": "success",
            "needs.build-wheel.result": "success",
            "needs.publish-wheel.result": "success",
        }

    def test_publish_requires_manual_non_dry_run(self) -> None:
        """Pushes and dry runs cannot request the production upload job."""
        for event in ("push", "workflow_dispatch"):
            for dry_run in (False, True):
                with self.subTest(event=event, dry_run=dry_run):
                    context = dict(self.context, **{"github.event_name": event, "inputs.dry-run": dry_run})
                    self.assertEqual(
                        evaluate(JOBS["publish-wheel"]["if"], context), event == "workflow_dispatch" and not dry_run
                    )
        self.context["github.repository"] = "example/fork"
        self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_missing_build_blocks_upload(self) -> None:
        """Failed, cancelled and skipped builds cannot reach publication."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.build-wheel.result"] = state
                self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_regression_failures_block_version_preparation(self) -> None:
        """Require the regression job to succeed before version preparation."""
        self.assertIn("release-workflow-tests", JOBS["bump"]["needs"])
        self.assertTrue(evaluate(JOBS["bump"]["if"], self.context))
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.release-workflow-tests.result"] = state
                self.assertFalse(evaluate(JOBS["bump"]["if"], self.context))

    def test_real_finalize_requires_upload(self) -> None:
        """A missing upload cannot finalize a real release; dry runs may continue."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.publish-wheel.result"] = state
                self.context["inputs.dry-run"] = False
                self.assertFalse(evaluate(JOBS["finalize"]["if"], self.context))
                self.context["inputs.dry-run"] = True
                self.assertTrue(evaluate(JOBS["finalize"]["if"], self.context))

    def test_summary_rejects_missing_required_jobs(self) -> None:
        """Run the actual shell summary against failures and permitted skips."""
        script = JOBS["release-summary"]["steps"][0]["run"]
        for publish, docs in product((False, True), repeat=2):
            baseline = {name: {"result": "success"} for name in JOBS["release-summary"]["needs"]}
            baseline["publish-wheel"]["result"] = "success" if publish else "skipped"
            baseline["rehearse-docs"]["result"] = "success" if docs and not publish else "skipped"
            baseline["publish-docs"]["result"] = "success" if docs and publish else "skipped"
            for job in (
                "release-workflow-tests",
                "validate-inputs",
                "bump",
                "build-wheel",
                "publish-wheel",
                "finalize",
                "rehearse-docs",
                "publish-docs",
            ):
                for state in ("success", "failure", "cancelled", "skipped"):
                    with self.subTest(publish=publish, docs=docs, job=job, state=state):
                        needs = {name: dict(value) for name, value in baseline.items()}
                        needs[job]["result"] = state
                        env = dict(
                            os.environ,
                            NEEDS=json.dumps(needs),
                            PUBLISH_EXPECTED=str(publish).lower(),
                            DOCS_EXPECTED=str(docs).lower(),
                            RELEASE_SKIPPED="false",
                        )
                        result = subprocess.run(
                            ["bash", "-euo", "pipefail", "-c", script], env=env, capture_output=True
                        )
                        expected = state == baseline[job]["result"]
                        self.assertEqual(result.returncode == 0, expected, result.stderr.decode())

    def test_exactly_one_docs_build_runs_when_requested_after_finalization(self) -> None:
        """Exercise rehearsal and real publication decisions independently of GitHub release creation."""
        for dispatch, dry_run, docs, create_release in product((False, True), repeat=4):
            with self.subTest(dispatch=dispatch, dry=dry_run, docs=docs, create_release=create_release):
                context = {
                    "github.event_name": "workflow_dispatch" if dispatch else "push",
                    "github.repository": "NVIDIA-NeMo/Megatron-Bridge",
                    "github.ref": "refs/heads/main" if dispatch else "refs/heads/pull-request/123",
                    "inputs.dry-run": dry_run,
                    "inputs.publish-docs": docs,
                    "inputs.create-gh-release": create_release,
                    "needs.finalize.result": "success",
                }
                self.assertEqual(evaluate(JOBS["rehearse-docs"]["if"], context), not dispatch or (dry_run and docs))
                self.assertEqual(evaluate(JOBS["publish-docs"]["if"], context), dispatch and not dry_run and docs)
                for state in ("failure", "cancelled", "skipped"):
                    context["needs.finalize.result"] = state
                    self.assertFalse(evaluate(JOBS["rehearse-docs"]["if"], context))
                    self.assertFalse(evaluate(JOBS["publish-docs"]["if"], context))
        production = self.context | {
            "github.ref": "refs/heads/main",
            "inputs.publish-docs": True,
            "needs.finalize.result": "success",
        }
        for mismatch in ({"github.repository": "example/fork"}, {"github.ref": "refs/heads/topic"}):
            with self.subTest(mismatch=mismatch):
                self.assertFalse(evaluate(JOBS["publish-docs"]["if"], production | mismatch))

    def test_release_docs_use_the_tagged_release_and_isolate_cloud_credentials(self) -> None:
        """Rehearse the release commit and verify the real tag before publishing the same build artifact."""
        finalizer = JOBS["finalize"]
        self.assertIs(finalizer["with"]["publish-docs"], False)
        rehearsal, publication = JOBS["rehearse-docs"], JOBS["publish-docs"]
        for job in (rehearsal, publication):
            self.assertEqual(job["uses"], "./.github/workflows/release-docs.yml")
            self.assertIn("finalize", job["needs"])
            self.assertEqual(job["permissions"], {"contents": "read"})
            self.assertEqual(job["with"]["docs-version-override"], finalizer["with"]["release-version"])
        self.assertNotIn("secrets", rehearsal)
        self.assertIs(rehearsal["with"]["dry-run"], True)
        self.assertEqual(rehearsal["with"]["github-ref"], finalizer["with"]["release-ref"])
        self.assertIs(publication["with"]["dry-run"], False)
        self.assertEqual(publication["with"]["expected-sha"], "${{ inputs.release-ref }}")
        self.assertEqual(
            publication["with"]["github-ref"],
            "${{ format('refs/tags/v{0}', needs.bump.outputs.release-version || needs.build-wheel.outputs.version) }}",
        )
        publisher = next(step for step in DOCS["publish-docs"]["steps"] if "/publish-docs@" in step.get("uses", ""))
        docs_credentials = {
            match[1]
            for value in publisher["with"].values()
            if isinstance(value, str) and (match := re.fullmatch(r"\$\{\{ secrets\.(\w+) \}\}", value))
        }
        self.assertTrue(docs_credentials)
        self.assertEqual(publication["secrets"], {name: "${{ secrets." + name + " }}" for name in docs_credentials})
        self.assertTrue(docs_credentials.isdisjoint(finalizer["secrets"]))

    def test_validate_ref_treats_input_as_data(self) -> None:
        """Only full immutable SHAs pass; branch names and shell payloads fail."""
        script = JOBS["validate-inputs"]["steps"][0]["run"]
        for ref in ("a" * 40, "main", "abc1234", "$(exit 0)", "a" * 40 + "\n"):
            with self.subTest(ref=ref):
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env=dict(os.environ, RELEASE_REF=ref),
                    capture_output=True,
                )
                self.assertEqual(result.returncode == 0, ref == "a" * 40)

    def test_upload_has_no_checkout_or_password(self) -> None:
        """Keep source execution and build credentials outside the OIDC job."""
        self.assertEqual(JOBS["build-wheel"]["permissions"], {"contents": "read"})
        self.assertNotIn("secrets", JOBS["build-wheel"])
        upload = JOBS["publish-wheel"]
        self.assertEqual(upload["permissions"], {"id-token": "write"})
        self.assertEqual(upload["environment"]["name"], "pypi")
        self.assertEqual(len(upload["steps"]), 2)
        for step in upload["steps"]:
            self.assertNotIn("run", step)
            self.assertNotIn("password", step.get("with", {}))
            self.assertNotIn("checkout", step["uses"])

    def test_manual_release_checks_run_on_docs_only_changes(self) -> None:
        """Every manual release validates source and target even after a docs-only push."""
        for event in ("push", "workflow_dispatch"):
            for docs_only in ("true", "false"):
                with self.subTest(event=event, docs_only=docs_only):
                    context = {
                        "github.event_name": event,
                        "needs.pre-flight.outputs.docs_only": docs_only,
                        "needs.pre-flight.outputs.is_deployment_workflow": "false",
                    }
                    self.assertEqual(
                        evaluate(JOBS["validate-inputs"]["if"], context),
                        event == "workflow_dispatch" or docs_only != "true",
                    )

    def test_dispatch_requires_main(self) -> None:
        """Allow push rehearsals and require a main-branch dispatch."""
        guard = next(step for step in JOBS["validate-inputs"]["steps"] if step["name"] == "Require dispatch from main")
        for event in ("push", "workflow_dispatch"):
            for ref in ("refs/heads/main", "refs/heads/r0.5.0", "refs/tags/main", "refs/heads/topic"):
                with self.subTest(event=event, ref=ref):
                    blocked = evaluate(guard["if"], {"github.event_name": event, "github.ref": ref})
                    self.assertEqual(blocked, event == "workflow_dispatch" and ref != "refs/heads/main")
        self.assertNotEqual(subprocess.run(["bash", "-c", guard["run"]], capture_output=True).returncode, 0)

    def test_release_target_requires_a_protected_history(self) -> None:
        """Run the release validation shell with read-only GitHub API fixtures."""
        script = next(
            step["run"] for step in JOBS["validate-inputs"]["steps"] if step["name"].startswith("Verify the release")
        )
        candidate = "1" * 40
        tip = "2" * 40
        mock_api = """
gh() {
    case "$2" in
        "repos/$GITHUB_REPOSITORY/branches/$TARGET_BRANCH") jq -r "$4" <<< "$BRANCH_REPLY" ;;
        "repos/$GITHUB_REPOSITORY/compare/$RELEASE_REF...$EXPECTED_TIP") jq -r "$4" <<< "$COMPARE_REPLY" ;;
        *) return 1 ;;
    esac
}
"""
        for branch, protected, status, expected in (
            ("main", True, "identical", True),
            ("r1.2.3", True, "ahead", True),
            ("v1.2.3", True, "behind", False),
            ("main", True, "diverged", False),
            ("r1.2.3", False, "ahead", False),
            ("topic", True, "ahead", False),
            ("r1.2.3-preview", True, "ahead", False),
        ):
            with self.subTest(branch=branch, status=status, protected=protected):
                env = dict(
                    os.environ,
                    RELEASE_REF=candidate,
                    EXPECTED_TIP=tip,
                    TARGET_BRANCH=branch,
                    GITHUB_REPOSITORY="NVIDIA-NeMo/Megatron-Bridge",
                    BRANCH_REPLY=json.dumps({"protected": protected, "commit": {"sha": tip}}),
                    COMPARE_REPLY=json.dumps({"status": status}),
                )
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", mock_api + script], env=env, capture_output=True
                )
                self.assertEqual(result.returncode == 0, expected)

    def test_freeze_docs_uses_reviewed_pull_requests(self) -> None:
        """Rehearse locally and open separate main and release docs PRs on a real freeze."""
        self.assertEqual(FREEZE["code-freeze"]["with"]["freeze-source-branch"], "main")
        for job in (FREEZE["code-freeze"], JOBS["bump"], JOBS["finalize"]):
            self.assertEqual(job["with"]["release-environment"], "megatron-bridge-release")
            self.assertNotIn("BOT_KEY", job.get("secrets", {}))
            self.assertNotIn("RELEASE_APP_PRIVATE_KEY", job.get("secrets", {}))
        docs = FREEZE["bump-docs-versions"]
        self.assertEqual(set(docs["strategy"]["matrix"]["target"]), {"main", "release"})
        self.assertEqual(docs["environment"]["name"], "${{ inputs.dry-run && 'public' || 'megatron-bridge-release' }}")
        token = next(step for step in docs["steps"] if "create-github-app-token" in step.get("uses", ""))
        create = next(step for step in docs["steps"] if "create-pull-request" in step.get("uses", ""))
        self.assertEqual(create["if"], "${{ !inputs.dry-run }}")
        self.assertEqual(token["if"], create["if"])
        self.assertNotIn("git push", "\n".join(step.get("run", "") for step in docs["steps"]))


class StandaloneDocsGuards(unittest.TestCase):
    """Cover real and rehearsal publication using the standalone docs entry points."""

    def run_source(self, *, branch: str = "main", comparison: str = "ahead", **env: str) -> tuple[int, dict[str, str]]:
        """Run the source step against GitHub-shaped commit, branch and compare fixtures."""
        candidate, tip = "a" * 40, "b" * 40
        mock_api = """
gh() {
    [[ "$1" == api ]] || return 64
    shift
    local path='' filter='' pages=false reply
    while (($#)); do
        case "$1" in
            --paginate) pages=true ;;
            --jq) shift; filter=$1 ;;
            *) path=$1 ;;
        esac
        shift
    done
    case "$path" in
        repos/"$GITHUB_REPOSITORY"/commits/missing) return 1 ;;
        repos/"$GITHUB_REPOSITORY"/commits/v1.2.3) reply='{"sha":"cccccccccccccccccccccccccccccccccccccccc"}' ;;
        repos/"$GITHUB_REPOSITORY"/commits/*) reply=$COMMIT_REPLY ;;
        repos/"$GITHUB_REPOSITORY"/branches\\?*protected=true*)
            if [[ "$pages" == true ]]; then reply=$BRANCH_REPLY; else reply='[]'; fi ;;
        repos/"$GITHUB_REPOSITORY"/compare/"$CANDIDATE"..."$TIP"*) reply=$COMPARE_REPLY ;;
        *) return 1 ;;
    esac
    jq -r "$filter" <<< "$reply"
}
"""
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            output.touch()
            fixture = {
                "PATH": os.environ["PATH"],
                "GITHUB_REPOSITORY": "NVIDIA-NeMo/Megatron-Bridge",
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SHA": candidate,
                "GITHUB_OUTPUT": str(output),
                "EXPECTED_SHA": "",
                "DOCS_REF": candidate,
                "DOCS_VERSION": "",
                "DRY_RUN": "false",
                "PUBLISH_LATEST": "true",
                "CANDIDATE": candidate,
                "TIP": tip,
                "COMMIT_REPLY": json.dumps({"sha": candidate}),
                "BRANCH_REPLY": json.dumps([{"name": branch, "commit": {"sha": tip}, "protected": True}]),
                "COMPARE_REPLY": json.dumps({"status": comparison}),
            }
            script = next(step["run"] for step in DOCS["source"]["steps"] if step.get("id") == "source")
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", mock_api + script], env=fixture | env, capture_output=True
            )
            return result.returncode, dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_publish_source_requires_a_trusted_invocation_and_history(self) -> None:
        """Allow only canonical main dispatches/schedules and commits on protected release history."""
        for changes in (
            {"comparison": "diverged"},
            {"comparison": "behind"},
            {"branch": "topic"},
            {"branch": "r1.2.3-preview"},
            {"GITHUB_EVENT_NAME": "push"},
            {"GITHUB_REPOSITORY": "example/Megatron-Bridge"},
            {"GITHUB_REF": "refs/tags/v1.2.3"},
            {"GITHUB_REF": "refs/heads/r1.2.3"},
            {"DOCS_REF": "missing", "DOCS_VERSION": "preview"},
            {"DOCS_REF": "refs/tags/undocumented"},
            {"DOCS_REF": "undocumented"},
            {"DOCS_REF": "r1.2.3"},
            {"PUBLISH_LATEST": "false"},
            {"DOCS_REF": "v1.2.3", "COMMIT_REPLY": '{"sha":"malformed"}'},
            {"DOCS_REF": "v1.2.3", "EXPECTED_SHA": "b" * 40},
            {"BRANCH_REPLY": "[]"},
            {"DOCS_VERSION": "nested/version"},
        ):
            with self.subTest(changes=changes):
                rc, _ = self.run_source(**changes)
                self.assertNotEqual(rc, 0)
        for branch in ("main", "r1.2.3"):
            with self.subTest(branch=branch):
                self.assertEqual(self.run_source(branch=branch), (0, {"sha": "a" * 40, "version": ""}))
        self.assertEqual(self.run_source(DOCS_REF="b" * 40), (0, {"sha": "b" * 40, "version": ""}))

    def test_source_versions_and_dry_rehearsals(self) -> None:
        """Keep tagged, scheduled and user-supplied versions while allowing untrusted builds in dry runs."""
        for context, version in (
            ({"DOCS_REF": "v1.2.3", "EXPECTED_SHA": "a" * 40}, "1.2.3"),
            ({"DOCS_REF": "v0.2.0rc1"}, "0.2.0rc1"),
            ({"DOCS_REF": "26.04-alpha.rc1"}, "26.04-alpha.rc1"),
            ({"DOCS_REF": "refs/tags/v1.2.3", "DOCS_VERSION": "preview"}, "preview"),
            ({"DOCS_REF": "r1.2.3", "DOCS_VERSION": "1.2.3", "PUBLISH_LATEST": "false"}, "1.2.3"),
            ({"GITHUB_EVENT_NAME": "schedule", "DOCS_VERSION": "nightly", "PUBLISH_LATEST": "false"}, "nightly"),
            ({"GITHUB_REF": "refs/heads/topic", "DRY_RUN": "true", "comparison": "diverged"}, ""),
        ):
            with self.subTest(context=context):
                self.assertEqual(self.run_source(**context), (0, {"sha": "a" * 40, "version": version}))

    def test_dry_run_checks_a_nonempty_built_artifact(self) -> None:
        """Run the credential-free dry-run validation on missing, empty and populated indexes."""
        rehearsal = next(job for job in DOCS.values() if job.get("if") == "inputs.dry-run")
        self.assertIn("build-docs", rehearsal["needs"])
        self.assertEqual(rehearsal["permissions"], {})
        self.assertNotIn("environment", rehearsal)
        download = next(step for step in rehearsal["steps"] if "download-artifact@" in step.get("uses", ""))
        self.assertEqual(download["with"]["name"], "docs-html")
        script = next(step["run"] for step in rehearsal["steps"] if "run" in step)
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / download["with"]["path"] / "index.html"
            index.parent.mkdir(parents=True)
            for content in (None, "", "<html>Documentation</html>"):
                with self.subTest(content=content):
                    if content is not None:
                        index.write_text(content)
                    check = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=temp, capture_output=True)
                    self.assertEqual(check.returncode == 0, bool(content))

    def test_only_real_docs_publication_receives_credentials(self) -> None:
        """Bind the build to the validated SHA and keep dry runs outside the publishing environment."""
        build = DOCS["build-docs"]
        self.assertIn("source", build["needs"])
        self.assertEqual(build["with"]["ref"], "${{ needs.source.outputs.sha }}")
        self.assertNotIn("secrets", build)
        publish = DOCS["publish-docs"]
        self.assertIn("build-docs", publish["needs"])
        self.assertEqual(publish["environment"]["name"], "megatron-bridge-release")
        self.assertTrue(evaluate(publish["if"], {"inputs.dry-run": False}))
        self.assertFalse(evaluate(publish["if"], {"inputs.dry-run": True}))
        publisher = next(step for step in publish["steps"] if "/publish-docs@" in step.get("uses", ""))
        self.assertEqual(publisher["with"]["docs-version-override"], "${{ needs.source.outputs.version }}")
        self.assertEqual(publisher["with"]["s3-target-path"], "nemo/megatron-bridge")
        self.assertEqual(publisher["with"]["artifacts-name"], "docs-html")
        for name, job in DOCS.items():
            if name != "publish-docs":
                self.assertNotRegex(json.dumps(job), r"secrets\.(?:AWS_|AKAMAI_|S3_)")
        nightly = yaml.safe_load((ROOT / ".github/workflows/release-nightly-docs.yml").read_text())
        self.assertEqual(
            nightly["jobs"]["call-release-docs"]["with"],
            {
                "dry-run": False,
                "publish-as-latest": False,
                "docs-version-override": "nightly",
                "update-version-picker": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
