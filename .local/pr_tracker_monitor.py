#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO = "NVIDIA-NeMo/Megatron-Bridge"
LOCAL_DIR = Path(__file__).resolve().parent
TRACKER_PATH = LOCAL_DIR / "pr-tracker.md"
STATE_PATH = LOCAL_DIR / "pr-tracker-state.json"
LOG_PATH = LOCAL_DIR / "pr-tracker-monitor.log"
PID_PATH = LOCAL_DIR / "pr-tracker-monitor.pid"

FAILED_CONCLUSIONS = {
    "ACTION_REQUIRED",
    "CANCELLED",
    "FAILURE",
    "STARTUP_FAILURE",
    "STALE",
    "TIMED_OUT",
}
PENDING_STATUSES = {"EXPECTED", "IN_PROGRESS", "PENDING", "QUEUED", "REQUESTED", "WAITING"}
SUCCESS_CONCLUSIONS = {"NEUTRAL", "SKIPPED", "SUCCESS"}

FLAKY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"OutOfMemoryError|CUDA out of memory|\bOOM\b",
        r"NCCL error|NCCL timeout|ncclSystemError|ncclInternalError",
        r"not enough GPUs|No GPUs available|srun: error",
        r"ConnectionError|Connection reset by peer",
        r"Timeout|timed out waiting",
        r"DGX Cloud|infrastructure|pool error",
        r"Signal 11 \(SIGSEGV\)",
        r"LocalEntryNotFoundError|couldn't connect to 'https://huggingface\.co'",
        r"job was cancelled|job was canceled|killed externally",
        r"container pull failure|node allocation failure",
        r"nvidia-resiliency-ext",
    )
]

RUN_ID_RE = re.compile(r"/actions/runs/(\d+)")
PR_NUM_RE = re.compile(r"#(\d+)")


@dataclass
class TrackedPR:
    number: int
    title: str
    status: str
    last_checked: str
    notes: str


def run_gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GH_PAGER"] = "cat"
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"gh {' '.join(args)} failed")
    return result


def gh_json(args: list[str], check: bool = True):
    result = run_gh(args, check=check)
    output = result.stdout.strip()
    if not output:
        return None
    return json.loads(output)


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def escape_cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|").strip()


def parse_tracker() -> list[TrackedPR]:
    if not TRACKER_PATH.exists():
        raise FileNotFoundError(f"Tracker file not found: {TRACKER_PATH}")

    rows: list[TrackedPR] = []
    for raw_line in TRACKER_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith("|----") or line.startswith("| PR "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        match = PR_NUM_RE.search(cells[0])
        if not match:
            continue
        rows.append(
            TrackedPR(
                number=int(match.group(1)),
                title=cells[1],
                status=cells[2],
                last_checked=cells[3],
                notes=cells[4],
            )
        )
    return rows


def write_tracker(rows: list[TrackedPR]) -> None:
    lines = [
        "# Tracked PRs",
        "",
        "| PR | Title | Status | Last Checked | Notes |",
        "|----|-------|--------|--------------|-------|",
    ]
    for row in rows:
        pr_cell = f"[#{row.number}](https://github.com/{REPO}/pull/{row.number})"
        lines.append(
            f"| {pr_cell} | {escape_cell(row.title)} | {escape_cell(row.status)} | "
            f"{escape_cell(row.last_checked)} | {escape_cell(row.notes)} |"
        )
    TRACKER_PATH.write_text("\n".join(lines) + "\n")


def load_state() -> dict[str, object]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_state(state: dict[str, object]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def get_rollup_field(check: dict, name: str, default: str = "") -> str:
    value = check.get(name)
    if value is None:
        return default
    return str(value)


def parse_run_id(details_url: str) -> str | None:
    match = RUN_ID_RE.search(details_url)
    if not match:
        return None
    return match.group(1)


def normalize_title(title: str) -> str:
    title = title.strip()
    if title.startswith("[") and "](" in title:
        parts = title.split("](", 1)
        if len(parts) == 2:
            return parts[0].lstrip("[")
    return title


def short_list(items: list[str], limit: int = 3) -> str:
    if not items:
        return ""
    if len(items) <= limit:
        return ", ".join(items)
    shown = ", ".join(items[:limit])
    return f"{shown}, +{len(items) - limit} more"


def extract_log_summary(log_text: str) -> str:
    summary_patterns = [
        r"AssertionError:.*",
        r"ModuleNotFoundError:.*",
        r"ImportError:.*",
        r"ValueError:.*",
        r"RuntimeError:.*",
        r"TypeError:.*",
        r"AttributeError:.*",
        r"KeyError:.*",
        r"NameError:.*",
        r"FAILED .*$",
        r"short test summary info.*",
        r"Process completed with exit code \d+",
    ]
    for pattern in summary_patterns:
        match = re.search(pattern, log_text, re.MULTILINE)
        if match:
            return match.group(0).strip()
    for line in reversed(log_text.splitlines()):
        cleaned = line.strip()
        if cleaned:
            return cleaned[:200]
    return "check logs"


def is_flaky_log(log_text: str) -> tuple[bool, str]:
    for pattern in FLAKY_PATTERNS:
        match = pattern.search(log_text)
        if match:
            return True, match.group(0)
    return False, ""


def get_latest_main_status(workflow_name: str) -> tuple[str, str]:
    if not workflow_name:
        return "", ""
    runs = gh_json(
        [
            "run",
            "list",
            "--repo",
            REPO,
            "--branch",
            "main",
            "--workflow",
            workflow_name,
            "--limit",
            "3",
            "--json",
            "databaseId,status,conclusion,createdAt,name",
        ],
        check=False,
    )
    if not runs:
        return "", ""
    for run in runs:
        status = str(run.get("status", ""))
        conclusion = str(run.get("conclusion", ""))
        if status == "completed":
            return status, conclusion
    latest = runs[0]
    return str(latest.get("status", "")), str(latest.get("conclusion", ""))


def should_skip_retrigger(pr_state: dict[str, object], failed_run_ids: list[str]) -> bool:
    last_retriggered = pr_state.get("last_retriggered_run_ids", [])
    if not isinstance(last_retriggered, list):
        return False
    return sorted(last_retriggered) == sorted(failed_run_ids)


def retrigger_ci(pr_number: int, head_sha: str) -> bool:
    if not head_sha:
        return False
    result = run_gh(
        [
            "pr",
            "comment",
            str(pr_number),
            "--repo",
            REPO,
            "--body",
            f"/ok to test {head_sha}",
        ],
        check=False,
    )
    return result.returncode == 0


def classify_failures(
    pr_number: int,
    head_sha: str,
    failed_checks: list[dict],
    pr_state: dict[str, object],
) -> tuple[str, str, dict[str, object]]:
    failed_names: list[str] = []
    flaky_checks: list[str] = []
    main_broken_checks: list[str] = []
    real_checks: list[str] = []
    failure_details: list[str] = []
    failed_run_ids: list[str] = []

    for check in failed_checks:
        name = get_rollup_field(check, "name", "unknown-check")
        workflow_name = get_rollup_field(check, "workflowName")
        details_url = get_rollup_field(check, "detailsUrl")
        failed_names.append(name)
        run_id = parse_run_id(details_url)
        if run_id:
            failed_run_ids.append(run_id)

        log_text = ""
        if run_id:
            result = run_gh(
                [
                    "run",
                    "view",
                    run_id,
                    "--repo",
                    REPO,
                    "--log-failed",
                ],
                check=False,
            )
            log_text = result.stdout

        is_flaky, flaky_reason = is_flaky_log(log_text)
        if is_flaky:
            flaky_checks.append(name)
            failure_details.append(f"{name}: {flaky_reason}")
            continue

        main_status, main_conclusion = get_latest_main_status(workflow_name)
        if main_status == "completed" and main_conclusion not in {"", "success"}:
            main_broken_checks.append(name)
            failure_details.append(f"{name}: main {workflow_name} is {main_conclusion}")
            continue

        real_checks.append(name)
        summary = extract_log_summary(log_text) if log_text else "check logs"
        failure_details.append(f"{name}: {summary}")

    next_state = dict(pr_state)
    if not failed_run_ids:
        next_state.pop("last_retriggered_run_ids", None)

    if real_checks:
        next_state["last_retriggered_run_ids"] = []
        return "needs-fix", f"failed: {short_list(failure_details, limit=2)}", next_state

    if flaky_checks:
        if should_skip_retrigger(pr_state, failed_run_ids):
            return "ci-failed-flaky", f"flaky: {short_list(failure_details, limit=2)}", next_state
        retriggered = retrigger_ci(pr_number, head_sha)
        next_state["last_retriggered_run_ids"] = failed_run_ids
        note = f"re-triggered flaky failure: {short_list(failure_details, limit=2)}"
        if not retriggered:
            return "ci-failed-flaky", note, next_state
        return "ci-pending", note, next_state

    if main_broken_checks:
        if should_skip_retrigger(pr_state, failed_run_ids):
            return "ci-failed-main", f"main also failing: {short_list(main_broken_checks)}", next_state
        retriggered = retrigger_ci(pr_number, head_sha)
        next_state["last_retriggered_run_ids"] = failed_run_ids
        note = f"main also broken on {short_list(main_broken_checks)}"
        if retriggered:
            note = f"{note}; re-triggered CI"
        return "ci-failed-main", note, next_state

    next_state["last_retriggered_run_ids"] = []
    return "needs-fix", f"failed: {short_list(failed_names)}", next_state


def classify_pr(row: TrackedPR, state: dict[str, object]) -> tuple[TrackedPR, dict[str, object]]:
    view = gh_json(
        [
            "pr",
            "view",
            str(row.number),
            "--repo",
            REPO,
            "--json",
            "number,title,state,headRefName,headRefOid,mergeable,reviewDecision,statusCheckRollup",
        ]
    )

    title = str(view.get("title") or normalize_title(row.title))
    last_checked = now_string()
    pr_state = dict(state.get(str(row.number), {}))
    head_sha = str(view.get("headRefOid", ""))
    status = row.status
    notes = row.notes

    state_value = str(view.get("state", ""))
    mergeable = str(view.get("mergeable", ""))
    review_decision = str(view.get("reviewDecision", ""))
    checks = view.get("statusCheckRollup") or []

    if state_value == "MERGED":
        pr_state["last_retriggered_run_ids"] = []
        return TrackedPR(row.number, title, "merged", last_checked, "merged"), pr_state

    if mergeable == "CONFLICTING":
        pr_state["last_retriggered_run_ids"] = []
        return TrackedPR(row.number, title, "needs-rebase", last_checked, "merge conflicts"), pr_state

    failed_checks = []
    pending_checks = []
    for check in checks:
        status_value = get_rollup_field(check, "status").upper()
        conclusion = get_rollup_field(check, "conclusion").upper()
        if conclusion in FAILED_CONCLUSIONS:
            failed_checks.append(check)
        elif status_value in PENDING_STATUSES or not conclusion:
            pending_checks.append(check)

    if failed_checks:
        status, notes, pr_state = classify_failures(row.number, head_sha, failed_checks, pr_state)
    elif pending_checks:
        pending_names = [get_rollup_field(check, "name", "pending-check") for check in pending_checks]
        status = "ci-pending"
        notes = f"waiting on {short_list(pending_names)}"
        pr_state["last_retriggered_run_ids"] = []
    else:
        status = "ci-green"
        notes = "all checks passed, approved" if review_decision == "APPROVED" else "all checks passed, review required"
        pr_state["last_retriggered_run_ids"] = []

    return TrackedPR(row.number, title, status, last_checked, notes), pr_state


def append_log(rows: list[TrackedPR]) -> None:
    timestamp = now_string()
    summary = ", ".join(f"#{row.number}:{row.status}" for row in rows)
    with LOG_PATH.open("a") as log_file:
        log_file.write(f"{timestamp} {summary}\n")


def sync_once() -> int:
    rows = parse_tracker()
    state = load_state()
    updated_rows: list[TrackedPR] = []
    updated_state: dict[str, object] = dict(state)

    for row in rows:
        updated_row, pr_state = classify_pr(row, state)
        updated_rows.append(updated_row)
        updated_state[str(row.number)] = pr_state

    write_tracker(updated_rows)
    save_state(updated_state)
    append_log(updated_rows)

    for row in updated_rows:
        print(f"#{row.number} {row.status} - {row.notes}")
    return 0


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_pidfile() -> None:
    if PID_PATH.exists():
        try:
            existing_pid = int(PID_PATH.read_text().strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0 and pid_is_running(existing_pid):
            raise RuntimeError(f"Monitor already running with pid {existing_pid}")
    PID_PATH.write_text(f"{os.getpid()}\n")


def cleanup_pidfile(*_: object) -> None:
    if PID_PATH.exists():
        try:
            current = PID_PATH.read_text().strip()
        except OSError:
            current = ""
        if current == str(os.getpid()):
            PID_PATH.unlink(missing_ok=True)


def loop(interval_seconds: int) -> int:
    acquire_pidfile()
    signal.signal(signal.SIGTERM, cleanup_pidfile)
    signal.signal(signal.SIGINT, cleanup_pidfile)
    try:
        while True:
            try:
                sync_once()
            except Exception as exc:
                message = f"{now_string()} ERROR {exc}"
                print(message, file=sys.stderr)
                with LOG_PATH.open("a") as log_file:
                    log_file.write(message + "\n")
            time.sleep(interval_seconds)
    finally:
        cleanup_pidfile()


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor tracked last-mile PRs.")
    parser.add_argument("--loop", action="store_true", help="Continuously monitor PRs.")
    parser.add_argument("--interval", type=int, default=1800, help="Loop interval in seconds.")
    args = parser.parse_args()

    if args.loop:
        return loop(args.interval)
    return sync_once()


if __name__ == "__main__":
    raise SystemExit(main())
