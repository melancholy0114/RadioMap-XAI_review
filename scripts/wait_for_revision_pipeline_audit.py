"""Wait for the revision pipeline and run its final completion audit.

This supervisor is intentionally outside the guarded training/evaluation source
set.  It can therefore be added while the already-running pipeline is waiting
without changing that pipeline's registered protocol fingerprints.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the completion audit after the revision pipeline exits"
    )
    parser.add_argument("--output-root", default="outputs/revision_pipeline")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _record(path, status, **details):
    payload = {
        "status": status,
        "updated_at": now_iso(),
        **details,
    }
    _atomic_json_dump(payload, path)


def ensure_existing_checkpoint_report(root, pipeline_status):
    reports = root / "reports"
    markdown = reports / "existing_checkpoint_evaluation.md"
    machine_readable = reports / "existing_checkpoint_evaluation.json"
    if markdown.is_file() and machine_readable.is_file():
        _load_json(machine_readable)
        return [markdown, machine_readable]

    stage = pipeline_status.get("stages", {}).get(
        "record_existing_checkpoint_evaluation", {}
    )
    if stage.get("status") != "completed":
        raise RuntimeError(
            "Pipeline is marked complete but the existing-checkpoint report "
            "stage is not complete"
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/wait_for_existing_checkpoint_report.py",
            "--output-root",
            os.fspath(root),
            "--poll-seconds",
            "1",
        ],
        cwd=_PROJECT_ROOT,
        check=True,
    )
    _load_json(machine_readable)
    return [markdown, machine_readable]


def run_completion_audit(root):
    reports = root / "reports"
    json_output = reports / "completion_audit.json"
    markdown_output = reports / "completion_audit.md"
    subprocess.run(
        [
            sys.executable,
            "analysis/audit_revision_pipeline_outputs.py",
            "--output-root",
            os.fspath(root),
            "--json-output",
            os.fspath(json_output),
            "--markdown-output",
            os.fspath(markdown_output),
        ],
        cwd=_PROJECT_ROOT,
        check=True,
    )
    result = _load_json(json_output)
    if result.get("passed") is not True:
        raise RuntimeError("Completion audit returned without a passing result")
    return result, [json_output, markdown_output]


def main():
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    root = Path(args.output_root)
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    watcher_status = reports / "completion_audit_watcher.json"

    lock_handle = open(root / "completion_audit_watcher.lock", "a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another completion-audit watcher is already running") from exc

    status_path = root / "status.json"
    _record(
        watcher_status,
        "waiting_for_pipeline",
        pipeline_status_path=os.fspath(status_path),
        watcher_pid=os.getpid(),
    )
    print(f"Waiting for completed pipeline status at {status_path}", flush=True)

    try:
        while True:
            if status_path.is_file():
                pipeline_status = _load_json(status_path)
                state = pipeline_status.get("status")
                if state == "completed":
                    break
                if state == "failed":
                    raise RuntimeError(
                        "Revision pipeline failed: "
                        + str(pipeline_status.get("error", "unknown error"))
                    )
            time.sleep(args.poll_seconds)

        _record(
            watcher_status,
            "auditing",
            pipeline_status_path=os.fspath(status_path),
            watcher_pid=os.getpid(),
        )
        existing_report_outputs = ensure_existing_checkpoint_report(
            root, pipeline_status
        )
        audit, audit_outputs = run_completion_audit(root)
        _record(
            watcher_status,
            "completed",
            pipeline_status_path=os.fspath(status_path),
            watcher_pid=os.getpid(),
            n_checks=len(audit.get("checks", [])),
            outputs=[
                os.fspath(path)
                for path in (*existing_report_outputs, *audit_outputs)
            ],
        )
        print("Revision pipeline completion audit passed", flush=True)
    except Exception as exc:
        _record(
            watcher_status,
            "failed",
            pipeline_status_path=os.fspath(status_path),
            watcher_pid=os.getpid(),
            error=str(exc),
        )
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
