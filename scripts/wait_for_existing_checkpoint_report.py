"""Wait for formal existing-checkpoint artifacts and render their report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Render report after existing evaluations")
    parser.add_argument("--output-root", default="outputs/revision_pipeline")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    root = Path(args.output_root)
    if not root.is_absolute():
        root = _PROJECT_ROOT / root

    status_path = root / "status.json"
    required_stage = "compare_existing_radiounet_xai"
    print(f"Waiting for stage {required_stage}", flush=True)
    while True:
        if status_path.is_file():
            status = _load_json(status_path)
            if status.get("status") == "failed":
                raise RuntimeError(
                    "Revision pipeline failed before the existing-checkpoint report: "
                    + str(status.get("error"))
                )
            stage_status = status.get("stages", {}).get(required_stage, {}).get("status")
            if stage_status == "completed":
                break
        time.sleep(args.poll_seconds)

    prediction = root / "prediction"
    xai = root / "xai"
    comparisons = root / "comparisons"
    reports = root / "reports"
    inputs = (
        prediction / "restormer_baseline_3seed.json",
        prediction / "restormer_physics_l1_3seed.json",
        comparisons / "restormer_baseline_vs_physics_prediction.json",
        xai / "restormer_baseline_3seed_ig.json",
        xai / "restormer_physics_l1_3seed_ig.json",
        comparisons / "restormer_baseline_vs_physics_xai.json",
        prediction / "radiounet_baseline_seed42.json",
        prediction / "radiounet_physics_l1_seed42.json",
        xai / "radiounet_baseline_seed42_ig.json",
        xai / "radiounet_physics_l1_seed42_ig.json",
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required report input not found: {path}")
        _load_json(path)

    command = [
        sys.executable,
        "analysis/build_existing_checkpoint_report.py",
        "--baseline-prediction", str(inputs[0]),
        "--physics-prediction", str(inputs[1]),
        "--prediction-comparison", str(inputs[2]),
        "--baseline-xai", str(inputs[3]),
        "--physics-xai", str(inputs[4]),
        "--xai-comparison", str(inputs[5]),
        "--radiounet-baseline-prediction", str(inputs[6]),
        "--radiounet-physics-prediction", str(inputs[7]),
        "--radiounet-baseline-xai", str(inputs[8]),
        "--radiounet-physics-xai", str(inputs[9]),
        "--output", str(reports / "existing_checkpoint_evaluation.md"),
        "--json-output", str(reports / "existing_checkpoint_evaluation.json"),
    ]
    subprocess.run(command, cwd=_PROJECT_ROOT, check=True)
    print("Existing-checkpoint formal report completed", flush=True)


if __name__ == "__main__":
    main()
