"""Requirement-by-requirement completion audit for the revision pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import torch


EXPECTED_SEEDS = [42, 123, 2016]
EXPECTED_XAI_METHODS = ["integrated_gradients"]
EXPECTED_XAI_SAMPLES = 50
EXPECTED_IG_STEPS = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Audit revision-pipeline completion")
    parser.add_argument("--output-root", default="outputs/revision_pipeline")
    parser.add_argument(
        "--json-output",
        default="outputs/revision_pipeline/reports/completion_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="outputs/revision_pipeline/reports/completion_audit.md",
    )
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as handle:
        handle.write(text)
    os.replace(temporary, path)


def validate_warmstart_lineage(warm_records, checkpoint_provenance, project_root):
    """Cross-check weights-only warm starts against formally evaluated baselines."""
    baseline_hashes = {
        int(record["training_seed"]): record.get("sha256")
        for record in checkpoint_provenance
        if record.get("arm") == "restormer_baseline"
    }
    records_by_seed = {
        int(record.get("seed", -1)): record
        for record in warm_records
    }
    errors = {}
    for seed in EXPECTED_SEEDS:
        record = records_by_seed.get(seed)
        if record is None:
            errors[f"seed_{seed}"] = "missing manifest record"
            continue
        destination = Path(record.get("destination", ""))
        if not destination.is_absolute():
            destination = Path(project_root) / destination
        try:
            if record.get("source_sha256") != baseline_hashes.get(seed):
                raise ValueError(
                    "warm-start source hash differs from the formally evaluated baseline"
                )
            actual_destination_hash = _sha256(destination)
            recorded_destination_hash = record.get("destination_sha256")
            if (
                recorded_destination_hash is not None
                and recorded_destination_hash != actual_destination_hash
            ):
                raise ValueError("warm-start destination hash differs from the manifest")
            warm = _torch_load(destination)
            if int(warm.get("epoch", -2)) != -1:
                raise ValueError(f"warm-start epoch={warm.get('epoch')}, expected -1")
            if int(warm.get("training_seed", -1)) != seed:
                raise ValueError(
                    f"warm-start training_seed={warm.get('training_seed')}, expected {seed}"
                )
            if int(warm.get("split_seed", -1)) != 42:
                raise ValueError(
                    f"warm-start split_seed={warm.get('split_seed')}, expected 42"
                )
            if warm.get("warm_start_source_sha256") != baseline_hashes.get(seed):
                raise ValueError("embedded source hash differs from the evaluated baseline")
            if "optimizer_state_dict" in warm or "scheduler_state_dict" in warm:
                raise ValueError("weights-only warm start contains optimizer/scheduler state")
            del warm
        except Exception as exc:
            errors[str(destination)] = str(exc)
    return errors


class Auditor:
    def __init__(self):
        self.checks = []

    def check(self, requirement, condition, evidence, detail=None):
        record = {
            "requirement": requirement,
            "passed": bool(condition),
            "evidence": evidence,
        }
        if detail is not None:
            record["detail"] = detail
        self.checks.append(record)
        return bool(condition)

    @property
    def passed(self):
        return bool(self.checks) and all(record["passed"] for record in self.checks)


def _prediction_paths(root):
    directory = root / "prediction"
    return {
        "baseline": directory / "restormer_baseline_3seed.json",
        "physics": directory / "restormer_physics_l1_3seed.json",
        "continuation": directory / "restormer_l1_continuation_3seed.json",
        "radio_baseline": directory / "radiounet_baseline_seed42.json",
        "radio_physics": directory / "radiounet_physics_l1_seed42.json",
    }


def _xai_paths(root):
    directory = root / "xai"
    return {
        "baseline": directory / "restormer_baseline_3seed_ig.json",
        "physics": directory / "restormer_physics_l1_3seed_ig.json",
        "continuation": directory / "restormer_l1_continuation_3seed_ig.json",
        "radio_baseline": directory / "radiounet_baseline_seed42_ig.json",
        "radio_physics": directory / "radiounet_physics_l1_seed42_ig.json",
    }


def audit_prediction_result(auditor, label, path, expected_seeds):
    try:
        result = _load_json(path)
    except Exception as exc:
        auditor.check(f"prediction:{label}:readable", False, str(path), str(exc))
        return None
    protocol = result.get("protocol", {})
    runs = result.get("runs", [])
    summary = result.get("summary", {})
    auditor.check(
        f"prediction:{label}:formal_full_test_set",
        protocol.get("full_test_set") is True and int(protocol.get("n_test_samples", 0)) > 0,
        str(path),
        {"full_test_set": protocol.get("full_test_set"), "n": protocol.get("n_test_samples")},
    )
    auditor.check(
        f"prediction:{label}:training_seeds",
        protocol.get("training_seeds") == expected_seeds
        and [int(run.get("training_seed")) for run in runs] == expected_seeds,
        str(path),
    )
    auditor.check(
        f"prediction:{label}:metrics_and_per_map_records",
        all(
            key in summary
            for key in ("global_rmse", "global_mae", "mean_sample_rmse", "mean_sample_mae")
        )
        and all(run.get("per_map") for run in runs),
        str(path),
    )
    return result


def audit_xai_result(auditor, label, path, expected_seeds):
    try:
        result = _load_json(path)
    except Exception as exc:
        auditor.check(f"xai:{label}:readable", False, str(path), str(exc))
        return None
    protocol = result.get("protocol", {})
    runs = result.get("runs", [])
    auditor.check(
        f"xai:{label}:fixed_formal_protocol",
        protocol.get("training_seeds") == expected_seeds
        and protocol.get("methods") == EXPECTED_XAI_METHODS
        and int(protocol.get("n_explanation_samples", 0)) == EXPECTED_XAI_SAMPLES
        and int(protocol.get("settings", {}).get("ig_steps", 0)) == EXPECTED_IG_STEPS,
        str(path),
        {
            "seeds": protocol.get("training_seeds"),
            "methods": protocol.get("methods"),
            "samples": protocol.get("n_explanation_samples"),
            "ig_steps": protocol.get("settings", {}).get("ig_steps"),
        },
    )
    identities = protocol.get("sample_identities", [])
    auditor.check(
        f"xai:{label}:raw_sample_records",
        len(runs) == len(expected_seeds)
        and len(identities) == EXPECTED_XAI_SAMPLES
        and all(len(run.get("samples", [])) == EXPECTED_XAI_SAMPLES for run in runs),
        str(path),
    )
    auditor.check(
        f"xai:{label}:seed_uncertainty_summary",
        bool(result.get("across_seed_summary")),
        str(path),
    )
    return result


def main():
    args = parse_args()
    root = Path(args.output_root)
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    auditor = Auditor()

    status_path = root / "status.json"
    try:
        status = _load_json(status_path)
    except Exception as exc:
        status = {}
        auditor.check("pipeline_status_readable", False, str(status_path), str(exc))
    else:
        auditor.check(
            "pipeline_completed",
            status.get("status") == "completed",
            str(status_path),
            status.get("status"),
        )
        stages = status.get("stages", {})
        auditor.check(
            "every_recorded_stage_completed",
            bool(stages) and all(stage.get("status") == "completed" for stage in stages.values()),
            str(status_path),
            {name: stage.get("status") for name, stage in stages.items()},
        )
        recorded_sources = status.get("metadata", {}).get("source_sha256", {})
        source_matches = {}
        for relative, recorded_hash in recorded_sources.items():
            path = _PROJECT_ROOT / relative
            source_matches[relative] = path.is_file() and _sha256(path) == recorded_hash
        auditor.check(
            "recorded_protocol_sources_unchanged",
            bool(source_matches) and all(source_matches.values()),
            str(status_path),
            {name: matches for name, matches in source_matches.items() if not matches},
        )
        stage_outputs = [
            Path(output)
            for stage in stages.values()
            for output in stage.get("outputs", [])
        ]
        missing_outputs = [
            str(path)
            for path in stage_outputs
            if not ((_PROJECT_ROOT / path) if not path.is_absolute() else path).exists()
        ]
        auditor.check(
            "all_stage_outputs_exist",
            bool(stage_outputs) and not missing_outputs,
            str(status_path),
            missing_outputs,
        )

    prediction_results = {}
    for label, path in _prediction_paths(root).items():
        seeds = [42] if label.startswith("radio_") else EXPECTED_SEEDS
        prediction_results[label] = audit_prediction_result(auditor, label, path, seeds)

    xai_results = {}
    for label, path in _xai_paths(root).items():
        seeds = [42] if label.startswith("radio_") else EXPECTED_SEEDS
        xai_results[label] = audit_xai_result(auditor, label, path, seeds)

    restormer_xai = [
        xai_results.get("baseline"),
        xai_results.get("physics"),
        xai_results.get("continuation"),
    ]
    protocols_available = all(result is not None for result in restormer_xai)
    same_identities = protocols_available and len(
        {
            json.dumps(result["protocol"]["sample_identities"], sort_keys=True)
            for result in restormer_xai
        }
    ) == 1
    auditor.check(
        "three_restormer_arms_use_identical_xai_samples",
        same_identities,
        [str(path) for path in _xai_paths(root).values()],
    )

    comparison_dir = root / "comparisons"
    required_comparisons = (
        "restormer_baseline_vs_physics_prediction.json",
        "restormer_baseline_vs_l1_continuation_prediction.json",
        "restormer_l1_continuation_vs_physics_prediction.json",
        "restormer_baseline_vs_physics_xai.json",
        "restormer_baseline_vs_l1_continuation_xai.json",
        "restormer_l1_continuation_vs_physics_xai.json",
        "radiounet_baseline_vs_physics_xai.json",
    )
    comparison_errors = {}
    for name in required_comparisons:
        path = comparison_dir / name
        try:
            result = _load_json(path)
            if not result.get("protocol"):
                raise ValueError("missing protocol")
        except Exception as exc:
            comparison_errors[name] = str(exc)
    auditor.check(
        "all_registered_paired_comparisons_readable",
        not comparison_errors,
        str(comparison_dir),
        comparison_errors,
    )

    existing_report_markdown = root / "reports/existing_checkpoint_evaluation.md"
    existing_report_json = root / "reports/existing_checkpoint_evaluation.json"
    try:
        existing_report = _load_json(existing_report_json)
    except Exception as exc:
        existing_report = {}
        existing_report_error = str(exc)
    else:
        existing_report_error = None
    auditor.check(
        "existing_checkpoints_formally_recorded_before_final_decision",
        existing_report_markdown.is_file()
        and existing_report.get("protocol", {}).get("full_test_set") is True
        and bool(existing_report.get("restormer_prediction"))
        and bool(existing_report.get("restormer_physical_alignment")),
        [str(existing_report_markdown), str(existing_report_json)],
        existing_report_error,
    )

    expected_checkpoint_units = [
        ("restormer_baseline", seed) for seed in EXPECTED_SEEDS
    ] + [
        ("restormer_physics_l1", seed) for seed in EXPECTED_SEEDS
    ] + [
        ("radiounet_baseline", 42),
        ("radiounet_physics_l1", 42),
    ]
    checkpoint_provenance = existing_report.get("provenance", {}).get(
        "checkpoints", []
    )
    actual_checkpoint_units = [
        (record.get("arm"), record.get("training_seed"))
        for record in checkpoint_provenance
    ]
    checkpoint_hash_errors = {}
    for record in checkpoint_provenance:
        path = Path(record.get("path", ""))
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        try:
            current_hash = _sha256(path)
            if current_hash != record.get("sha256"):
                raise ValueError("current SHA-256 differs from the recorded value")
        except Exception as exc:
            checkpoint_hash_errors[str(path)] = str(exc)
    auditor.check(
        "existing_checkpoint_hash_provenance_matches",
        actual_checkpoint_units == expected_checkpoint_units
        and not checkpoint_hash_errors
        and all(
            len(str(record.get("sha256", ""))) == 64
            and int(record.get("size_bytes", 0)) > 0
            for record in checkpoint_provenance
        ),
        str(existing_report_json),
        {
            "units": actual_checkpoint_units,
            "hash_errors": checkpoint_hash_errors,
        },
    )

    auxiliary_provenance = existing_report.get("provenance", {})
    auxiliary_records = list(auxiliary_provenance.get("configs", [])) + list(
        auxiliary_provenance.get("evaluation_artifacts", {}).values()
    )
    auxiliary_hash_errors = {}
    for record in auxiliary_records:
        path = Path(record.get("path", ""))
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        try:
            if _sha256(path) != record.get("sha256"):
                raise ValueError("current SHA-256 differs from the recorded value")
        except Exception as exc:
            auxiliary_hash_errors[str(path)] = str(exc)
    auditor.check(
        "evaluation_inputs_and_configs_have_matching_provenance",
        len(auxiliary_provenance.get("configs", [])) == 4
        and len(auxiliary_provenance.get("evaluation_artifacts", {})) == 10
        and not auxiliary_hash_errors,
        str(existing_report_json),
        auxiliary_hash_errors,
    )

    stages = status.get("stages", {})
    record_stage = stages.get("record_existing_checkpoint_evaluation", {})
    first_control_stage = stages.get("train_l1_continuation_seed_42", {})
    try:
        record_finished_at = datetime.fromisoformat(record_stage["finished_at"])
        control_started_at = datetime.fromisoformat(first_control_stage["started_at"])
        existing_record_precedes_control = record_finished_at <= control_started_at
        ordering_detail = {
            "record_finished_at": record_stage["finished_at"],
            "control_started_at": first_control_stage["started_at"],
        }
    except Exception as exc:
        existing_record_precedes_control = False
        ordering_detail = str(exc)
    auditor.check(
        "existing_checkpoint_record_completed_before_continuation_training",
        record_stage.get("status") == "completed"
        and first_control_stage.get("status") == "completed"
        and existing_record_precedes_control,
        str(status_path),
        ordering_detail,
    )

    warm_manifest_path = root / "l1_continuation_warmstarts.json"
    try:
        warm_manifest = _load_json(warm_manifest_path)
        warm_records = warm_manifest.get("checkpoints", [])
    except Exception as exc:
        warm_records = []
        warm_error = str(exc)
    else:
        warm_error = None
    auditor.check(
        "three_weights_only_warmstarts_recorded",
        [record.get("seed") for record in warm_records] == EXPECTED_SEEDS
        and all(record.get("source_sha256") for record in warm_records),
        str(warm_manifest_path),
        warm_error,
    )
    warm_lineage_errors = validate_warmstart_lineage(
        warm_records,
        checkpoint_provenance,
        _PROJECT_ROOT,
    )
    control_command_errors = {}
    for seed in EXPECTED_SEEDS:
        stage_name = f"train_l1_continuation_seed_{seed}"
        command = status.get("stages", {}).get(stage_name, {}).get("command", "")
        expected_resume = f"outputs/l1_continuation/warm_starts/seed_{seed}.pth"
        if expected_resume not in command or "--full-resume" in command:
            control_command_errors[stage_name] = command
    auditor.check(
        "l1_continuation_uses_evaluated_baseline_weights_with_fresh_optimizers",
        not warm_lineage_errors and not control_command_errors,
        [str(warm_manifest_path), str(status_path)],
        {
            "lineage_errors": warm_lineage_errors,
            "command_errors": control_command_errors,
        },
    )

    control_errors = {}
    for seed in EXPECTED_SEEDS:
        path = _PROJECT_ROOT / "outputs/l1_continuation/checkpoints" / f"seed_{seed}" / "final_model.pth"
        try:
            checkpoint = _torch_load(path)
            if int(checkpoint.get("epoch", -1)) != 49:
                raise ValueError(f"epoch={checkpoint.get('epoch')}")
            if int(checkpoint.get("training_seed", -1)) != seed:
                raise ValueError(f"training_seed={checkpoint.get('training_seed')}")
            if int(checkpoint.get("split_seed", -1)) != 42:
                raise ValueError(f"split_seed={checkpoint.get('split_seed')}")
            if "optimizer_state_dict" not in checkpoint:
                raise ValueError("missing optimizer state in completed checkpoint")
            del checkpoint
        except Exception as exc:
            control_errors[str(path)] = str(exc)
    auditor.check(
        "three_l1_continuation_epoch50_checkpoints_valid",
        not control_errors,
        "outputs/l1_continuation/checkpoints/seed_*/final_model.pth",
        control_errors,
    )

    decision_markdown = root / "reports/follow_up_decision.md"
    decision_json = root / "reports/follow_up_decision.json"
    try:
        decision = _load_json(decision_json)
    except Exception as exc:
        decision = {}
        decision_error = str(exc)
    else:
        decision_error = None
    auditor.check(
        "follow_up_decision_recorded",
        decision_markdown.is_file()
        and bool(decision.get("radiounet", {}).get("recommendation"))
        and bool(decision.get("second_dataset", {}).get("recommendation")),
        [str(decision_markdown), str(decision_json)],
        decision_error,
    )

    result = {
        "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": auditor.passed,
        "checks": auditor.checks,
    }
    _atomic_write(json.dumps(result, indent=2) + "\n", args.json_output)
    markdown = [
        "# Revision pipeline completion audit",
        "",
        f"Overall: **{'PASS' if auditor.passed else 'FAIL'}**",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for record in auditor.checks:
        markdown.append(
            f"| {record['requirement']} | {'PASS' if record['passed'] else 'FAIL'} | "
            f"`{record['evidence']}` |"
        )
    markdown.append("")
    _atomic_write("\n".join(markdown), args.markdown_output)
    print(json.dumps({"passed": auditor.passed, "n_checks": len(auditor.checks)}, indent=2))
    if not auditor.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
