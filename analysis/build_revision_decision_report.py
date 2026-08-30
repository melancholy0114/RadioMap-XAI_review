"""Build an evidence-linked decision report after the automated evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build revision experiment decision report")
    parser.add_argument("--baseline-prediction", required=True)
    parser.add_argument("--physics-prediction", required=True)
    parser.add_argument("--continuation-prediction", required=True)
    parser.add_argument("--baseline-vs-physics", required=True)
    parser.add_argument("--baseline-vs-continuation", required=True)
    parser.add_argument("--continuation-vs-physics", required=True)
    parser.add_argument("--baseline-xai", required=True)
    parser.add_argument("--physics-xai", required=True)
    parser.add_argument("--continuation-xai", required=True)
    parser.add_argument("--baseline-vs-physics-xai", required=True)
    parser.add_argument("--baseline-vs-continuation-xai", required=True)
    parser.add_argument("--continuation-vs-physics-xai", required=True)
    parser.add_argument("--radiounet-baseline-prediction", required=True)
    parser.add_argument("--radiounet-physics-prediction", required=True)
    parser.add_argument("--radiounet-baseline-xai", required=True)
    parser.add_argument("--radiounet-physics-xai", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--independent-dataset-available", action="store_true")
    return parser.parse_args()


def _load(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _atomic_write_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _atomic_json_dump(payload, path):
    _atomic_write_text(json.dumps(payload, indent=2) + "\n", path)


def _prediction_summary(result, metric="global_rmse"):
    return result["summary"][metric]


def _prediction_difference(comparison, metric="global_rmse"):
    return comparison["seed_level"][metric]["candidate_minus_baseline"]


def _xai_mean(result, prior, metric, method="integrated_gradients"):
    return result["across_seed_summary"][method][prior][metric]["mean"]


def _xai_difference(comparison, prior, metric, method="integrated_gradients"):
    return comparison["seed_level"][method][prior][metric]["candidate_minus_baseline"]


def _ci_excludes_zero(summary):
    low = summary.get("ci95_low")
    high = summary.get("ci95_high")
    return low is not None and high is not None and (high < 0 or low > 0)


def _fmt_summary(summary, digits=6):
    mean = summary["mean"]
    std = summary.get("sample_std")
    low = summary.get("ci95_low")
    high = summary.get("ci95_high")
    if std is None:
        return f"{mean:.{digits}f} (single seed)"
    return (
        f"{mean:.{digits}f} ± {std:.{digits}f}; "
        f"95% CI [{low:.{digits}f}, {high:.{digits}f}]"
    )


def build_report(args):
    baseline_prediction = _load(args.baseline_prediction)
    physics_prediction = _load(args.physics_prediction)
    continuation_prediction = _load(args.continuation_prediction)
    baseline_vs_physics = _load(args.baseline_vs_physics)
    baseline_vs_continuation = _load(args.baseline_vs_continuation)
    continuation_vs_physics = _load(args.continuation_vs_physics)
    baseline_xai = _load(args.baseline_xai)
    physics_xai = _load(args.physics_xai)
    continuation_xai = _load(args.continuation_xai)
    baseline_vs_physics_xai = _load(args.baseline_vs_physics_xai)
    baseline_vs_continuation_xai = _load(args.baseline_vs_continuation_xai)
    continuation_vs_physics_xai = _load(args.continuation_vs_physics_xai)
    radio_baseline_prediction = _load(args.radiounet_baseline_prediction)
    radio_physics_prediction = _load(args.radiounet_physics_prediction)
    radio_baseline_xai = _load(args.radiounet_baseline_xai)
    radio_physics_xai = _load(args.radiounet_physics_xai)

    baseline_rmse = _prediction_summary(baseline_prediction)
    continuation_rmse = _prediction_summary(continuation_prediction)
    physics_rmse = _prediction_summary(physics_prediction)
    physics_minus_baseline = _prediction_difference(baseline_vs_physics)
    continuation_minus_baseline = _prediction_difference(baseline_vs_continuation)
    physics_minus_continuation = _prediction_difference(continuation_vs_physics)

    radio_baseline_rmse = _prediction_summary(radio_baseline_prediction)["mean"]
    radio_physics_rmse = _prediction_summary(radio_physics_prediction)["mean"]
    radio_delta = radio_physics_rmse - radio_baseline_rmse
    restormer_delta = physics_minus_baseline["mean"]
    prediction_direction_consistent = (
        (radio_delta < 0 and restormer_delta < 0)
        or (radio_delta > 0 and restormer_delta > 0)
        or (radio_delta == 0 and restormer_delta == 0)
    )

    priors = ("los", "obstruction", "directional")
    xai_rows = []
    for prior in priors:
        for metric in ("iou", "pearson_corr", "spearman_corr"):
            xai_rows.append(
                {
                    "prior": prior,
                    "metric": metric,
                    "baseline": _xai_mean(baseline_xai, prior, metric),
                    "continuation": _xai_mean(continuation_xai, prior, metric),
                    "physics": _xai_mean(physics_xai, prior, metric),
                    "physics_minus_baseline": _xai_difference(
                        baseline_vs_physics_xai, prior, metric
                    ),
                    "continuation_minus_baseline": _xai_difference(
                        baseline_vs_continuation_xai, prior, metric
                    ),
                    "physics_minus_continuation": _xai_difference(
                        continuation_vs_physics_xai, prior, metric
                    ),
                }
            )

    restormer_hierarchy = sorted(
        priors,
        key=lambda prior: _xai_mean(physics_xai, prior, "iou"),
        reverse=True,
    )
    radio_hierarchy = sorted(
        priors,
        key=lambda prior: _xai_mean(radio_physics_xai, prior, "iou"),
        reverse=True,
    )
    hierarchy_consistent = restormer_hierarchy == radio_hierarchy

    if not prediction_direction_consistent or not hierarchy_consistent:
        radio_decision = "strongly_recommended"
        radio_reason = (
            "The single-seed RadioUNet result does not reproduce every primary "
            "Restormer direction/hierarchy; two additional seeds are needed to "
            "separate backbone effects from seed noise."
        )
    else:
        radio_decision = "recommended_if_cross_backbone_claim_is_retained"
        radio_reason = (
            "The seed-42 RadioUNet direction is consistent, but a single seed has "
            "no training-seed uncertainty. Add seeds 123 and 2016 for a strong "
            "cross-backbone claim, or explicitly present RadioUNet as a one-seed "
            "supporting case."
        )

    if args.independent_dataset_available:
        dataset_decision = "run_registered_cross_dataset_experiment"
        dataset_reason = (
            "An independent dataset is available; pre-register one controlled "
            "Baseline/Physics-L1 matrix before inspecting its outcomes."
        )
    else:
        dataset_decision = "narrow_claim_or_acquire_independent_dataset"
        dataset_reason = (
            "Only RadioMapSeer and its DPM/IRT target variants are locally available. "
            "IRT2/IRT4 can provide a cross-simulator stress test but are not a literal "
            "second dataset."
        )

    evidence = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prediction": {
            "baseline_rmse": baseline_rmse,
            "l1_continuation_rmse": continuation_rmse,
            "physics_l1_rmse": physics_rmse,
            "physics_minus_baseline": physics_minus_baseline,
            "l1_continuation_minus_baseline": continuation_minus_baseline,
            "physics_minus_l1_continuation": physics_minus_continuation,
            "physics_improvement_vs_baseline_ci_excludes_zero": (
                _ci_excludes_zero(physics_minus_baseline)
                and physics_minus_baseline["mean"] < 0
            ),
            "physics_improvement_vs_continuation_ci_excludes_zero": (
                _ci_excludes_zero(physics_minus_continuation)
                and physics_minus_continuation["mean"] < 0
            ),
        },
        "physical_alignment": xai_rows,
        "radiounet": {
            "baseline_global_rmse": radio_baseline_rmse,
            "physics_global_rmse": radio_physics_rmse,
            "physics_minus_baseline": radio_delta,
            "prediction_direction_consistent_with_restormer": prediction_direction_consistent,
            "restormer_physics_iou_hierarchy": restormer_hierarchy,
            "radiounet_physics_iou_hierarchy": radio_hierarchy,
            "prior_hierarchy_consistent": hierarchy_consistent,
            "recommendation": radio_decision,
            "reason": radio_reason,
        },
        "second_dataset": {
            "independent_dataset_available": bool(args.independent_dataset_available),
            "recommendation": dataset_decision,
            "reason": dataset_reason,
        },
        "source_files": vars(args),
    }

    lines = [
        "# Revision experiment decision report",
        "",
        f"Generated: {evidence['generated_at']}",
        "",
        "## Prediction evidence (normalized [0, 1])",
        "",
        f"- Baseline L1: {_fmt_summary(baseline_rmse)}",
        f"- Matched-budget L1 continuation: {_fmt_summary(continuation_rmse)}",
        f"- Physics-L1: {_fmt_summary(physics_rmse)}",
        f"- Physics-L1 − Baseline: {_fmt_summary(physics_minus_baseline)}",
        f"- L1 continuation − Baseline: {_fmt_summary(continuation_minus_baseline)}",
        f"- Physics-L1 − L1 continuation: {_fmt_summary(physics_minus_continuation)}",
        "",
        "## Integrated-Gradients physical alignment",
        "",
        "| Prior | Metric | Baseline | L1 continuation | Physics-L1 | Physics − continuation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in xai_rows:
        lines.append(
            f"| {row['prior']} | {row['metric']} | {row['baseline']:.6f} | "
            f"{row['continuation']:.6f} | {row['physics']:.6f} | "
            f"{row['physics_minus_continuation']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Follow-up decisions",
            "",
            f"- **RadioUNet multi-seed:** `{radio_decision}`. {radio_reason}",
            f"- **Second dataset:** `{dataset_decision}`. {dataset_reason}",
            "",
            "The report makes recommendations only; it does not automatically launch "
            "the optional RadioUNet or external-dataset jobs.",
            "",
        ]
    )
    return evidence, "\n".join(lines)


def main():
    args = parse_args()
    evidence, markdown = build_report(args)
    _atomic_json_dump(evidence, args.json_output)
    _atomic_write_text(markdown, args.output)
    print(f"Saved decision report to {args.output}", flush=True)
    print(f"Saved machine-readable decision evidence to {args.json_output}", flush=True)


if __name__ == "__main__":
    main()
