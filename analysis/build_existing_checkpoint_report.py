"""Render the formal existing-checkpoint evaluation as Markdown and JSON."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


DB_SCALE = 139.0
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Build existing-checkpoint report")
    parser.add_argument("--baseline-prediction", required=True)
    parser.add_argument("--physics-prediction", required=True)
    parser.add_argument("--prediction-comparison", required=True)
    parser.add_argument("--baseline-xai", required=True)
    parser.add_argument("--physics-xai", required=True)
    parser.add_argument("--xai-comparison", required=True)
    parser.add_argument("--radiounet-baseline-prediction", required=True)
    parser.add_argument("--radiounet-physics-prediction", required=True)
    parser.add_argument("--radiounet-baseline-xai", required=True)
    parser.add_argument("--radiounet-physics-xai", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    return parser.parse_args()


def _load(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path):
    resolved = _resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Provenance input not found: {resolved}")
    return {
        "path": os.fspath(path),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def checkpoint_provenance(arm, evaluation):
    records = []
    for run in evaluation["runs"]:
        record = file_provenance(run["checkpoint"])
        record.update(
            {
                "arm": arm,
                "training_seed": int(run["training_seed"]),
                "model_name": run["model_name"],
                "checkpoint_epoch": int(run["checkpoint_epoch"]),
                "best_val_loss": run.get("best_val_loss"),
            }
        )
        records.append(record)
    return records


def _atomic_write(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _fmt(summary, digits=6, scale=1.0):
    mean = summary["mean"] * scale
    std = summary.get("sample_std")
    low = summary.get("ci95_low")
    high = summary.get("ci95_high")
    if std is None:
        return f"{mean:.{digits}f} (single seed)"
    return (
        f"{mean:.{digits}f} ± {std * scale:.{digits}f}; "
        f"95% CI [{low * scale:.{digits}f}, {high * scale:.{digits}f}]"
    )


def _xai_mean(result, prior, metric):
    return result["across_seed_summary"]["integrated_gradients"][prior][metric]["mean"]


def build_report(args):
    baseline_prediction = _load(args.baseline_prediction)
    physics_prediction = _load(args.physics_prediction)
    prediction_comparison = _load(args.prediction_comparison)
    baseline_xai = _load(args.baseline_xai)
    physics_xai = _load(args.physics_xai)
    xai_comparison = _load(args.xai_comparison)
    radio_baseline_prediction = _load(args.radiounet_baseline_prediction)
    radio_physics_prediction = _load(args.radiounet_physics_prediction)
    radio_baseline_xai = _load(args.radiounet_baseline_xai)
    radio_physics_xai = _load(args.radiounet_physics_xai)

    checkpoint_records = []
    for arm, evaluation in (
        ("restormer_baseline", baseline_prediction),
        ("restormer_physics_l1", physics_prediction),
        ("radiounet_baseline", radio_baseline_prediction),
        ("radiounet_physics_l1", radio_physics_prediction),
    ):
        checkpoint_records.extend(checkpoint_provenance(arm, evaluation))

    evaluation_input_names = (
        "baseline_prediction",
        "physics_prediction",
        "prediction_comparison",
        "baseline_xai",
        "physics_xai",
        "xai_comparison",
        "radiounet_baseline_prediction",
        "radiounet_physics_prediction",
        "radiounet_baseline_xai",
        "radiounet_physics_xai",
    )
    evaluation_artifacts = {
        name: file_provenance(getattr(args, name))
        for name in evaluation_input_names
    }
    config_paths = list(
        dict.fromkeys(
            evaluation["protocol"]["config"]
            for evaluation in (
                baseline_prediction,
                physics_prediction,
                radio_baseline_prediction,
                radio_physics_prediction,
            )
        )
    )
    config_records = [file_provenance(path) for path in config_paths]

    baseline_rmse = baseline_prediction["summary"]["global_rmse"]
    physics_rmse = physics_prediction["summary"]["global_rmse"]
    baseline_mae = baseline_prediction["summary"]["global_mae"]
    physics_mae = physics_prediction["summary"]["global_mae"]
    rmse_comparison = prediction_comparison["seed_level"]["global_rmse"]
    mae_comparison = prediction_comparison["seed_level"]["global_mae"]
    map_rmse = prediction_comparison["map_level_after_seed_averaging"]["rmse"]
    map_mae = prediction_comparison["map_level_after_seed_averaging"]["mae"]

    xai_rows = []
    for prior in ("los", "obstruction", "directional"):
        for metric in ("iou", "pearson_corr", "spearman_corr"):
            difference = xai_comparison["seed_level"]["integrated_gradients"][prior][metric]
            xai_rows.append(
                {
                    "prior": prior,
                    "metric": metric,
                    "baseline": _xai_mean(baseline_xai, prior, metric),
                    "physics": _xai_mean(physics_xai, prior, metric),
                    "difference": difference["candidate_minus_baseline"],
                    "paired_t_pvalue": difference.get("paired_t_pvalue"),
                }
            )

    radio_baseline_rmse = radio_baseline_prediction["summary"]["global_rmse"]
    radio_physics_rmse = radio_physics_prediction["summary"]["global_rmse"]
    radio_rows = []
    for prior in ("los", "obstruction", "directional"):
        radio_rows.append(
            {
                "prior": prior,
                "baseline_iou": _xai_mean(radio_baseline_xai, prior, "iou"),
                "physics_iou": _xai_mean(radio_physics_xai, prior, "iou"),
            }
        )

    evidence = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "restormer_training_seeds": baseline_prediction["protocol"]["training_seeds"],
            "split_seed": baseline_prediction["protocol"]["split_seed"],
            "full_test_set": baseline_prediction["protocol"]["full_test_set"],
            "n_test_samples": baseline_prediction["protocol"]["n_test_samples"],
            "xai_sample_identities": baseline_xai["protocol"]["sample_identities"],
            "xai_settings": baseline_xai["protocol"]["settings"],
            "db_scale": DB_SCALE,
        },
        "restormer_prediction": {
            "baseline_global_rmse": baseline_rmse,
            "physics_global_rmse": physics_rmse,
            "baseline_global_mae": baseline_mae,
            "physics_global_mae": physics_mae,
            "rmse_comparison": rmse_comparison,
            "mae_comparison": mae_comparison,
            "map_rmse_comparison": map_rmse,
            "map_mae_comparison": map_mae,
        },
        "restormer_physical_alignment": xai_rows,
        "radiounet": {
            "baseline_global_rmse": radio_baseline_rmse,
            "physics_global_rmse": radio_physics_rmse,
            "physical_alignment_iou": radio_rows,
            "scope": "single training seed (42)",
        },
        "provenance": {
            "checkpoint_selection": "minimum validation loss (best_model.pth)",
            "checkpoints": checkpoint_records,
            "configs": config_records,
            "evaluation_artifacts": evaluation_artifacts,
        },
        "source_files": vars(args),
    }

    lines = [
        "# Formal evaluation of existing checkpoints",
        "",
        f"Generated: {evidence['generated_at']}",
        "",
        "This report was generated before interpreting the matched-budget L1 "
        "continuation. Checkpoint selection follows minimum validation loss.",
        "",
        "## Evaluated checkpoint provenance",
        "",
        "Full SHA-256 values and input-artifact hashes are retained in the "
        "machine-readable report.",
        "",
        "| Arm | Seed | Best epoch (0-based) | Best val loss | Checkpoint SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for record in checkpoint_records:
        best_val_loss = record["best_val_loss"]
        formatted_loss = "—" if best_val_loss is None else f"{best_val_loss:.8f}"
        lines.append(
            f"| {record['arm']} | {record['training_seed']} | "
            f"{record['checkpoint_epoch']} | {formatted_loss} | "
            f"`{record['sha256'][:16]}…` |"
        )
    lines.extend(
        [
        "",
        "## Restormer prediction (seeds 42, 123, 2016)",
        "",
        "| Outcome | Baseline L1 | Physics-L1 | Physics − Baseline |",
        "|---|---:|---:|---:|",
        f"| Global RMSE (normalized) | {_fmt(baseline_rmse)} | {_fmt(physics_rmse)} | {_fmt(rmse_comparison['candidate_minus_baseline'])} |",
        f"| Global RMSE (dB, ×139) | {_fmt(baseline_rmse, 3, DB_SCALE)} | {_fmt(physics_rmse, 3, DB_SCALE)} | {_fmt(rmse_comparison['candidate_minus_baseline'], 3, DB_SCALE)} |",
        f"| Global MAE (normalized) | {_fmt(baseline_mae)} | {_fmt(physics_mae)} | {_fmt(mae_comparison['candidate_minus_baseline'])} |",
        "",
        f"Seed-paired RMSE t-test p-value: `{rmse_comparison.get('paired_t_pvalue')}`.  ",
        f"Map-level RMSE paired t-test p-value: `{map_rmse.get('paired_t_pvalue')}`; "
        f"Wilcoxon p-value: `{map_rmse.get('wilcoxon_pvalue')}`.  ",
        f"Map-level MAE paired t-test p-value: `{map_mae.get('paired_t_pvalue')}`; "
        f"Wilcoxon p-value: `{map_mae.get('wilcoxon_pvalue')}`.",
        "",
        "## Restormer Integrated-Gradients alignment",
        "",
        "| Prior | Metric | Baseline | Physics-L1 | Difference | Seed-paired p |",
        "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in xai_rows:
        lines.append(
            f"| {row['prior']} | {row['metric']} | {row['baseline']:.6f} | "
            f"{row['physics']:.6f} | {row['difference']['mean']:.6f} | "
            f"{row['paired_t_pvalue']} |"
        )
    lines.extend(
        [
            "",
            "## RadioUNet supporting case (seed 42 only)",
            "",
            f"- Baseline normalized RMSE: `{radio_baseline_rmse['mean']:.6f}` "
            f"(`{radio_baseline_rmse['mean'] * DB_SCALE:.3f}` dB).",
            f"- Physics-L1 normalized RMSE: `{radio_physics_rmse['mean']:.6f}` "
            f"(`{radio_physics_rmse['mean'] * DB_SCALE:.3f}` dB).",
            "",
            "| Prior | Baseline IoU | Physics-L1 IoU |",
            "|---|---:|---:|",
        ]
    )
    for row in radio_rows:
        lines.append(
            f"| {row['prior']} | {row['baseline_iou']:.6f} | {row['physics_iou']:.6f} |"
        )
    lines.extend(
        [
            "",
            "RadioUNet has one training seed at this stage; its result cannot be "
            "reported with training-seed uncertainty.",
            "",
        ]
    )
    return evidence, "\n".join(lines)


def main():
    args = parse_args()
    evidence, markdown = build_report(args)
    _atomic_write(json.dumps(evidence, indent=2) + "\n", args.json_output)
    _atomic_write(markdown, args.output)
    print(f"Saved existing-checkpoint report to {args.output}", flush=True)
    print(f"Saved report evidence to {args.json_output}", flush=True)


if __name__ == "__main__":
    main()
