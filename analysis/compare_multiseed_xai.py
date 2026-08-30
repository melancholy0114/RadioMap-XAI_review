"""Paired comparison of two fixed-sample multi-seed XAI evaluations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from scipy import stats

from analysis.evaluate_multiseed import summary_statistics
from analysis.evaluate_multiseed_xai import ALIGNMENT_METRICS, PRIORS


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multi-seed XAI results")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    return parser.parse_args()


def _atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def paired_summary(baseline_values, candidate_values):
    baseline = np.asarray(baseline_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    if baseline.ndim != 1 or baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError("paired values must be non-empty one-dimensional arrays")
    differences = candidate - baseline
    result = {
        "n_pairs": int(baseline.size),
        "baseline": summary_statistics(baseline),
        "candidate": summary_statistics(candidate),
        "candidate_minus_baseline": summary_statistics(differences),
        "paired_t_statistic": None,
        "paired_t_pvalue": None,
        "wilcoxon_statistic": None,
        "wilcoxon_pvalue": None,
    }
    if baseline.size >= 2:
        if np.allclose(differences, 0.0):
            t_statistic, t_pvalue = 0.0, 1.0
            w_statistic, w_pvalue = 0.0, 1.0
        else:
            if np.isclose(differences.std(ddof=1), 0.0):
                t_statistic, t_pvalue = None, 0.0
            else:
                t_statistic, t_pvalue = stats.ttest_rel(candidate, baseline)
            w_statistic, w_pvalue = stats.wilcoxon(candidate, baseline)
        result.update(
            {
                "paired_t_statistic": (
                    float(t_statistic) if t_statistic is not None else None
                ),
                "paired_t_pvalue": float(t_pvalue),
                "wilcoxon_statistic": float(w_statistic),
                "wilcoxon_pvalue": float(w_pvalue),
            }
        )
    return result


def _runs_by_seed(result):
    runs = {int(run["training_seed"]): run for run in result["runs"]}
    if len(runs) != len(result["runs"]):
        raise ValueError("evaluation contains duplicate training seeds")
    return runs


def _validate_protocols(baseline, candidate):
    baseline_protocol = baseline["protocol"]
    candidate_protocol = candidate["protocol"]
    exact_fields = (
        "split_seed",
        "evaluation_seed",
        "test_split",
        "sample_identities",
        "methods",
        "settings",
        "scalar_attribution_target",
    )
    mismatched = [
        field
        for field in exact_fields
        if baseline_protocol.get(field) != candidate_protocol.get(field)
    ]
    if mismatched:
        raise ValueError(
            "XAI evaluation protocols differ for: " + ", ".join(mismatched)
        )


def _samples_by_identity(run):
    samples = {
        (int(sample["dataset_index"]), str(sample["map_id"]), int(sample["tx_idx"])): sample
        for sample in run["samples"]
    }
    if len(samples) != len(run["samples"]):
        raise ValueError("run contains duplicate sample identities")
    return samples


def compare_results(baseline, candidate):
    _validate_protocols(baseline, candidate)
    baseline_runs = _runs_by_seed(baseline)
    candidate_runs = _runs_by_seed(candidate)
    seeds = sorted(set(baseline_runs) & set(candidate_runs))
    if not seeds:
        raise ValueError("evaluations have no matching training seeds")

    methods = baseline["protocol"]["methods"]
    seed_level = {}
    map_level = {}
    for method in methods:
        seed_level[method] = {}
        map_level[method] = {}
        for prior in PRIORS:
            seed_level[method][prior] = {}
            map_level[method][prior] = {}
            for metric in ALIGNMENT_METRICS:
                seed_level[method][prior][metric] = paired_summary(
                    [
                        baseline_runs[seed]["summary"][method][prior][metric]["mean"]
                        for seed in seeds
                    ],
                    [
                        candidate_runs[seed]["summary"][method][prior][metric]["mean"]
                        for seed in seeds
                    ],
                )

                baseline_maps = defaultdict(list)
                candidate_maps = defaultdict(list)
                for seed in seeds:
                    baseline_samples = _samples_by_identity(baseline_runs[seed])
                    candidate_samples = _samples_by_identity(candidate_runs[seed])
                    if set(baseline_samples) != set(candidate_samples):
                        raise ValueError(f"sample identities differ for seed {seed}")
                    for identity in baseline_samples:
                        map_id = identity[1]
                        baseline_maps[map_id].append(
                            baseline_samples[identity]["alignment"][method][prior][metric]
                        )
                        candidate_maps[map_id].append(
                            candidate_samples[identity]["alignment"][method][prior][metric]
                        )
                map_ids = sorted(set(baseline_maps) & set(candidate_maps))
                map_level[method][prior][metric] = {
                    "map_ids": map_ids,
                    **paired_summary(
                        [float(np.mean(baseline_maps[map_id])) for map_id in map_ids],
                        [float(np.mean(candidate_maps[map_id])) for map_id in map_ids],
                    ),
                }
    return seeds, seed_level, map_level


def main():
    args = parse_args()
    with open(args.baseline, "r") as handle:
        baseline = json.load(handle)
    with open(args.candidate, "r") as handle:
        candidate = json.load(handle)

    seeds, seed_level, map_level = compare_results(baseline, candidate)
    result = {
        "protocol": {
            "baseline": args.baseline,
            "candidate": args.candidate,
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "paired_training_seeds": seeds,
            "difference_direction": "candidate minus baseline",
            "seed_level_unit": "mean across fixed samples within each training seed",
            "map_level_unit": "mean across selected Tx samples and training seeds within each map",
        },
        "seed_level": seed_level,
        "map_level_after_seed_averaging": map_level,
    }
    _atomic_json_dump(result, args.output)
    print(f"Saved paired XAI comparison to {args.output}", flush=True)


if __name__ == "__main__":
    main()
