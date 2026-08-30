"""Paired statistical comparison of two multi-seed evaluation files."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from scipy import stats

from analysis.evaluate_multiseed import METRIC_KEYS, summary_statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Compare paired multi-seed results")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-label", default="L1")
    parser.add_argument("--candidate-label", default="Physics-L1")
    return parser.parse_args()


def _paired_test(baseline_values, candidate_values):
    baseline = np.asarray(baseline_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired values must be one-dimensional and have matching shapes")

    differences = candidate - baseline
    result = {
        "baseline": summary_statistics(baseline),
        "candidate": summary_statistics(candidate),
        "candidate_minus_baseline": summary_statistics(differences),
        "relative_change_percent": summary_statistics(
            100.0 * differences / np.maximum(np.abs(baseline), 1e-12)
        ),
        "paired_t_statistic": None,
        "paired_t_pvalue": None,
        "paired_t_note": None,
    }
    if baseline.size >= 2:
        if np.allclose(differences, 0.0):
            statistic, pvalue = 0.0, 1.0
        elif np.isclose(differences.std(ddof=1), 0.0):
            statistic = None
            pvalue = 0.0
            result["paired_t_note"] = (
                "All paired differences are the same non-zero value; "
                "the finite t statistic is undefined."
            )
        else:
            statistic, pvalue = stats.ttest_rel(candidate, baseline)
        result["paired_t_statistic"] = (
            float(statistic) if statistic is not None else None
        )
        result["paired_t_pvalue"] = float(pvalue)
    return result


def _runs_by_seed(result):
    runs = {int(run["training_seed"]): run for run in result["runs"]}
    if len(runs) != len(result["runs"]):
        raise ValueError("evaluation file contains duplicate training seeds")
    return runs


def compare_seed_metrics(baseline_result, candidate_result):
    baseline_runs = _runs_by_seed(baseline_result)
    candidate_runs = _runs_by_seed(candidate_result)
    common_seeds = sorted(set(baseline_runs) & set(candidate_runs))
    if len(common_seeds) < 2:
        raise ValueError("at least two matching seeds are required for a paired comparison")

    comparisons = {}
    for metric in METRIC_KEYS:
        comparisons[metric] = _paired_test(
            [baseline_runs[seed]["metrics"][metric] for seed in common_seeds],
            [candidate_runs[seed]["metrics"][metric] for seed in common_seeds],
        )
    return common_seeds, comparisons


def _mean_per_map(runs, seeds, metric):
    values = defaultdict(list)
    for seed in seeds:
        for record in runs[seed]["per_map"]:
            values[str(record["map_id"])].append(float(record[metric]))
    return {
        map_id: float(np.mean(map_values))
        for map_id, map_values in values.items()
        if len(map_values) == len(seeds)
    }


def compare_maps(baseline_result, candidate_result, seeds):
    baseline_runs = _runs_by_seed(baseline_result)
    candidate_runs = _runs_by_seed(candidate_result)
    comparisons = {}

    for metric in ("rmse", "mae"):
        baseline_maps = _mean_per_map(baseline_runs, seeds, metric)
        candidate_maps = _mean_per_map(candidate_runs, seeds, metric)
        map_ids = sorted(set(baseline_maps) & set(candidate_maps))
        if len(map_ids) < 2:
            raise ValueError("at least two matching test maps are required")
        baseline_values = np.asarray([baseline_maps[map_id] for map_id in map_ids])
        candidate_values = np.asarray([candidate_maps[map_id] for map_id in map_ids])
        comparison = _paired_test(baseline_values, candidate_values)
        differences = candidate_values - baseline_values
        if np.allclose(differences, 0.0):
            wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
        else:
            wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(
                candidate_values,
                baseline_values,
            )
        comparison.update(
            {
                "n_maps": len(map_ids),
                "wilcoxon_statistic": float(wilcoxon_statistic),
                "wilcoxon_pvalue": float(wilcoxon_pvalue),
            }
        )
        comparisons[metric] = comparison
    return comparisons


def main():
    args = parse_args()
    with open(args.baseline, "r") as handle:
        baseline_result = json.load(handle)
    with open(args.candidate, "r") as handle:
        candidate_result = json.load(handle)

    baseline_split = int(baseline_result["protocol"]["split_seed"])
    candidate_split = int(candidate_result["protocol"]["split_seed"])
    if baseline_split != candidate_split:
        raise ValueError(
            f"split seeds differ: baseline={baseline_split}, candidate={candidate_split}"
        )
    if not baseline_result["protocol"].get("full_test_set", False):
        raise ValueError("baseline result is not a formal full-test-set evaluation")
    if not candidate_result["protocol"].get("full_test_set", False):
        raise ValueError("candidate result is not a formal full-test-set evaluation")

    seeds, seed_comparisons = compare_seed_metrics(baseline_result, candidate_result)
    result = {
        "protocol": {
            "baseline": args.baseline,
            "candidate": args.candidate,
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "paired_training_seeds": seeds,
            "split_seed": baseline_split,
            "difference_direction": "candidate minus baseline; negative favors candidate for error metrics",
        },
        "seed_level": seed_comparisons,
        "map_level_after_seed_averaging": compare_maps(
            baseline_result,
            candidate_result,
            seeds,
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["seed_level"], indent=2), flush=True)
    print(f"Saved paired comparison to {output_path}", flush=True)


if __name__ == "__main__":
    main()
