import tempfile
import inspect
from pathlib import Path
import unittest

import torch
import yaml

from analysis.compare_multiseed_xai import compare_results
from analysis.evaluate_multiseed_xai import (
    ALIGNMENT_METRICS,
    PRIORS,
    select_sample_indices,
    summarize_across_seeds,
    summarize_samples,
)
from scripts.prepare_l1_continuation_warmstarts import prepare_one
from scripts.run_post_training_revision_pipeline import (
    main as pipeline_main,
    source_fingerprints,
    validate_control_protocol,
)


def _alignment(value):
    return {
        "integrated_gradients": {
            prior: {metric: float(value) for metric in ALIGNMENT_METRICS}
            for prior in PRIORS
        }
    }


def _xai_result(seed_values):
    identities = [
        {"dataset_index": 3, "map_id": "11", "tx_idx": 1},
        {"dataset_index": 7, "map_id": "12", "tx_idx": 2},
    ]
    runs = []
    for seed, values in seed_values.items():
        samples = []
        for identity, value in zip(identities, values):
            samples.append(
                {
                    **identity,
                    "prediction_rmse": 0.1,
                    "prediction_mae": 0.05,
                    "alignment": _alignment(value),
                }
            )
        runs.append(
            {
                "training_seed": seed,
                "samples": samples,
                "summary": summarize_samples(samples, ["integrated_gradients"]),
            }
        )
    return {
        "protocol": {
            "split_seed": 42,
            "evaluation_seed": 42,
            "test_split": "test",
            "sample_identities": identities,
            "methods": ["integrated_gradients"],
            "settings": {
                "ig_steps": 2,
                "occlusion_window": 16,
                "occlusion_stride": 8,
                "top_k_percent": 20.0,
            },
            "scalar_attribution_target": "sum",
        },
        "runs": runs,
        "across_seed_summary": summarize_across_seeds(
            runs, ["integrated_gradients"]
        ),
    }


class WarmStartTests(unittest.TestCase):
    def test_weights_only_warm_start_discards_optimizer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            destination = root / "warm.pth"
            torch.save(
                {
                    "epoch": 48,
                    "model_state_dict": {"weight": torch.tensor([1.0])},
                    "optimizer_state_dict": {"state": {1: "not retained"}},
                    "scheduler_state_dict": {"last_epoch": 49},
                    "best_val_loss": 0.01,
                    "training_seed": 123,
                    "split_seed": 42,
                    "model_name": "restormer",
                },
                source,
            )

            record = prepare_one(source, destination, 123, 42)
            warm = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertEqual(record["status"], "created")
            self.assertEqual(warm["epoch"], -1)
            self.assertEqual(warm["training_seed"], 123)
            self.assertNotIn("optimizer_state_dict", warm)
            self.assertNotIn("scheduler_state_dict", warm)
            self.assertEqual(warm["warm_start_source_epoch"], 48)

            reused = prepare_one(source, destination, 123, 42, allow_existing=True)
            self.assertEqual(reused["status"], "reused")


class XaiEvaluationTests(unittest.TestCase):
    def test_fixed_sample_selection_is_deterministic(self):
        first = select_sample_indices(100, 12, 42)
        second = select_sample_indices(100, 12, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_paired_xai_comparison_preserves_seed_and_map_units(self):
        baseline = _xai_result({42: [0.1, 0.2], 123: [0.2, 0.3]})
        candidate = _xai_result({42: [0.2, 0.3], 123: [0.3, 0.4]})
        seeds, seed_level, map_level = compare_results(baseline, candidate)
        self.assertEqual(seeds, [42, 123])
        seed_diff = seed_level["integrated_gradients"]["los"]["iou"]
        self.assertAlmostEqual(seed_diff["candidate_minus_baseline"]["mean"], 0.1)
        map_diff = map_level["integrated_gradients"]["los"]["iou"]
        self.assertEqual(map_diff["n_pairs"], 2)
        self.assertAlmostEqual(map_diff["candidate_minus_baseline"]["mean"], 0.1)


class PipelineProtocolTests(unittest.TestCase):
    def test_control_config_matches_training_protocol(self):
        validate_control_protocol(42)
        with open("configs/config_l1_continuation.yaml", "r") as handle:
            control = yaml.safe_load(handle)
        self.assertEqual(control["loss"]["primary"], "l1")
        self.assertEqual(control["training"]["epochs"], 50)
        self.assertIn("l1_continuation", control["output"]["checkpoint_dir"])

    def test_all_fingerprinted_sources_exist(self):
        fingerprints = source_fingerprints()
        self.assertIn("training/train.py", fingerprints)
        self.assertIn("analysis/evaluate_multiseed_xai.py", fingerprints)
        self.assertIn("analysis/build_existing_checkpoint_report.py", fingerprints)
        self.assertTrue(all(len(value) == 64 for value in fingerprints.values()))

    def test_existing_checkpoint_report_precedes_control_training(self):
        source = inspect.getsource(pipeline_main)
        report_position = source.index('"record_existing_checkpoint_evaluation"')
        training_position = source.index('f"train_l1_continuation_seed_{seed}"')
        self.assertLess(report_position, training_position)


if __name__ == "__main__":
    unittest.main()
