"""Regression tests for controlled multi-seed experiments."""

from argparse import Namespace
import os
import unittest

from analysis.compare_multiseed import compare_seed_metrics
from analysis.evaluate_multiseed import resolve_checkpoint_paths, summary_statistics
from scripts.run_multi_seed import build_command
from utils import (
    configure_seeded_run,
    get_split_seed,
    seed_metadata,
    validate_seed_metadata,
)


class ReproducibilityTests(unittest.TestCase):
    @staticmethod
    def _legacy_config():
        return {
            "data": {},
            "training": {"seed": 42},
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
                "physics_checkpoint_dir": "outputs/improved_checkpoints",
            },
        }

    def test_training_seed_override_keeps_legacy_split_fixed(self):
        original = self._legacy_config()
        resolved = configure_seeded_run(
            original,
            training_seed=123,
            isolate_outputs=True,
        )

        self.assertEqual(resolved["training"]["seed"], 123)
        self.assertEqual(get_split_seed(resolved), 42)
        self.assertNotIn("split_seed", original["data"])
        for key, path in resolved["output"].items():
            if key.endswith("_dir"):
                self.assertEqual(os.path.basename(path), "seed_123")

    def test_split_seed_can_be_overridden_independently(self):
        resolved = configure_seeded_run(
            self._legacy_config(),
            training_seed=7,
            split_seed=99,
        )
        self.assertEqual(seed_metadata(resolved), {"training_seed": 7, "split_seed": 99})

        validate_seed_metadata({"training_seed": 7, "split_seed": 99}, resolved)
        validate_seed_metadata({}, resolved)
        with self.assertRaisesRegex(ValueError, "training_seed"):
            validate_seed_metadata({"training_seed": 8}, resolved)

    def test_checkpoint_template_resolves_one_path_per_seed(self):
        paths = resolve_checkpoint_paths(
            [42, 123, 2026],
            template="outputs/checkpoints/seed_{seed}/best_model.pth",
        )
        self.assertEqual(paths[1], "outputs/checkpoints/seed_123/best_model.pth")
        with self.assertRaisesRegex(ValueError, "contain exactly one path"):
            resolve_checkpoint_paths([42, 123], checkpoints=["only_one.pth"])
        with self.assertRaisesRegex(ValueError, "distinct checkpoint"):
            resolve_checkpoint_paths(
                [42, 123],
                checkpoints=["same.pth", "same.pth"],
            )

    def test_summary_uses_sample_std_and_student_t_interval(self):
        summary = summary_statistics([1.0, 2.0, 3.0])
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["sample_std"], 1.0)
        self.assertAlmostEqual(summary["ci95_low"], -0.4841377, places=5)
        self.assertAlmostEqual(summary["ci95_high"], 4.4841377, places=5)

    def test_launcher_passes_seed_without_mixing_backbone_logic(self):
        args = Namespace(
            trainer="physics",
            config="configs/config_radiounet_ablation.yaml",
            nproc_per_node=4,
            split_seed=42,
            subset=1.0,
            smoke_test_batches=None,
            log_interval=None,
            resume_template="outputs/radiounet_c/l1/checkpoints/seed_{seed}/best_model.pth",
            full_resume=False,
        )
        command = build_command(args, 123)
        self.assertIn("training/train_physics.py", command)
        self.assertEqual(command[command.index("--seed") + 1], "123")
        self.assertEqual(command[command.index("--split-seed") + 1], "42")
        self.assertIn(
            "outputs/radiounet_c/l1/checkpoints/seed_123/best_model.pth",
            command,
        )

    def test_seed_level_comparison_matches_seeds_not_file_order(self):
        def make_result(order, offset):
            return {
                "runs": [
                    {
                        "training_seed": seed,
                        "metrics": {
                            "global_rmse": seed / 10000 + offset,
                            "global_mae": seed / 20000 + offset,
                            "mean_sample_rmse": seed / 15000 + offset,
                            "mean_sample_mae": seed / 25000 + offset,
                        },
                    }
                    for seed in order
                ]
            }

        seeds, comparison = compare_seed_metrics(
            make_result([123, 42, 2026], 0.0),
            make_result([2026, 123, 42], -0.001),
        )
        self.assertEqual(seeds, [42, 123, 2026])
        self.assertAlmostEqual(
            comparison["global_rmse"]["candidate_minus_baseline"]["mean"],
            -0.001,
        )


if __name__ == "__main__":
    unittest.main()
