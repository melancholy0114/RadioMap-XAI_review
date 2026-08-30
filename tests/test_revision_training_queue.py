"""Tests for the fail-fast major-revision training queue."""

from argparse import Namespace
import unittest

from scripts.run_revision_training_queue import (
    build_queue_steps,
    validate_args,
    validate_protocol,
)


class RevisionTrainingQueueTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "restormer_seeds": [123, 2016],
            "split_seed": 42,
            "wait_pid": 12345,
            "poll_seconds": 30.0,
            "nproc_per_node": 4,
            "gpus": "0,1,2,3",
        }
        values.update(overrides)
        return Namespace(**values)

    def test_current_configs_match_controlled_protocol(self):
        validate_protocol(42)

    def test_completed_seed_42_cannot_be_queued_again(self):
        with self.assertRaisesRegex(ValueError, "Do not include seed 42"):
            validate_args(self._args(restormer_seeds=[42, 123, 2016]))

    def test_queue_order_and_seed_isolation(self):
        steps = build_queue_steps(
            python_executable="/env/bin/python",
            restormer_seeds=[123, 2016],
            split_seed=42,
            gpus="0,1,2,3",
            nproc_per_node=4,
        )

        self.assertEqual(len(steps), 4)
        self.assertIn("RadioUNet", steps[0].name)
        self.assertIn("configs/config_radiounet_ablation_50ep.yaml", steps[0].command)
        self.assertIn("--full-resume", steps[0].command)
        self.assertNotIn("--seed", steps[0].command)

        baseline = steps[1].command
        self.assertIn("configs/config.yaml", baseline)
        seed_index = baseline.index("--seeds")
        self.assertEqual(baseline[seed_index + 1 : seed_index + 3], ("123", "2016"))
        self.assertNotIn("42", baseline[seed_index + 1 : seed_index + 3])

        physics20 = steps[2].command
        self.assertIn("configs/config_ablation.yaml", physics20)
        self.assertIn("outputs/checkpoints/seed_{seed}/best_model.pth", physics20)
        self.assertNotIn("--full-resume", physics20)

        physics50 = steps[3].command
        self.assertIn("configs/config_ablation_50ep.yaml", physics50)
        self.assertIn(
            "outputs/improved_checkpoints/seed_{seed}/final_model.pth",
            physics50,
        )
        self.assertIn("--full-resume", physics50)

    def test_world_size_requires_enough_gpu_ids(self):
        with self.assertRaisesRegex(ValueError, "GPU IDs"):
            validate_args(self._args(gpus="0,1", nproc_per_node=4))


if __name__ == "__main__":
    unittest.main()
