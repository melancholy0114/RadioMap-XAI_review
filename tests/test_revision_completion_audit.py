import json
from pathlib import Path
import tempfile
import unittest

import torch

from analysis.audit_revision_pipeline_outputs import (
    Auditor,
    audit_prediction_result,
    audit_xai_result,
    validate_warmstart_lineage,
)


class CompletionAuditTests(unittest.TestCase):
    def _write_json(self, root, name, payload):
        path = Path(root) / name
        with open(path, "w") as handle:
            json.dump(payload, handle)
        return path

    def test_formal_prediction_result_requires_full_test_and_per_map_records(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = {
                "protocol": {
                    "full_test_set": True,
                    "n_test_samples": 100,
                    "training_seeds": [42, 123, 2016],
                },
                "runs": [
                    {"training_seed": seed, "per_map": [{"map_id": "1"}]}
                    for seed in (42, 123, 2016)
                ],
                "summary": {
                    key: {"mean": 0.1}
                    for key in (
                        "global_rmse",
                        "global_mae",
                        "mean_sample_rmse",
                        "mean_sample_mae",
                    )
                },
            }
            valid_path = self._write_json(directory, "valid.json", valid)
            auditor = Auditor()
            result = audit_prediction_result(
                auditor, "valid", valid_path, [42, 123, 2016]
            )
            self.assertIsNotNone(result)
            self.assertTrue(auditor.passed)

            invalid = dict(valid)
            invalid["protocol"] = dict(valid["protocol"], full_test_set=False)
            invalid_path = self._write_json(directory, "invalid.json", invalid)
            invalid_auditor = Auditor()
            audit_prediction_result(
                invalid_auditor, "invalid", invalid_path, [42, 123, 2016]
            )
            self.assertFalse(invalid_auditor.passed)

    def test_xai_result_requires_registered_samples_and_raw_records(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = [
                {"dataset_index": index, "map_id": str(index), "tx_idx": 0}
                for index in range(50)
            ]
            valid = {
                "protocol": {
                    "training_seeds": [42, 123, 2016],
                    "methods": ["integrated_gradients"],
                    "n_explanation_samples": 50,
                    "settings": {"ig_steps": 50},
                    "sample_identities": samples,
                },
                "runs": [
                    {"training_seed": seed, "samples": samples}
                    for seed in (42, 123, 2016)
                ],
                "across_seed_summary": {"integrated_gradients": {"los": {}}},
            }
            valid_path = self._write_json(directory, "valid_xai.json", valid)
            auditor = Auditor()
            result = audit_xai_result(
                auditor, "valid", valid_path, [42, 123, 2016]
            )
            self.assertIsNotNone(result)
            self.assertTrue(auditor.passed)

            invalid = dict(valid)
            invalid["runs"] = [
                {"training_seed": seed, "samples": samples[:10]}
                for seed in (42, 123, 2016)
            ]
            invalid_path = self._write_json(directory, "invalid_xai.json", invalid)
            invalid_auditor = Auditor()
            audit_xai_result(
                invalid_auditor, "invalid", invalid_path, [42, 123, 2016]
            )
            self.assertFalse(invalid_auditor.passed)

    def test_warmstart_lineage_requires_matching_baseline_hash_and_no_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = []
            warm_records = []
            for seed in (42, 123, 2016):
                source_hash = f"{seed:064x}"
                provenance.append(
                    {
                        "arm": "restormer_baseline",
                        "training_seed": seed,
                        "sha256": source_hash,
                    }
                )
                destination = root / f"seed_{seed}.pth"
                torch.save(
                    {
                        "epoch": -1,
                        "training_seed": seed,
                        "split_seed": 42,
                        "model_state_dict": {"weight": torch.tensor([1.0])},
                        "warm_start_source_sha256": source_hash,
                    },
                    destination,
                )
                warm_records.append(
                    {
                        "seed": seed,
                        "source_sha256": source_hash,
                        "destination": str(destination),
                    }
                )
            self.assertEqual(
                validate_warmstart_lineage(warm_records, provenance, root), {}
            )

            warm = torch.load(
                warm_records[0]["destination"], map_location="cpu", weights_only=False
            )
            warm["optimizer_state_dict"] = {"state": {}}
            torch.save(warm, warm_records[0]["destination"])
            errors = validate_warmstart_lineage(warm_records, provenance, root)
            self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
