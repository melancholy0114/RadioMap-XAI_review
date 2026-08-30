import hashlib
from pathlib import Path
import tempfile
import unittest

from analysis.build_existing_checkpoint_report import (
    checkpoint_provenance,
    file_provenance,
)


class ExistingCheckpointReportTests(unittest.TestCase):
    def test_file_provenance_records_size_and_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            payload = b"immutable checkpoint bytes"
            path.write_bytes(payload)
            record = file_provenance(path)
            self.assertEqual(record["size_bytes"], len(payload))
            self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())

    def test_checkpoint_provenance_retains_selection_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model.pth"
            path.write_bytes(b"weights")
            evaluation = {
                "runs": [
                    {
                        "training_seed": 123,
                        "checkpoint": str(path),
                        "checkpoint_epoch": 37,
                        "best_val_loss": 0.007,
                        "model_name": "restormer",
                    }
                ]
            }
            record = checkpoint_provenance("restormer_baseline", evaluation)[0]
            self.assertEqual(record["arm"], "restormer_baseline")
            self.assertEqual(record["training_seed"], 123)
            self.assertEqual(record["checkpoint_epoch"], 37)
            self.assertEqual(record["best_val_loss"], 0.007)


if __name__ == "__main__":
    unittest.main()
