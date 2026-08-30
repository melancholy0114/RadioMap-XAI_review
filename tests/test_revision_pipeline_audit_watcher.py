import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.wait_for_revision_pipeline_audit import (
    ensure_existing_checkpoint_report,
    run_completion_audit,
)


class RevisionPipelineAuditWatcherTests(unittest.TestCase):
    def test_existing_report_is_reused_when_both_artifacts_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            (reports / "existing_checkpoint_evaluation.md").write_text("report\n")
            (reports / "existing_checkpoint_evaluation.json").write_text(
                json.dumps({"protocol": {"full_test_set": True}})
            )
            with mock.patch("subprocess.run") as run:
                outputs = ensure_existing_checkpoint_report(root, {})
            self.assertEqual(len(outputs), 2)
            run.assert_not_called()

    def test_missing_report_requires_completed_existing_evaluation_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "stage is not complete"):
                ensure_existing_checkpoint_report(
                    root,
                    {"stages": {"compare_existing_radiounet_xai": {"status": "running"}}},
                )

    def test_completion_audit_rejects_nonpassing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()

            def fake_run(command, **kwargs):
                json_path = Path(command[command.index("--json-output") + 1])
                json_path.write_text(json.dumps({"passed": False, "checks": []}))

            with mock.patch("subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "passing result"):
                    run_completion_audit(root)


if __name__ == "__main__":
    unittest.main()
