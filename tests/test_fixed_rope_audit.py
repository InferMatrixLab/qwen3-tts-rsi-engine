import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.audit_fixed_rope_campaign import checked_cohort, cohort_gate


class FixedRopeAuditTests(unittest.TestCase):
    def test_primary_requires_every_repeat_to_improve(self):
        control = [{"throughput_frames_s": 100, "ttfa_p95_ms": 10}] * 3
        candidate = [{"throughput_frames_s": v, "ttfa_p95_ms": 10} for v in (110, 110, 99)]
        self.assertEqual(cohort_gate(1, control, candidate)["gate"], "FAIL")

    def test_conservative_tail_gate_is_not_paired_average(self):
        control = [{"throughput_frames_s": 100, "ttfa_p95_ms": v} for v in (10, 11, 12)]
        candidate = [{"throughput_frames_s": 110, "ttfa_p95_ms": v} for v in (9, 10, 11)]
        self.assertEqual(cohort_gate(32, control, candidate)["gate"], "FAIL")

    def test_incomplete_requests_fail(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            checked_cohort({"requests": []})

    def test_missing_evidence_is_inconclusive_not_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "report.json"
            result = subprocess.run([sys.executable, "-m", "tools.audit_fixed_rope_campaign",
                                     "--evidence", folder, "--output", str(output)], capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(output.read_text())["status"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
