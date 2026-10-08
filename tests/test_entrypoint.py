"""User-facing CLI and model onboarding regression tests."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "experiment" / "train_unified.py"


class EntrypointTests(unittest.TestCase):
    def run_cli(self, *args, no_packages=False):
        command = [sys.executable]
        if no_packages:
            command.append("-S")
        command.extend([str(ENTRYPOINT), *args])
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                  timeout=90, env={**os.environ, "HF_HUB_OFFLINE": "1"})

    def test_help_needs_no_third_party_dependencies(self):
        result = self.run_cli("--help", no_packages=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--smoke-test", result.stdout)

    def test_synthetic_model_paths(self):
        for feature in ("opensmile", "mel"):
            for mode in ("normal", "vib"):
                with self.subTest(feature=feature, mode=mode):
                    result = self.run_cli("--smoke-test", "--feature_type", feature, "--mode", mode)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f"PASS: {feature}/{mode}", result.stdout)

    def test_smoke_rejects_dp_without_importing_dependencies(self):
        result = self.run_cli("--smoke-test", "--mode", "dp", no_packages=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not validate DP training", result.stderr)

    def test_invalid_training_arguments_fail_early(self):
        for flag in ("--epochs", "--batch_size"):
            with self.subTest(flag=flag):
                result = self.run_cli(flag, "0", no_packages=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be positive", result.stderr)

    def test_missing_real_data_returns_failure(self):
        result = self.run_cli("--feature_type", "mel", "--mode", "normal", "--epochs", "1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Required data files are missing", result.stderr)
        self.assertNotIn("Training completed successfully", result.stdout)


if __name__ == "__main__":
    unittest.main()
