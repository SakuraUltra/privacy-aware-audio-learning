import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment"))
from utils.reporting import summarize_fold_metrics


class ReportingTests(unittest.TestCase):
    def test_mean_and_sample_std(self):
        result = summarize_fold_metrics({"accuracy": [0.6, 0.7, 0.8, 0.9, 1.0]})
        self.assertAlmostEqual(result["accuracy"], 0.8)
        self.assertAlmostEqual(result["accuracy_std"], math.sqrt(0.025))

    def test_identical_folds_have_zero_std(self):
        self.assertEqual(summarize_fold_metrics({"f1": [0.5] * 5}),
                         {"f1": 0.5, "f1_std": 0.0})

    def test_missing_fold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 5 folds"):
            summarize_fold_metrics({"accuracy": [0.5] * 4})

    def test_nonfinite_metric_is_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                summarize_fold_metrics({"f1": [0.5] * 4 + [value]})

    def test_empty_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No cross-validation metrics"):
            summarize_fold_metrics({})


if __name__ == "__main__":
    unittest.main()
