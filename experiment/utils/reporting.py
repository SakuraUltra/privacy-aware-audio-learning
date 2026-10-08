"""Cross-validation summaries with explicit sample standard deviations."""
import math
from statistics import mean, stdev


def summarize_fold_metrics(fold_metrics, expected_folds=5):
    """Keep mean keys compatible and add <metric>_std keys (sample std, ddof=1)."""
    if expected_folds < 2:
        raise ValueError("At least two folds are required for sample standard deviation")
    if not fold_metrics:
        raise ValueError("No cross-validation metrics were provided")
    summary = {}
    for metric, values in fold_metrics.items():
        if len(values) != expected_folds:
            raise ValueError(f"{metric}: expected {expected_folds} folds, received {len(values)}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{metric}: non-finite fold metric")
        summary[metric] = mean(values)
        summary[f"{metric}_std"] = stdev(values)
    return summary
