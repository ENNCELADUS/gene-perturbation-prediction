"""
Evaluation utilities for reverse perturbation prediction.

Available modules:
- report: Comparison report generation
- aggregator: Multi-run aggregation and leaderboards
- metrics: Evaluation metrics

Note: Evaluator classes (CellRetrievalEvaluator, ClassifierEvaluator, etc.)
are not implemented in this codebase.
"""

from .report import generate_report, generate_comparison_table
from .aggregator import RunAggregator, RunInfo

__all__ = [
    "generate_report",
    "generate_comparison_table",
    "RunAggregator",
    "RunInfo",
]
