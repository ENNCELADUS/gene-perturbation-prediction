"""
Evaluation utilities for reverse perturbation prediction.

Available modules:
- aggregator: Multi-run aggregation and leaderboards
- metrics: Evaluation metrics

Note: Evaluator classes (CellRetrievalEvaluator, ClassifierEvaluator, etc.)
are not implemented in this codebase.
"""

from .aggregator import RunAggregator, RunInfo

__all__ = [
    "RunAggregator",
    "RunInfo",
]
