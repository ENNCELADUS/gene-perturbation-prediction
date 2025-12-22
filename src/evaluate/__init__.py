"""Evaluation module for reverse perturbation prediction."""

from .cell_eval import CellRetrievalEvaluator
from .classifier_eval import ClassifierEvaluator
from .scgpt_classifier_eval import ScGPTClassifierEvaluator
from .metrics import compute_all_metrics
from .report import (
    aggregate_results,
    generate_comparison_table,
    generate_report,
)
from .confidence import (
    ConfidenceScorer,
    compute_margin_confidence,
    compute_entropy_confidence,
    coverage_accuracy_curve,
)
from .error_analysis import (
    build_confusion_matrix,
    find_commonly_confused_pairs,
    generate_error_report,
)
from .aggregator import RunAggregator

__all__ = [
    # Evaluators
    "CellRetrievalEvaluator",
    "ClassifierEvaluator",
    "ScGPTClassifierEvaluator",
    # Metrics
    "compute_all_metrics",
    # Reporting
    "aggregate_results",
    "generate_comparison_table",
    "generate_report",
    # Confidence
    "ConfidenceScorer",
    "compute_margin_confidence",
    "compute_entropy_confidence",
    "coverage_accuracy_curve",
    # Error analysis
    "build_confusion_matrix",
    "find_commonly_confused_pairs",
    "generate_error_report",
    # Aggregation
    "RunAggregator",
]
