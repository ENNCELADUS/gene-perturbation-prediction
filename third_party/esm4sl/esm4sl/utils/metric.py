import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, auc, average_precision_score, balanced_accuracy_score


def mean_loss(losses: list[tuple[float, float]]) -> float:
    """
    Mean loss of an epoch.
    Each item is a tuple of (loss, batch_size).
    """
    total_loss = sum([loss * batch_size for loss, batch_size in losses])
    total_batch_size = sum([batch_size for _, batch_size in losses])
    return total_loss / total_batch_size


def compute_metrics(labels: torch.Tensor, probs: torch.Tensor) -> dict[str, float]:
    preds = probs > 0.5

    metrics = dict()
    metrics['auroc'] = roc_auc_score(labels, probs)
    metrics['auprc'] = average_precision_score(labels, probs)  # an estimate
    metrics['acc'] = accuracy_score(labels, preds)
    metrics['bacc'] = balanced_accuracy_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds)
    metrics['recall'] = recall_score(labels, preds)
    metrics['f1'] = f1_score(labels, preds)

    return metrics
