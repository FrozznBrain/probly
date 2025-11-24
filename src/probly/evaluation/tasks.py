"""Collection of downstream tasks to evaluate the performance of uncertainty pipelines."""

from __future__ import annotations

import numpy as np

# TODO(mmshlk): remove sklearn dependency - https://github.com/pwhofman/probly/issues/132
import sklearn.metrics as sm


def selective_prediction(criterion: np.ndarray, losses: np.ndarray, n_bins: int = 50) -> tuple[float, np.ndarray]:
    """Selective prediction downstream task for evaluation.

    Perform selective prediction based on criterion and losses.
    The criterion is used the sort the losses. In line with uncertainty
    literature the sorting is done in descending order, i.e.
    the losses with the largest criterion are rejected first.

    Args:
        criterion: numpy.ndarray shape (n_instances,), criterion values
        losses: numpy.ndarray shape (n_instances,), loss values
        n_bins: int, number of bins
    Returns:
        auroc: float, area under the loss curve
        bin_losses: numpy.ndarray shape (n_bins,), loss per bin

    """
    if n_bins > len(losses):
        msg = "The number of bins can not be larger than the number of elements criterion"
        raise ValueError(msg)
    sort_idxs = np.argsort(criterion)[::-1]
    losses_sorted = losses[sort_idxs]
    bin_len = len(losses) // n_bins
    bin_losses = np.empty(n_bins)
    for i in range(n_bins):
        bin_losses[i] = np.mean(losses_sorted[(i * bin_len) :])

    # Also compute the area under the loss curve based on the bin losses.
    auroc = sm.auc(np.linspace(0, 1, n_bins), bin_losses)
    return auroc, bin_losses


def out_of_distribution_detection(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using prediction functionals from id and ood data.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals
    Returns:
        auroc: float, area under the roc curve

    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    auroc = sm.roc_auc_score(labels, preds)
    return float(auroc)


import numpy as np
import sklearn.metrics as sm


def fnr_at_tpr95(in_distribution: np.ndarray,
                 out_distribution: np.ndarray) -> float:
    """
    Compute FNR@95%TPR for OOD detection.

    Args:
        in_distribution: scores for ID samples (higher = more ID-like)
        out_distribution: scores for OOD samples (higher = more ID-like)

    Returns:
        fnr95: False Negative Rate at 95% True Positive Rate
    """


    scores = np.concatenate([in_distribution, out_distribution])
    labels = np.concatenate([np.zeros(len(in_distribution)),np.ones(len(out_distribution))])

    # ROC curve
    fpr, tpr, thresholds = sm.roc_curve(labels, scores)

    # Find first index where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        raise ValueError("Could not achieve 95% TPR with given scores.")

    tpr95 = tpr[idx[0]]

    # FNR = 1 - TPR
    fnr95 = 1 - tpr95

    return float(fnr95)