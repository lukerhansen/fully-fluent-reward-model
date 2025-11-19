"""Evaluation metrics."""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: True scores
        y_pred: Predicted scores

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Correlation metrics
    metrics['pearson'], _ = pearsonr(y_true, y_pred)
    metrics['spearman'], _ = spearmanr(y_true, y_pred)

    # Error metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics
