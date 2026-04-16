"""
Pure statistical helpers used by the alt-test pipeline.

Based on the AltTest implementation: https://github.com/nitaytech/AltTest
All functions are stateless and accept only plain Python / NumPy / SciPy types.
"""

from typing import Any, Callable, List, Union
import numpy as np
from scipy.stats import ttest_1samp


def by_procedure(p_values: List[float], q: float) -> List[int]:
    """
    Perform Benjamini-Yekutieli procedure for FDR control under arbitrary dependence.

    Parameters
    ----------
    p_values : List[float]
        List of p-values.
    q : float
        Desired FDR level.

    Returns
    -------
    List[int]
        Indices of rejected hypotheses.
    """
    p_values = np.array(p_values, dtype=float)
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # Harmonic sum H_m = 1 + 1/2 + ... + 1/m
    H_m = np.sum(1.0 / np.arange(1, m + 1))

    # BY thresholds for each rank i
    by_thresholds = (np.arange(1, m + 1) / m) * (q / H_m)

    max_i = -1
    for i in range(m):
        if sorted_pvals[i] <= by_thresholds[i]:
            max_i = i
    if max_i == -1:
        return []
    return list(sorted_indices[:max_i + 1])


def accuracy(pred: Any, annotations: List[Any]) -> float:
    """Fraction of annotations equal to pred."""
    return float(np.mean([pred == ann for ann in annotations]))


def neg_rmse(
    pred: Union[int, float],
    annotations: List[Union[int, float]],
) -> float:
    """Negative RMSE (higher is better)."""
    return -1.0 * float(np.sqrt(np.mean([(pred - ann) ** 2 for ann in annotations])))


def sim(pred: str, annotations: List[str], similarity_func: Callable) -> float:
    """Mean similarity between pred and each annotation."""
    return float(np.mean([similarity_func(pred, ann) for ann in annotations]))


def ttest(indicators: List[float], epsilon: float) -> float:
    """One-sample t-test against epsilon (alternative='less')."""
    return ttest_1samp(indicators, epsilon, alternative='less').pvalue
