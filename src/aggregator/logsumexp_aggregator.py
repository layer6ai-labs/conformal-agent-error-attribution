from typing import List

from .base_aggregator import BaseAggregator
from scipy.special import logsumexp
import numpy as np

# Default hyperparameters for sweep (used in run_conformal_experiments.py):
# LOGSUMEXP_BETAS = [0.0001, 1.0, 10.0]


def g_lse(scores: np.ndarray, beta: float) -> float:
    """
    scores: array of shape (ell,) containing LLM(c_j:k)
    beta: inverse temperature
    """
    if beta == 0:
        raise ValueError("beta must be nonzero.")
    return logsumexp(beta * scores) / beta

class LogSumExpAggregator(BaseAggregator):
    """LogSumExp aggregator: g_lse(scores, beta) = logsumexp(beta*scores) / (n*beta)."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def prefix(self, probs: List[float]) -> List[float]:
        """result[j] = g_lse(probs[0:j+1], beta)"""
        n = len(probs)
        if n == 0:
            return []
        return [g_lse(np.array(probs[:j + 1]), beta=self.beta) for j in range(n)]

    def suffix(self, probs: List[float]) -> List[float]:
        """result[j] = g_lse(probs[j:], beta)"""
        n = len(probs)
        if n == 0:
            return []
        return [g_lse(np.array(probs[j:]), beta=self.beta) for j in range(n)]
    

    def combine(self, a: float, b: float) -> float:
        return a + b

    def score(self, probs: List[float]) -> float:
        n = len(probs)
        return g_lse(np.array(probs), beta=self.beta) if n > 0 else 0.0
