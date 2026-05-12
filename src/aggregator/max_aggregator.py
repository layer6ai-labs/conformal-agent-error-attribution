from typing import List

from .base_aggregator import BaseAggregator


class MaxAggregator(BaseAggregator):
    """Aggregator using running maximums, normalised by trajectory length."""

    def prefix(self, probs: List[float]) -> List[float]:
        """result[j] = max(probs[0:j+1]) / n"""
        n = len(probs)
        if n == 0:
            return []
        return [max(probs[: j + 1]) / n for j in range(n)]

    def suffix(self, probs: List[float]) -> List[float]:
        """result[j] = max(probs[j:]) / n"""
        n = len(probs)
        if n == 0:
            return []
        return [max(probs[j:]) / n for j in range(n)]

    def combine(self, a: float, b: float) -> float:
        return max(a, b)

    def score(self, probs: List[float]) -> float:
        return max(probs) if probs else 0.0
