from typing import List

from .base_aggregator import BaseAggregator


class SumAggregator(BaseAggregator):
    """Aggregator using cumulative sums, normalised by trajectory length."""

    def prefix(self, probs: List[float]) -> List[float]:
        """result[j] = sum(probs[0:j+1]) / n"""
        n = len(probs)
        if n == 0:
            return []
        return [sum(probs[: j + 1]) / n for j in range(n)]

    def suffix(self, probs: List[float]) -> List[float]:
        """result[j] = sum(probs[j:]) / n"""
        n = len(probs)
        if n == 0:
            return []
        return [sum(probs[j:]) / n for j in range(n)]

    def combine(self, a: float, b: float) -> float:
        return a + b

    def score(self, probs: List[float]) -> float:
        n = len(probs)
        return sum(probs) / n if n > 0 else 0.0
