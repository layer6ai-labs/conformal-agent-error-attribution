from typing import List

from .base_aggregator import BaseAggregator
import numpy as np

# Default hyperparameters for sweep (used in run_conformal_experiments.py):
# LENGTH_PENALTY_LAMBDAS = [0.01, 0.02, 0.05, 0.1]


    # RF, left-dense data, qwen3_ce_1_7b_uniform, llm_naive
def length_penalized_with_max(scores: np.ndarray, lambda_length_penalty: float) -> float:
    return sum(scores) + lambda_length_penalty * np.log(scores.shape[0])

class LengthPenalizedWithMaxAggregator(BaseAggregator):
    """Aggregator using cumulative sums, normalised by trajectory length."""

    def __init__(self, lambda_length_penalty: float = 0.1):
        super().__init__()
        self.lambda_length_penalty = lambda_length_penalty

    def prefix(self, probs: List[float]) -> List[float]:
        """result[j] = length_penalized_with_max(probs[0:j+1], lambda)"""
        n = len(probs)
        if n == 0:
            return []
        return [length_penalized_with_max(np.array(probs[:j + 1]), self.lambda_length_penalty) for j in range(n)]

    def suffix(self, probs: List[float]) -> List[float]:
        """result[j] = length_penalized_with_max(probs[j:], lambda)"""
        n = len(probs)
        if n == 0:
            return []
        return [length_penalized_with_max(np.array(probs[j:]), self.lambda_length_penalty) for j in range(n)]
    

    def combine(self, a: float, b: float) -> float:
        return a + b

    def score(self, probs: List[float]) -> float:
        n = len(probs)
        return length_penalized_with_max(np.array(probs), self.lambda_length_penalty) if n > 0 else 0.0
