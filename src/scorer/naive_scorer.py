from typing import Any, Dict, List

from .base_scorer import IScorer
from ..evaluator.base_evaluator import BaseEvaluator
from ..logger import get_logger


class NaiveScorer(IScorer):
    """
    Naive scorer that directly returns the evaluator's score.
    Can work with node evaluator or logprobs evaluator.
    """

    def __init__(self, evaluator: BaseEvaluator, logger_name: str = "agentic_conformal") -> None:
        self.evaluator = evaluator
        self.logger = get_logger(logger_name)

    def score(self, data: Dict[str, Any]) -> float:
        """
        Directly return the evaluator's score for the given data.
        
        Args:
            data: Dictionary containing evaluation data
        
        Returns:
            Float score from the evaluator
        """
        try:
            score = self.evaluator.evaluate(data)
            self.logger.info(f"Naive scorer returning score: {score:.3f}")
            return float(score)
        except Exception as e:
            self.logger.error(f"Error in naive scorer evaluation: {e}")
            return 0.5  # Default fallback score