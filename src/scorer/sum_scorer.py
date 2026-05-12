from typing import Any, Dict, List

from .base_scorer import IScorer
from ..evaluator.base_evaluator import BaseEvaluator
from ..logger import get_logger


class SumScorer(IScorer):
    """
    Sum scorer that evaluates each position in a range individually and returns the sum of scores.
    For a range [start, end], it evaluates (start, start+1), (start+1, start+2), ..., (end-1, end)
    and returns the total score across the evaluated positions.
    """

    def __init__(self, evaluator: BaseEvaluator, logger_name: str = "agentic_conformal") -> None:
        self.evaluator = evaluator
        self.logger = get_logger(logger_name)

    def score(self, data: Dict[str, Any]) -> float:
        """
        Evaluate each position in the range individually and return the maximum score.
        
        Args:
            data: Dictionary containing evaluation data with 'start' and 'end' keys
        
        Returns:
            Maximum float score from all individual evaluations
        """
        try:
            start = data.get('start', 0)
            end = data.get('end', 0)
            eval_cache = data.get('eval_cache', None)
            if eval_cache is not None and not isinstance(eval_cache, dict):
                eval_cache = None  # ensure only dict-like caches are used
            
            # If start == end, evaluate just that single position
            if start == end:
                key = (start, end)
                if eval_cache is not None and key in eval_cache:
                    score = float(eval_cache[key])
                else:
                    score = self.evaluator.evaluate(data)
                    if eval_cache is not None:
                        eval_cache[key] = float(score)
                self.logger.info(f"Sum scorer (single position {start}): {score:.3f}")
                return float(score)
            
            # Evaluate each consecutive pair in the range
            scores = []
            for i in range(start, end):
                # Create a copy of data with modified start/end for individual evaluation
                eval_data = data.copy()
                eval_data['start'] = i
                eval_data['end'] = i + 1
                key = (i, i + 1)
                
                try:
                    if eval_cache is not None and key in eval_cache:
                        score = float(eval_cache[key])
                    else:
                        score = self.evaluator.evaluate(eval_data)
                        if eval_cache is not None:
                            eval_cache[key] = float(score)
                    scores.append(float(score))
                    self.logger.debug(f"Position ({i}, {i+1}): score = {score:.3f}")
                except Exception as e:
                    self.logger.warning(f"Error evaluating position ({i}, {i+1}): {e}")
                    # Continue with other positions
                    continue
            
            if not scores:
                self.logger.error(f"No valid scores obtained for range [{start}, {end}]")
                return 0.5  # Default fallback score
            
            sum_score = sum(scores)
            self.logger.info(f"Sum scorer range [{start}, {end}): sum score = {sum_score:.3f} (from {len(scores)} evaluations)")
            return sum_score
            
        except Exception as e:
            self.logger.error(f"Error in sum scorer evaluation: {e}")
            return 0.5  # Default fallback score


# Backward compatibility for configs still referencing MaxScorer
class MaxScorer(SumScorer):
    pass
