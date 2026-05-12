from typing import Any, Dict

from ..evaluator.base_evaluator import BaseEvaluator
from ..scorer.hierarchical_conformal_scorer import HierarchicalConformalScorer


class DummyEvaluator(BaseEvaluator):
    """A dummy evaluator that deterministically prefers the half containing a preset index."""

    def __init__(self, failing_index: int, p_confident: float = 0.8) -> None:
        self.failing_index = failing_index
        self.p_confident = p_confident

    def evaluate(self, input_data: Any) -> float:
        start = int(input_data["start"])  # inclusive
        end = int(input_data["end"])      # inclusive
        mid = start + (end - start) // 2
        # If failing index is in left half, return high left prob, else low
        if self.failing_index <= mid:
            return self.p_confident
        else:
            return 1.0 - self.p_confident


def run_smoke() -> Dict[str, float]:
    # Build a toy conversation of length 8
    history = [{"name": f"A{k}", "content": f"msg{k}"} for k in range(8)]
    failing_idx = 5
    evaluator = DummyEvaluator(failing_index=failing_idx, p_confident=0.8)
    scorer = HierarchicalConformalScorer(evaluator)

    # Whole range - create data object for new interface
    score_data_all = {
        "agents_responses": history,
        "start": 0,
        "end": len(history) - 1,
        "true_fail": 5,
        "problem": "toy",
        "answer": "ans"
    }
    s_all = scorer.score(score_data_all)

    # A range that isolates the failing idx - create data object for new interface
    score_data_sub = {
        "agents_responses": history,
        "start": 4,
        "end": 7,
        "true_fail": 5,
        "problem": "toy", 
        "answer": "ans"
    }
    s_sub = scorer.score(score_data_sub)

    return {"s_all": s_all, "s_sub": s_sub}


if __name__ == "__main__":
    out = run_smoke()
    print(out)
