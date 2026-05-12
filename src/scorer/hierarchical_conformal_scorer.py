from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Dict

from .base_scorer import IScorer
from ..evaluator.base_evaluator import BaseEvaluator
from ..logger import get_logger


@dataclass
class Node:
    start: int
    end: int


class HierarchicalConformalScorer(IScorer):
    """
    Performs a binary-search traversal over a sequence of agent steps, using an evaluator
    that estimates P(first_failure in left half | within current node). At each decision,
    we compute the normalized transition probability and multiply along the path.

    The score returned for a given inclusive range [start, end] is the product of
    transition probabilities along the path to the last node searched (leaf or stop).
    """

    def __init__(self, evaluator: BaseEvaluator, logger_name: str = "agentic_conformal") -> None:
        self.evaluator = evaluator
        self.logger = get_logger(logger_name)

    def _step(self, problem: str, answer: str, history: List[Any], node: Node, path_prob: float, extra_eval_kwargs: Optional[Dict[str, Any]] = None) -> tuple[Node, float]:
        start, end = node.start, node.end
        #always search [start, mid) and [mid, end+1)
        mid = start + (end - start + 1) // 2

        data = {
            "problem": problem,
            "answer": answer,
            "history": history,
            "path_prob": path_prob,
            "start": start,
            "end": end,
        }
        # Add n_logprobs if evaluator supports it
        if hasattr(self.evaluator, '__class__') and 'Logprobs' in self.evaluator.__class__.__name__:
            data["n_logprobs"] = getattr(self.evaluator, 'n_logprobs', 5)

        p_left = self.evaluator.evaluate(data)
        p_right = max(path_prob - p_left, 0.0)

        if p_left > p_right:
            next_node = Node(start=start, end=mid-1)
            trans = p_left / (p_left + p_right) if (p_left + p_right) > 0 else 0.5
            move = "left"
        else:
            next_node = Node(start=mid, end=end)
            trans = p_right / (p_left + p_right) if (p_left + p_right) > 0 else 0.5
            move = "right"

        self.logger.info(f"Binary step {start}-{end}, mid={mid}, p_left={p_left:.3f}, p_right={p_right:.3f}, move={move}, trans={trans:.3f}")
        return next_node, float(max(0.0, min(1.0, trans)))

    def _traverse(self, agents_responses: List[Any], start: int, end: int, *, problem: str = "", answer: str = "", max_depth: Optional[int] = None, evaluator_kwargs: Optional[Dict[str, Any]] = None, true_fail: Optional[float] = None, stop_on_fail: bool = False) -> Any:
        """
        Shared traversal logic for score/infer. If stop_on_fail is True, will stop when true_fail is out of range.
        Returns a dict for infer, or a list for score.
        """
        n = len(agents_responses)
        if n == 0 or start < 0 or end >= n or start > end:
            self.logger.warning(f"Invalid traversal range start={start}, end={end}, len={n}.")
            if stop_on_fail:
                return 0.0
            else:
                return {"depth": 0, "trans": [], "range": [start, start+1]}

        node = Node(start=start, end=end)
        depth = 0
        path_prob = 1.0
        trans_list = []
        score_list = []

        while node.start < node.end:
            next_node, trans = self._step(problem, answer, agents_responses, node, path_prob, evaluator_kwargs)
            node = next_node
            original_trans_prob = round(path_prob * trans, 3)
            path_prob *= trans
            trans_list.append(trans)
            depth += 1
            if node.start > node.end:
                break
            if stop_on_fail and true_fail is not None:
                if true_fail < node.start or true_fail > node.end:
                    break
            score_list.append({"depth": depth, "prob": original_trans_prob, "range": [node.start, node.end]})

        return score_list

    def score(self, data: Dict[str, Any]) -> List[float]:
        """
        Compute the conformal score (path probability product) for the range [start, end].
        Stops when true_fail is out of range.
        
        Args:
            data: Dictionary containing:
                - agents_responses: List[Any] - sequence of agent steps
                - start: int - start index
                - end: int - end index
                - true_fail: float - true failure step
                - problem: str (optional) - problem description
                - answer: str (optional) - answer description
                - max_depth: int (optional) - maximum traversal depth
                - evaluator_kwargs: Dict (optional) - additional evaluator arguments
        """
        agents_responses = data["agents_responses"]
        start = data["start"]
        end = data["end"]
        true_fail = data["true_fail"]
        problem = data.get("problem", "")
        answer = data.get("answer", "")
        max_depth = data.get("max_depth", None)
        evaluator_kwargs = data.get("evaluator_kwargs", None)
        
        return self._traverse(
            agents_responses, start, end,
            problem=problem, answer=answer,
            max_depth=max_depth, evaluator_kwargs=evaluator_kwargs,
            true_fail=true_fail, stop_on_fail=True
        )

    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform binary traversal without stopping for true_fail.
        
        Args:
            data: Dictionary containing:
                - agents_responses: List[Any] - sequence of agent steps
                - start: int - start index
                - end: int - end index
                - problem: str (optional) - problem description
                - answer: str (optional) - answer description
                - max_depth: int (optional) - maximum traversal depth
                - evaluator_kwargs: Dict (optional) - additional evaluator arguments
        """
        agents_responses = data["agents_responses"]
        start = data["start"]
        end = data["end"]
        problem = data.get("problem", "")
        answer = data.get("answer", "")
        max_depth = data.get("max_depth", None)
        evaluator_kwargs = data.get("evaluator_kwargs", None)
        
        return self._traverse(
            agents_responses, start, end,
            problem=problem, answer=answer,
            max_depth=max_depth, evaluator_kwargs=evaluator_kwargs,
            true_fail=None, stop_on_fail=False
        )