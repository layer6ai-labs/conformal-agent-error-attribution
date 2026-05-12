import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from .base_processor import BaseDataProcessor
from ..evaluator.utils import _normalize_history

class FilteringDataProcessor(BaseDataProcessor):
    def __init__(self, temperature: float = 1.0):
        # temperature kept for API parity with other processors
        self.temperature = temperature

    def _compute_q_star(
        self,
        row: Dict[str, Any],
        scorer: Optional[Any],
        is_right_filter: bool,
        eval_cache: Optional[Dict[Any, float]] = None,
    ) -> Tuple[float, List[Dict[str, Any]], List[float]]:
        """Compute q* for a single row given direction.

        Returns (q_star, nodes_metadata, scaled_probabilities).
        """
        history = row['history']
        problem = row.get('question', '')
        answer = row.get('ground_truth') or row.get('groundtruth', '')
        true_fail = int(row.get('mistake_step', -1))
        norm_history = _normalize_history(history)

        node_result: List[float] = []
        nodes: List[Dict[str, Any]] = []
        cache: Dict[Any, float] = eval_cache if eval_cache is not None else {}

        for node_idx in range(len(history)):
            eval_data = {
                "problem": problem,
                "answer": answer,
                "history": norm_history,
                "start": 0,
                "end": node_idx + 1,
                "eval_cache": cache,
            }
            if not is_right_filter:
                eval_data["start"] = node_idx
                eval_data["end"] = len(history)

            prob = scorer.score(eval_data) if scorer else 0.0
            nodes.append({"node_content": norm_history[node_idx], "logit": prob})
            node_result.append(prob)

        if true_fail < 0 or true_fail >= len(nodes):
            return 0.0, nodes, node_result

        # Scale the node_result by trajectory length
        node_result = [prob / len(node_result) for prob in node_result]

        q_star = node_result[true_fail]

        return q_star, nodes, node_result

    def process_single_row(
        self,
        idx: int,
        row,
        scorer: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single row for filtering evaluation.
        
        Args:
            idx: Row index
            row: Row data from DataFrame
            scorer: Scorer instance
            
        Returns:
            Processed result dictionary or None to skip
        """
        # Shared cache for evaluator to reuse computations when possible
        shared_cache: Dict[Any, float] = {}

        # Compute right direction (prefix 0..j)
        right_q, nodes_right, right_probs = self._compute_q_star(
            row, scorer, True, eval_cache=shared_cache
        )

        # Compute left direction (suffix i..end)
        left_q, nodes_left, left_probs = self._compute_q_star(
            row, scorer, False, eval_cache=shared_cache
        )

        # If both empty, skip
        if not nodes_right and not nodes_left:
            return None

        # Compute advance 2-way range/score based on right_probs
        adv_left = 0
        adv_right = len(right_probs) - 1
        true_fail_idx = int(row.get('mistake_step', -1))

        if adv_right >= 0:
            while adv_left < adv_right:
                n = adv_right - adv_left + 1
                mid = adv_left + (n + 1) // 2  # left half gets more if odd
                left_sum = sum(right_probs[adv_left:mid])
                right_sum = sum(right_probs[mid:adv_right + 1])
                if left_sum < right_sum:
                    adv_left += 1
                else:
                    adv_right -= 1

                if not (adv_left <= true_fail_idx <= adv_right):
                    if true_fail_idx < adv_left:
                        adv_left -= 1
                    else:
                        adv_right += 1
                    break

            adv_2way_range = [adv_left, adv_right]
            adv_2way_score = sum(right_probs[adv_left:adv_right + 1])
        else:
            adv_2way_range = [0, -1]
            adv_2way_score = 0.0

        # Build unified nodes list with per-node left/right logits
        max_len = max(len(nodes_right), len(nodes_left))
        nodes_combined: List[Dict[str, Any]] = []
        for i in range(max_len):
            node_content = None
            right_logit = None
            left_logit = None

            if i < len(nodes_right):
                node_content = nodes_right[i].get("node_content")
                right_logit = nodes_right[i].get("logit")
            if i < len(nodes_left):
                # Prefer node_content from right if present; else take from left
                node_content = node_content if node_content is not None else nodes_left[i].get("node_content")
                left_logit = nodes_left[i].get("logit")

            nodes_combined.append({
                "node_content": node_content,
                "right_logit": right_logit,
                "left_logit": left_logit,
            })

        return {
            "conversation_id": idx,
            "question": row.get('question', ''),
            "answer": row.get('ground_truth') or row.get('groundtruth', ''),
            "true_fail": int(row.get('mistake_step', -1)),
            "nodes_logit": nodes_combined,
            "right_sum_probabilities": right_probs,
            "left_sum_probabilities": left_probs,
            "right_score": right_q,
            "left_score": left_q,
            "adv_2way_range": adv_2way_range,
            "adv_2way_score": adv_2way_score,
        }
