import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from .base_processor import BaseDataProcessor
from ..evaluator.utils import _normalize_history

class NodeDataProcessor(BaseDataProcessor):
    """
    Data processor for node-level evaluation on conversation data.
    Processes individual conversation nodes for evaluation.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        super().__init__()
        
    
    def process_single_row(
        self,
        idx: int,
        row,
        scorer: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single row for node evaluation.
        
        Args:
            idx: Row index
            row: Row data from DataFrame
            scorer: Node evaluator instance
            
        Returns:
            Processed result dictionary or None to skip
        """
        history = row['history']
        problem = row.get('question', '')
        answer = row.get('ground_truth') or row.get('groundtruth', '')
        true_fail = int(row.get('mistake_step', -1))
        norm_history = _normalize_history(history)
        node_result = []
        nodes = []
        
        # Process each node in the conversation
        for node_idx in range(len(history)):
            # Create data object for the evaluator interface
            eval_data = {
                "problem": problem,
                "answer": answer,
                "history": norm_history,
                "start": node_idx,
                "end": node_idx + 1
            }
            
            # Get evaluation result from evaluator
            if scorer:
                prob = scorer.score(eval_data)
            else:
                prob = 0.5  # Default probability if no evaluator
            nodes.append({
                "node_content": norm_history[node_idx],
                "logit": prob
            })
            node_result.append(prob)
            
        # do a softmax with temperature control
        exp_logits = np.exp(np.array(node_result) / self.temperature)
        nodes_probs = exp_logits / np.sum(exp_logits)

        if true_fail < 0 or true_fail >= len(nodes):
            return None  # Skip invalid true_fail indices
        
        # Structure the result
        return {
            "conversation_id": idx,
            "question": problem,
            "answer": answer,
            "true_fail": true_fail,
            "nodes_logit": nodes,
            "probability": nodes_probs.tolist(),
            "score": nodes_probs[true_fail]
        }