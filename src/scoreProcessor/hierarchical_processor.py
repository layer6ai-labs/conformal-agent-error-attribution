import json
import numpy as np
from typing import List, Dict, Any, Optional
from .base_processor import BaseDataProcessor


class HierarchicalDataProcessor(BaseDataProcessor):
    """
    Data processor for hierarchical conformal prediction on conversation data.
    Implements the inference logic for binary tree traversal of agent conversations.
    """
    
    def __init__(self):
        """
        Initialize the hierarchical data processor.
        
        Args:
            random_seed: Random seed for reproducibility
            noise_sigma: Standard deviation for noise added to probabilities
        """
    
    def process_single_row(
        self,
        idx: int,
        row,
        scorer: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single row for hierarchical evaluation.
        
        Args:
            idx: Row index
            row: Row data from DataFrame
            scorer: HierarchicalConformalScorer instance
            
        Returns:
            Processed result dictionary or None to skip
        """
        history = row['history']
        problem = row.get('question', '')
        answer = row.get('ground_truth') or row.get('groundtruth', '')
        true_fail = int(row.get('mistake_step', -1))
        
        # Create data object for the scorer interface
        infer_data = {
            "agents_responses": history,
            "start": 0,
            "end": len(history) - 1,
            "problem": problem,
            "answer": answer
        }
        
        # Get inference results from scorer
        result = scorer.infer(infer_data)
        
        # Add noise and metadata to each transition
        for r in result:
            r['mistake_step'] = true_fail
        
        # Structure the result
        return {
            "conversation_id": idx,
            "question": problem,
            "answer": answer,
            "true_fail": true_fail,
            "transitions": result
        }

    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get basic statistics about the inference results.
        
        Args:
            results: List of inference results
            
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}
        
        total_results = len(results)
        total_transitions = sum(len(r.get("transitions", [])) for r in results)
        avg_transitions = total_transitions / total_results if total_results > 0 else 0
        
        # Get probability statistics
        all_probs = [
            t["prob"] for r in results 
            for t in r.get("transitions", [])
        ]
        
        stats = {
            "total_results": total_results,
            "total_transitions": total_transitions,
            "avg_transitions_per_result": avg_transitions,
        }
        
        if all_probs:
            stats.update({
                "min_prob": min(all_probs),
                "max_prob": max(all_probs),
                "mean_prob": np.mean(all_probs),
                "std_prob": np.std(all_probs)
            })
        
        return stats