import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from src.conformal.utils import compute_threshold
from src.conformal.base_conformal import BaseConformal
from src.logger import get_logger

class FilteringConformal(BaseConformal):
    def __init__(self, random_seed: int = 42, is_right_filter: bool = True):
        super().__init__()
        log_name = f"logs/{__name__}_{datetime.now().timestamp()}.log"
        self.logger = get_logger(name=f"{__name__}_{datetime.now().timestamp()}", log_file=log_name)
        self.random_seed = random_seed
        self.is_right_filter = is_right_filter


    def initialize(self, data, seed=None):
        shuffled_results = copy.deepcopy(data)
        if seed is not None:
            random.seed(seed)
        random.shuffle(shuffled_results)

        mid = len(shuffled_results) // 2
        calib_data = shuffled_results[:mid]
        test_data = shuffled_results[mid:]
        return calib_data, test_data
    
    def compute_tau_star(self, alpha, calib_data):
        
        calibration_scores = []
        for d in calib_data:
            # Use score from appropriate side based on is_right_filter
            if self.is_right_filter:
                score = d.get('right_score', 0.0)
            else:
                score = d.get('left_score', 0.0)
            noise = d['noise'] if 'noise' in d else 0.0
            score = np.clip(score + noise, 0, 1)
            calibration_scores.append(score)

        self._record_calibration_scores(calibration_scores)
        tau_star = compute_threshold(alpha=alpha, r_scores=calibration_scores)
        return tau_star

    def find_cut_index(self, data, tau_star, is_right_filter):
        """Find the cut index in forward direction for both right and left filters.
        
        Returns a forward index in [0, n-1] if a cut is found; otherwise returns n (no cut).
        Behavior is equivalent to original implementation which searched from the right by
        reversing probabilities for the right filter and searched forward for the left filter.
        """
        if is_right_filter:
            probs = data.get('right_sum_probabilities', [])
            noise = data['noise'] if 'noise' in data else 0.0
            probs = np.clip(np.array(probs) + noise, 0, 1)
            # For right filter: choose the rightmost index k where probs[k] <= tau_star.
            last_idx = None
            for i, p in enumerate(probs):
                if p <= tau_star:
                    last_idx = i
            return last_idx if last_idx is not None else len(probs)
        else:
            probs = data.get('left_sum_probabilities', [])
            noise = data['noise'] if 'noise' in data else 0.0
            probs = np.clip(np.array(probs) + noise, 0, 1)

            # For left filter: choose the leftmost index k where probs[k] <= tau_star.
            for i, p in enumerate(probs):
                if p <= tau_star:
                    return i
            return len(probs)
    
    def evaluate_all(self, tau_star, test_data):
        # Filter out data true_failure is out of bounds
        #test_data = [d for d in test_data if 0 <= d['true_fail'] < len(d.get('nodes_logit', []))]

        # Filter transitions based on tau_star
        positives = []
        removal = []
        for i, d in enumerate(test_data):
            # Get probabilities from appropriate side
            probs = d.get('right_sum_probabilities', []) if self.is_right_filter else d.get('left_sum_probabilities', [])
            noise = d['noise'] if 'noise' in d else 0.0
            probs = np.clip(np.array(probs) + noise, 0, 1)

            # Compute forward cut index via helper
            cut_index = self.find_cut_index(d, tau_star, self.is_right_filter)
            d['cut_index'] = cut_index

            if self.is_right_filter:
                if cut_index < len(probs):
                    #[1, 2, 3, 4, 5, 6, 7, 8, 9]
                    #[ Concormal Set ^][ Removed ]
                    #                cut_index
                    # Positives: len-1-true_fail >= j  ==> true_fail <= cut_index
                    if d['true_fail'] <= cut_index:
                        positives.append(d)
                    # Conformal set: indices from cut_index to the end
                    d['conformal_set'] = list(range(0, cut_index + 1))
                    # Removal rate: (j+1)/n with j = (n-1) - cut_index  ==> (n - cut_index - 1)/n
                    removal.append((len(probs) - cut_index - 1) / len(probs) if len(probs) > 0 else 0.0)
                else:
                    # No cut found when scanning right side
                    d['conformal_set'] = []
                    removal.append(1.0)
                    if d['true_fail'] == -1:
                        positives.append(d)
            else:
                    #[1, 2, 3, 4, 5, 6, 7, 8, 9]
                    #[Removed][^ Conformal Set ]
                    #          cut_index
                if cut_index < len(probs):
                    # Positives: d['true_fail'] >= cut_index
                    if d['true_fail'] >= cut_index or d['true_fail'] < 0:
                        positives.append(d)
                    # Conformal set: [0, cut_index]
                    d['conformal_set'] = list(range(cut_index, len(probs)))
                    removal.append(cut_index / len(probs) if len(probs) > 0 else 0.0)
                else:
                    # No cut found for left filter
                    removal.append(1.0)
                    if d['true_fail'] < 0 or d['true_fail'] >= len(probs):
                        positives.append(d)
                    d['conformal_set'] = []

        avg_removal_rate = np.mean(removal) if removal else 0.0
        self.logger.info(f"Average removal rate from conformal set: {avg_removal_rate:.2f}")

        total = len(test_data)
        correct = len(positives)
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, total, correct, avg_removal_rate


class TwoWayFilteringConformal(FilteringConformal):
    """Two-way filtering conformal: use max(left_score, right_score) for tau_star,
    and conformal_set is the intersection of left and right conformal sets.
    """
    
    def compute_tau_star(self, alpha, calib_data):
        """Compute tau_star using max(left_score, right_score) from calibration data."""
        calibration_scores = []
        for d in calib_data:
            left_score = d.get('left_score', 0.0)
            right_score = d.get('right_score', 0.0)
            # Use max of both scores
            max_score = max(left_score, right_score)
            noise = d['noise'] if 'noise' in d else 0.0
            max_score = np.clip(max_score + noise, 0, 1)
            calibration_scores.append(max_score)

        self._record_calibration_scores(calibration_scores)
        tau_star = compute_threshold(alpha=alpha, r_scores=calibration_scores)
        return tau_star
    
    def evaluate_all(self, tau_star, test_data):
        """Evaluate using intersection of left and right conformal sets."""
        positives = []
        removal = []
        
        for i, d in enumerate(test_data):
            # Compute cut indices for both sides
            cut_index_left = self.find_cut_index(d, tau_star, is_right_filter=False)
            cut_index_right = self.find_cut_index(d, tau_star, is_right_filter=True)
            
            n = len(d.get('nodes_logit', []))
            # Left conformal set: [0, cut_index_left]
            # Right conformal set: [cut_index_right, n-1]
            # Intersection: [cut_index_right, cut_index_left] if cut_index_right <= cut_index_left, else empty
            
            if cut_index_left <= cut_index_right:
                # Non-empty intersection
                d['conformal_set'] = list(range(cut_index_left, cut_index_right + 1))
                
                # Check if true_fail is in conformal set
                if cut_index_left <= d['true_fail'] <= cut_index_right or d['true_fail'] < 0:
                    positives.append(d)
                
                # Removal rate: fraction of indices removed
                # Kept: cut_index_left - cut_index_right + 1 out of n
                kept = cut_index_right - cut_index_left + 1
                removal.append(1.0 - (kept / n if n > 0 else 0.0))
            else:
                # Empty intersection
                d['conformal_set'] = []
                
                # Check if true_fail is -1 (no mistake)
                if d['true_fail'] == -1:
                    positives.append(d)
                
                removal.append(1.0)
        
        avg_removal_rate = np.mean(removal) if removal else 0.0
        self.logger.info(f"Average removal rate from two-way conformal set: {avg_removal_rate:.2f}")
        
        total = len(test_data)
        correct = len(positives)
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, total, correct, avg_removal_rate
