import numpy as np
import random
import copy
import math
from tqdm import tqdm
from src.conformal.utils import compute_threshold
from src.conformal.base_conformal import BaseConformal

class TreeBinaryConformal(BaseConformal):
    def __init__(self, random_seed: int = 42):
        super().__init__()
        self.random_seed = random_seed

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
        filtered_calib_data = self.filter_results(calib_data)
        calibration_scores = []
        for d in filtered_calib_data:
            if len(d['transitions']) > 0:
                calibration_scores.append(min([e['prob'] for e in d['transitions']]))
            else:
                calibration_scores.append(1.0)  # Root node is the answer
        self._record_calibration_scores(calibration_scores)
        tau_star = compute_threshold(alpha=alpha, r_scores=calibration_scores)
        return tau_star
    
    def filter_results(self, results):
        """
        Keep question/answer/true_fail as is,
        but replace 'transitions' with only those events
        where mistake_step is inside the range.
        """
        return [
            {
                **res,
                "transitions": [
                    event
                    for event in res.get("transitions", [])
                    if int(event["mistake_step"]) >= event["range"][0]
                    and int(event["mistake_step"]) <= event["range"][1]
                ],
            }
            for res in results
        ]


    def evaluate_all(self, tau_star, test_data):
        total_count = len(test_data)
        kept_transitions_weighted = 0
        positives = []

        for d in test_data:
            filtered = [e for e in d.get('transitions', []) if e['prob'] > tau_star]
            d['transitions'] = filtered

            kept_count = 1 / math.pow(len(filtered), 2) if len(filtered) > 0 else 1
            kept_transitions_weighted += kept_count

            if all(
                int(t["mistake_step"]) >= t["range"][0]
                and int(t["mistake_step"]) <= t["range"][1]
                for t in filtered
            ):
                positives.append(d)

        correct = len(positives)
        accuracy = correct / total_count if total_count > 0 else 0.0
        removal_rate = (total_count - kept_transitions_weighted) / total_count if total_count > 0 else 0.0
        return accuracy, total_count, correct, removal_rate
