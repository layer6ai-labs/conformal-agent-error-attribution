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

class AdvancedFilteringConformal(BaseConformal):
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
            score = d.get('adv_2way_score', 0.0)
            noise = d.get('noise', 0.0)
            calibration_scores.append(np.clip(score + noise, 0, 1))

        self._record_calibration_scores(calibration_scores)
        tau_star = compute_threshold(alpha=alpha, r_scores=calibration_scores)
        return tau_star
    
    def find_adv_2way_range(self, data, tau_star):
        """Find the advanced 2-way range based on adv_2way_score and tau_star.
        
        Returns the infer range [left, right] based on tau_star that inf(sum(probabilities[left:right])) <= tau_star.
        Noise is added to probabilities to match the noise applied during calibration.
        """
        noise = data.get('noise', 0.0)
        probability = data.get('probability', [])
        probability = list(np.clip(np.array(probability) + noise, 0, 1))
        n_nodes = len(probability)
        left = 0
        right = n_nodes - 1
        while(sum(probability[left:right + 1]) > tau_star and left < right):
            n = right - left + 1
            mid = left + (n + 1) // 2  # left half gets more if odd
            left_sum = sum(probability[left:mid])
            right_sum = sum(probability[mid:right + 1])
            if left_sum < right_sum:
                left += 1
            else:
                right -= 1
        return [left, right]
    
    def evaluate_all(self, tau_star, test_data):
        positives = []
        removal = []

        for _, d in enumerate(test_data):
            probability = d.get('probability', [])
            n = len(probability)
            true_fail = d.get('true_fail', -1)

            if n <= 0:
                d['adv_2way_range'] = [0, -1]
                d['conformal_set'] = []
                removal.append(1.0)
                if true_fail < 0 or true_fail >= n:
                    positives.append(d)
                continue

            adv_range = self.find_adv_2way_range(d, tau_star)
            left, right = adv_range[0], adv_range[1]
            d['adv_2way_range'] = [left, right]

            if left <= right:
                d['conformal_set'] = list(range(left, right + 1))
                kept = right - left + 1
                removal.append(1.0 - (kept / n))
                if (left <= true_fail <= right) or true_fail < 0 or true_fail >= n:
                    positives.append(d)
            else:
                d['conformal_set'] = []
                removal.append(1.0)
                if true_fail < 0 or true_fail >= n:
                    positives.append(d)

        avg_removal_rate = np.mean(removal) if removal else 0.0
        self.logger.info(f"Average removal rate from advanced conformal set: {avg_removal_rate:.2f}")

        total = len(test_data)
        correct = len(positives)
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, total, correct, avg_removal_rate