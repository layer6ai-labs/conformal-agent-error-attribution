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

class VanillaConformal(BaseConformal):
    def __init__(self, random_seed: int = 42):
        super().__init__()
        log_name = f"logs/{__name__}_{datetime.now().timestamp()}.log"
        self.logger = get_logger(name=f"{__name__}_{datetime.now().timestamp()}", log_file=log_name)
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
        
        calibration_scores = []
        for d in calib_data:
            noise = d['noise'] if 'noise' in d else 0.0
            calibration_scores.append(np.clip(1 - d['score'] + noise, 0, 1))

        self._record_calibration_scores(calibration_scores)
        tau_star = compute_threshold(alpha=alpha, r_scores=calibration_scores)
        return tau_star
    
    def evaluate_all(self, tau_star, test_data):
        for d in test_data:
            noise = d['noise'] if 'noise' in d else 0.0
            d['conformal_set'] = [i for i, e in enumerate(np.clip(np.array(d['probability']) + noise, 0, 1)) if e >= 1 - tau_star]

        positives = [d for d in test_data if d['true_fail'] in d['conformal_set']]
        avg_removal_rate = np.mean([(len(d['nodes_logit']) - len(d['conformal_set'])) / len(d['nodes_logit']) if len(d['nodes_logit']) > 0 else 0.0 for d in test_data]) if test_data else 0.0
        self.logger.info(f"Average removal rate from conformal set: {avg_removal_rate:.2f}")

        total = len(test_data)
        correct = len(positives)
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, total, correct, avg_removal_rate

