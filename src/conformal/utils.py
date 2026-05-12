import json
import csv
import numpy as np
from math import ceil
from collections import defaultdict

CORRECT_ANNOTATIONS = ["Y", "S"]


def load_subclaim_data(file_path):
    """Load calibration data from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def append_result_to_csv(csv_filename, label, y, yerr):
    """Append calibration results to CSV file"""
    formatted_results = [f"{y:.4f} ± {yerr:.4f}" for y, yerr in zip(y, yerr)]
    formatted_results.reverse()
    row = [label] + formatted_results
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def _get_accepted_subclaims(entry, threshold, confidence_method):
    """Helper function to get accepted subclaims based on threshold"""
    return [
        subclaim
        for subclaim in entry["subclaims"]
        if subclaim["scores"][confidence_method] + subclaim["scores"]["noise"]
        >= threshold
    ]


def _calculate_entailed_fraction(subclaims):
    """Helper function to calculate fraction of entailed/correct subclaims"""
    if not subclaims:
        return 1.0
    return np.mean(
        [
            subclaim["annotations"]["gpt"] in CORRECT_ANNOTATIONS
            for subclaim in subclaims
        ]
    )




def compute_threshold(alpha, r_scores):
    """
    Computes the quantile/threshold from conformal prediction.
    # alpha: float in (0, 1)
    # conformal scores need to calculate threshold
    # a: as in paper, required fraction correct, section 4.1
    # confidence_method: string
    """
    # Compute threshold for conformal prection. The quantile is ceil((n+1)*(1-alpha))/n, and
    # We map this to the index by dropping the division by n and subtracting one (for zero-index).
    quantile_target_index = ceil((len(r_scores) + 1) * (1 - alpha))
    threshold = sorted(r_scores)[quantile_target_index - 1]
    return threshold

def compute_thresholds_exlcude(alpha, r_scores):
    """
    Computes the quantile/threshold from calibration set build exclude the true failure case.
    # alpha: float in (0, 1)
    # conformal scores need to calculate threshold
    # a: as in paper, required fraction correct, section 4.1
    # confidence_method: string
    """
    # Compute threshold for conformal prection. The quantile is  $\tfrac{\lceil{(n+1)(1-\alpha)}\rceil}{n} + 1$, and
    # We map this to the index by dropping the division by n and subtracting one (for zero-index).
    quantile_target_index = ceil((len(r_scores) + 1) * (1 - alpha)) + 1
    threshold = r_scores[quantile_target_index - 1]
    return threshold

    
# Make sure the split calibrate_range ratio are all same not just in overall level but in group level
# not return data in list but in a map with each group name as key
def split_group(data, calibrate_range=0.5):
    group_data = defaultdict(list)
    calibration_data = defaultdict(list)
    test_data = []

    for entry in data:
        group = entry["groups"][0]  # Use first group as default
        group_data[group].append(entry)

    for group, group_entries in group_data.items():
        split_index = ceil(len(group_entries) * calibrate_range)
        calibration_data[group].extend(group_entries[:split_index])
        test_data.extend(group_entries[split_index:])

    return calibration_data, test_data

# Analyze Functions #

def percentage_highest_not_S(data, key="relavance"):
    count_total = 0
    count_not_S = 0

    for item in data:
        subclaims = item.get("subclaims", [])
        if not subclaims:
            continue

        # Sort subclaims by (score[key] + score[noise]), descending
        subclaims_sorted = sorted(
            subclaims,
            key=lambda sc: sc["scores"].get(key, 0) + sc["scores"].get("noise", 0),
            reverse=True
        )

        top_annotation = subclaims_sorted[0].get("annotations", {}).get("gpt", None)

        count_total += 1
        if top_annotation != "S":
            count_not_S += 1

    if count_total == 0:
        return 0.0  # Avoid division by zero

    return (count_not_S / count_total) * 100