from typing import List, Dict, Any

from .base_aggregator import BaseAggregator


def _compute_adv_2way(prefix: List[float], true_fail: int):
    """
    Compute the advanced 2-way oracle score and range from prefix-aggregated
    probabilities and the known failure position (true_fail).

    Mirrors the bisection in FilteringDataProcessor: shrink [adv_left, adv_right]
    toward the lighter half while keeping true_fail inside the window.

    Returns (adv_2way_score, [adv_left, adv_right]).
    """
    n = len(prefix)
    if n == 0 or true_fail < 0 or true_fail >= n:
        return 0.0, [0, -1]

    adv_left, adv_right = 0, n - 1
    while adv_left < adv_right:
        seg_n   = adv_right - adv_left + 1
        mid     = adv_left + (seg_n + 1) // 2
        l_sum   = sum(prefix[adv_left:mid])
        r_sum   = sum(prefix[mid:adv_right + 1])
        if l_sum < r_sum:
            adv_left += 1
        else:
            adv_right -= 1
        if not (adv_left <= true_fail <= adv_right):
            if true_fail < adv_left:
                adv_left -= 1
            else:
                adv_right += 1
            break

    return sum(prefix[adv_left:adv_right + 1]), [adv_left, adv_right]


def preprocess_data(data: List[Dict[str, Any]], aggregator: BaseAggregator) -> List[Dict[str, Any]]:
    """Inject aggregated fields into each record from the raw ``probability`` array.

    This must be called before passing data to ``initialize()`` on any conformal
    method that relies on pre-stored directional probability fields.

    Fields written (overwriting any prior values):
        - ``right_sum_probabilities``: prefix aggregate array (length n)
        - ``left_sum_probabilities``:  suffix aggregate array (length n)
        - ``right_score``:   prefix_agg[true_fail]  (0.0 if true_fail is OOB)
        - ``left_score``:    suffix_agg[true_fail]  (0.0 if true_fail is OOB)
        - ``score``:         global aggregate scalar (used by vanilla calibration)
        - ``adv_2way_score``: oracle bisection score for AdvancedFilteringConformal
        - ``adv_2way_range``: [adv_left, adv_right] containing true_fail

    Note: noise is NOT applied here; conformal methods add noise at runtime as
    they always have.
    """
    for d in data:
        probability = d.get('probability', [])
        n = len(probability)
        true_fail = int(d.get('true_fail', -1))

        prefix = aggregator.prefix(probability)
        suffix = aggregator.suffix(probability)

        d['right_sum_probabilities'] = prefix
        d['left_sum_probabilities']  = suffix
        d['right_score'] = prefix[true_fail] if 0 <= true_fail < n else 0.0
        d['left_score']  = suffix[true_fail] if 0 <= true_fail < n else 0.0
        d['score']       = aggregator.score(probability)

        adv_score, adv_range = _compute_adv_2way(prefix, true_fail)
        d['adv_2way_score'] = adv_score
        d['adv_2way_range'] = adv_range

    return data
