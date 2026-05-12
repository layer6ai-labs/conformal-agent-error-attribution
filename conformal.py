#!/usr/bin/env python3
"""
Phase 2 – Conformal prediction from cached node_results.

No model or API calls are made. Reads pre-computed node_results files,
applies the chosen aggregator (overwriting directional probability fields),
runs the selected conformal method, and saves accuracy / removal results.

Output directory convention:
    empirical_data/{timestamp}-{backbone}-{method}-{aggregator}-{evaluator}-{task}/

Usage:
    # Run experiments
    python conformal.py \\
        --methods right_filter left_filter \\
        --aggregators sum \\
        --evaluators llm_naive agent_echo \\
        --tasks dylan_gsm8k_uniformleft macnet_gsm8k_uniformleft

    # Run + collect summary CSV
    python conformal.py \\
        --methods right_filter left_filter 2way_filter \\
        --aggregators sum \\
        --evaluators llm_naive \\
        --tasks dylan_math_uniform macnet_math_uniform whoandwhen \\
        --collect-results

    # Only collect results (no new experiments)
    python conformal.py \\
        --collect-results-only \\
        --methods right_filter --aggregators sum \\
        --evaluators llm_naive \\
        --tasks dylan_gsm8k_uniformleft
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.getcwd())

from src.conformal.vanilla_conformal import VanillaConformal
from src.conformal.filtering_conformal import FilteringConformal, TwoWayFilteringConformal
from src.conformal.advanced_filtering_conformal import AdvancedFilteringConformal
from src.conformal.tree_crsvp_conformal import TreeHierarchicalRestrictConformal
from src.aggregator import (
    SumAggregator, MaxAggregator,
    LogSumExpAggregator, NormalizedLogSumExpAggregator,
    LengthPenalizedAggregator, LengthPenalizedWithMaxAggregator,
    preprocess_data,
)
from src.logger import get_logger

# Same convention as score.py
NODE_RESULTS_DIR  = "empirical_data"
NODE_RESULTS_TMPL = "node_results-{task}-{backbone}-{evaluator}.jsonl"
DEFAULT_BACKBONE  = "gpt-4o-mini"
DEFAULT_ALPHA     = 0.2

VALID_METHODS = [
    "vanilla", "right_filter", "left_filter",
    "2way_filter", "adv_2way_filter", "tree_crsvp",
]
VALID_EVALUATORS = [
    "llm_naive", "llm_logprobs", "llm_echo", "agent_echo",
    "qwen3_ce_1_7b", "qwen3_ce_1_7b_uniform",
    "qwen3_ce_8b",   "qwen3_ce_8b_uniform",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def node_results_path(task: str, backbone: str, evaluator: str) -> str:
    filename = NODE_RESULTS_TMPL.format(task=task, backbone=backbone, evaluator=evaluator)
    return os.path.join(NODE_RESULTS_DIR, filename)


def load_config(config_path: str = "conf/conformal_method.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_dir(base: str, timestamp: str, backbone: str, suffix: str) -> str:
    d = os.path.join(base, f"{timestamp}-{backbone}-{suffix}")
    os.makedirs(d, exist_ok=True)
    return d


# ── Factories ──────────────────────────────────────────────────────────────────

def build_aggregator(name: str):
    """Parse aggregator name and return an aggregator instance."""
    if name == "sum":
        return SumAggregator()
    if name == "max":
        return MaxAggregator()
    if name.startswith("normalizedlogsumexp_"):
        return NormalizedLogSumExpAggregator(beta=float(name[len("normalizedlogsumexp_"):]))
    if name.startswith("logsumexp_"):
        return LogSumExpAggregator(beta=float(name[len("logsumexp_"):]))
    if name.startswith("lengthpenalizedwithmax_"):
        return LengthPenalizedWithMaxAggregator(
            lambda_length_penalty=float(name[len("lengthpenalizedwithmax_"):])
        )
    if name.startswith("lengthpenalized_"):
        return LengthPenalizedAggregator(
            lambda_length_penalty=float(name[len("lengthpenalized_"):])
        )
    raise ValueError(
        f"Unknown aggregator '{name}'. Supported: sum, max, "
        "logsumexp_<b>, normalizedlogsumexp_<b>, "
        "lengthpenalized_<l>, lengthpenalizedwithmax_<l>"
    )


def build_conformal(method: str, aggregator, seed: int):
    """Return a conformal predictor instance for the given method."""
    if method == "vanilla":
        return VanillaConformal(random_seed=seed)
    if method == "right_filter":
        return FilteringConformal(random_seed=seed, is_right_filter=True)
    if method == "left_filter":
        return FilteringConformal(random_seed=seed, is_right_filter=False)
    if method == "2way_filter":
        return TwoWayFilteringConformal(random_seed=seed)
    if method == "adv_2way_filter":
        return AdvancedFilteringConformal(random_seed=seed)
    if method == "tree_crsvp":
        # tree_crsvp uses the aggregator structurally during tree construction
        return TreeHierarchicalRestrictConformal(random_seed=seed, aggregator=aggregator)
    raise ValueError(f"Unknown method '{method}'. Valid: {VALID_METHODS}")


# ── Single experiment ──────────────────────────────────────────────────────────

def run_single(
    method: str,
    aggregator_name: str,
    evaluator: str,
    task: str,
    backbone: str,
    config: dict,
) -> Optional[Dict]:
    """
    Run one (method, aggregator, evaluator, task) experiment.

    Returns a dict with 'log_dir' and 'data_dir', or None if the node_results
    file is missing (caller should run score.py first).
    """
    data_file = node_results_path(task, backbone, evaluator)
    if not os.path.exists(data_file):
        print(
            f"  [SKIP] node_results not found: {data_file}\n"
            "         Run score.py first to generate it."
        )
        return None

    data = load_jsonl(data_file)
    if not data:
        print(f"  [SKIP] empty data file: {data_file}")
        return None

    # Aggregator fills in right_sum_probabilities, left_sum_probabilities,
    # right_score, left_score, score, adv_2way_score from the raw probability array.
    aggregator = build_aggregator(aggregator_name)
    preprocess_data(data, aggregator)

    seed      = config["experiment"]["random_seed"]
    conformal = build_conformal(method, aggregator, seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label     = f"{method}-{aggregator_name}-{evaluator}-{task}"
    log_dir   = _make_dir("logs",          timestamp, backbone, label)
    data_dir  = _make_dir("empirical_data", timestamp, backbone, label)

    log_file = os.path.join(log_dir, f"{label}.log")
    logger   = get_logger(
        name=f"conformal_{method}_{aggregator_name}_{timestamp}", log_file=log_file
    )
    logger.info(
        f"method={method}  aggregator={aggregator_name}  "
        f"evaluator={evaluator}  task={task}"
    )
    logger.info(f"Loaded {len(data)} records from {data_file}")

    alphas   = np.array(config["experiment"]["alphas"])
    n_trials = config["experiment"]["n_trials"]

    accuracy_csv  = os.path.join(data_dir, "accuracy_results.csv")
    accuracy_plot = os.path.join(data_dir, "accuracy_plot.png")
    removal_csv   = os.path.join(data_dir, "removal_results.csv")
    removal_plot  = os.path.join(data_dir, "removal_plot.png")
    updated_file  = os.path.join(
        data_dir, f"updated_{method}_{aggregator_name}_{evaluator}.jsonl"
    )

    accuracy_results, removal_results = conformal.compute_results(
        data=data,
        alphas=alphas,
        n_trials=n_trials,
        save_accuracy_csv_path = accuracy_csv   if config["output"]["save_results"] else None,
        save_accuracy_fig_path = accuracy_plot  if config["output"]["save_plots"]   else None,
        save_removal_csv_path  = removal_csv    if config["output"]["save_results"] else None,
        save_removal_fig_path  = removal_plot   if config["output"]["save_plots"]   else None,
        updated_data_file      = updated_file,
    )
    logger.info("Experiment completed.")
    print(f"  Saved → {data_dir}")
    return {"log_dir": log_dir, "data_dir": data_dir}


# ── Result collection ──────────────────────────────────────────────────────────

def _find_result_dirs(
    backbone: str, method: str, aggregator: str, evaluator: str, task: str
) -> List[str]:
    """Return latest-first result directories matching the given combo."""
    pattern = f"{backbone}-{method}-{aggregator}-{evaluator}-{task}"
    matches = [
        d for d in os.listdir("empirical_data")
        if pattern in d and os.path.isdir(os.path.join("empirical_data", d))
    ]
    matches.sort(reverse=True)
    return [os.path.join("empirical_data", d) for d in matches]


def _get_removal_rate(result_dir: str, alpha: float) -> Optional[str]:
    csv_path = os.path.join(result_dir, "removal_results.csv")
    if not os.path.exists(csv_path):
        return None
    df  = pd.read_csv(csv_path)
    row = df[df["alpha"] == alpha]
    if row.empty:
        return None
    mean = row["removal_rate_mean"].values[0]
    std  = row["removal_rate_std"].values[0]
    return f"{mean:.3f} ({std:.3f})"


def collect_results(
    backbone: str,
    methods: List[str],
    aggregators: List[str],
    evaluators: List[str],
    tasks: List[str],
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """
    Scan empirical_data/ for result directories and build a summary DataFrame.

    Rows:    {evaluator}_{method}_{aggregator}
    Columns: task names
    Values:  removal_rate_mean (removal_rate_std) at the given alpha
    """
    rows = [
        f"{ev}_{m}_{agg}"
        for ev  in sorted(evaluators)
        for m   in methods
        for agg in aggregators
    ]
    summary = pd.DataFrame(index=rows, columns=tasks)

    for row_name in rows:
        m = agg = ev = None
        for known_m in sorted(methods, key=len, reverse=True):
            for known_agg in aggregators:
                suffix = f"_{known_m}_{known_agg}"
                if row_name.endswith(suffix):
                    m   = known_m
                    agg = known_agg
                    ev  = row_name[: -len(suffix)]
                    break
            if m:
                break
        if m is None:
            continue

        for task in tasks:
            dirs = _find_result_dirs(backbone, m, agg, ev, task)
            if dirs:
                summary.loc[row_name, task] = _get_removal_rate(dirs[0], alpha)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Run conformal prediction from cached node_results."
    )
    parser.add_argument(
        "--methods", nargs="+", required=True, choices=VALID_METHODS, metavar="METHOD",
    )
    parser.add_argument(
        "--aggregators", nargs="+", required=True, metavar="AGG",
        help=(
            "sum, max, logsumexp_<b>, normalizedlogsumexp_<b>, "
            "lengthpenalized_<l>, lengthpenalizedwithmax_<l>"
        ),
    )
    parser.add_argument(
        "--evaluators", nargs="+", required=True, choices=VALID_EVALUATORS,
        metavar="EVALUATOR",
    )
    parser.add_argument(
        "--tasks", nargs="+", required=True, metavar="TASK",
        help="Task names, e.g.  dylan_gsm8k_uniformleft  macnet_math_uniform  whoandwhen",
    )
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--config",   default="conf/conformal_method.yaml")
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="Alpha level used for result collection (default 0.2)",
    )
    parser.add_argument(
        "--collect-results", action="store_true",
        help="Collect and save a summary CSV after all experiments finish",
    )
    parser.add_argument(
        "--collect-results-only", action="store_true",
        help="Skip experiments; only build the summary CSV from existing result dirs",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if not args.collect_results_only:
        combos = [
            (m, agg, ev, task)
            for m    in args.methods
            for agg  in args.aggregators
            for ev   in args.evaluators
            for task in args.tasks
        ]
        total = len(combos)
        for i, (method, agg, ev, task) in enumerate(combos, 1):
            print(f"\n{'='*70}")
            print(
                f"[{i}/{total}]  method={method}  agg={agg}  "
                f"evaluator={ev}  task={task}"
            )
            print(f"{'='*70}")
            run_single(method, agg, ev, task, args.backbone, config)

    if args.collect_results or args.collect_results_only:
        print(f"\n{'='*70}")
        print("Collecting results...")
        df = collect_results(
            backbone=args.backbone,
            methods=args.methods,
            aggregators=args.aggregators,
            evaluators=args.evaluators,
            tasks=args.tasks,
            alpha=args.alpha,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = f"empirical_data/final_results_{timestamp}.csv"
        df.to_csv(out_path)
        print(f"Results saved → {out_path}")
        print(df.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
