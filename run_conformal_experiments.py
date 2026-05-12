#!/usr/bin/env python3
"""
Batch runner for conformal prediction experiments.

Orchestrates the two-phase pipeline:
  Phase 1 (score.py)     – Ensures all node_results files are complete.
                           Loads evaluator models/APIs only when a file is missing.
  Phase 2 (conformal.py) – Runs conformal prediction from cached node_results.
                           No model or API calls.

Edit the Configuration block below, then run:
    python run_conformal_experiments.py

To skip Phase 1 (scoring) and jump straight to conformal:
    python run_conformal_experiments.py --skip-scoring

To skip Phase 2 (experiments) and only collect results into a CSV:
    python run_conformal_experiments.py --collect-only
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import List

import pandas as pd

sys.path.append(os.getcwd())

# ── Configuration ──────────────────────────────────────────────────────────────
# [ "vanilla", "left_filter", "right_filter", "2way_filter", "adv_2way_filter", "tree_crsvp"]
METHODS = [ "vanilla", "tree_crsvp", "left_filter", "right_filter", "2way_filter", ]

# Uncomment additional aggregators as needed:
LOGSUMEXP_BETAS        = [0.01, 0.1, 1.0, 10.0]
LENGTH_PENALTY_LAMBDAS = [0.01, 0.02, 0.05, 0.1]

AGGREGATORS = (
    ["sum"]
    # + ["max"]
    # + [f"logsumexp_{b}"             for b in LOGSUMEXP_BETAS]
    # + [f"normalizedlogsumexp_{b}"   for b in LOGSUMEXP_BETAS]
    # + [f"lengthpenalized_{l}"        for l in LENGTH_PENALTY_LAMBDAS]
    # + [f"lengthpenalizedwithmax_{l}" for l in LENGTH_PENALTY_LAMBDAS]
)

# Datasets to process  (score.py discovers tasks automatically from source JSONLs)
# Choices: "uniform", "uniformleft", "uniformmid", "uniformright", "whoandwhen"
DATASETS_TO_USE = ["whoandwhen", "uniform", "uniformleft", "uniformmid", "uniformright"]

# Evaluators for Who&When (DEFAULT) and JSONL datasets (FINETUNE)
# DEFAULT: ["llm_naive", "llm_logprobs", "llm_echo", "agent_echo"]
# FINETUNE: ["llm_naive", "agent_echo", "qwen3_ce_1_7b_uniform", ...]
DEFAULT_EVALUATORS  = ["llm_naive", "agent_echo"]
FINETUNE_EVALUATORS = ["llm_naive", "agent_echo", "qwen3_ce_1_7b_uniform"]

BACKBONE = "gpt-4o-mini"
ALPHA    = 0.2

# ── Task registry (must stay in sync with score.py DATASET_REGISTRY) ──────────
# Maps dataset name → list of task names produced from that source JSONL.
DATASET_TASKS = {
    "uniform":      ["dylan_math_uniform",       "macnet_math_uniform"],
    "uniformleft":  ["dylan_gsm8k_uniformleft",  "macnet_gsm8k_uniformleft"],
    "uniformmid":   ["dylan_gsm8k_uniformmid",   "macnet_gsm8k_uniformmid"],
    "uniformright": ["dylan_gsm8k_uniformright",  "macnet_gsm8k_uniformright"],
    "whoandwhen":   ["whoandwhen"],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_subprocess(cmd: List[str]) -> bool:
    """Run a subprocess with real-time output streaming. Returns True on success."""
    print(f"\n{'─'*70}")
    print(f"$ {' '.join(cmd)}")
    print(f"{'─'*70}\n")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    rc = process.wait()
    if rc != 0:
        print(f"\n[ERROR] Process exited with code {rc}")
    return rc == 0


def _all_tasks_for_datasets(datasets: List[str]) -> List[str]:
    tasks = []
    for ds in datasets:
        tasks.extend(DATASET_TASKS.get(ds, []))
    return tasks


# ── Phase 1: scoring ───────────────────────────────────────────────────────────

def phase1_scoring(
    who_and_when_datasets: List[str],
    jsonl_datasets: List[str],
    default_evals: List[str],
    finetune_evals: List[str],
    backbone: str,
) -> None:
    """Call score.py to ensure all node_results files are complete."""
    # Who&When uses default evaluators
    if who_and_when_datasets and default_evals:
        cmd = (
            [sys.executable, "score.py"]
            + ["--datasets"] + who_and_when_datasets
            + ["--evaluators"] + default_evals
            + ["--backbone", backbone]
        )
        if not _run_subprocess(cmd):
            print("Warning: scoring phase had errors for Who&When datasets.")

    # JSONL datasets use finetune evaluators
    if jsonl_datasets and finetune_evals:
        cmd = (
            [sys.executable, "score.py"]
            + ["--datasets"] + jsonl_datasets
            + ["--evaluators"] + finetune_evals
            + ["--backbone", backbone]
        )
        if not _run_subprocess(cmd):
            print("Warning: scoring phase had errors for JSONL datasets.")


# ── Phase 2: conformal ─────────────────────────────────────────────────────────

def phase2_conformal(
    methods: List[str],
    aggregators: List[str],
    who_and_when_tasks: List[str],
    jsonl_tasks: List[str],
    default_evals: List[str],
    finetune_evals: List[str],
    backbone: str,
    collect: bool,
) -> None:
    """Call conformal.py for all combinations."""
    all_combos = []

    # Who&When: methods × aggregators × default_evals × whoandwhen tasks
    if who_and_when_tasks and default_evals:
        all_combos.append((who_and_when_tasks, default_evals))

    # JSONL datasets: methods × aggregators × finetune_evals × jsonl_tasks
    if jsonl_tasks and finetune_evals:
        all_combos.append((jsonl_tasks, finetune_evals))

    for tasks, evals in all_combos:
        cmd = (
            [sys.executable, "conformal.py"]
            + ["--methods"]     + methods
            + ["--aggregators"] + aggregators
            + ["--evaluators"]  + evals
            + ["--tasks"]       + tasks
            + ["--backbone",     backbone]
            + (["--collect-results"] if collect else [])
        )
        if not _run_subprocess(cmd):
            print("Warning: conformal phase had errors.")


# ── Result collection (standalone) ────────────────────────────────────────────

def collect_results(
    methods: List[str],
    aggregators: List[str],
    evaluators: List[str],
    tasks: List[str],
    backbone: str,
    alpha: float,
) -> None:
    """Call conformal.py --collect-results-only to build summary CSV."""
    cmd = (
        [sys.executable, "conformal.py", "--collect-results-only"]
        + ["--methods"]     + methods
        + ["--aggregators"] + aggregators
        + ["--evaluators"]  + evaluators
        + ["--tasks"]       + tasks
        + ["--backbone",     backbone]
        + ["--alpha",        str(alpha)]
    )
    _run_subprocess(cmd)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch conformal experiment runner.")
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Skip Phase 1 (assume all node_results files exist)",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="Skip experiments; only collect existing results into a summary CSV",
    )
    args = parser.parse_args()

    # Separate datasets by type
    who_and_when_datasets = [d for d in DATASETS_TO_USE if d == "whoandwhen"]
    jsonl_datasets        = [d for d in DATASETS_TO_USE if d != "whoandwhen"]

    who_and_when_tasks = _all_tasks_for_datasets(who_and_when_datasets)
    jsonl_tasks        = _all_tasks_for_datasets(jsonl_datasets)
    all_tasks          = who_and_when_tasks + jsonl_tasks
    all_evaluators     = sorted(set(DEFAULT_EVALUATORS + FINETUNE_EVALUATORS))

    print("=" * 70)
    print("CONFORMAL EXPERIMENTS BATCH RUNNER")
    print("=" * 70)
    print(f"  Methods:             {METHODS}")
    print(f"  Aggregators:         {AGGREGATORS}")
    print(f"  Default evaluators:  {DEFAULT_EVALUATORS}")
    print(f"  Finetune evaluators: {FINETUNE_EVALUATORS}")
    print(f"  Datasets:            {DATASETS_TO_USE}")
    print(f"  Tasks ({len(all_tasks)}):  {all_tasks}")
    print(f"  Backbone:            {BACKBONE}")
    print(f"  Alpha:               {ALPHA}")
    print("=" * 70)

    if args.collect_only:
        print("\n[collect-only mode] Skipping scoring and experiments.\n")
        collect_results(
            methods=METHODS,
            aggregators=AGGREGATORS,
            evaluators=all_evaluators,
            tasks=all_tasks,
            backbone=BACKBONE,
            alpha=ALPHA,
        )
        return

    # ── Phase 1: Score generation ──────────────────────────────────────────
    if not args.skip_scoring:
        print("\n" + "=" * 70)
        print("PHASE 1: Score generation")
        print("=" * 70)
        phase1_scoring(
            who_and_when_datasets=who_and_when_datasets,
            jsonl_datasets=jsonl_datasets,
            default_evals=DEFAULT_EVALUATORS,
            finetune_evals=FINETUNE_EVALUATORS,
            backbone=BACKBONE,
        )
    else:
        print("\n[--skip-scoring] Phase 1 skipped.\n")

    # ── Phase 2: Conformal experiments ────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Conformal experiments")
    print("=" * 70)
    phase2_conformal(
        methods=METHODS,
        aggregators=AGGREGATORS,
        who_and_when_tasks=who_and_when_tasks,
        jsonl_tasks=jsonl_tasks,
        default_evals=DEFAULT_EVALUATORS,
        finetune_evals=FINETUNE_EVALUATORS,
        backbone=BACKBONE,
        collect=True,
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
