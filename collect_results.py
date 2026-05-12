#!/usr/bin/env python3
"""
Script to collect and aggregate conformal prediction results from experiment directories.

This script reads existing removal_results.csv files from all experiment directories
and generates a comprehensive results table at a specified alpha level.

Usage:
    python collect_results.py [--alpha 0.2]
"""

import os
import re
import pandas as pd
from datetime import datetime
import argparse
from typing import Optional, Dict, Tuple


def find_latest_result_dir(method: str, evaluator: str, task_name: Optional[str] = None, 
                           base_dir: str = "empirical_data") -> Optional[str]:
    """Find the latest result directory for a given method, evaluator, and optional task name."""
    # Build the expected suffix pattern
    suffix_pattern = f"-gpt-4o-mini-{method}-{evaluator}"
    if task_name:
        suffix_pattern += f"-{task_name}"
    
    matching_dirs = []
    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        
        # Check if directory ends with the expected suffix pattern
        if dirname.endswith(suffix_pattern):
            matching_dirs.append(dirname)
    
    if not matching_dirs:
        return None
    
    # Sort by timestamp (first part of dirname)
    matching_dirs.sort(reverse=True)
    return os.path.join(base_dir, matching_dirs[0])


def get_removal_rate_at_alpha(result_dir: str, alpha: float = 0.2) -> Optional[str]:
    """Get removal rate at specific alpha from removal_results.csv."""
    csv_path = os.path.join(result_dir, "removal_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        row = df[df['alpha'] == alpha]
        
        if len(row) == 0:
            print(f"Warning: No data for alpha={alpha} in {csv_path}")
            return None
        
        mean = row['removal_rate_mean'].values[0]
        std = row['removal_rate_std'].values[0]
        
        return f"{mean:.2f}({std:.3f})"
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def collect_all_results(alpha: float = 0.2, base_dir: str = "empirical_data") -> pd.DataFrame:
    """
    Collect all experiment results at specified alpha level.
    
    Returns DataFrame with:
    - Rows: evaluator_method combinations
    - Columns: datasets (Who&When + task-specific)
    """
    
    # Define the structure
    methods = ["vanilla_max", "tree_crsvp_max", "tree_crsvp_sum"]
    evaluators = ["llm_naive", "llm_logprobs", "qwen3_ce_1_7b", "qwen3_ce_8b"]
    
    # Task names
    single_fm_tasks = ["dylan_gsm8k_2_3", "dylan_math_2_3", "macnet_gsm8k_2_3", "macnet_math_2_3"]
    mix_fm_tasks = ["dylan_gsm8k_mix", "dylan_math_mix", "macnet_gsm8k_mix", "macnet_math_mix"]
    all_tasks = single_fm_tasks + mix_fm_tasks
    
    # Column names
    columns = ["Who&When"] + all_tasks
    
    # Row names: evaluator_method combinations
    rows = []
    for evaluator in evaluators:
        for method in methods:
            rows.append(f"{evaluator}_{method}")
    
    # Initialize results dataframe
    results_df = pd.DataFrame(index=rows, columns=columns)
    
    print(f"Collecting results at alpha={alpha}")
    print("=" * 80)
    
    # Collect results
    for row_name in rows:
        # Parse row_name like "llm_naive_vanilla_max" or "qwen3_ce_1_7b_tree_crsvp_max"
        parts = row_name.split('_')
        
        # Method is always the last 2 or 3 parts
        if 'crsvp' in parts or 'binary' in parts:
            # tree_crsvp_max or tree_binary has 3 parts
            method = '_'.join(parts[-3:])
            evaluator = '_'.join(parts[:-3])
        else:
            # vanilla_max has 2 parts
            method = '_'.join(parts[-2:])
            evaluator = '_'.join(parts[:-2])
        
        print(f"\nProcessing: {evaluator} + {method}")
        
        # For Who&When (only llm_naive and llm_logprobs)
        if evaluator in ["llm_naive", "llm_logprobs"]:
            result_dir = find_latest_result_dir(method, evaluator, task_name=None, base_dir=base_dir)
            if result_dir:
                removal_rate = get_removal_rate_at_alpha(result_dir, alpha=alpha)
                results_df.loc[row_name, 'Who&When'] = removal_rate
                print(f"  Who&When: {removal_rate} (from {os.path.basename(result_dir)})")
            else:
                print(f"  Who&When: No directory found")
        
        # For all task-specific datasets
        for task in all_tasks:
            result_dir = find_latest_result_dir(method, evaluator, task_name=task, base_dir=base_dir)
            if result_dir:
                removal_rate = get_removal_rate_at_alpha(result_dir, alpha=alpha)
                results_df.loc[row_name, task] = removal_rate
                print(f"  {task}: {removal_rate} (from {os.path.basename(result_dir)})")
            else:
                print(f"  {task}: No directory found")
    
    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect conformal prediction experiment results"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.2,
        help="Alpha level for coverage guarantee (default: 0.2 for 80%% coverage)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="empirical_data",
        help="Base directory containing experiment results (default: empirical_data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: empirical_data/final_results_TIMESTAMP.csv)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CONFORMAL PREDICTION RESULTS COLLECTOR")
    print("=" * 80)
    print(f"Alpha level: {args.alpha} ({(1-args.alpha)*100:.0f}% coverage guarantee)")
    print(f"Base directory: {args.base_dir}")
    print("=" * 80)
    
    # Collect results
    results_df = collect_all_results(alpha=args.alpha, base_dir=args.base_dir)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{args.base_dir}/final_results_{timestamp}.csv"
    
    # Save results
    results_df.to_csv(output_path)
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    print("\nResults Summary:")
    print(results_df)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Coverage Statistics:")
    print("=" * 80)
    total_cells = results_df.size
    filled_cells = results_df.notna().sum().sum()
    print(f"Total cells: {total_cells}")
    print(f"Filled cells: {filled_cells}")
    print(f"Missing cells: {total_cells - filled_cells}")
    print(f"Completion: {filled_cells/total_cells*100:.1f}%")
    
    return results_df


if __name__ == "__main__":
    main()
