# Conformal Agent Error Attribution

**ArXiv**: [arXiv:2605.06788](https://arxiv.org/abs/2605.06788v1)

**HuggingFace**: [huggingface:2605.06788](https://huggingface.co/papers/2605.06788)

**Finetuned Model**: [checkpoint.zip](https://drive.google.com/file/d/1_VwNu5IKSUNa8GWW8rFu0s-V9ZX-1IJC/view?usp=sharing)

## Abstract

When multi-agent systems (MAS) fail, identifying where the decisive error occurred is the first step for automated recovery to an earlier state. Error attribution remains a fundamental challenge due to the long interaction traces that large language model-based MAS generate. This paper presents a framework for error attribution based on conformal prediction (CP) which provides finite-sample, distribution-free coverage guarantees. We introduce new algorithms for filtration-based CP designed for sequential data such as agent trajectories. Unlike existing CP algorithms, our approach predicts sets that are contiguous sequences to enable efficient recovery and debugging. We verify our theoretical guarantees on a variety of agents and datasets, show that errors can be precisely isolated, then use prediction sets to rollback MAS to correct their own errors. Our overall approach is model-agnostic, and offers a principled uncertainty layer for MAS error attribution.

---

## Pipeline Overview

The codebase is organized as a two-phase pipeline:

```
Phase 1 (score.py)      Phase 2 (conformal.py)
─────────────────       ──────────────────────────
Source JSONL            node_results-*-*.jsonl
    │                          │
    ▼                          ▼
Evaluator model        preprocess_data(aggregator)
(API / Qwen3)               │
    │                        ▼
    ▼              Conformal method (compute_results)
node_results-               │
{task}-{backbone}-           ▼
{evaluator}.jsonl   empirical_data/{ts}-…/
                      accuracy_results.csv
                      removal_results.csv
```

**Key separation**: evaluators and API/model calls only appear in Phase 1. Phase 2 is pure Python — no GPU, no network.

---

## Repository Structure

```
├── conf/                   # Configuration files
│   └── conformal_method.yaml
├── empirical_data/         # Processed data and experiment results
├── logs/                   # Experiment logs organized by timestamp-model-method
├── src/
│   ├── conformal/          # Conformal prediction implementations
│   │   ├── vanilla_conformal.py         # Node-level conformal prediction
│   │   ├── filtering_conformal.py       # Left/right/two-way filtering conformal prediction
│   │   └── tree_binary_conformal.py     # Tree-based conformal prediction
│   ├── scoreProcessor/     # Data processing pipelines
│   │   ├── node_processor.py            # Node-level data processing
│   │   ├── hierarchical_processor.py    # Tree-based data processing
│   │   └── filtering_processor.py       # Filtering data processing
│   ├── evaluator/          # LLM-based evaluators
│   │   ├── llm_naive_evaluator.py       # Naive LLM-as-a-judge
│   │   ├── llm_logprobs_evaluator.py    # Logprobs-based LLM evaluator
│   │   ├── llm_ECHOprompt_evaluator.py  # Multi-perspective LLM evaluator
│   │   └── qwen3_crossentropy_evaluator.py  # Fine-tuned Qwen3 evaluator
│   └── scorer/             # Scoring mechanisms (sum and max aggregate scorer)
├── baseline_stat.ipynb     # Supplementary statistics notebook
├── main.py                 # Main experiment runner
└── run_conformal_experiments.py  # Batch orchestrator and report generator
```

---

## Node-Results Filename Convention

Scored output files follow the pattern:

```
empirical_data/node_results-{task}-{backbone}-{evaluator}.jsonl
```

Examples:
```
empirical_data/node_results-dylan_math_uniform-gpt-4o-mini-llm_naive.jsonl
empirical_data/node_results-macnet_gsm8k_uniformleft-gpt-4o-mini-qwen3_ce_1_7b_uniform.jsonl
empirical_data/node_results-whoandwhen-gpt-4o-mini-agent_echo.jsonl
```

---

## Usage

### Phase 1: `score.py`

Checks each `node_results-{task}-{backbone}-{evaluator}.jsonl` file. If a file is **missing or incomplete** (line count < expected source records), loads the required evaluator and runs `NodeDataProcessor` to fill it. Evaluators are loaded once per type even when many tasks are pending, so a Qwen3 checkpoint is loaded at most once per invocation. Uses `NaiveScorer` → `NodeDataProcessor` (temperature-softmax over nodes), writing `probability`, `score`, `nodes_logit`, and `noise` fields per record.

```bash
# By dataset name (auto-discovers tasks)
python score.py --datasets uniform uniformleft uniformmid uniformright \
                --evaluators llm_naive agent_echo --backbone gpt-4o-mini

# Who&When
python score.py --datasets whoandwhen --evaluators llm_naive

# Explicit task names
python score.py --tasks dylan_gsm8k_uniformleft macnet_math_uniform \
                --evaluators qwen3_ce_1_7b_uniform

# Force re-generation (even if files are complete)
python score.py --datasets uniform --evaluators llm_naive --force
```

| Argument | Default | Description |
|---|---|---|
| `--datasets` | — | Dataset name(s); mutually exclusive with `--tasks` |
| `--tasks` | — | Explicit task name(s); auto-resolves to source JSONL |
| `--evaluators` | required | Evaluator name(s) |
| `--backbone` | `gpt-4o-mini` | Model name embedded in the output filename |
| `--config` | `conf/conformal_method.yaml` | YAML config (evaluator checkpoints) |
| `--force` | `False` | Re-generate even if complete |

---

### Phase 2: `conformal.py`

Loads `node_results-{task}-{backbone}-{evaluator}.jsonl`, calls `preprocess_data(aggregator)` to compute prefix/suffix sums and conformal scores, instantiates the selected conformal class, and saves results to `empirical_data/{timestamp}-{backbone}-{method}-{aggregator}-{evaluator}-{task}/`.

Computed fields from `preprocess_data`:
- `right_sum_probabilities`, `left_sum_probabilities` (prefix/suffix sums)
- `right_score`, `left_score`, `score`
- `adv_2way_score`, `adv_2way_range` (for `adv_2way_filter`)

**Methods supported**

| Method key | Conformal class |
|---|---|
| `vanilla` | `VanillaConformal` |
| `right_filter` | `FilteringConformal(is_right_filter=True)` |
| `left_filter` | `FilteringConformal(is_right_filter=False)` |
| `2way_filter` | `TwoWayFilteringConformal` |
| `adv_2way_filter` | `AdvancedFilteringConformal` |
| `tree_crsvp` | `TreeHierarchicalRestrictConformal` |

**Aggregators supported**

| Name pattern | Description |
|---|---|
| `sum` | Prefix/suffix sum normalised by n |
| `max` | Running max normalised by n |
| `logsumexp_<b>` | LogSumExp with temperature β |
| `normalizedlogsumexp_<b>` | Normalized LogSumExp |
| `lengthpenalized_<l>` | Sum + λ·log(n) |
| `lengthpenalizedwithmax_<l>` | Length-penalised with max |

```bash
# Run experiments for a set of combos
python conformal.py \
    --methods right_filter left_filter 2way_filter \
    --aggregators sum \
    --evaluators llm_naive agent_echo \
    --tasks dylan_gsm8k_uniformleft macnet_gsm8k_uniformleft

# Run + collect summary CSV (alpha=0.2 by default)
python conformal.py \
    --methods vanilla right_filter left_filter 2way_filter adv_2way_filter tree_crsvp \
    --aggregators sum \
    --evaluators llm_naive \
    --tasks dylan_math_uniform macnet_math_uniform whoandwhen \
    --collect-results

# Only collect results from existing dirs (no new runs)
python conformal.py --collect-results-only \
    --methods right_filter --aggregators sum \
    --evaluators llm_naive \
    --tasks dylan_gsm8k_uniformleft macnet_gsm8k_uniformleft \
    --alpha 0.2
```

| Argument | Default | Description |
|---|---|---|
| `--methods` | required | Conformal method(s) |
| `--aggregators` | required | Aggregator(s) |
| `--evaluators` | required | Evaluator name(s) (selects node_results file) |
| `--tasks` | required | Task name(s) |
| `--backbone` | `gpt-4o-mini` | Backbone name in file lookup |
| `--config` | `conf/conformal_method.yaml` | YAML config (alphas, n_trials, seed) |
| `--alpha` | `0.2` | Alpha for summary CSV |
| `--collect-results` | `False` | Save summary CSV after experiments |
| `--collect-results-only` | `False` | Skip experiments; only collect |

---

### Batch Orchestrator: `run_conformal_experiments.py`

Edit the **Configuration block** at the top of the file, then run:

```bash
python run_conformal_experiments.py
```

```python
METHODS             = ["right_filter", "left_filter", "2way_filter",
                        "adv_2way_filter", "tree_crsvp"]
AGGREGATORS         = ["sum"]               # extend as needed
DATASETS_TO_USE     = ["uniform",           # source JSONL datasets
                        "uniformleft", "uniformmid", "uniformright"]
DEFAULT_EVALUATORS  = ["llm_naive"]         # for Who&When
FINETUNE_EVALUATORS = ["llm_naive"]         # for JSONL datasets
BACKBONE            = "gpt-4o-mini"
ALPHA               = 0.2
```

```bash
# Full pipeline (scoring + experiments + collect results)
python run_conformal_experiments.py

# Skip scoring (all node_results already exist)
python run_conformal_experiments.py --skip-scoring

# Only collect results from existing runs
python run_conformal_experiments.py --collect-only
```

---

### Parallel Execution

Use `run_conformal_experiments_from_cache_parallel.sh` for parallel Phase 2 runs (calls `main.py` directly). Alternatively, run `conformal.py` with `xargs` or `parallel`:

```bash
# Example: 4 parallel workers over task list
printf '%s\n' dylan_math_uniform macnet_math_uniform \
              dylan_gsm8k_uniformleft macnet_gsm8k_uniformleft \
  | xargs -P 4 -I{} python conformal.py \
      --methods right_filter left_filter \
      --aggregators sum \
      --evaluators llm_naive \
      --tasks {}
```

---

## Notes

`preprocess_data()` in `src/aggregator/utils.py` computes two additional fields required by `adv_2way_filter`:

- **`adv_2way_score`** – oracle bisection score used by `AdvancedFilteringConformal` during calibration (`compute_tau_star`). Derived from `right_sum_probabilities` using the same shrinking-window algorithm as `FilteringDataProcessor`.
- **`adv_2way_range`** – `[adv_left, adv_right]` interval containing `true_fail`.

This means `adv_2way_filter` works with `NodeDataProcessor` output (via `score.py`) without needing the legacy `FilteringDataProcessor` / `filtering_sum_node_results-*` files.

---

## License

MIT

