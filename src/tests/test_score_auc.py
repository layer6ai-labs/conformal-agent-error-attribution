import random
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from src.evaluator.llm_naive_evaluator import LLMNaiveEvaluator
from src.scorer.naive_scorer import NaiveScorer
from tqdm import tqdm
from datetime import datetime

# create more test data from one row by select a segement size and cut history within the size, 
# e.g., segment size = 3, and history size 5, we can create 3 new test data with history size 3 {[1,2,3], [2,3,4], [3,4,5]}
def generate_segmented_tests(row, segment_size):
    history = row['history']
    total_size = len(history)
    segmented_tests = []
    
    for start_idx in range(total_size - segment_size + 1):
        end_idx = start_idx + segment_size
        if end_idx >= total_size:
            end_idx = total_size
        new_row = row.copy()
        new_row['history'] = history
        new_row['groundtruth'] = row['groundtruth'] if row['groundtruth'] else row['ground_truth']
        new_row['mistake_agent'] = row.get('mistake_agent', None)
        new_row['mistake_step'] = row.get('mistake_step', None)
        new_row['segment_start'] = start_idx
        new_row['segment_end'] = end_idx
        segmented_tests.append(new_row)
    
    return segmented_tests


class ScorerAucTest:
    """Binary Classification: Segment Size Analysis with AUC-ROC"""
    
    def __init__(self, scorer=None, model='gpt-4o-mini'):
        """
        Initialize the ScorerAucRocTest.
        
        Args:
            scorer: NaiveScorer instance (optional)
        """
        self.scorer = scorer
        self.model = model
        self.colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    def prepare_data(self, df, segment_sizes=[5, 8, 10]):
        """Prepare segmented data for binary classification."""
        all_segments = {}
        
        for size in segment_sizes:
            segments = []
            for row in df.iloc:
                segs = generate_segmented_tests(row, segment_size=size)
                for seg in segs:
                    mistake_step = int(seg.get('mistake_step', -1))
                    segment_start = seg['segment_start']
                    segment_end = seg['segment_end']
                    seg['is_positive'] = (segment_start <= mistake_step < segment_end)
                    segments.append(seg)
            all_segments[size] = segments
        
        return all_segments

    def score_segments(self, segments, sample_size=None):
        """Score all segments using NaiveScorer."""
        if self.scorer is None:
            print("No scorer provided. Cannot score segments.")
            return None, None
        
        # Randomly choose sample_size / 2 'is_positive' samples and rest negative samples
        positive_segments = [s for s in segments if s["is_positive"]]
        negative_segments = [s for s in segments if not s["is_positive"]]
        # Half positive, half negative
        half_size = sample_size // 2

        # Randomly sample (with safety for smaller lists)
        sampled_positives = random.sample(positive_segments, min(half_size, len(positive_segments)))
        sampled_negatives = random.sample(negative_segments, min(half_size, len(negative_segments)))
        print(f"Sampled {len(sampled_positives)} positive and {len(sampled_negatives)} negative segments for scoring.")
        test_segments = sampled_positives + sampled_negatives
        scores = []
        labels = []
        
        for segment in tqdm(test_segments, desc="Scoring segments", leave=False):
            try:
                history = segment['history']
                problem = segment.get('question', '')
                answer = segment.get('groundtruth', segment.get('ground_truth', ''))
                segment_start = segment['segment_start']
                segment_end = segment['segment_end']
                
                score_data = {
                    "problem": problem,
                    "answer": answer,
                    "history": history,
                    "start": segment_start,
                    "end": segment_end,
                    "client": None
                }
                
                score = self.scorer.score(score_data)
                noise_sigma = 1e-3
                rng = np.random.default_rng()
                score = np.clip(score + rng.normal(0, noise_sigma), 0, 1)
                
                scores.append(score)
                labels.append(1 if segment['is_positive'] else 0)
            except Exception:
                continue
        
        return np.array(scores), np.array(labels)

    def plot_roc_curves(self, results_dict, save_path='empirical_data/segment_size_roc_curves.png'):
        """Plot ROC curves for different segment sizes."""
        plt.figure(figsize=(10, 8))
        
        for i, (size, (scores, labels, auc_score)) in enumerate(results_dict.items()):
            if scores is not None and labels is not None:
                fpr, tpr, _ = roc_curve(labels, scores)
                plt.plot(fpr, tpr, 
                        color=self.colors[i % len(self.colors)], 
                        linewidth=2,
                        label=f'Segment Size {size} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Different Segment Sizes\nBinary Classification: Mistake Detection', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return plt.gcf()

    def plot_prc_curves(self, results_dict, save_path='empirical_data/segment_size_prc_curves.png'):
        """Plot Precision-Recall Curves for different segment sizes."""
        plt.figure(figsize=(10, 8))
        
        for i, (size, (scores, labels, _)) in enumerate(results_dict.items()):
            if scores is not None and labels is not None:
                precision, recall, _ = precision_recall_curve(labels, scores)
                # Calculate PR AUC
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, 
                        color=self.colors[i % len(self.colors)], 
                        linewidth=2,
                        label=f'Segment Size {size} (PR-AUC = {pr_auc:.3f})')
        
        # Baseline is the proportion of positive samples
        if len(results_dict) > 0:
            # Get first result to calculate baseline
            first_result = next(iter(results_dict.values()))
            if first_result[1] is not None:
                baseline = np.mean(first_result[1])
                plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                           alpha=0.6, label=f'Baseline (Random: {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves for Different Segment Sizes\nBinary Classification: Mistake Detection', fontsize=14)
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return plt.gcf()

    def run_analysis(self, df, mode = 'roc', segment_sizes=[5, 8, 10], sample_size=50, 
                     save_path='empirical_data'):
        """
        Run the complete AUC-ROC analysis.
        
        Args:
            df: DataFrame with the data
            segment_sizes: List of segment sizes to test
            sample_size: Number of samples per segment size
            save_summary_path: Path to save summary statistics
            save_path: Path to save ROC curve plot
            
        Returns:
            dict: Results dictionary with scores, labels, and AUC for each size
        """
        #get a datatime string for unique file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluator_name = self.scorer.evaluator.__class__.__name__ if self.scorer else 'NoScorer'
        save_path = os.path.join(save_path, f'{timestamp}-{self.model}-{evaluator_name}-segment_size_analysis')
        os.makedirs(save_path, exist_ok=True)
        save_summary_path = os.path.join(save_path, f'segment_sample_size_{sample_size}_summary_stats.json')
        save_plot_path = os.path.join(save_path, f'segment_sample_size_{sample_size}_{mode}_curves.png')
        if len(df) == 0:
            print("No data available for analysis.")
            return {}
        
        print("=== GENERATING SEGMENTS ===")
        segment_data = self.prepare_data(df, segment_sizes=segment_sizes)
        
        for size, segments in segment_data.items():
            positive_count = sum(1 for s in segments if s['is_positive'])
            negative_count = len(segments) - positive_count
            print(f"Size {size}: {len(segments)} segments ({positive_count} positive, {negative_count} negative)")
        
        print("\n=== SCORING SEGMENTS ===")
        results = {}
        
        for size, segments in segment_data.items():
            print(f"Processing {len(segments)} segments of size {size}...")
            actual_sample_size = min(sample_size, len(segments))
            scores, labels = self.score_segments(segments, sample_size=actual_sample_size)
            #save scores and labels only if we have valid data
            data_valid_path = os.path.join(save_path, f'segment_size_{size}_scores_labels.json')
            if scores is not None and labels is not None:
                with open(data_valid_path, 'w') as f:
                    json.dump({
                        'scores': scores.tolist(),
                        'labels': labels.tolist()
                    }, f, indent=2)
                print(f"Saved scores and labels to: {data_valid_path}")
            
            if scores is not None and labels is not None and len(scores) > 0:
                auc_score = roc_auc_score(labels, scores)
                results[size] = (scores, labels, auc_score)
        
        if results:
            if mode == 'roc':
                # Plot ROC curves
                fig = self.plot_roc_curves(results, save_path=save_plot_path)
            elif mode == 'prc':
                # Plot PR curves
                fig = self.plot_prc_curves(results, save_path=save_plot_path)
            else:
                print(f"Unknown mode: {mode}. Skipping plot generation.")
                fig = None

            # Save summary statistics
            summary_stats = {
                size: {
                    'auc_score': float(auc),
                    'num_samples': len(scores),
                    'positive_samples': int(np.sum(labels)),
                    'negative_samples': int(len(labels) - np.sum(labels))
                }
                for size, (scores, labels, auc) in results.items()
            }
            
            with open(save_summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            print(f"\n curves saved to: {save_plot_path}")
            print(f"Summary statistics saved to: {save_summary_path}")
        
        return results