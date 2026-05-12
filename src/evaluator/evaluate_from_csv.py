"""
Evaluation script for base evaluators using CSV data.
Adapts evaluators to work with the FailureDetectionFinetune evaluation pipeline.
"""
import pandas as pd
import numpy as np
import torch
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any
import os
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class EvaluatorCSVDataset(Dataset):
    """
    Dataset that loads CSV data and formats it for evaluator input.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize dataset from CSV file.
        
        Args:
            csv_path: Path to the CSV file with evaluation data
        """
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Parse history column (stored as string representation of list)
        print("Parsing history data...")
        self.df['history'] = self.df['history'].apply(self._parse_history)
        
        print(f"Loaded {len(self.df)} samples")
        print(f"  - Positive samples (c_t=1): {self.df['c_t'].sum()}")
        print(f"  - Negative samples (c_t=0): {(self.df['c_t']==0).sum()}")
    
    def _parse_history(self, history_str):
        """Parse history string to list of dicts."""
        try:
            return ast.literal_eval(history_str)
        except:
            # If parsing fails, return empty list
            return []
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns dict with:
            - problem: the task/question
            - answer: ground truth answer
            - history: conversation history
            - start: start index (t)
            - end: end index (t+1 for single step)
            - label: ground truth label (c_t)
        """
        row = self.df.iloc[idx]
        
        return {
            'problem': row['r_i'],
            'answer': row['ground_truth'],
            'history': row['history'],
            'start': int(row['t']),
            'end': int(row['t']) + 1,  # Single step evaluation
            'label': float(row['c_t']),
            'task_id': row['task_id']
        }


class EvaluatorModelWrapper:
    """
    Wrapper that makes an evaluator work like a PyTorch model for evaluation.
    """
    
    def __init__(self, evaluator, client=None, use_sigmoid=False):
        """
        Initialize wrapper.
        
        Args:
            evaluator: BaseEvaluator instance
            client: Optional OpenAI client for LLM-based evaluators
            use_sigmoid: If True, apply sigmoid to outputs (for logit-based evaluators like Qwen3)
        """
        self.evaluator = evaluator
        self.client = client
        self.use_sigmoid = use_sigmoid
        self.eval()  # Set to eval mode by default
    
    def eval(self):
        """Set to eval mode (for compatibility with PyTorch models)."""
        self._is_training = False
        return self
    
    def train(self):
        """Set to train mode (for compatibility with PyTorch models)."""
        self._is_training = True
        return self
    
    def predict_batch(self, batch):
        """
        Make predictions for a batch.
        
        Args:
            batch: Dictionary with 'problem', 'answer', 'history', 'start', 'end' keys
            
        Returns:
            predictions: numpy array of probabilities
        """
        predictions = []
        
        for i in range(len(batch['problem'])):
            # Prepare data for evaluator
            data = {
                'problem': batch['problem'][i],
                'answer': batch['answer'][i],
                'history': batch['history'][i],
                'start': batch['start'][i],
                'end': batch['end'][i],
                'client': self.client
            }
            
            # Get prediction
            try:
                pred = self.evaluator.evaluate(data)
                
                # Apply sigmoid if needed (for logit-based evaluators)
                if self.use_sigmoid:
                    pred = 1.0 / (1.0 + np.exp(-pred))
                
                predictions.append(pred)
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                predictions.append(0.0)  # Default to 0 on error
        
        return np.array(predictions)


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with batched data
    """
    return {
        'problem': [item['problem'] for item in batch],
        'answer': [item['answer'] for item in batch],
        'history': [item['history'] for item in batch],
        'start': [item['start'] for item in batch],
        'end': [item['end'] for item in batch],
        'labels': torch.tensor([item['label'] for item in batch], dtype=torch.float32),
        'task_id': [item['task_id'] for item in batch]
    }


def evaluate_evaluator(
    evaluator_wrapper: EvaluatorModelWrapper,
    data_loader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Evaluate an evaluator model on a dataset.
    Reuses evaluation logic from FailureDetectionFinetune.
    
    Args:
        evaluator_wrapper: EvaluatorModelWrapper instance
        data_loader: DataLoader with evaluation data
        device: Device (not used for evaluators, but kept for compatibility)
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator_wrapper.eval()
    all_preds = []
    all_labels = []
    
    print("\nRunning evaluation...")
    for batch in tqdm(data_loader, desc="Evaluating"):
        # Get predictions
        preds = evaluator_wrapper.predict_batch(batch)
        labels = batch['labels'].numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics (reusing logic from FailureDetectionFinetune)
    binary_preds = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    prc_auc = auc(recall, precision)
    
    print(f"\nEvaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PRC AUC: {prc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'prc_auc': prc_auc,
        'predictions': all_preds,
        'labels': all_labels,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }


def plot_roc_prc(results: Dict[str, Any], output_dir: str, evaluator_name: str):
    """
    Plot ROC and PRC curves.
    Reuses plotting logic from FailureDetectionFinetune.
    
    Args:
        results: Results dictionary from evaluate_evaluator
        output_dir: Directory to save plots
        evaluator_name: Name of evaluator for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ROC Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['fpr'], results['tpr'], 
             label=f"ROC (AUC = {results['roc_auc']:.4f})", 
             color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {evaluator_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # PRC Plot
    plt.subplot(1, 2, 2)
    plt.plot(results['recall'], results['precision'],
             label=f"PRC (AUC = {results['prc_auc']:.4f})",
             color='blue', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {evaluator_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{evaluator_name}_roc_prc.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {plot_path}")


def save_results(results: Dict[str, Any], output_dir: str, evaluator_name: str):
    """
    Save evaluation results to CSV.
    
    Args:
        results: Results dictionary from evaluate_evaluator
        output_dir: Directory to save results
        evaluator_name: Name of evaluator
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'evaluator': [evaluator_name],
        'accuracy': [results['accuracy']],
        'roc_auc': [results['roc_auc']],
        'prc_auc': [results['prc_auc']]
    })
    
    results_path = os.path.join(output_dir, f'{evaluator_name}_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"Results saved to: {results_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'prediction': results['predictions'],
        'label': results['labels']
    })
    
    predictions_path = os.path.join(output_dir, f'{evaluator_name}_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    print(f"Predictions saved to: {predictions_path}")


def main():
    """
    Main function to run evaluation.
    """
    # Example usage
    from src.evaluator.llm_naive_evaluator import LLMNaiveEvaluator
    from openai import OpenAI
    import os
    
    # Configuration
    csv_path = "empirical_data/experiment_20251229_105821/data/test_data.csv"
    output_dir = "empirical_data/evaluator_evaluation_results"
    batch_size = 8
    
    # Create dataset and dataloader
    dataset = EvaluatorCSVDataset(csv_path)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize evaluator
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in .env file")
    client = OpenAI()
    evaluator = LLMNaiveEvaluator(model="gpt-4o-mini")
    
    # Wrap evaluator
    evaluator_wrapper = EvaluatorModelWrapper(evaluator, client=client)
    
    # Evaluate
    results = evaluate_evaluator(evaluator_wrapper, data_loader)
    
    # Plot results
    plot_roc_prc(results, output_dir, "llm_naive_evaluator")
    
    # Save results
    save_results(results, output_dir, "llm_naive_evaluator")


if __name__ == "__main__":
    main()
