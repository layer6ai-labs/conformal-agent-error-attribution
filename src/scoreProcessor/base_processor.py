from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processors that handle inference and calibration data preparation.
    """
    
    @abstractmethod
    def process_single_row(
        self,
        idx: int,
        row: pd.Series,
        scorer: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single row of data.
        
        Args:
            idx: Row index
            row: Row data from DataFrame
            scorer: Scorer instance for computing results
            
        Returns:
            Processed result dictionary with 'conversation_id' key, or None to skip
        """
        pass
    
    def process_data(
        self, 
        df: pd.DataFrame, 
        scorer: Optional[Any], 
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process raw dataset into inference results with resume capability.
        
        Args:
            df: Input DataFrame containing conversation data
            scorer: Scorer instance for computing inference results
            output_file: Optional path to save results as JSONL
            
        Returns:
            List of processed inference results
        """
        results = []
        processed_indices = set()
        start_idx = 0
        
        # Check if output file exists and load existing results
        if output_file:
            if os.path.exists(output_file):
                print(f"Found existing output file: {output_file}")
                try:
                    with open(output_file, 'r') as f:
                        for line in f:
                            result = json.loads(line.strip())
                            results.append(result)
                            processed_indices.add(result['conversation_id'])
                    print(f"Loaded {len(results)} existing results")
                    start_idx = len(results)
                except Exception as e:
                    print(f"Error loading existing results: {e}")
                    results = []
                    processed_indices = set()
                    start_idx = 0

        # Create progress bar starting from the correct position
        pbar = tqdm(df.iloc[start_idx:].iterrows(), total=df.shape[0], initial=start_idx)
        
        for idx, row in pbar:
            # Skip if already processed (shouldn't happen with correct slicing)
            if idx in processed_indices:
                continue
            
            # Process single row
            result_entry = self.process_single_row(idx, row, scorer)
            
            # Skip if process_single_row returns None
            if result_entry is None:
                continue
            
            # Ensure conversation_id is set
            if 'conversation_id' not in result_entry:
                result_entry['conversation_id'] = idx

            # Add Gaussian noise N(0, 0.001) to score when using LLMNaiveEvaluator or Qwen3CrossEntropyEvaluator
            if (
                scorer is not None
                and hasattr(scorer, 'evaluator')
                and type(scorer.evaluator).__name__ in ['LLMNaiveEvaluator', 'Qwen3CrossEntropyEvaluator']
                and 'score' in result_entry
            ):
                result_entry['noise'] = float(np.random.normal(0, 1e-3))

            results.append(result_entry)
            
            # Save incrementally after each result
            if output_file:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result_entry) + '\n')
        
        # Close progress bar
        pbar.close()
            
        return results

    def _save_to_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save data to a JSONL file.
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')