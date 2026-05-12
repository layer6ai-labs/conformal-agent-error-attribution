"""
Qwen3 Cross-Entropy Evaluator for Failure Detection

This evaluator uses a fine-tuned Qwen3 model with LoRA adapters to evaluate
trajectory segments from the Who_and_When dataset format. It returns raw logits
for cross-entropy based conformal prediction.
"""

from typing import Any, Dict
import pandas as pd
import torch

from .base_evaluator import BaseEvaluator
from ..logger import get_logger
from ..finetune.Qwen3LoraFailureClassifier import Qwen3LoraFailureClassifier


class Qwen3CrossEntropyEvaluator(BaseEvaluator):
    """
    Evaluator that uses a fine-tuned Qwen3 model to score trajectory segments.
    Returns raw logits for use in cross-entropy based conformal prediction.
    """

    def __init__(
        self, 
        checkpoint_path: str,
        device: str = None,
        max_length: int = 4096,
        logger_name: str = "agentic_conformal"
    ) -> None:
        """
        Initialize the Qwen3 Cross-Entropy Evaluator.
        
        Args:
            checkpoint_path: Path to the fine-tuned model checkpoint (.pt file)
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
            max_length: Maximum sequence length for tokenization
            logger_name: Name for the logger
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_length = max_length
        self.logger = get_logger(logger_name)
        
        # Initialize the classifier
        self.logger.info(f"Loading Qwen3 classifier from {checkpoint_path}")
        self.classifier = Qwen3LoraFailureClassifier(
            checkpoint_path=checkpoint_path,
            device=device,
            max_length=max_length
        )
        self.logger.info("Qwen3 classifier loaded successfully")
    
    def evaluate(self, input_data: Any) -> float:
        """
        Evaluate a single trajectory step from Who_and_When dataset format.
        
        Args:
            input_data: Dictionary containing trajectory information in Who_and_When format
                Required keys:
                    - "history": List of conversation steps (list of dicts with 'name' and 'content')
                    - "problem": The task/request string (r_i)
                    - "start": Starting index of the segment to evaluate (t)
                    - "end": Ending index of the segment (must be start + 1)
                
        Returns:
            Raw logit value from the classifier (float)
            
        Raises:
            ValueError: If end - start != 1 (this evaluator only handles single steps)
            ValueError: If required keys are missing from input_data
        """
        data: Dict[str, Any] = input_data
        
        # Validate required keys
        required_keys = ["history", "problem", "start", "end"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in input_data: {missing_keys}. "
                f"Required keys are: {required_keys}"
            )
        
        history = data["history"]
        problem = data["problem"]
        start = int(data["start"])
        end = int(data["end"])
        
        # Validate single trajectory constraint
        if end - start != 1:
            raise ValueError(
                f"This evaluator only supports single trajectory evaluation at a time. "
                f"Expected end - start = 1, but got end={end}, start={start}, "
                f"difference={end - start}. Please provide data for a single step only."
            )
        
        # Validate segment bounds
        if start < 0 or start >= len(history):
            raise ValueError(
                f"Invalid start index: {start}. Must be in range [0, {len(history) - 1}]"
            )
        
        # The current step is at index `start` (since end = start + 1)
        current_step_idx = start
        
        # Format the input following Who8WhenDataPreparation.format_step_input logic
        formatted_input = self._format_step_input(
            question=problem,
            history=history,
            current_step_idx=current_step_idx
        )
        
        # Get logit from classifier
        logit = self.classifier.predict_logit(formatted_input)
        
        self.logger.debug(
            f"Evaluated step {current_step_idx}/{len(history)}: logit={logit:.4f}"
        )
        
        #return sigmoid of logit for cross-entropy based conformal prediction
        return torch.sigmoid(torch.tensor(logit)).item()
    
    def _format_step_input(
        self, 
        question: str, 
        history: list, 
        current_step_idx: int
    ) -> str:
        """
        Format a single step with historical context for model input.
        Follows the same format as Who8WhenDataPreparation.format_step_input.
        
        Includes up to 3 steps before and after the current step for context.
        
        Args:
            question: The task/request (r_i)
            history: List of conversation steps
            current_step_idx: Index of the current step to evaluate (t)
            
        Returns:
            Formatted prompt string
        """
        # Normalize history to ensure consistent format
        normalized_history = self._normalize_history(history)
        
        current_step = current_step_idx
        total_steps = len(normalized_history)
        
        # Build context from previous steps (up to 3)
        context_before = []
        for i in range(max(0, current_step - 3), current_step):
            step = normalized_history[i]
            content_truncated = step.get('content', '')[:1000]
            context_before.append(
                f"[Step {i}] {step.get('name', 'unknown')}: {content_truncated}..."
            )
        
        # Build context from next steps (up to 3)
        context_after = []
        for i in range(current_step + 1, min(total_steps, current_step + 4)):
            step = normalized_history[i]
            content_truncated = step.get('content', '')[:1000]
            context_after.append(
                f"[Step {i}] {step.get('name', 'unknown')}: {content_truncated}..."
            )
        
        # Get current step info
        current_step_data = normalized_history[current_step]
        agent_name = current_step_data.get('name', 'unknown')
        step_content = current_step_data.get('content', '')
        
        # Format the prompt (matching Who8WhenDataPreparation format)
        prompt = f"""Task: {question}

{'--- Previous Steps ---' if context_before else ''}
{chr(10).join(context_before) if context_before else '(No previous steps)'}

--- Current Step (Step {current_step} of {total_steps}) ---
Agent: {agent_name}
{step_content}

{'--- Next Steps ---' if context_after else ''}
{chr(10).join(context_after) if context_after else '(No next steps)'}

Question: Is this step likely to be the failure point in the multi-agent system execution?
Answer (0 for correct, 1 for failure):"""
        
        return prompt
    
    @staticmethod
    def _normalize_history(history: list) -> list:
        """
        Normalize history entries to ensure consistent dict format.
        
        Args:
            history: List of conversation steps (can be strings or dicts)
            
        Returns:
            List of dicts with 'name' and 'content' keys
        """
        normalized = []
        for i, entry in enumerate(history):
            if isinstance(entry, str):
                # Parse "Agent: content" format
                if ": " in entry:
                    parts = entry.split(": ", 1)
                    normalized.append({
                        "name": parts[0].strip(), 
                        "content": parts[1].strip()
                    })
                else:
                    normalized.append({
                        "name": f"Agent_{i}", 
                        "content": entry
                    })
            elif isinstance(entry, dict):
                # Use existing dict format
                name = entry.get("name") or entry.get("role") or f"Agent_{i}"
                content = entry.get("content", "")
                normalized.append({
                    "name": name, 
                    "content": content
                })
            else:
                # Fallback for unexpected types
                normalized.append({
                    "name": f"Agent_{i}",
                    "content": str(entry)
                })
        
        return normalized
