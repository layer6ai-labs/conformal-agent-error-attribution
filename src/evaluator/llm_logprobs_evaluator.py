from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from .base_evaluator import BaseEvaluator
from ..logger import get_logger
from .utils import _normalize_history

class LLMLogprobsEvaluator(BaseEvaluator):
    """
    Evaluator that queries an LLM to get a binary response (1 or 0) about whether
    the critical error is in a conversation segment, using logprobs for scoring.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, logger_name: str = "agentic_conformal") -> None:
        self.model = model
        self.api_key = api_key
        self.logger = get_logger(logger_name)
        self._client = None

    def _client_or_init(self, provided_client: Any) -> Any:
        if provided_client is not None:
            return provided_client
        if self._client is not None:
            return self._client
        if OpenAI is None:
            raise ImportError("openai package not installed; pass a client or install openai.")
        self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def _construct_prompt(self, problem, answer, history, chat_segment_content, evaluate_content):
        """Constructs the prompt for binary classification."""
        return (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify whether the most critical mistake is within the provided segment.\n"
            f"The problem to address is as follows: {problem}\n"
            f"The Answer for the problem is: {answer}\n"
            f"The wrong answer given in the end of the conversation is: {history[-1]}\n"
            f"Review the following conversation segment: {evaluate_content} \n"
            f"Cutting from entire history:\n\n{chat_segment_content}\n"
            f"Based on your analysis, determine if the most critical error is located in this segment.\n"
            f"Please respond with ONLY '1' if the critical error is in this segment, or '0' if it is not. Do not include any explanation or additional text."
        )

    def evaluate(self, input_data: Any) -> float:
        data: Dict[str, Any] = input_data
        problem: str = data.get("problem", "")
        history = data.get("history", [])
        answer = data.get("answer", "")
        start: int = int(data.get("start", 0))
        end: int = int(data.get("end", len(history) - 1))
        model: str = data.get("model", self.model)
        client = self._client_or_init(data.get("client"))
        n_logprobs: int = data.get("n_logprobs", 5)

        normalized_history = _normalize_history(history)
        if start < 0 or end > len(normalized_history) or start > end:
            self.logger.warning(f"Invalid segment range start={start}, end={end}, len={len(normalized_history)}. Returning 0.5.")
            return 0.5

        segment = normalized_history[start : end]
        chat_segment_content = "\n".join(f"{entry['name']}: {entry['content']}" for entry in normalized_history)

        prompt = self._construct_prompt(
            problem=problem,
            answer=answer,
            history=normalized_history,
            chat_segment_content=chat_segment_content,
            evaluate_content=segment,
        )
        
        #self.logger.info(f"LLM evaluate segment {start}-{end} (len={len(segment)})")
        
        messages = [
            {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
            {"role": "user", "content": prompt},
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model, 
                    messages=messages, 
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20  # Get top 20 logprobs
                )
                
                # Extract logprobs from the first token and compute probability
                if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                    token_logprobs = resp.choices[0].logprobs.content[0]
                    
                    if token_logprobs.top_logprobs:
                        probability = self._compute_probability_from_logprobs(
                            token_logprobs.top_logprobs, n_logprobs
                        )
                        #self.logger.info(f"Computed probability: {probability:.3f}")
                        return probability
                
                self.logger.warning("No logprobs available in response, returning 0.5")
                return 0.5
                    
            except Exception as e:
                self.logger.error(f"LLM call failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying LLM call due to exception (attempt {attempt+2})...")
                    continue
                return 0.5

    def _compute_probability_from_logprobs(self, top_logprobs, n_logprobs: int) -> float:
        """
        Compute probability using: 1*logprob(1) + 0.5*logprob(others except 1 and 0)
        """
        import math
        
        # Take first n_logprobs
        relevant_logprobs = top_logprobs[:n_logprobs]
        
        probability = 0.0
        for logprob_item in relevant_logprobs:
            token = logprob_item.token.strip()
            logprob = logprob_item.logprob
            
            if token == "1":
                probability += 1.0 * math.exp(logprob)
            elif token != "0":  # Others except 1 and 0
                probability += 0.5 * math.exp(logprob)
        
        return min(1.0, max(0.0, probability))