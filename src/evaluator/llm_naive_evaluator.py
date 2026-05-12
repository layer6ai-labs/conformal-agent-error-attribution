from datetime import datetime, timezone

from .base_evaluator import BaseEvaluator
from ..logger import get_logger
from typing import Any, Optional
from openai import OpenAI  # type: ignore
from .utils import _extract_probability, _normalize_history

class LLMNaiveEvaluator(BaseEvaluator):
    def __init__(self, model, api_key: Optional[str] = None):
        super().__init__()
        #get timestamp and class name for log name
        log_name = f"logs/{__name__}_{datetime.now().timestamp()}.log"
        self.logger = get_logger(name = __name__, log_file=log_name)
        self.client = None
        self.api_key = api_key
        self.model = model

    def _client_or_init(self, provided_client: Any) -> Any:
        if provided_client is not None:
            return provided_client
        if self.client is not None:
            return self.client
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def evaluate(self, data):
        client = self._client_or_init(data.get("client", None))
        problem = data.get("problem")
        answer = data.get("answer")
        start = data.get("start")
        end = data.get("end")
        history = data.get("history", [])
        # Normalize history to ensure consistent structure
        normalized_history = _normalize_history(history)
        # Implement evaluation logic here

        chat_segment_content = "\n".join(f"{entry['name']}: {entry['content']}" for entry in normalized_history)
        evaluate_content = normalized_history[start:end]
        prompt = self._construct_binary_search_prompt(problem, answer, normalized_history, chat_segment_content, evaluate_content)

        messages = [
            {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
            {"role": "user", "content": prompt},
        ]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if "gpt-5" in self.model:
                    resp = client.chat.completions.create(model=self.model, messages=messages)
                elif "gpt-4o" in self.model:
                    resp = client.chat.completions.create(model=self.model, messages=messages, max_tokens=80)
                else:
                    resp = client.chat.completions.create(model=self.model, messages=messages, max_tokens=80)
                text = resp.choices[0].message.content.strip()
                #self.logger.info(f"LLM raw response (attempt {attempt+1}): {text}")

                result = _extract_probability(text)
                return result
            except Exception as e:
                self.logger.error(f"LLM call failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying LLM call due to exception (attempt {attempt+2})...")
                    continue
                return 0.0

    def _construct_binary_search_prompt(self, problem, answer, history, chat_segment_content, evaluate_content) -> str:
        """Constructs the prompt for the binary search step."""
        return (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
            f"The problem to address is as follows: {problem}\n"
            f"The final correct Answer for the problem is: {answer}\n" # Included as per original code - remove if ground truth shouldn't be used
            f"The wrong answer given in the end of the conversation is: {history[-1]}\n"
            f"Review the following conversation segment: {evaluate_content} \n"
            f"Cutting from entire history:\n\n{chat_segment_content}\n"
            f"Based on your analysis please respond with ONLY a single probability value which is an estimated probability between 0 and 1 of the critical error being in this conversation segment you reviewed. Do not include any explanation or additional text."
        )
    

