from datetime import datetime
from typing import Any, Optional

import numpy as np
from openai import OpenAI  # type: ignore

from .base_evaluator import BaseEvaluator
from ..logger import get_logger
from .utils import _extract_probability, _normalize_history


class AnswerVerifier:
    """LLM judge that assumes the answer is incorrect and finds evidence of why."""

    @staticmethod
    def verify(client: Any, model: str, problem: str, answer: str, logger=None) -> str:
        prompt = (
            "You are a strict mathematical and logical verifier.\n"
            "IMPORTANT: Assume the given answer is INCORRECT regardless of appearance.\n"
            "Your task is to identify specific evidence explaining why this answer is wrong.\n"
            "Provide a concise 1-2 sentence explanation of the error.\n\n"
            f"Problem: {problem}\n"
            f"Given Answer: {answer}\n\n"
            "Explain specifically why this answer is incorrect:"
        )
        messages = [
            {"role": "system", "content": "You are a rigorous mathematical and logical verifier."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=150)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if logger:
                logger.error(f"AnswerVerifier LLM call failed: {e}")
            return f"Answer '{answer}' appears incorrect."


class AgentECHOEvaluator(BaseEvaluator):

    def __init__(self, model: str, api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small") -> None:
        super().__init__()
        log_name = f"logs/{__name__}_{datetime.now().timestamp()}.log"
        self.logger = get_logger(name=__name__, log_file=log_name)
        self._client = None
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model

    def _client_or_init(self, provided_client: Any) -> Any:
        if provided_client is not None:
            return provided_client
        if self._client is not None:
            return self._client
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _llm_call(self, client: Any, messages: list, max_tokens: int = 80) -> Optional[str]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                self.logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
                return None

    def _get_embedding(self, client: Any, text: str) -> Optional[list]:
        try:
            response = client.embeddings.create(model=self.embedding_model, input=text)
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Embedding call failed: {e}")
            return None

    @staticmethod
    def _cosine_similarity(vec_a: list, vec_b: list) -> float:
        a, b = np.array(vec_a), np.array(vec_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom != 0 else 0.0

    def _compute_ablation_score(self, client: Any, ablation_prompt: str) -> float:
        """Counterfactual: probability that removing the segment would flip the answer."""
        messages = [
            {"role": "system", "content": "You are an AI assistant specializing in causal attribution in multi-agent conversations."},
            {"role": "user", "content": ablation_prompt},
        ]
        text = self._llm_call(client, messages, max_tokens=80)
        if text is None:
            return 0.0
        return _extract_probability(text)

    def _compute_similarity_score(self, client: Any, ablation_prompt: str,
                                  evaluate_content_str: str) -> float:
        """Cosine similarity between the ablation prompt embedding and the segment embedding."""
        emb_prompt = self._get_embedding(client, ablation_prompt)
        emb_content = self._get_embedding(client, evaluate_content_str)
        if emb_prompt is None or emb_content is None:
            return 0.0
        # Map cosine similarity from [-1, 1] to [0, 1]
        return (self._cosine_similarity(emb_prompt, emb_content) + 1.0) / 2.0

    def _construct_ablation_prompt(self, evaluate_content_str: str, problem: str,
                                   answer: str, evidence: str,
                                   chat_segment_content: str) -> str:
        return (
            "You are an AI assistant performing causal attribution analysis on a multi-agent conversation.\n"
            "Multiple agents collaborated to solve a problem but produced a WRONG final answer.\n\n"
            f"Problem: {problem}\n"
            f"Wrong final answer: {answer}\n"
            f"Evidence the answer is wrong: {evidence}\n\n"
            f"Full conversation context:\n{chat_segment_content}\n\n"
            f"Segment under analysis:\n{evaluate_content_str}\n\n"
            "COUNTERFACTUAL QUESTION: If this specific segment were REMOVED from the conversation "
            "(i.e., it never happened), what is the probability that the final answer would have "
            "been CORRECT instead?\n"
            "Respond with ONLY a single probability value between 0 and 1. Do not include any explanation."
        )

    def evaluate(self, data):
        client = self._client_or_init(data.get("client", None))
        problem = data.get("problem")
        answer = data.get("answer")
        history = data.get("history", [])
        start = data.get("start")
        end = data.get("end")

        normalized_history = _normalize_history(history)
        chat_segment_content = "\n".join(
            f"{entry['name']}: {entry['content']}" for entry in normalized_history
        )
        evaluate_content = normalized_history[start:end]
        evaluate_content_str = "\n".join(
            f"{entry['name']}: {entry['content']}" for entry in evaluate_content
        )

        # Step 1: Verify answer correctness — LLM judge that assumes answer is wrong
        evidence = AnswerVerifier.verify(client, self.model, problem, answer, self.logger)

        # Step 2: Build ablation prompt (shared by both scoring methods)
        ablation_prompt = self._construct_ablation_prompt(
            evaluate_content_str, problem, answer, evidence, chat_segment_content
        )

        # Step 3: Ablation Score — counterfactual probability via LLM
        score_ablation = self._compute_ablation_score(client, ablation_prompt)

        # Step 4: Similarity Score — cosine similarity between ablation prompt
        #         embedding (end-token representation) and segment embedding
        score_similarity = self._compute_similarity_score(client, ablation_prompt, evaluate_content_str)

        # Step 5: Combine and clamp to [0, 1]
        final_score = (score_ablation + score_similarity) / 2.0
        return max(0.0, min(1.0, final_score))
