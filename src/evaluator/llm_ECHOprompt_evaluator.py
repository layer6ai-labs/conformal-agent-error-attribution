import json
from datetime import datetime
from time import sleep
from typing import Optional, Any, List, Dict
from openai import OpenAI
from statistics import mean
from src.logger import get_logger
from .llm_naive_evaluator import LLMNaiveEvaluator
from .utils import _normalize_history

class LLMECHOPromptEvaluator(LLMNaiveEvaluator):
    """
    LLM evaluator using role-based prompt ensemble (Conservative, Liberal,
    Skeptical, Pattern) and evidence-anchored confidence calibration.
    """

    ROLES = ["Conservative", "Liberal", "Pattern"]

    def __init__(self, model, api_key: Optional[str] = None, delta_threshold: float = 0.3):
        super().__init__(model=model, api_key=api_key)
        log_name = f"logs/ECHO_roles_{datetime.now().timestamp()}.log"
        self.logger = get_logger(name=__name__, log_file=log_name)
        self.delta_threshold = delta_threshold

    def evaluate(self, data):
        client = self._client_or_init(data.get("client", None))
        problem = data.get("problem")
        answer = data.get("answer")
        history = _normalize_history(data.get("history", []))
        start, end = data.get("start"), data.get("end")
        chat_segment_content = "\n".join(f"{entry['name']}: {entry['content']}" for entry in history)
        evaluate_content = history[start:end]

        # Run ensemble of role-based prompts
        results = []
        for role in self.ROLES:
            max_retries = 3
            for attempt in range(max_retries):
                prompt = self._construct_role_prompt(role, problem, answer, chat_segment_content, evaluate_content)
                messages = [
                    {"role": "system", "content": f"You are a {role} Analyst specializing in identifying failures in multi-agent reasoning."},
                    {"role": "user", "content": prompt},
                ]
                try:
                    resp = client.chat.completions.create(model=self.model, messages=messages)
                    text = resp.choices[0].message.content.strip()
                    parsed = self._safe_parse_json(text)
                    if parsed and parsed.get("primary_conclusion", {}).get("confidence", 0) >= self.delta_threshold:
                        results.append(parsed)
                    break  # Exit retry loop on success
                except Exception as e:
                    self.logger.error(f"LLM call for {role} failed: {e}")
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying LLM call due to exception (attempt {attempt + 1})...")
                        # wait for a short period before retrying
                        sleep(2**attempt)
                        continue
        if not results:
            return 0.0

        return self._aggregate_confidences(results)

    def _construct_role_prompt(self, role, problem, answer, chat_segment_content, evaluate_content) -> str:
        """Construct prompt for a specific analytical role."""
        role_instructions = {
            "Conservative": (
                "- Attribute an error only when explicit contradiction or logical failure is visible.\n"
                "- Assign high confidence only if the evidence is direct and quotes are exact.\n"
            ),
            "Liberal": (
                "- Consider potential upstream or downstream causes even if indirect.\n"
                "- Assign moderate confidence (0.5–0.7) for plausible but not proven causes.\n"
            ),
            "Skeptical": (
                "- Always include at least one alternative explanation and note missing evidence.\n"
                "- Lower confidence if evidence is incomplete.\n"
            ),
            "Pattern": (
                "- Focus on repeated reasoning or coordination issues.\n"
                "- Identify patterns across multiple steps rather than isolated mistakes.\n"
            ),
        }

        schema = json.dumps({
            "investigation_summary": "short summary",
            "primary_conclusion": {
                "agent": "Agent-X",
                "mistake_step": "int",
                "confidence": "float(0-1)",
                "evidence": ["quoted spans or step references"],
                "reason": "brief reason",
                "alternative_explanations": ["optional list of other possibilities"]
            }
        }, indent=2)

        return (
            f"Problem to solve: {problem}\n"
            f"Expected correct answer: {answer}\n\n"
            f"Conversation segment under review:\n{evaluate_content}\n\n"
            f"Full chat context:\n{chat_segment_content}\n\n"
            f"ROLE: {role} Analyst\n"
            f"{role_instructions[role]}\n"
            "TASK: Identify if this segment likely contains the key reasoning failure "
            "that led to the wrong final answer. Provide your analysis strictly in JSON format "
            "matching the schema below. Quote evidence spans exactly from the text and assign a numeric confidence "
            "between 0 and 1. Map confidence as follows:\n"
            "0.0–0.2 = Very unlikely, 0.2–0.4 = Unlikely, 0.4–0.6 = Possible, "
            "0.6–0.8 = Likely, 0.8–1.0 = Very likely.\n\n"
            f"Return ONLY valid JSON (no extra text):\n{schema}\n"
        )

    def _safe_parse_json(self, text: str) -> Optional[Dict]:
        """Parse and sanitize JSON output from LLM."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            parsed = json.loads(text[start:end])
            return parsed
        except Exception:
            self.logger.warning(f"Failed to parse JSON output:\n{text}")
            return None

    def _aggregate_confidences(self, results: List[Dict]) -> float:
        """Weighted average of confidences from multiple role analyses."""
        confs = []
        for r in results:
            conf = r.get("primary_conclusion", {}).get("confidence", 0)
            confs.append(conf)
        return mean(confs) if confs else 0.0
