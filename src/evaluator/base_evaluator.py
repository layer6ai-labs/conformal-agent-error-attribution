from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, input_data: Any) -> float:
        pass