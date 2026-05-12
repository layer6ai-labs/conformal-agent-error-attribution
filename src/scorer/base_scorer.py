from typing import List
from abc import ABC, abstractmethod
from typing import Any

class IScorer(ABC):
    @abstractmethod
    #evaluate conformal score in a range of responses
    def score(self, data):
        pass