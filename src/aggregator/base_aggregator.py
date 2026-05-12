from abc import ABC, abstractmethod
from typing import List


class BaseAggregator(ABC):
    """Base class for aggregating per-node probabilities into directional conformal scores.

    Each aggregator must implement four operations:
      - ``prefix``: sliding prefix aggregate (used by right-filter / vanilla / tree)
      - ``suffix``: sliding suffix aggregate (used by left-filter)
      - ``combine``: pairwise reduction used when building internal tree nodes
      - ``score``: single global score from a full probability array (used by vanilla calibration)

    All returned arrays are normalised by trajectory length *n* so that values
    remain in roughly the same range regardless of sequence length.
    """

    @abstractmethod
    def prefix(self, probs: List[float]) -> List[float]:
        """Compute prefix aggregation.

        Returns a list of length *n* where ``result[j] = agg(probs[0:j+1]) / n``.
        """

    @abstractmethod
    def suffix(self, probs: List[float]) -> List[float]:
        """Compute suffix aggregation.

        Returns a list of length *n* where ``result[j] = agg(probs[j:]) / n``.
        """

    @abstractmethod
    def combine(self, a: float, b: float) -> float:
        """Combine two node values when building binary tree internal nodes."""

    @abstractmethod
    def score(self, probs: List[float]) -> float:
        """Compute a single global score from a full probability array."""
