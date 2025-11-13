"""Base class for field matching."""

from abc import ABC, abstractmethod
from typing import Any

import dspy


class BaseMatcher(ABC):
    """Base class for field matching."""

    def __init__(self, field_name: str):
        self.field_name = field_name

    @abstractmethod
    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        """Return (score, feedback)."""
        pass

    def __call__(self, example, pred, trace=None) -> dspy.Prediction:
        """DSPy metric interface."""
        gold = self._extract(example, self.field_name)
        pred_val = self._extract(pred, self.field_name)

        if gold is None and pred_val is None:
            return dspy.Prediction(score=1.0, feedback=f"✓ {self.field_name}: Both None")
        if gold is None:
            return dspy.Prediction(score=0.0, feedback=f"✗ {self.field_name}: Hallucination")
        if pred_val is None:
            return dspy.Prediction(score=0.0, feedback=f"✗ {self.field_name}: Missing")

        score, feedback = self.match(gold, pred_val)
        return dspy.Prediction(score=score, feedback=feedback)

    @staticmethod
    def _extract(obj, field_name):
        if isinstance(obj, dict):
            return obj.get(field_name)
        return getattr(obj, field_name, None)
