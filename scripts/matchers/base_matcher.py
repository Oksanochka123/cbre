"""Base class for field matching."""

from abc import ABC, abstractmethod
from typing import Any

import dspy

from components.constants import NULL_VALUES


class BaseMatcher(ABC):
    """Base class for field matching."""

    def __init__(self, field_name: str, field_type: str = "string"):
        self.field_name = field_name
        self.field_type = field_type

    @abstractmethod
    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        """Return (score, feedback)."""
        pass

    def __call__(self, example, pred, trace=None) -> dspy.Prediction:
        """DSPy metric interface.

        Enhanced to provide richer feedback when predictions fail type validation.
        """
        gold = self._extract(example, self.field_name)
        pred_val = self._extract(pred, self.field_name)

        # Use _is_null() to handle both None and string representations ("null", "None", etc.)
        gold_is_null = self._is_null(gold)
        pred_is_null = self._is_null(pred_val)

        if gold_is_null and pred_is_null:
            return dspy.Prediction(score=1.0, feedback=f"✓ {self.field_name}: Both null")
        if gold_is_null and not pred_is_null:
            return dspy.Prediction(score=0.0, feedback=f"✗ {self.field_name}: Hallucination (predicted {str(pred_val)[:50]} when gold is null)")
        if not gold_is_null and pred_is_null:
            return dspy.Prediction(score=0.0, feedback=f"✗ {self.field_name}: Missing (expected {str(gold)[:50]}, got null)")

        score, feedback = self.match(gold, pred_val)
        return dspy.Prediction(score=score, feedback=feedback)

    @staticmethod
    def _extract(obj, field_name):
        if isinstance(obj, dict):
            return obj.get(field_name)
        return getattr(obj, field_name, None)

    @staticmethod
    def _is_null(val: Any) -> bool:
        """
        Check if a value is null/None/empty using NULL_VALUES from constants.

        Handles:
        - None
        - Values in NULL_VALUES set (case-insensitive)
        - Empty list []
        - Empty dict {}
        """
        if val is None:
            return True
        if isinstance(val, str):
            s = val.strip().lower()
            if s in NULL_VALUES:
                return True
        if isinstance(val, (list, dict)) and len(val) == 0:
            return True
        return False
