"""Float matcher with tolerance."""

from typing import Any

from matchers.base_matcher import BaseMatcher


class FloatMatcher(BaseMatcher):
    """Match floats with tolerance."""

    def __init__(self, field_name: str, tolerance: float = 5e-4, **kwargs):
        super().__init__(field_name, **kwargs)
        self.tolerance = tolerance

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        try:
            g_val = float(str(gold).strip())
            p_val = float(str(pred).strip())
        except (ValueError, AttributeError):
            return 0.0, f"✗ {self.field_name}: Parse error"

        diff = abs(g_val - p_val)
        if diff < self.tolerance:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        return 0.0, f"✗ {self.field_name}: {g_val} vs {p_val} (diff: {diff:.6f})"
