"""String matcher with null handling."""

from difflib import SequenceMatcher

from matchers.base_matcher import BaseMatcher


class StringMatcher(BaseMatcher):
    """Match strings with fuzzy similarity and null detection."""

    def __init__(self, field_name: str, threshold: float = 0.6):
        super().__init__(field_name)
        self.threshold = threshold

    def match(self, gold: str, pred: str) -> tuple[float, str]:
        # Null handling first
        gold_null = self._is_null(gold)
        pred_null = self._is_null(pred)

        if gold_null and pred_null:
            return 1.0, f"✓ {self.field_name}: Both null"
        if gold_null and not pred_null:
            return 0.0, f"✗ {self.field_name}: Hallucination (predicted '{pred}' when gold is null)"
        if not gold_null and pred_null:
            return 0.0, f"✗ {self.field_name}: Missing (expected '{gold}', got null)"

        # Normalize and compare
        g_norm = str(gold).strip().lower()
        p_norm = str(pred).strip().lower()

        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: '{gold}'"

        sim = SequenceMatcher(None, g_norm, p_norm).ratio()
        score = sim if sim >= self.threshold else 0.0

        return score, f"✗ {self.field_name}: {sim:.2f} sim | '{gold}' vs '{pred}'"
