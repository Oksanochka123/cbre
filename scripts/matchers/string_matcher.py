"""String matcher."""

from difflib import SequenceMatcher

from matchers.base_matcher import BaseMatcher


class StringMatcher(BaseMatcher):
    """Match strings with fuzzy similarity."""

    def __init__(self, field_name: str, threshold: float = 0.6):
        super().__init__(field_name)
        self.threshold = threshold

    def match(self, gold: str, pred: str) -> tuple[float, str]:
        g_norm = str(gold).strip().lower()
        p_norm = str(pred).strip().lower()

        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: '{gold}'"

        sim = SequenceMatcher(None, g_norm, p_norm).ratio()
        score = sim if sim >= self.threshold else 0.0

        return score, f"✗ {self.field_name}: {sim:.2f} sim\n  '{gold}' → '{pred}'"
