"""Address matcher."""

import re
from difflib import SequenceMatcher

from matchers.base_matcher import BaseMatcher


class AddressMatcher(BaseMatcher):
    """Match addresses with fuzzy similarity."""

    def __init__(self, field_name: str, threshold: float = 0.7):
        super().__init__(field_name)
        self.threshold = threshold

    def _normalize(self, addr: str) -> str:
        s = str(addr).strip().lower()
        s = re.sub(r"\bst\b", "street", s)
        s = re.sub(r"\bave\b", "avenue", s)
        s = re.sub(r"\brd\b", "road", s)
        s = re.sub(r"\bdr\b", "drive", s)
        s = re.sub(r"\bblvd\b", "boulevard", s)
        s = re.sub(r"[.,#]", "", s)
        return " ".join(s.split())

    def match(self, gold: str, pred: str) -> tuple[float, str]:
        g_norm = self._normalize(gold)
        p_norm = self._normalize(pred)

        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: {gold}"

        sim = SequenceMatcher(None, g_norm, p_norm).ratio()
        score = sim if sim >= self.threshold else 0.0

        return score, f"✗ {self.field_name}: {sim:.2f} sim\n  '{gold}' → '{pred}'"
