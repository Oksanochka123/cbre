"""Phone number matcher."""

import re

from constants import NULL_VALUES
from matchers.base_matcher import BaseMatcher


class PhoneMatcher(BaseMatcher):
    """Match phone numbers."""

    def _normalize(self, phone: str) -> str | None:
        s = str(phone).strip().lower()
        if not s or s in NULL_VALUES:
            return None

        digits = re.sub(r"\D", "", s)
        if not digits or len(digits) < 7 or len(digits) > 15:
            return None
        if all(d == "0" for d in digits) or all(d == "8" for d in digits):
            return None

        return digits

    def match(self, gold: str, pred: str) -> tuple[float, str]:
        g_norm = self._normalize(gold)
        p_norm = self._normalize(pred)

        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        if g_norm and p_norm:
            if g_norm in p_norm or p_norm in g_norm:
                if g_norm[-7:] == p_norm[-7:]:
                    return 1.0, f"✓ {self.field_name}: {gold} → {pred}"
            if g_norm[-7:] == p_norm[-7:]:
                return 0.7, f"✗ {self.field_name}: Partial match\n  {gold} → {pred}"

        return 0.0, f"✗ {self.field_name}: Mismatch\n  {gold} → {pred}"
