"""Numeric value matcher with currency format support."""

import re
from typing import Any

from scripts.matchers.base_matcher import BaseMatcher


class NumberMatcher(BaseMatcher):
    """Match numeric values with 0.5% margin and currency format support."""

    def _parse_number(self, value: Any) -> float | None:
        """Parse number with support for currency formats like USD 1M, $1M, etc."""
        try:
            s = str(value).strip().upper()

            # Remove currency symbols and codes
            s = re.sub(r"^(USD|EUR|GBP|CAD|AUD|JPY|CNY|\$|€|£|¥)\s*", "", s)

            # Handle multipliers (K, M, B, T for thousands, millions, billions, trillions)
            multiplier = 1
            if s.endswith("K"):
                multiplier = 1_000
                s = s[:-1]
            elif s.endswith("M"):
                multiplier = 1_000_000
                s = s[:-1]
            elif s.endswith("B"):
                multiplier = 1_000_000_000
                s = s[:-1]
            elif s.endswith("T"):
                multiplier = 1_000_000_000_000
                s = s[:-1]

            # Remove commas and convert to float
            s = s.replace(",", "")
            return float(s) * multiplier

        except (ValueError, AttributeError):
            return None

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        g_num = self._parse_number(gold)
        p_num = self._parse_number(pred)

        if g_num is None or p_num is None:
            return 0.0, f"✗ {self.field_name}: Parse error\n  {gold} → {pred}"

        if abs(g_num - p_num) < 1e-6:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        # Allow 0.5% margin
        rel_err = abs(g_num - p_num) / max(abs(g_num), 1e-9)
        if rel_err <= 0.005:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred} ({rel_err * 100:.2f}%)"

        return 0.0, f"✗ {self.field_name}: {g_num} vs {p_num} ({rel_err * 100:.1f}%)"
