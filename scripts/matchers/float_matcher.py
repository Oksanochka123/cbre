"""Float matcher with tolerance."""

import re
from typing import Any

from matchers.base_matcher import BaseMatcher


class FloatMatcher(BaseMatcher):
    """Match floats with tolerance, supporting percentages and currency."""

    def __init__(self, field_name: str, tolerance: float = 5e-4,
                 allow_percentage: bool = True, allow_currency: bool = True, **kwargs):
        super().__init__(field_name, field_type="float")
        self.tolerance = tolerance
        self.allow_percentage = allow_percentage
        self.allow_currency = allow_currency

    def _parse_float(self, val: Any) -> float | None:
        """
        Parse float from various formats.

        Supports:
        - Plain numbers: "123.45", "123"
        - Percentages: "5%", "3.25%", "five percent"
        - Currency: "$1,234.56", "€1000"
        - No value: None, "None", "null", "", "n/a"
        """
        if self._is_null(val):
            return None

        s = str(val).strip()

        # Handle percentage notation: "5%" → 5.0
        if self.allow_percentage and '%' in s:
            s = s.replace('%', '').strip()

        # Handle currency: "$1,234.56" → 1234.56
        if self.allow_currency:
            # Remove common currency symbols
            s = re.sub(r'[$€¥£]', '', s)
            # Remove thousands separators
            s = s.replace(',', '').replace(' ', '')

        # Handle text percentages: "five percent" → 5.0
        text_nums = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10'
        }
        for word, num in text_nums.items():
            s = re.sub(r'\b' + word + r'\b', num, s, flags=re.IGNORECASE)

        # Handle fractions: "1/2" → 0.5
        frac_match = re.match(r'^\s*(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*$', s)
        if frac_match:
            try:
                num, denom = float(frac_match.group(1)), float(frac_match.group(2))
                return num / denom if denom != 0 else None
            except (ValueError, ZeroDivisionError):
                return None

        try:
            return float(s)
        except ValueError:
            return None

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        from components.parse_feedback import format_parse_error_feedback, is_json_string

        # Check if pred is a JSON string (JSON object/array instead of float)
        if is_json_string(pred):
            feedback = format_parse_error_feedback(
                self.field_name,
                expected_type="float",
                actual_value=pred,
            )
            return 0.0, feedback

        # Null handling first
        gold_null = self._is_null(gold)
        pred_null = self._is_null(pred)

        if gold_null and pred_null:
            return 1.0, f"✓ {self.field_name}: Both null"
        if gold_null and not pred_null:
            return 0.0, f"✗ {self.field_name}: Hallucination (predicted {pred} when gold is null)"
        if not gold_null and pred_null:
            return 0.0, f"✗ {self.field_name}: Missing (expected {gold}, got null)"

        # Parse values
        g_val = self._parse_float(gold)
        p_val = self._parse_float(pred)

        if g_val is None or p_val is None:
            return 0.0, f"✗ {self.field_name}: Parse error (gold={gold}, pred={pred})"

        # Compare with tolerance
        diff = abs(g_val - p_val)
        if diff < self.tolerance:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        return 0.0, f"✗ {self.field_name}: {g_val} vs {p_val} (diff: {diff:.6f})"
