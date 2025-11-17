"""Enum matcher with exact value matching and null handling."""

from typing import Any

from components.constants import NULL_VALUES
from matchers.base_matcher import BaseMatcher


class EnumMatcher(BaseMatcher):
    """Match fixed value sets with exact matching and null detection."""

    PRESETS = {
        "yes_no": ["yes", "no"],
        "yes_no_na": ["yes", "no", "n/a"],
        "boolean": ["true", "false"],
    }

    def __init__(
        self,
        field_name: str,
        valid_values: list | None = None,
        preset: str | None = None,
        case_sensitive: bool = False,
        treat_null_as_none: bool = True,
        **kwargs,
    ):
        super().__init__(field_name)

        if preset:
            self.valid_values = self.PRESETS[preset]
        elif valid_values:
            self.valid_values = valid_values
        else:
            raise ValueError("Need valid_values or preset")

        self.case_sensitive = case_sensitive
        self.treat_null_as_none = treat_null_as_none

        if case_sensitive:
            self.valid_values_norm = [str(v).strip() for v in self.valid_values]
        else:
            self.valid_values_norm = [str(v).strip().lower() for v in self.valid_values]

    def _normalize(self, val: Any) -> str | None:
        """Normalize enum value, returning None for null values."""
        if self._is_null(val):
            return None
        s = str(val).strip()
        return s if self.case_sensitive else s.lower()

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        # Null handling first
        gold_null = self._is_null(gold)
        pred_null = self._is_null(pred)

        if gold_null and pred_null:
            return 1.0, f"✓ {self.field_name}: Both null"
        if gold_null and not pred_null:
            return 0.0, f"✗ {self.field_name}: Hallucination (predicted '{pred}' when gold is null)"
        if not gold_null and pred_null:
            return 0.0, f"✗ {self.field_name}: Missing (expected one of {self.valid_values}, got null)"

        # Normalize values
        g_norm = self._normalize(gold)
        p_norm = self._normalize(pred)

        # Exact match
        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: '{pred}'"

        # Check if both are valid but different
        g_valid = g_norm in self.valid_values_norm
        p_valid = p_norm in self.valid_values_norm

        if g_valid and p_valid:
            return 0.0, f"✗ {self.field_name}: Wrong value | expected '{gold}', got '{pred}'"

        # Invalid prediction
        return 0.0, f"✗ {self.field_name}: Invalid value '{pred}' (valid: {self.valid_values})"
