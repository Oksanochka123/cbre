"""Enum matcher with exact value matching."""

from typing import Any

from constants import NULL_VALUES
from matchers.base_matcher import BaseMatcher


class EnumMatcher(BaseMatcher):
    """Match fixed value sets with exact matching."""

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
        super().__init__(field_name, **kwargs)

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
        if val is None:
            return None
        s = str(val).strip()
        if self.treat_null_as_none and s.lower() in NULL_VALUES:
            return None
        return s if self.case_sensitive else s.lower()

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        g_norm = self._normalize(gold)
        p_norm = self._normalize(pred)

        if g_norm is None and p_norm is None:
            return 1.0, f"✓ {self.field_name}: Both null"
        if g_norm is None:
            return 0.0, f"✗ {self.field_name}: Gold null, pred='{pred}'"
        if p_norm is None:
            return 0.0, f"✗ {self.field_name}: Pred null, gold='{gold}'"

        # Exact match
        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: {pred}"

        # Check if both are valid but different
        g_valid = g_norm in self.valid_values_norm
        p_valid = p_norm in self.valid_values_norm

        if g_valid and p_valid:
            return 0.0, f"✗ {self.field_name}: '{gold}' vs '{pred}'"

        # Invalid prediction
        valid_str = ", ".join(f"'{v}'" for v in self.valid_values[:3])
        if len(self.valid_values) > 3:
            valid_str += "..."

        return 0.0, f"✗ {self.field_name}: Invalid '{pred}' ({valid_str})"
