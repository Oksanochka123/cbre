"""Boolean matcher."""

from typing import Any

from matchers.base_matcher import BaseMatcher


class BooleanMatcher(BaseMatcher):
    """Match boolean strings."""

    TRUE_VALS = {"yes", "true", "t", "y", "1"}
    FALSE_VALS = {"no", "false", "f", "n", "0"}

    def __init__(self, field_name: str):
        super().__init__(field_name, field_type="boolean")

    def _to_bool(self, val: Any) -> bool | None:
        s = str(val).strip().lower()
        if s in self.TRUE_VALS:
            return True
        if s in self.FALSE_VALS:
            return False
        return None

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        from components.parse_feedback import format_parse_error_feedback, is_json_string

        # Check if pred is a JSON string (JSON object/array instead of boolean)
        if is_json_string(pred):
            feedback = format_parse_error_feedback(
                self.field_name,
                expected_type="boolean",
                actual_value=pred,
            )
            return 0.0, feedback

        g_bool = self._to_bool(gold)
        p_bool = self._to_bool(pred)

        if g_bool is None or p_bool is None:
            return 0.0, f"✗ {self.field_name}: Parse error\n  {gold} → {pred}"

        if g_bool == p_bool:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        return 0.0, f"✗ {self.field_name}: Mismatch\n  {gold} → {pred}"
