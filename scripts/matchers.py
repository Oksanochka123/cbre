"""Field matchers for extraction validation."""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import dspy


class BaseMatcher(ABC):
    """Base class for field matching."""

    def __init__(self, field_name: str, exact_weight: float = 0.8, sim_weight: float = 0.2):
        self.field_name = field_name
        self.exact_weight = exact_weight
        self.sim_weight = sim_weight

    @abstractmethod
    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        """Return (score, feedback)."""
        pass

    def __call__(self, example, pred, trace=None) -> dspy.Prediction:
        """DSPy metric interface."""
        gold = self._extract(example, self.field_name)
        pred_val = self._extract(pred, self.field_name)

        if gold is None and pred_val is None:
            return dspy.Prediction(score=1.0, feedback=f"✓ {self.field_name}: Both None")
        if gold is None:
            return dspy.Prediction(score=0.0, feedback=f"✗ {self.field_name}: Hallucination")
        if pred_val is None:
            return dspy.Prediction(score=0.0, feedback=f"✗ {self.field_name}: Missing")

        score, feedback = self.match(gold, pred_val)
        return dspy.Prediction(score=score, feedback=feedback)

    @staticmethod
    def _extract(obj, field_name):
        if isinstance(obj, dict):
            return obj.get(field_name)
        return getattr(obj, field_name, None)


class PhoneMatcher(BaseMatcher):
    """Match phone numbers."""

    def _normalize(self, phone: str) -> str | None:
        s = str(phone).strip().lower()
        if not s or s in {"unknown", "tbd", "na", "n/a", "none", "pending"}:
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


class DateMatcher(BaseMatcher):
    """Match dates."""

    FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d", "%m-%d-%Y"]

    def _parse(self, date_str: str) -> datetime | None:
        s = str(date_str).strip()
        for fmt in self.FORMATS:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    def match(self, gold: str, pred: str) -> tuple[float, str]:
        g_date = self._parse(gold)
        p_date = self._parse(pred)

        if not g_date or not p_date:
            return 0.0, f"✗ {self.field_name}: Parse error\n  {gold} → {pred}"

        if g_date == p_date:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        days_diff = abs((g_date - p_date).days)
        if days_diff <= 1:
            return 0.9, f"✗ {self.field_name}: Off by {days_diff} day\n  {gold} → {pred}"

        return 0.0, f"✗ {self.field_name}: Off by {days_diff} days\n  {gold} → {pred}"


class NumberMatcher(BaseMatcher):
    """Match numeric values with 0.5% margin."""

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        try:
            g_num = float(str(gold).strip().replace(",", ""))
            p_num = float(str(pred).strip().replace(",", ""))
        except (ValueError, AttributeError):
            return 0.0, f"✗ {self.field_name}: Parse error\n  {gold} → {pred}"

        if abs(g_num - p_num) < 1e-6:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        # Allow 0.5% margin
        rel_err = abs(g_num - p_num) / max(abs(g_num), 1e-9)
        if rel_err <= 0.005:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred} ({rel_err * 100:.2f}%)"

        return 0.0, f"✗ {self.field_name}: {g_num} vs {p_num} ({rel_err * 100:.1f}%)"


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


class StringMatcher(BaseMatcher):
    """Match strings."""

    def match(self, gold: str, pred: str) -> tuple[float, str]:
        g_norm = str(gold).strip().lower()
        p_norm = str(pred).strip().lower()

        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: '{gold}'"

        sim = SequenceMatcher(None, g_norm, p_norm).ratio()
        score = self.exact_weight * 0.0 + self.sim_weight * sim

        return score, f"✗ {self.field_name}: {sim:.2f} sim\n  '{gold}' → '{pred}'"


class BooleanMatcher(BaseMatcher):
    """Match boolean strings."""

    TRUE_VALS = {"yes", "true", "t", "y", "1"}
    FALSE_VALS = {"no", "false", "f", "n", "0"}

    def _to_bool(self, val: Any) -> bool | None:
        s = str(val).strip().lower()
        if s in self.TRUE_VALS:
            return True
        if s in self.FALSE_VALS:
            return False
        return None

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        g_bool = self._to_bool(gold)
        p_bool = self._to_bool(pred)

        if g_bool is None or p_bool is None:
            return 0.0, f"✗ {self.field_name}: Parse error\n  {gold} → {pred}"

        if g_bool == p_bool:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        return 0.0, f"✗ {self.field_name}: Mismatch\n  {gold} → {pred}"


class EnumMatcher(BaseMatcher):
    """Match fixed value sets with fuzzy matching."""

    PRESETS = {
        "yes_no": ["yes", "no"],
        "yes_no_na": ["yes", "no", "n/a"],
        "boolean": ["true", "false"],
    }

    NULL_VALUES = {"", "null", "none", "na", "n/a", "unknown", "tbd", "tba", "unk", "pending", "to be determined", "0"}

    def __init__(
        self,
        field_name: str,
        valid_values: list | None = None,
        preset: str | None = None,
        case_sensitive: bool = False,
        fuzzy_threshold: float = 0.85,
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
        self.fuzzy_threshold = fuzzy_threshold
        self.treat_null_as_none = treat_null_as_none

        if case_sensitive:
            self.valid_values_norm = [str(v).strip() for v in self.valid_values]
        else:
            self.valid_values_norm = [str(v).strip().lower() for v in self.valid_values]

    def _normalize(self, val: Any) -> str | None:
        if val is None:
            return None
        s = str(val).strip()
        if self.treat_null_as_none and s.lower() in self.NULL_VALUES:
            return None
        return s if self.case_sensitive else s.lower()

    def _find_best(self, val_norm: str) -> tuple[str | None, float]:
        best_match, best_score = None, 0.0
        for i, valid_norm in enumerate(self.valid_values_norm):
            sim = SequenceMatcher(None, val_norm, valid_norm).ratio()
            if sim > best_score:
                best_score = sim
                best_match = self.valid_values[i]
        return best_match, best_score

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        g_norm = self._normalize(gold)
        p_norm = self._normalize(pred)

        if g_norm is None and p_norm is None:
            return 1.0, f"✓ {self.field_name}: Both null"
        if g_norm is None:
            return 0.0, f"✗ {self.field_name}: Gold null, pred='{pred}'"
        if p_norm is None:
            return 0.0, f"✗ {self.field_name}: Pred null, gold='{gold}'"

        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: {pred}"

        g_valid = g_norm in self.valid_values_norm
        p_valid = p_norm in self.valid_values_norm

        if g_valid and p_valid:
            return 0.0, f"✗ {self.field_name}: '{gold}' vs '{pred}'"

        best_match, sim = self._find_best(p_norm)

        if sim >= self.fuzzy_threshold:
            score = 0.9 if sim >= 0.95 else 0.7
            return score, f"✗ {self.field_name}: '{pred}' → '{best_match}' ({sim:.2f})"

        valid_str = ", ".join(f"'{v}'" for v in self.valid_values[:3])
        if len(self.valid_values) > 3:
            valid_str += "..."

        return 0.0, f"✗ {self.field_name}: Invalid '{pred}' ({valid_str})"


class AddressMatcher(BaseMatcher):
    """Match addresses."""

    def __init__(self, field_name: str, **kwargs):
        kwargs.setdefault("sim_weight", 0.85)
        super().__init__(field_name, **kwargs)

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
        if sim >= 0.9:
            score = 0.95
        else:
            score = self.exact_weight * 0.0 + self.sim_weight * sim

        return score, f"✗ {self.field_name}: {sim:.2f} sim\n  '{gold}' → '{pred}'"


class JSONMatcher(BaseMatcher):
    """Match JSON lists."""

    def __init__(
        self,
        field_name: str,
        judge_lm: dspy.LM | None = None,
        programmatic_weight: float = 0.3,
        llm_weight: float = 0.7,
        **kwargs,
    ):
        super().__init__(field_name, **kwargs)
        self.judge_lm = judge_lm
        self.programmatic_weight = programmatic_weight
        self.llm_weight = llm_weight

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        from scripts.json_metrics import hybrid_json_score, json_match_score

        if self.judge_lm:
            score, details = hybrid_json_score(
                gold,
                pred,
                judge_lm=self.judge_lm,
                programmatic_weight=self.programmatic_weight,
                llm_weight=self.llm_weight,
            )
            prog_det = details.get("programmatic_details", {})
            if "error" in prog_det:
                return score, f"⚠ {self.field_name}: {prog_det['error']}"

            n_matched = prog_det.get("n_matched", 0)
            feedback = (
                f"{'✓' if score >= 0.95 else '✗'} {self.field_name}:\n"
                f"  Prog: {details['programmatic_score']:.2f}, "
                f"LLM: {details['llm_score']:.2f}, Final: {score:.2f}\n"
                f"  {prog_det['n_gold']}g/{prog_det['n_pred']}p/{n_matched}m"
            )
        else:
            score, details = json_match_score(gold, pred)
            if "error" in details:
                return score, f"⚠ {self.field_name}: {details['error']}"

            n_matched = details.get("n_matched", 0)
            feedback = (
                f"{'✓' if score >= 0.95 else '✗'} {self.field_name}:\n"
                f"  Score: {score:.2f}\n"
                f"  {details['n_gold']}g/{details['n_pred']}p/{n_matched}m"
            )

        return score, feedback


class MatcherRegistry:
    """Registry for matcher types."""

    _matchers = {
        "phone": PhoneMatcher,
        "date": DateMatcher,
        "number": NumberMatcher,
        "float": FloatMatcher,
        "string": StringMatcher,
        "boolean": BooleanMatcher,
        "boolean_string": BooleanMatcher,
        "enum": EnumMatcher,
        "address": AddressMatcher,
        "json": JSONMatcher,
    }

    @classmethod
    def create(cls, field_name: str, field_type: str, **kwargs) -> BaseMatcher:
        """Create matcher for field type."""
        matcher_cls = cls._matchers.get(field_type.lower(), StringMatcher)
        return matcher_cls(field_name, **kwargs)

    @classmethod
    def register(cls, field_type: str, matcher_cls: type):
        """Register custom matcher."""
        cls._matchers[field_type.lower()] = matcher_cls
