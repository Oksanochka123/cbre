"""Date matcher."""

from datetime import datetime

from scripts.matchers.base_matcher import BaseMatcher


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
